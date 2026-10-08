import argparse
import csv
import json
import os
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the first Apertus media-framing LoRA adapter. "
            "This script is intended for a GPU environment after train adjudication is complete."
        )
    )
    parser.add_argument("--config", default="configs/apertus_5c_lora_media_first_run.json")
    return parser.parse_args()


def require_training_dependencies() -> None:
    missing = []
    for module in ["torch", "datasets", "transformers", "peft"]:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(
            "Missing training dependencies: "
            + ", ".join(missing)
            + ". Install requirements-training.txt in a GPU-capable environment before training."
        )


def validate_privacy_policy(config: dict) -> None:
    privacy = config.get("privacy", {})
    if privacy.get("training_data_visibility") != "private_only":
        raise ValueError(
            "Training config must declare privacy.training_data_visibility='private_only'."
        )
    if privacy.get("model_adapter_visibility") != "private_only":
        raise ValueError(
            "Training config must declare privacy.model_adapter_visibility='private_only'."
        )
    if privacy.get("locked_test_upload_allowed") is not False:
        raise ValueError("Training config must set privacy.locked_test_upload_allowed=false.")

    dataset_paths = [
        str(value)
        for key, value in config.get("dataset", {}).items()
        if key.endswith("_jsonl") or key in {"train_jsonl", "dev_jsonl"}
    ]
    locked_test_pattern = re.compile(r"(gold100|locked_reference|locked[_-]?test|test)", re.I)
    suspicious = [path for path in dataset_paths if locked_test_pattern.search(path)]
    if suspicious:
        raise ValueError(
            "Training dataset config appears to reference locked/test data: "
            + ", ".join(suspicious)
        )

    hub = config.get("hub", {})
    if hub.get("push_to_hub", False) and not hub.get("private", False):
        raise ValueError("Hub upload is enabled, but hub.private is not true.")


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_privacy_policy(config)
    require_training_dependencies()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )
    from torch.nn.utils.rnn import pad_sequence

    # Seed every generator before the base model is loaded and the adapter is created, so that
    # the LoRA initialisation, the data order and dropout all follow config["seed"]. GPU kernels
    # can still be non-deterministic, so retraining is seed-controlled but not guaranteed to give
    # a bit-identical adapter.
    import random

    import numpy as np
    from transformers import set_seed

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = config["dataset"]["train_jsonl"]
    dev_jsonl = config["dataset"]["dev_jsonl"]
    for path in [train_jsonl, dev_jsonl]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing prepared dataset: {path}")

    dataset = load_dataset(
        "json",
        data_files={
            "train": train_jsonl,
            "validation": dev_jsonl,
        },
    )

    hf_token = os.environ.get(config.get("token_env", "HF_TOKEN"))
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        trust_remote_code=True,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if config.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.get("bf16", False) else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=torch.bfloat16 if config.get("bf16", False) else torch.float16,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
        token=hf_token,
    )
    if config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    if config.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)

    continue_from_adapter_repo = config.get("continue_from_adapter_repo")
    if continue_from_adapter_repo:
        model = PeftModel.from_pretrained(
            model,
            continue_from_adapter_repo,
            is_trainable=True,
            token=hf_token,
        )
        print(f"Continuing trainable PEFT adapter from: {continue_from_adapter_repo}")
    else:
        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            target_modules=config["lora"]["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)

    def summarize(values: list[int]) -> dict[str, float | int | None]:
        if not values:
            return {"n": 0, "min": None, "max": None, "mean": None}
        ordered = sorted(values)

        def percentile(p: float) -> int:
            index = round((len(ordered) - 1) * p)
            return int(ordered[index])

        return {
            "n": len(ordered),
            "min": int(ordered[0]),
            "max": int(ordered[-1]),
            "mean": sum(ordered) / len(ordered),
            "median": percentile(0.5),
            "p90": percentile(0.9),
            "p95": percentile(0.95),
        }

    def tokenize_example(example):
        messages = example["messages"]
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not full_text.startswith(prompt_text):
            raise ValueError(
                "Chat template mismatch: prompt text is not a prefix of full training text."
            )

        assistant_text = full_text[len(prompt_text):]
        prompt_ids = tokenizer(
            prompt_text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        assistant_ids = tokenizer(
            assistant_text,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]

        if not assistant_ids:
            raise ValueError("Training example is missing assistant label tokens.")
        if len(assistant_ids) >= config["max_seq_length"]:
            raise ValueError(
                "Assistant label sequence alone exceeds max_seq_length. "
                "Reduce output format length or increase the sequence limit."
            )

        available_prompt_tokens = config["max_seq_length"] - len(assistant_ids)
        kept_prompt_ids = prompt_ids[:available_prompt_tokens]
        input_ids = kept_prompt_ids + assistant_ids
        attention_mask = [1] * len(input_ids)
        labels = ([-100] * len(kept_prompt_ids)) + assistant_ids

        if all(label == -100 for label in labels):
            raise ValueError(
                "A training example was truncated before the assistant label. "
                "Increase max_seq_length or shorten the input article text."
            )

        if len(kept_prompt_ids) < len(prompt_ids):
            original_prompt_tokens = len(prompt_ids)
            kept_prompt_tokens = len(kept_prompt_ids)
            print(
                "Warning: truncated prompt tokens for one example "
                f"from {original_prompt_tokens} to {kept_prompt_tokens} "
                f"to preserve the assistant label within max_seq_length."
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "doc_id": str(example.get("doc_id", "")),
            "sft_split": str(example.get("sft_split", "")),
            "original_prompt_tokens": len(prompt_ids),
            "assistant_tokens": len(assistant_ids),
            "full_tokens_before_cap": len(prompt_ids) + len(assistant_ids),
            "kept_prompt_tokens": len(kept_prompt_ids),
            "dropped_prompt_tokens": len(prompt_ids) - len(kept_prompt_ids),
            "truncated_by_max_seq_length": len(kept_prompt_ids) < len(prompt_ids),
        }

    tokenized_dataset = dataset.map(
        tokenize_example,
        remove_columns=dataset["train"].column_names,
    )

    audit_rows = []
    for split_name, split_dataset in tokenized_dataset.items():
        for item in split_dataset:
            audit_rows.append(
                {
                    "split": split_name,
                    "doc_id": item["doc_id"],
                    "sft_split": item["sft_split"],
                    "original_prompt_tokens": int(item["original_prompt_tokens"]),
                    "assistant_tokens": int(item["assistant_tokens"]),
                    "full_tokens_before_cap": int(item["full_tokens_before_cap"]),
                    "kept_prompt_tokens": int(item["kept_prompt_tokens"]),
                    "dropped_prompt_tokens": int(item["dropped_prompt_tokens"]),
                    "truncated_by_max_seq_length": bool(item["truncated_by_max_seq_length"]),
                }
            )

    audit_csv_path = output_dir / "tokenization_audit.csv"
    audit_json_path = output_dir / "tokenization_audit_summary.json"
    with audit_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    truncated_rows = [row for row in audit_rows if row["truncated_by_max_seq_length"]]
    audit_summary = {
        "max_seq_length": int(config["max_seq_length"]),
        "n_examples": len(audit_rows),
        "n_truncated_by_max_seq_length": len(truncated_rows),
        "truncated_doc_ids": [row["doc_id"] for row in truncated_rows],
        "prompt_tokens": summarize([row["original_prompt_tokens"] for row in audit_rows]),
        "assistant_tokens": summarize([row["assistant_tokens"] for row in audit_rows]),
        "full_tokens_before_cap": summarize([row["full_tokens_before_cap"] for row in audit_rows]),
        "dropped_prompt_tokens_among_truncated": summarize(
            [row["dropped_prompt_tokens"] for row in truncated_rows]
        ),
    }
    audit_json_path.write_text(
        json.dumps(audit_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if config.get("fail_on_prompt_truncation", False) and truncated_rows:
        raise ValueError(
            "Prompt truncation detected despite fail_on_prompt_truncation=true. "
            f"Increase max_seq_length or use a documented packer. See {audit_csv_path}."
        )

    def data_collator(features):
        input_ids = [
            torch.tensor(feature["input_ids"], dtype=torch.long)
            for feature in features
        ]
        attention_mask = [
            torch.tensor(feature["attention_mask"], dtype=torch.long)
            for feature in features
        ]
        labels = [
            torch.tensor(feature["labels"], dtype=torch.long)
            for feature in features
        ]
        return {
            "input_ids": pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
        }

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        save_strategy=config["save_strategy"],
        eval_strategy="epoch",
        bf16=config.get("bf16", False),
        seed=config["seed"],
        optim=config.get("optim", "adamw_torch"),
        max_steps=config.get("max_steps", -1),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))
    Path(output_dir / "training_config_used.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
