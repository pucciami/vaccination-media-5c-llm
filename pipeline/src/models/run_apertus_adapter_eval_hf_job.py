import argparse
import json
import os
import re
import time
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


CONSTRUCTS = [
    "confidence",
    "complacency",
    "constraints",
    "calculation",
    "collective_responsibility",
]
CONSTRUCT_TO_VALENCE_FIELD = {
    "confidence": "valence_confidence_d1",
    "complacency": "valence_complacency_d1",
    "constraints": "valence_constraints_d1",
    "calculation": "valence_calculation_d1",
    "collective_responsibility": "valence_collective_d1",
}
ALLOWED_DISEASES = {"covid19", "influenza", "measles", "hpv", "unspecified", "other"}
ALLOWED_VALENCES = {-1, 0, 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run locked batch inference and evaluation for Apertus base, optionally plus "
            "a private PEFT adapter, inside an HF Job."
        )
    )
    parser.add_argument("--base-model", default="swiss-ai/Apertus-8B-Instruct-2509")
    parser.add_argument("--adapter-repo", default=None)
    parser.add_argument("--tokenizer-repo", default=None)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--gold-csv", required=True)
    parser.add_argument("--system-prompt-path", required=True)
    parser.add_argument("--evaluator-script", required=True)
    parser.add_argument("--output-dir", default="/job-output/apertus_adapter_eval")
    parser.add_argument("--private-output-repo", required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--retry-max-new-tokens", default="1024,2048")
    parser.add_argument("--generation-max-time", type=float, default=0.0)
    parser.add_argument(
        "--assistant-json-prefill-mode",
        choices=["none", "relevant_hierarchy_prefix"],
        default="none",
        help="Optional schema-control prefix for difficult forced-relevant hierarchy rows.",
    )
    parser.add_argument("--output-prefix", default="gold100_apertus_lora_clean_v2_al_v1_stage2_forced")
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the local evaluator. Use for subset repair jobs whose predictions are merged later.",
    )
    parser.add_argument(
        "--force-relevant-output",
        action="store_true",
        help=(
            "Force relevance='relevant' before hierarchy normalization. Use only for "
            "stage-2 extractor runs where an external stage-1 gate controls relevance."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Pass enable_thinking=True to the chat template so Apertus emits a reasoning "
            "trace before the JSON object (the trace is stripped by strip_think_blocks). "
            "Default off preserves bit-identical behaviour of all previously locked runs. "
            "Side-experiment lever for the reasoning-mode parity check against Qwen3-32B."
        ),
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help=(
            "Use stochastic sampling instead of greedy decoding. Off by default to preserve "
            "bit-identical behaviour of the locked greedy runs. Enable for the vendor-recommended "
            "decoding robustness check (Apertus model card: temperature 0.8, top_p 0.9)."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (only used with --do-sample).")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling (only used with --do-sample).")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible sampling.")
    return parser.parse_args()


def resolve_assistant_json_prefill(mode: str) -> str:
    if mode == "relevant_hierarchy_prefix":
        return '{"relevance": "relevant", '
    return ""


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for idx, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start : idx + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        start = None
                        continue
                    if isinstance(parsed, dict):
                        objects.append(parsed)
                    start = None
    return objects


def parse_prediction(raw_text: str) -> dict[str, Any]:
    text = strip_think_blocks(raw_text.strip())
    preferred_fields = {"relevance", "disease_1", "has_relevant_5c", "c5_presence_d1"}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            categorization = parsed.get("categorization")
            if isinstance(categorization, dict) and (preferred_fields & set(categorization)):
                return categorization
            return parsed
    except json.JSONDecodeError:
        pass

    for parsed in reversed(extract_json_objects(text)):
        if preferred_fields & set(parsed):
            return parsed
        categorization = parsed.get("categorization")
        if isinstance(categorization, dict) and (preferred_fields & set(categorization)):
            return categorization

    raise ValueError("Could not extract a full-hierarchy JSON object from model output.")


def retry_budgets(raw: str, base_budget: int) -> list[int]:
    budgets = [base_budget]
    for part in (raw or "").split(","):
        part = part.strip()
        if part:
            value = int(part)
            if value > 0 and value not in budgets:
                budgets.append(value)
    return budgets


# --- Stage-2 output validation -------------------------------------------------------------
# The same block is kept in each Stage-2 runner (run_qwen_longitudinal_stage2_hf_job.py,
# run_qwen_merged_eval_hf_job.py, run_apertus_adapter_eval_hf_job.py), because each runner is
# submitted to a job as a single file. An output is accepted only if it satisfies the output
# schema and the hierarchy rules of the extractor prompt; otherwise it is rejected, retried with
# the next token budget, and counted. Only format-level repairs are made (for example a valence
# written as the string "1"), and each one is recorded with the prediction; no label is ever
# filled in or substituted.

REQUIRED_FIELDS = [
    "relevance",
    "disease_1",
    "has_relevant_5c",
    "c5_presence_d1",
    *CONSTRUCT_TO_VALENCE_FIELD.values(),
]
NUMERIC_VALENCE_STRINGS = {"-1": -1, "0": 0, "1": 1, "+1": 1}


class Stage2OutputError(ValueError):
    """Model output that does not satisfy the Stage-2 output schema."""


def parse_prediction_checked(raw_text: str) -> tuple[dict[str, Any], list[str]]:
    """Parse the final JSON object; return it with the list of format repairs applied."""
    repairs: list[str] = []
    if "<think>" in raw_text and "</think>" not in raw_text:
        raise Stage2OutputError("reasoning trace not closed (output truncated before the JSON object)")
    text = strip_think_blocks(raw_text.strip())
    try:
        direct = json.loads(text)
    except json.JSONDecodeError:
        direct = None
    try:
        parsed = parse_prediction(raw_text)
    except ValueError as exc:
        raise Stage2OutputError(str(exc)) from exc
    if not isinstance(direct, dict):
        repairs.append("JSON object taken from surrounding text")
    elif parsed != direct:
        repairs.append("JSON object taken from a nested 'categorization' field")
    return parsed, repairs


def validate_stage2_output(
    prediction: dict[str, Any], *, force_relevant: bool
) -> tuple[dict[str, Any], list[str]]:
    """Check one parsed output against the schema and hierarchy rules of the extractor prompt.

    Returns the normalised prediction and the list of format repairs applied. Raises
    Stage2OutputError naming every problem when a required field is missing, disease_1 is
    missing or outside the allowed set, has_relevant_5c is not yes/no, a construct label is
    unknown, a present construct has no valence, or the fields contradict the hierarchy rules.
    """
    problems: list[str] = []
    repairs: list[str] = []
    missing = [field for field in REQUIRED_FIELDS if field not in prediction]
    if missing:
        problems.append("missing required field(s): " + ", ".join(missing))

    relevance = prediction.get("relevance")
    if force_relevant:
        if relevance != "relevant":
            repairs.append(f"relevance {relevance!r} set to 'relevant' (relevance is decided by stage 1)")
        relevance = "relevant"
    elif relevance not in {"relevant", "not_relevant"}:
        problems.append(f"invalid relevance {relevance!r}")

    raw_presence = prediction.get("c5_presence_d1")
    presence: list[Any] = []
    if isinstance(raw_presence, list):
        presence = raw_presence
    elif isinstance(raw_presence, str) and raw_presence.strip().startswith("["):
        try:
            parsed_presence = json.loads(raw_presence)
        except json.JSONDecodeError:
            parsed_presence = None
        if isinstance(parsed_presence, list):
            presence = parsed_presence
            repairs.append("c5_presence_d1 given as a JSON string")
        else:
            problems.append(f"c5_presence_d1 is not a list: {raw_presence!r}")
    elif raw_presence is not None or "c5_presence_d1" in prediction:
        problems.append(f"c5_presence_d1 is not a list: {raw_presence!r}")
    unknown = [item for item in presence if item not in CONSTRUCT_TO_VALENCE_FIELD]
    if unknown:
        problems.append(f"unknown construct label(s) in c5_presence_d1: {unknown}")
    if len(set(map(str, presence))) != len(presence):
        repairs.append("duplicate construct label removed from c5_presence_d1")
    present = sorted({item for item in presence if item in CONSTRUCT_TO_VALENCE_FIELD}, key=CONSTRUCTS.index)

    valences: dict[str, int | None] = {}
    for construct, field in CONSTRUCT_TO_VALENCE_FIELD.items():
        value = prediction.get(field)
        valences[construct] = None
        if value is None:
            continue
        if isinstance(value, bool):
            problems.append(f"invalid {field} {value!r}")
        elif isinstance(value, int) and value in ALLOWED_VALENCES:
            valences[construct] = value
        elif isinstance(value, float) and value.is_integer() and int(value) in ALLOWED_VALENCES:
            valences[construct] = int(value)
            repairs.append(f"{field} given as a number with decimals")
        elif isinstance(value, str) and value.strip() in NUMERIC_VALENCE_STRINGS:
            valences[construct] = NUMERIC_VALENCE_STRINGS[value.strip()]
            repairs.append(f"{field} given as a string")
        elif isinstance(value, str) and value.strip() in {"", "null", "None"}:
            repairs.append(f"{field} given as an empty string (read as null)")
        else:
            problems.append(f"invalid {field} {value!r}")

    normalized = dict(prediction)
    normalized["relevance"] = relevance
    if relevance == "not_relevant":
        extra = [field for field in ["disease_1", "has_relevant_5c"] if prediction.get(field) is not None]
        if present or any(value is not None for value in valences.values()) or extra:
            repairs.append("disease and 5C fields cleared for an article the model judged not relevant")
        normalized["disease_1"] = None
        normalized["has_relevant_5c"] = None
        normalized["c5_presence_d1"] = []
        for field in CONSTRUCT_TO_VALENCE_FIELD.values():
            normalized[field] = None
    else:
        disease_1 = prediction.get("disease_1")
        if "disease_1" in prediction and disease_1 not in ALLOWED_DISEASES:
            problems.append(f"disease_1 not in the allowed set: {disease_1!r}")
        has_relevant_5c = prediction.get("has_relevant_5c")
        if "has_relevant_5c" in prediction and has_relevant_5c not in {"yes", "no"}:
            problems.append(f"invalid has_relevant_5c {has_relevant_5c!r}")
        elif has_relevant_5c == "no":
            if present:
                problems.append("has_relevant_5c is no but c5_presence_d1 lists constructs")
            if any(value is not None for value in valences.values()):
                problems.append("has_relevant_5c is no but a valence is given")
        elif has_relevant_5c == "yes":
            if not present:
                problems.append("has_relevant_5c is yes but c5_presence_d1 is empty")
            for construct in CONSTRUCTS:
                if construct in present and valences[construct] is None:
                    problems.append(f"no valence for present construct {construct}")
                if construct not in present and valences[construct] is not None:
                    problems.append(f"valence given for absent construct {construct}")
        normalized["c5_presence_d1"] = present
        for construct, field in CONSTRUCT_TO_VALENCE_FIELD.items():
            normalized[field] = valences[construct]

    if problems:
        raise Stage2OutputError("; ".join(problems))
    return normalized, repairs


def summarise_validation(
    n_rows: int,
    n_accepted: int,
    n_failed: int,
    n_invalid_attempts: int,
    n_generation_errors: int,
    n_accepted_after_retry: int,
    n_accepted_with_repairs: int,
    problem_counts: dict[str, int],
) -> dict[str, Any]:
    """Counts reported in the run manifest."""
    return {
        "n_rows": int(n_rows),
        "n_accepted": int(n_accepted),
        "n_failed_after_all_budgets": int(n_failed),
        "n_invalid_output_attempts": int(n_invalid_attempts),
        "n_generation_error_attempts": int(n_generation_errors),
        "n_accepted_after_retry": int(n_accepted_after_retry),
        "n_accepted_with_format_repairs": int(n_accepted_with_repairs),
        "invalid_output_problem_counts": dict(sorted(problem_counts.items())),
    }


def problem_keys(message: str) -> list[str]:
    """Short keys for counting problems by type (the text before any quoted value)."""
    keys = []
    for part in message.split("; "):
        key = part.split("'")[0].split(":")[0].split("[")[0].strip()
        keys.append(key or part)
    return keys
# --- end of Stage-2 output validation ---------------------------------------------------------


def main() -> None:
    args = parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"Missing Hugging Face token in environment variable: {args.token_env}")

    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "gold100_apertus_lora_predictions.jsonl"
    errors_path = output_dir / "gold100_apertus_lora_errors.jsonl"
    summary_dir = output_dir / "eval_report"
    summary_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = Path(args.system_prompt_path).read_text(encoding="utf-8").strip()
    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
    assistant_json_prefill = resolve_assistant_json_prefill(args.assistant_json_prefill_mode)

    tokenizer_repo = args.tokenizer_repo or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, trust_remote_code=True, token=token)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=token,
    )
    if args.adapter_repo:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, args.adapter_repo, token=token)
        generation_mode = "transformers_base_plus_adapter"
    else:
        model = base_model
        generation_mode = "transformers_base_only"
    model.eval()

    template_kwargs: dict[str, Any] = {}
    if args.enable_thinking:
        template_kwargs["enable_thinking"] = True

    if args.do_sample:
        from transformers import set_seed

        set_seed(args.seed)

    success_count = 0
    error_count = 0
    n_invalid_attempts = 0
    n_generation_errors = 0
    n_accepted_after_retry = 0
    n_accepted_with_repairs = 0
    problem_counts: dict[str, int] = {}
    budgets = retry_budgets(args.retry_max_new_tokens, args.max_new_tokens)
    with predictions_path.open("w", encoding="utf-8") as ok_handle, errors_path.open("w", encoding="utf-8") as err_handle:
        for idx, row in enumerate(df.to_dict(orient="records"), start=1):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Article:\n{row['text']}"},
            ]
            last_error: Exception | None = None
            last_raw = ""
            row_started = time.time()
            attempt_errors: list[str] = []
            for attempt, budget in enumerate(budgets, start=1):
                try:
                    rendered_prompt = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                        **template_kwargs,
                    )
                    if assistant_json_prefill:
                        rendered_prompt = rendered_prompt + assistant_json_prefill
                    inputs = tokenizer(rendered_prompt, return_tensors="pt").to(model.device)
                    input_len = inputs["input_ids"].shape[-1]
                    generation_kwargs: dict[str, Any] = {
                        "max_new_tokens": budget,
                        "do_sample": bool(args.do_sample),
                        "pad_token_id": tokenizer.eos_token_id,
                    }
                    if args.do_sample:
                        generation_kwargs["temperature"] = args.temperature
                        generation_kwargs["top_p"] = args.top_p
                    if args.generation_max_time and args.generation_max_time > 0:
                        generation_kwargs["max_time"] = args.generation_max_time
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            **generation_kwargs,
                        )
                    generated = outputs[0][input_len:]
                    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
                    raw_text = (
                        f"{assistant_json_prefill}{generated_text}"
                        if assistant_json_prefill
                        else generated_text
                    )
                    last_raw = raw_text
                    parsed, repairs = parse_prediction_checked(raw_text)
                    prediction, validation_repairs = validate_stage2_output(parsed, force_relevant=args.force_relevant_output)
                    repairs += validation_repairs
                    prediction["doc_id"] = row["doc_id"]
                    prediction["used_max_new_tokens"] = budget
                    prediction["attempt_number"] = attempt
                    prediction["n_attempts"] = attempt
                    prediction["n_retries"] = attempt - 1
                    prediction["row_latency_seconds"] = round(time.time() - row_started, 4)
                    prediction["validation_repairs"] = repairs
                    prediction["failed_attempts"] = list(attempt_errors)
                    prediction["raw_output"] = raw_text
                    ok_handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    ok_handle.flush()
                    success_count += 1
                    n_accepted_after_retry += int(attempt > 1)
                    n_accepted_with_repairs += int(bool(repairs))
                    print(f"Saved prediction {idx}/{len(df)} for doc_id={row['doc_id']}", flush=True)
                    break
                except Stage2OutputError as exc:
                    last_error = exc
                    n_invalid_attempts += 1
                    for key in problem_keys(str(exc)):
                        problem_counts[key] = problem_counts.get(key, 0) + 1
                    attempt_errors.append(f"max_new_tokens {budget}: invalid output: {exc}")
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    n_generation_errors += 1
                    attempt_errors.append(f"max_new_tokens {budget}: generation error: {exc}")
            else:
                error = {
                    "doc_id": row["doc_id"],
                    "error": str(last_error),
                    "attempt_errors": attempt_errors,
                    "raw_text_excerpt": last_raw[:2000],
                    "used_retry_budgets": budgets,
                    "n_attempts": len(budgets),
                    "n_retries": max(len(budgets) - 1, 0),
                    "row_latency_seconds": round(time.time() - row_started, 4),
                }
                err_handle.write(json.dumps(error, ensure_ascii=False) + "\n")
                err_handle.flush()
                error_count += 1
                print(f"Failed prediction {idx}/{len(df)} for doc_id={row['doc_id']}: {error['error']}", flush=True)

    if not error_count and not args.skip_eval:
        subprocess.run(
            [
                sys.executable,
                str(Path(args.evaluator_script).resolve()),
                "--gold",
                str(Path(args.gold_csv).resolve()),
                "--predictions",
                str(predictions_path),
                "--output-dir",
                str(summary_dir),
                "--output-prefix",
                args.output_prefix,
            ],
            check=True,
        )

    manifest = {
        "base_model": args.base_model,
        "adapter_repo": args.adapter_repo,
        "tokenizer_repo": tokenizer_repo,
        "input_csv": str(args.input_csv),
        "gold_csv": str(args.gold_csv),
        "system_prompt_path": str(args.system_prompt_path),
        "n_success": success_count,
        "n_errors": error_count,
        "output_validation": summarise_validation(
            len(df), success_count, error_count, n_invalid_attempts, n_generation_errors,
            n_accepted_after_retry, n_accepted_with_repairs, problem_counts,
        ),
        "force_relevant_output": bool(args.force_relevant_output),
        "enable_thinking": bool(args.enable_thinking),
        "decoding": {
            "do_sample": bool(args.do_sample),
            "temperature": args.temperature if args.do_sample else 0.0,
            "top_p": args.top_p if args.do_sample else 1.0,
            "seed": args.seed if args.do_sample else None,
        },
        "skip_eval": bool(args.skip_eval),
        "assistant_json_prefill_mode": args.assistant_json_prefill_mode,
        "assistant_json_prefill": assistant_json_prefill,
        "generation_mode": generation_mode,
        "max_new_tokens": args.max_new_tokens,
        "retry_max_new_tokens": budgets[1:],
        "generation_max_time": args.generation_max_time,
        "predictions": str(predictions_path),
        "errors": str(errors_path),
        "eval_report": str(summary_dir),
    }
    (output_dir / "apertus_adapter_eval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.private_output_repo,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=args.private_output_repo,
        repo_type="dataset",
        commit_message="Upload Apertus adapter locked evaluation outputs",
    )

    print(f"Uploaded evaluation outputs to private dataset repo: {args.private_output_repo}")

    print(f"Output validation: {json.dumps(manifest['output_validation'], ensure_ascii=False)}")

    if error_count:
        raise RuntimeError(f"Apertus stage-2 evaluation produced {error_count} parse/generation errors.")


if __name__ == "__main__":
    main()
