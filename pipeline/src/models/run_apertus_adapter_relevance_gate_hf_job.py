import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ALLOWED_REASON_CODES = {
    "explicit_vaccination_content",
    "explicit_vaccination_discussion",
    "vaccination_policy_logistics_or_programme",
    "vaccine_preventable_disease_with_vaccination_angle",
    "disease_only_no_vaccination",
    "public_health_general_no_vaccination",
    "pandemic_or_institution_context_only",
    "treatment_diagnostics_outbreak_politics_without_vaccine_angle",
    "vaccination_only_passing_or_background_reference",
    "unclear_default_not_relevant",
    "unclear_default_relevant",
    "no_vaccination_lexical_signal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run relevance-only gating with Apertus base, optionally plus a private PEFT adapter, inside an HF Job."
    )
    parser.add_argument("--base-model", default="swiss-ai/Apertus-8B-Instruct-2509")
    parser.add_argument("--adapter-repo", default=None)
    parser.add_argument("--tokenizer-repo", default=None)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--prompt-path", required=True)
    parser.add_argument("--output-dir", default="/job-output/apertus_adapter_relevance_gate")
    parser.add_argument("--private-output-repo", required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="doc_id")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--retry-max-new-tokens", default="1024,2048")
    parser.add_argument("--run-name", default="apertus_adapter_relevance_gold100_clean_v2_al_v1")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Pass enable_thinking=True to the chat template so Apertus deliberates before the "
            "relevance JSON (trace stripped by strip_think_blocks / ignored by the JSON extractor). "
            "Default off preserves bit-identical behaviour of the locked gate runs."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_prompt(prompt_template: str, article_text: str) -> str:
    return prompt_template.replace("{{ARTICLE_TEXT}}", str(article_text).strip())


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
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            categorization = parsed.get("categorization")
            if isinstance(categorization, dict) and "relevance" in categorization:
                return categorization
            return parsed
    except json.JSONDecodeError:
        pass

    for parsed in extract_json_objects(text):
        if "relevance" in parsed:
            return parsed
        categorization = parsed.get("categorization")
        if isinstance(categorization, dict) and "relevance" in categorization:
            return categorization

    label_match = re.search(r'"relevance"\s*:\s*"([^"]+)"', text)
    if label_match:
        reason_match = re.search(r'"reason_code"\s*:\s*"([^"]+)"', text)
        return {
            "relevance": label_match.group(1),
            "reason_code": reason_match.group(1) if reason_match else None,
        }
    raise ValueError("Could not extract a relevance JSON object from model output.")


def normalize_relevance_label(value: Any) -> str:
    text = str(value).strip().lower().replace("-", " ").replace("_", " ")
    text = " ".join(text.split())
    mapping = {
        "relevant": "relevant",
        "not relevant": "not_relevant",
        "notrelevant": "not_relevant",
        "irrelevant": "not_relevant",
    }
    if text in mapping:
        return mapping[text]
    raise ValueError(f"Invalid relevance value: {value}")


def normalize_prediction(prediction: dict[str, Any]) -> dict[str, str]:
    relevance = normalize_relevance_label(prediction.get("relevance"))
    reason_code = prediction.get("reason_code")
    if reason_code not in ALLOWED_REASON_CODES:
        reason_code = "explicit_vaccination_discussion" if relevance == "relevant" else "unclear_default_not_relevant"
    return {"relevance": relevance, "reason_code": reason_code}


def retry_budgets(raw: str, base_budget: int) -> list[int]:
    budgets = [base_budget]
    for part in (raw or "").split(","):
        part = part.strip()
        if part:
            value = int(part)
            if value > 0 and value not in budgets:
                budgets.append(value)
    return budgets


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
    predictions_path = output_dir / f"{args.run_name}.jsonl"
    errors_path = output_dir / f"{args.run_name}_errors.jsonl"
    manifest_path = output_dir / f"{args.run_name}_manifest.json"

    prompt_path = Path(args.prompt_path).resolve()
    prompt_template = prompt_path.read_text(encoding="utf-8")
    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)

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

    budgets = retry_budgets(args.retry_max_new_tokens, args.max_new_tokens)
    started = time.time()
    success_count = 0
    error_count = 0

    with predictions_path.open("w", encoding="utf-8") as ok_handle, errors_path.open("w", encoding="utf-8") as err_handle:
        for idx, row in enumerate(df.to_dict(orient="records"), start=1):
            doc_id = str(row[args.id_column])
            prompt = build_prompt(prompt_template, str(row[args.text_column]))
            messages = [{"role": "user", "content": prompt}]
            last_error: Exception | None = None
            last_raw = ""
            row_started = time.time()

            for attempt, budget in enumerate(budgets, start=1):
                try:
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
                        **template_kwargs,
                    ).to(model.device)
                    input_len = inputs["input_ids"].shape[-1]
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=budget,
                            do_sample=False,
                        )
                    generated = outputs[0][input_len:]
                    raw_text = tokenizer.decode(generated, skip_special_tokens=True)
                    last_raw = raw_text
                    normalized = normalize_prediction(parse_prediction(raw_text))
                    normalized["doc_id"] = doc_id
                    normalized["used_max_new_tokens"] = budget
                    normalized["attempt_number"] = attempt
                    normalized["n_attempts"] = attempt
                    normalized["n_retries"] = attempt - 1
                    normalized["row_latency_seconds"] = round(time.time() - row_started, 4)
                    normalized["raw_output"] = raw_text
                    ok_handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    ok_handle.flush()
                    success_count += 1
                    print(f"Saved prediction {idx}/{len(df)} for doc_id={doc_id}", flush=True)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
            else:
                error = {
                    "doc_id": doc_id,
                    "error": str(last_error),
                    "raw_text_excerpt": last_raw[:1000],
                    "used_retry_budgets": budgets,
                    "n_attempts": len(budgets),
                    "n_retries": max(len(budgets) - 1, 0),
                    "row_latency_seconds": round(time.time() - row_started, 4),
                }
                err_handle.write(json.dumps(error, ensure_ascii=False) + "\n")
                err_handle.flush()
                error_count += 1
                print(f"Failed prediction {idx}/{len(df)} for doc_id={doc_id}: {error['error']}", flush=True)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "base_model": args.base_model,
        "adapter_repo": args.adapter_repo,
        "tokenizer_repo": tokenizer_repo,
        "input_csv": str(args.input_csv),
        "prompt_path": str(prompt_path),
        "prompt_sha256": sha256_file(prompt_path),
        "text_column": args.text_column,
        "id_column": args.id_column,
        "n_input_rows": int(len(df)),
        "n_success": int(success_count),
        "n_errors": int(error_count),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "retry_max_new_tokens": budgets[1:],
            "mode": generation_mode,
            "enable_thinking": bool(args.enable_thinking),
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "outputs": {
            "predictions": str(predictions_path),
            "errors": str(errors_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.private_output_repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=args.private_output_repo,
        repo_type="dataset",
        commit_message="Upload Apertus adapter relevance-gate outputs",
    )
    print(f"Uploaded adapted Apertus relevance-gate outputs to private dataset repo: {args.private_output_repo}")


if __name__ == "__main__":
    main()
