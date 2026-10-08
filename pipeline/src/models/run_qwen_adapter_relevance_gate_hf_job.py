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

HIERARCHY_FIELDS = {
    "disease_1",
    "has_relevant_5c",
    "c5_presence_d1",
    "valence_confidence_d1",
    "valence_complacency_d1",
    "valence_constraints_d1",
    "valence_calculation_d1",
    "valence_collective_d1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run relevance-only gating with Qwen base plus private PEFT adapter inside an HF Job."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-repo", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--prompt-path", required=True)
    parser.add_argument("--output-dir", default="/job-output/qwen_adapter_relevance_gate")
    parser.add_argument("--private-output-repo", required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="doc_id")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--retry-max-new-tokens", default="0")
    parser.add_argument(
        "--generation-max-time",
        type=float,
        default=0.0,
        help=(
            "Optional per-attempt wall-clock soft limit passed to transformers.generate. "
            "Use 0 to disable. This is an operational guard against pathological generations."
        ),
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-errors", type=int, default=10)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--prompt-layout",
        choices=["single_user", "system_user"],
        default="single_user",
        help=(
            "Chat layout for the same prompt content. single_user preserves the original runner. "
            "system_user moves task instructions before the Article block into the system role and "
            "keeps the article in the user role, matching the SFT adapter chat shape more closely."
        ),
    )
    parser.add_argument(
        "--assistant-json-prefill",
        default="",
        help=(
            "Optional assistant-side text appended after the generation prompt before decoding. "
            "The prefix is prepended back to raw_output for parsing, and is intended only as a "
            "schema-control guard for relevance-only JSON."
        ),
    )
    parser.add_argument(
        "--assistant-json-prefill-mode",
        choices=["none", "relevance_key", "relevance_string"],
        default="none",
        help="Named schema-control prefix mode that avoids shell quoting issues for JSON fragments.",
    )
    parser.add_argument(
        "--allow-hierarchy-relevance-salvage",
        action="store_true",
        help=(
            "Diagnostic only. If the adapted model emits full-hierarchy JSON "
            "without a relevance field, infer the implied relevance instead of "
            "counting the row as a parse failure."
        ),
    )
    parser.add_argument("--run-name", default="qwen_adapter_relevance_gate")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_prompt(prompt_template: str, article_text: str) -> str:
    return prompt_template.replace("{{ARTICLE_TEXT}}", str(article_text).strip())


def build_messages(prompt_template: str, article_text: str, *, prompt_layout: str) -> list[dict[str, str]]:
    article = str(article_text).strip()
    if prompt_layout == "single_user" or "{{ARTICLE_TEXT}}" not in prompt_template:
        return [{"role": "user", "content": build_prompt(prompt_template, article)}]

    prefix, suffix = prompt_template.split("{{ARTICLE_TEXT}}", 1)
    return [
        {"role": "system", "content": prefix.strip()},
        {"role": "user", "content": f"Article:\n\n{article}{suffix}".strip()},
    ]


def resolve_assistant_json_prefill(raw_prefill: str, mode: str) -> str:
    if mode == "relevance_key":
        return '{"relevance": '
    if mode == "relevance_string":
        return '{"relevance": "'
    return raw_prefill


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


def has_meaningful_hierarchy_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() not in {"", "null", "none", "None", "NULL", "[]"}
    if isinstance(value, list):
        return len(value) > 0
    return True


def hierarchy_to_relevance(parsed: dict[str, Any]) -> dict[str, Any] | None:
    """Salvage adapted full-hierarchy JSON as the relevance decision it implies."""
    if not (HIERARCHY_FIELDS & set(parsed)):
        return None

    if any(has_meaningful_hierarchy_value(parsed.get(field)) for field in HIERARCHY_FIELDS):
        return {
            "relevance": "relevant",
            "reason_code": "explicit_vaccination_discussion",
            "parser_note": "inferred_relevant_from_hierarchy_output",
        }

    return {
        "relevance": "not_relevant",
        "reason_code": "unclear_default_not_relevant",
        "parser_note": "inferred_not_relevant_from_empty_hierarchy_output",
    }


def parse_prediction(raw_text: str, *, allow_hierarchy_salvage: bool = False) -> dict[str, Any]:
    text = strip_think_blocks(raw_text.strip())
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            categorization = parsed.get("categorization")
            if isinstance(categorization, dict) and "relevance" in categorization:
                return categorization
            if "relevance" in parsed:
                return parsed
            if allow_hierarchy_salvage:
                hierarchy_prediction = hierarchy_to_relevance(parsed)
                if hierarchy_prediction:
                    return hierarchy_prediction
    except json.JSONDecodeError:
        pass

    for parsed in extract_json_objects(text):
        if "relevance" in parsed:
            return parsed
        categorization = parsed.get("categorization")
        if isinstance(categorization, dict) and "relevance" in categorization:
            return categorization
        if allow_hierarchy_salvage:
            hierarchy_prediction = hierarchy_to_relevance(parsed)
            if hierarchy_prediction:
                return hierarchy_prediction

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
    normalized = {"relevance": relevance, "reason_code": reason_code}
    parser_note = prediction.get("parser_note")
    if parser_note:
        normalized["parser_note"] = str(parser_note)
    return normalized


def retry_budgets(raw: str, base_budget: int) -> list[int]:
    budgets = [base_budget]
    for part in (raw or "").split(","):
        part = part.strip()
        if part:
            value = int(part)
            if value > 0 and value not in budgets:
                budgets.append(value)
    return budgets


def apply_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool,
    assistant_json_prefill: str = "",
) -> tuple[Any, str]:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    else:
        try:
            rendered = tokenizer.apply_chat_template(messages, chat_template_kwargs={"enable_thinking": False}, **kwargs)
        except TypeError:
            rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if assistant_json_prefill:
        rendered = rendered + assistant_json_prefill
    return tokenizer(rendered, return_tensors="pt"), rendered


def chat_template_mode(enable_thinking: bool) -> str:
    if enable_thinking:
        return "qwen_default_thinking_compatible_template"
    return "qwen_disable_thinking_template_kwarg_when_available"


def main() -> None:
    args = parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"Missing Hugging Face token in environment variable: {args.token_env}")

    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{args.run_name}.jsonl"
    errors_path = output_dir / f"{args.run_name}_errors.jsonl"
    manifest_path = output_dir / f"{args.run_name}_manifest.json"

    prompt_path = Path(args.prompt_path).resolve()
    prompt_template = prompt_path.read_text(encoding="utf-8")
    assistant_json_prefill = resolve_assistant_json_prefill(
        args.assistant_json_prefill,
        args.assistant_json_prefill_mode,
    )
    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, token=token)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=token,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_repo, token=token)
    model.eval()

    budgets = retry_budgets(args.retry_max_new_tokens, args.max_new_tokens)
    started = time.time()
    success_count = 0
    error_count = 0
    stopped_early_reason = None

    with predictions_path.open("w", encoding="utf-8") as ok_handle, errors_path.open("w", encoding="utf-8") as err_handle:
        for idx, row in enumerate(df.to_dict(orient="records"), start=1):
            doc_id = str(row[args.id_column])
            messages = build_messages(prompt_template, str(row[args.text_column]), prompt_layout=args.prompt_layout)
            last_error = None
            last_raw = ""
            row_started = time.time()

            for attempt, budget in enumerate(budgets, start=1):
                try:
                    inputs, rendered_prompt = apply_template(
                        tokenizer,
                        messages,
                        enable_thinking=args.enable_thinking,
                        assistant_json_prefill=assistant_json_prefill,
                    )
                    inputs = inputs.to(model.device)
                    input_len = inputs["input_ids"].shape[-1]
                    generation_kwargs: dict[str, Any] = {
                        "max_new_tokens": budget,
                        "do_sample": False,
                        "pad_token_id": tokenizer.eos_token_id,
                    }
                    if args.generation_max_time and args.generation_max_time > 0:
                        generation_kwargs["max_time"] = args.generation_max_time
                    print(
                        f"Starting prediction {idx}/{len(df)} for doc_id={doc_id} "
                        f"attempt={attempt} max_new_tokens={budget} input_tokens={input_len} "
                        f"prompt_chars={len(rendered_prompt)}",
                        flush=True,
                    )
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
                    normalized = normalize_prediction(
                        parse_prediction(
                            raw_text,
                            allow_hierarchy_salvage=args.allow_hierarchy_relevance_salvage,
                        )
                    )
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
                print(
                    f"Failed prediction {idx}/{len(df)} for doc_id={doc_id}: {error['error']} "
                    f"raw_excerpt={error['raw_text_excerpt'][:500]!r}",
                    flush=True,
                )
                if args.max_errors is not None and error_count >= args.max_errors:
                    stopped_early_reason = f"max_errors_reached_{args.max_errors}"
                    print(f"Stopping early: {stopped_early_reason}", flush=True)
                    break

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "base_model": args.base_model,
        "adapter_repo": args.adapter_repo,
        "input_csv": str(args.input_csv),
        "prompt_path": str(prompt_path),
        "prompt_sha256": sha256_file(prompt_path),
        "text_column": args.text_column,
        "id_column": args.id_column,
        "n_input_rows": int(len(df)),
        "n_success": int(success_count),
        "n_errors": int(error_count),
        "stopped_early_reason": stopped_early_reason,
        "max_rows": args.max_rows,
        "max_errors": args.max_errors,
        "enable_thinking": bool(args.enable_thinking),
        "chat_template_mode": chat_template_mode(args.enable_thinking),
        "prompt_layout": args.prompt_layout,
        "assistant_json_prefill": assistant_json_prefill,
        "assistant_json_prefill_mode": args.assistant_json_prefill_mode,
        "allow_hierarchy_relevance_salvage": bool(args.allow_hierarchy_relevance_salvage),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "retry_max_new_tokens": budgets[1:],
            "generation_max_time": args.generation_max_time,
            "mode": "transformers_base_plus_adapter",
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
        commit_message="Upload Qwen adapter relevance-gate outputs",
    )
    print(f"Uploaded adapted relevance-gate outputs to private dataset repo: {args.private_output_repo}")


if __name__ == "__main__":
    main()
