from __future__ import annotations

import argparse
import ast
import json
import os
import time
from datetime import datetime, timezone
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
        description="Run checkpointed stage-2 active-learning inference for the merged private Qwen media model."
    )
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--tokenizer-repo", default=None)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--system-prompt-path", required=True)
    parser.add_argument("--output-dir", default="/job-output/qwen_active_learning_stage2")
    parser.add_argument("--private-output-repo", required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--text-column", default="model_text")
    parser.add_argument("--id-column", default="doc_id")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--retry-max-new-tokens", default="512")
    parser.add_argument("--run-name", default="qwen_active_learning_stage2_v1")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def normalize_presence_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text in {"", "null", "None"}:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(text)
            if not isinstance(parsed, list):
                raise ValueError(f"c5_presence_d1 string did not parse to a list: {value}")
            return parsed
        return [text]
    raise ValueError(f"c5_presence_d1 must be list-like, got {type(value).__name__}")


def normalize_valence_value(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and value in ALLOWED_VALENCES:
        return value
    if isinstance(value, float) and int(value) == value and int(value) in ALLOWED_VALENCES:
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text in {"", "null", "None"}:
            return None
        mapping = {
            "-1": -1,
            "-1.0": -1,
            "0": 0,
            "0.0": 0,
            "1": 1,
            "+1": 1,
            "1.0": 1,
            "+1.0": 1,
            "negative": -1,
            "positive": 1,
            "neutral": 0,
            "mixed": 0,
            "descriptive": 0,
        }
        if text in mapping:
            return mapping[text]
    raise ValueError(f"Invalid valence value: {value}")


def parse_prediction(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.rfind("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw_text[start : end + 1])
        raise


def normalize_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(prediction)
    normalized["c5_presence_d1"] = normalize_presence_value(normalized.get("c5_presence_d1", []))

    relevance = normalized.get("relevance")
    disease_1 = normalized.get("disease_1")
    has_relevant_5c = normalized.get("has_relevant_5c")
    presence = set(normalized["c5_presence_d1"])
    parsed_valences = {
        construct: normalize_valence_value(normalized.get(field))
        for construct, field in CONSTRUCT_TO_VALENCE_FIELD.items()
    }
    present_from_valence = {construct for construct, value in parsed_valences.items() if value is not None}

    if relevance not in {"relevant", "not_relevant"}:
        raise ValueError(f"Invalid relevance value: {relevance}")

    if any(item not in CONSTRUCT_TO_VALENCE_FIELD for item in presence):
        raise ValueError(f"Invalid construct labels in c5_presence_d1: {normalized['c5_presence_d1']}")
    normalized["c5_presence_d1"] = sorted(presence, key=CONSTRUCTS.index)

    if relevance == "not_relevant":
        normalized["disease_1"] = None
        normalized["has_relevant_5c"] = None
        normalized["c5_presence_d1"] = []
        for field in CONSTRUCT_TO_VALENCE_FIELD.values():
            normalized[field] = None
        return normalized

    if disease_1 is None:
        normalized["disease_1"] = "unspecified"
        disease_1 = "unspecified"
    if disease_1 not in ALLOWED_DISEASES:
        normalized["disease_1"] = "other"

    if has_relevant_5c is None:
        normalized["has_relevant_5c"] = "yes" if presence or present_from_valence else "no"
        has_relevant_5c = normalized["has_relevant_5c"]

    if has_relevant_5c == "no":
        normalized["c5_presence_d1"] = []
        for field in CONSTRUCT_TO_VALENCE_FIELD.values():
            normalized[field] = None
        return normalized

    if not normalized["c5_presence_d1"] and present_from_valence:
        normalized["c5_presence_d1"] = sorted(present_from_valence, key=CONSTRUCTS.index)
        presence = set(normalized["c5_presence_d1"])

    for construct, field in CONSTRUCT_TO_VALENCE_FIELD.items():
        value = parsed_valences[construct]
        normalized[field] = value if construct in presence else None

    return normalized


def parse_retry_budgets(raw: str, base_budget: int) -> list[int]:
    budgets = [base_budget]
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value > 0 and value not in budgets:
            budgets.append(value)
    return budgets


def write_chunk(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def upload_checkpoint_files(api: Any, repo_id: str, files: list[Path]) -> None:
    for path in files:
        if path.exists():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=repo_id,
                repo_type="dataset",
            )


def write_progress_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    n_input_rows: int,
    success_count: int,
    error_count: int,
    last_processed_index: int,
    checkpoint_index: int,
    uploaded_prediction_chunks: list[str],
    uploaded_error_chunks: list[str],
    started: float,
) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "role": "active_learning_hierarchy_selector",
        "benchmark_equivalent": False,
        "model_repo": args.model_repo,
        "tokenizer_repo": args.tokenizer_repo,
        "input_csv": str(args.input_csv),
        "system_prompt_path": str(args.system_prompt_path),
        "text_column": args.text_column,
        "id_column": args.id_column,
        "checkpoint_every": int(args.checkpoint_every),
        "n_input_rows": int(n_input_rows),
        "n_success": int(success_count),
        "n_errors": int(error_count),
        "last_processed_index": int(last_processed_index),
        "checkpoint_index": int(checkpoint_index),
        "scientific_use_note": (
            "This run is for an exploratory active-learning stage-2 subset only. "
            "It must not be reported as the locked benchmark result or used to estimate corpus prevalence."
        ),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "retry_max_new_tokens": parse_retry_budgets(args.retry_max_new_tokens, args.max_new_tokens)[1:],
            "mode": "merged_qwen_transformers_generate",
            "stored_uncertainty_signals": [
                "raw_output",
                "used_max_new_tokens",
                "attempt_number",
                "n_attempts",
                "n_retries",
                "row_latency_seconds",
            ],
        },
        "uploaded_prediction_chunks": uploaded_prediction_chunks,
        "uploaded_error_chunks": uploaded_error_chunks,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


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
    progress_manifest_path = output_dir / f"{args.run_name}_progress.json"
    manifest_path = output_dir / f"{args.run_name}_manifest.json"

    system_prompt = Path(args.system_prompt_path).read_text(encoding="utf-8").strip()
    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    tokenizer_repo = args.tokenizer_repo or args.model_repo
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, trust_remote_code=True, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_repo,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=token,
    )
    model.eval()

    api = HfApi(token=token)
    api.create_repo(repo_id=args.private_output_repo, repo_type="dataset", private=True, exist_ok=True)

    budgets = parse_retry_budgets(args.retry_max_new_tokens, args.max_new_tokens)
    started = time.time()
    success_count = 0
    error_count = 0
    checkpoint_index = 0
    uploaded_prediction_chunks: list[str] = []
    uploaded_error_chunks: list[str] = []
    chunk_predictions: list[dict[str, Any]] = []
    chunk_errors: list[dict[str, Any]] = []

    predictions_path.write_text("", encoding="utf-8")
    errors_path.write_text("", encoding="utf-8")

    with predictions_path.open("w", encoding="utf-8") as ok_handle, errors_path.open("w", encoding="utf-8") as err_handle:
        for idx, row in enumerate(df.to_dict(orient="records"), start=1):
            doc_id = str(row[args.id_column])
            row_started = time.time()
            raw_text = ""
            last_exc: Exception | None = None

            for attempt_number, budget in enumerate(budgets, start=1):
                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Article:\n{row[args.text_column]}"},
                    ]
                    inputs = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt",
                        return_dict=True,
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
                    prediction = normalize_prediction(parse_prediction(raw_text))
                    prediction["doc_id"] = doc_id
                    if "source_doc_id" in row:
                        prediction["source_doc_id"] = row["source_doc_id"]
                    prediction["used_max_new_tokens"] = budget
                    prediction["attempt_number"] = attempt_number
                    prediction["n_attempts"] = attempt_number
                    prediction["n_retries"] = attempt_number - 1
                    prediction["row_latency_seconds"] = round(time.time() - row_started, 4)
                    prediction["raw_output"] = raw_text
                    if "stage1_relevance" in row:
                        prediction["stage1_relevance"] = row["stage1_relevance"]
                    ok_handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    ok_handle.flush()
                    chunk_predictions.append(prediction)
                    success_count += 1
                    print(f"Saved stage-2 prediction {idx}/{len(df)} for doc_id={doc_id}", flush=True)
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc

            if last_exc is not None:
                error_row = {
                    "doc_id": doc_id,
                    "source_doc_id": row.get("source_doc_id"),
                    "error": str(last_exc),
                    "raw_text_excerpt": raw_text[:500] if raw_text else "",
                    "used_retry_budgets": budgets,
                    "n_attempts": len(budgets),
                    "n_retries": max(len(budgets) - 1, 0),
                    "row_latency_seconds": round(time.time() - row_started, 4),
                }
                err_handle.write(json.dumps(error_row, ensure_ascii=False) + "\n")
                err_handle.flush()
                chunk_errors.append(error_row)
                error_count += 1
                print(f"Failed stage-2 prediction {idx}/{len(df)} for doc_id={doc_id}: {error_row['error']}", flush=True)

            should_checkpoint = bool(args.checkpoint_every) and (idx % args.checkpoint_every == 0 or idx == len(df))
            if should_checkpoint:
                checkpoint_index += 1
                uploaded_files: list[Path] = []
                if chunk_predictions:
                    pred_chunk_path = output_dir / f"{args.run_name}_predictions_chunk_{checkpoint_index:05d}.jsonl"
                    write_chunk(pred_chunk_path, chunk_predictions)
                    uploaded_files.append(pred_chunk_path)
                    uploaded_prediction_chunks.append(pred_chunk_path.name)
                    chunk_predictions = []
                if chunk_errors:
                    err_chunk_path = output_dir / f"{args.run_name}_errors_chunk_{checkpoint_index:05d}.jsonl"
                    write_chunk(err_chunk_path, chunk_errors)
                    uploaded_files.append(err_chunk_path)
                    uploaded_error_chunks.append(err_chunk_path.name)
                    chunk_errors = []

                write_progress_manifest(
                    progress_manifest_path,
                    args=args,
                    n_input_rows=len(df),
                    success_count=success_count,
                    error_count=error_count,
                    last_processed_index=idx,
                    checkpoint_index=checkpoint_index,
                    uploaded_prediction_chunks=uploaded_prediction_chunks,
                    uploaded_error_chunks=uploaded_error_chunks,
                    started=started,
                )
                uploaded_files.append(progress_manifest_path)
                upload_checkpoint_files(api, args.private_output_repo, uploaded_files)

    write_progress_manifest(
        manifest_path,
        args=args,
        n_input_rows=len(df),
        success_count=success_count,
        error_count=error_count,
        last_processed_index=len(df),
        checkpoint_index=checkpoint_index,
        uploaded_prediction_chunks=uploaded_prediction_chunks,
        uploaded_error_chunks=uploaded_error_chunks,
        started=started,
    )
    upload_checkpoint_files(api, args.private_output_repo, [manifest_path])
    print(f"Uploaded stage-2 outputs to private dataset repo: {args.private_output_repo}", flush=True)


if __name__ == "__main__":
    main()
