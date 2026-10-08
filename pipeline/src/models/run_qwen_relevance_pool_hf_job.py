import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch


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
        description="Run stage-1 Qwen relevance inference over the post-gold media pool inside an HF Job."
    )
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--prompt-path", required=True)
    parser.add_argument("--schema-path", required=True)
    parser.add_argument("--output-dir", default="/job-output/qwen_relevance_pool_first_run")
    parser.add_argument("--private-output-repo", required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--text-column", default="model_text")
    parser.add_argument("--id-column", default="doc_id")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--max-errors", type=int, default=0, help="Abort after this many errors; 0 disables the guard.")
    parser.add_argument("--run-name", default="qwen_stage1_relevance_pool_v1")
    parser.add_argument("--disable-thinking", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_prompt(prompt_template: str, article_text: str) -> str:
    return prompt_template.replace("{{ARTICLE_TEXT}}", str(article_text).strip())


def parse_prediction(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.rfind("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def normalize_prediction(prediction: dict[str, Any]) -> dict[str, str]:
    relevance = prediction.get("relevance")
    reason_code = prediction.get("reason_code")
    if relevance not in {"relevant", "not_relevant"}:
        raise ValueError(f"Invalid relevance value: {relevance}")
    if reason_code not in ALLOWED_REASON_CODES:
        raise ValueError(f"Invalid reason_code value: {reason_code}")
    return {
        "relevance": relevance,
        "reason_code": reason_code,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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
    prompt_path: Path,
    schema_path: Path,
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
        "role": "longitudinal_2021_stage1_relevance_gate",
        "benchmark_equivalent_settings": True,
        "model_repo": args.model_repo,
        "input_csv": str(args.input_csv),
        "prompt_path": str(prompt_path),
        "schema_path": str(schema_path),
        "prompt_sha256": sha256_file(prompt_path),
        "schema_sha256": sha256_file(schema_path),
        "text_column": args.text_column,
        "id_column": args.id_column,
        "n_input_rows": int(n_input_rows),
        "n_success": int(success_count),
        "n_errors": int(error_count),
        "last_processed_index": int(last_processed_index),
        "checkpoint_index": int(checkpoint_index),
        "checkpoint_every": int(args.checkpoint_every),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "mode": "direct_transformers_chat_template_retry",
            "disable_thinking": args.disable_thinking,
        },
        "scientific_use_note": (
            "Stage-1 broad vaccination relevance gate for the 2021 longitudinal "
            "Eurosurveillance proof of concept. Uses the ESCAIDE G0 settings: "
            "base Qwen/Qwen3-32B, broad relevance prompt v2, thinking enabled unless "
            "explicitly disabled, deterministic decoding, and relevance-only output."
        ),
        "uploaded_prediction_chunks": uploaded_prediction_chunks,
        "uploaded_error_chunks": uploaded_error_chunks,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def build_chat_inputs(tokenizer, prompt: str, disable_thinking: bool):
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if disable_thinking:
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("chat_template_kwargs", None)
        rendered = tokenizer.apply_chat_template(messages, **kwargs)

    return tokenizer(rendered, return_tensors="pt")


def try_generate_prediction(model, tokenizer, prompt: str, args: argparse.Namespace) -> tuple[dict[str, str], str]:
    variants = [
        {"disable_thinking": args.disable_thinking, "do_sample": False, "temperature": 0.0, "top_p": 1.0},
        {"disable_thinking": False, "do_sample": False, "temperature": 0.0, "top_p": 1.0},
    ]

    last_error: Exception | None = None
    last_raw_text = ""

    for variant in variants:
        try:
            inputs = build_chat_inputs(tokenizer, prompt, disable_thinking=variant["disable_thinking"]).to(model.device)
            input_len = inputs["input_ids"].shape[-1]
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": variant["do_sample"],
                "pad_token_id": tokenizer.eos_token_id,
            }
            if variant["do_sample"]:
                generation_kwargs["temperature"] = variant["temperature"]
                generation_kwargs["top_p"] = variant["top_p"]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            generated = outputs[0][input_len:]
            raw_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            last_raw_text = raw_text
            if not raw_text:
                raise ValueError("empty_generation")
            normalized = normalize_prediction(parse_prediction(raw_text))
            return normalized, raw_text
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    if last_error is None:
        raise RuntimeError("generation_failed_without_exception")
    raise RuntimeError(f"{last_error}; raw={last_raw_text[:500]!r}") from last_error


def main() -> None:
    args = parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"Missing Hugging Face token in environment variable: {args.token_env}")

    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{args.run_name}.jsonl"
    errors_path = output_dir / f"{args.run_name}_errors.jsonl"
    progress_manifest_path = output_dir / f"{args.run_name}_progress.json"
    manifest_path = output_dir / f"{args.run_name}_manifest.json"

    prompt_path = Path(args.prompt_path).resolve()
    schema_path = Path(args.schema_path).resolve()
    prompt_template = prompt_path.read_text(encoding="utf-8")
    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    tokenizer = AutoTokenizer.from_pretrained(args.model_repo, trust_remote_code=True, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_repo,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=token,
    )
    model.eval()

    started = time.time()
    success_count = 0
    error_count = 0
    checkpoint_index = 0
    uploaded_prediction_chunks: list[str] = []
    uploaded_error_chunks: list[str] = []
    chunk_predictions: list[dict[str, Any]] = []
    chunk_errors: list[dict[str, Any]] = []

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.private_output_repo,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )

    with predictions_path.open("w", encoding="utf-8") as ok_handle, errors_path.open("w", encoding="utf-8") as err_handle:
        for idx, row in enumerate(df.to_dict(orient="records"), start=1):
            doc_id = str(row[args.id_column])
            article_text = str(row[args.text_column])
            prompt = build_prompt(prompt_template, article_text)
            try:
                normalized, raw_text = try_generate_prediction(model, tokenizer, prompt, args)
                normalized["doc_id"] = doc_id
                if "source_doc_id" in row:
                    normalized["source_doc_id"] = row["source_doc_id"]
                ok_handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                ok_handle.flush()
                chunk_predictions.append(dict(normalized))
                success_count += 1
                print(f"Saved prediction {idx}/{len(df)} for doc_id={doc_id}", flush=True)
            except Exception as exc:  # noqa: BLE001
                error_row = {
                    "doc_id": doc_id,
                    "source_doc_id": row.get("source_doc_id"),
                    "error": str(exc),
                    "prompt_preview": prompt[:500],
                }
                err_handle.write(json.dumps(error_row, ensure_ascii=False) + "\n")
                err_handle.flush()
                chunk_errors.append(error_row)
                error_count += 1
                print(f"Failed prediction {idx}/{len(df)} for doc_id={doc_id}: {exc}", flush=True)
                if args.max_errors and error_count >= args.max_errors:
                    print(f"Stopping early after reaching --max-errors={args.max_errors}", flush=True)
                    break

            should_checkpoint = bool(args.checkpoint_every) and (idx % args.checkpoint_every == 0 or idx == len(df))
            if should_checkpoint:
                checkpoint_index += 1
                uploaded_files: list[Path] = []
                if chunk_predictions:
                    pred_chunk_path = output_dir / f"{args.run_name}_predictions_chunk_{checkpoint_index:05d}.jsonl"
                    write_jsonl(pred_chunk_path, chunk_predictions)
                    uploaded_files.append(pred_chunk_path)
                    uploaded_prediction_chunks.append(pred_chunk_path.name)
                    chunk_predictions = []
                if chunk_errors:
                    err_chunk_path = output_dir / f"{args.run_name}_errors_chunk_{checkpoint_index:05d}.jsonl"
                    write_jsonl(err_chunk_path, chunk_errors)
                    uploaded_files.append(err_chunk_path)
                    uploaded_error_chunks.append(err_chunk_path.name)
                    chunk_errors = []
                write_progress_manifest(
                    progress_manifest_path,
                    args=args,
                    prompt_path=prompt_path,
                    schema_path=schema_path,
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

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_repo": args.model_repo,
        "input_csv": str(args.input_csv),
        "prompt_path": str(prompt_path),
        "schema_path": str(schema_path),
        "prompt_sha256": sha256_file(prompt_path),
        "schema_sha256": sha256_file(schema_path),
        "text_column": args.text_column,
        "id_column": args.id_column,
        "n_input_rows": int(len(df)),
        "n_success": int(success_count),
        "n_errors": int(error_count),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "mode": "direct_transformers_chat_template_retry",
            "disable_thinking": args.disable_thinking,
        },
        "elapsed_seconds": round(time.time() - started, 2),
        "outputs": {
            "predictions": str(predictions_path),
            "errors": str(errors_path),
        },
        "uploaded_prediction_chunks": uploaded_prediction_chunks,
        "uploaded_error_chunks": uploaded_error_chunks,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    upload_checkpoint_files(api, args.private_output_repo, [predictions_path, errors_path, manifest_path])

    print(f"Uploaded pool relevance outputs to private dataset repo: {args.private_output_repo}", flush=True)


if __name__ == "__main__":
    main()
