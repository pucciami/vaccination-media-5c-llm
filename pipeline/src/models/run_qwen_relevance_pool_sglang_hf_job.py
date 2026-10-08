import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
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
        description="Run stage-1 Qwen relevance inference over the post-gold media pool inside an HF Job using a local sglang server."
    )
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--tokenizer-repo", default=None)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--prompt-path", required=True)
    parser.add_argument("--schema-path", required=True)
    parser.add_argument("--output-dir", default="/job-output/qwen_relevance_pool_sglang_first_run")
    parser.add_argument("--private-output-repo", required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--text-column", default="model_text")
    parser.add_argument("--id-column", default="doc_id")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--retry-max-new-tokens", default="1024,2048,4096")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--presence-penalty", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--server-port", type=int, default=30000)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-start-timeout", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--binary-only", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--disable-reasoning-parser", action="store_true")
    parser.add_argument("--run-name", default="qwen_stage1_relevance_pool_v1")
    parser.add_argument("--run-role", default="active_learning_selector")
    parser.add_argument("--benchmark-equivalent-settings", action="store_true")
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


def normalize_prediction(prediction: dict[str, Any], *, binary_only: bool = False) -> dict[str, str]:
    relevance = normalize_relevance_label(prediction.get("relevance"))
    if binary_only:
        return {"relevance": relevance}
    reason_code = prediction.get("reason_code")
    if reason_code not in ALLOWED_REASON_CODES:
        raise ValueError(f"Invalid reason_code value: {reason_code}")
    return {"relevance": relevance, "reason_code": reason_code}


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


def call_local_openai_chat(
    endpoint_url: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    timeout: float = 180.0,
    enable_thinking: bool = True,
) -> dict[str, Any]:
    url = endpoint_url.rstrip("/") + "/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "presence_penalty": presence_penalty,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    started = time.time()
    response = httpx.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    latency_seconds = round(time.time() - started, 4)
    response.raise_for_status()
    data = response.json()
    try:
        choice = data["choices"][0]
        message = choice["message"]
        return {
            "raw_text": message["content"],
            "reasoning_text": message.get("reasoning_content"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
            "response_id": data.get("id"),
            "model": data.get("model"),
            "latency_seconds": latency_seconds,
        }
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected local OpenAI-compatible response: {data}") from exc


def wait_for_server(endpoint_url: str, timeout_seconds: int, poll_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last_error = None
    health_url = endpoint_url.rstrip("/") + "/health"
    while time.time() < deadline:
        try:
            response = httpx.get(health_url, timeout=10.0)
            if response.status_code == 200:
                return
            last_error = f"status={response.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(poll_seconds)
    raise TimeoutError(f"Local sglang server did not become healthy in time: {last_error}")


def start_sglang_server(args: argparse.Namespace, token: str, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env[args.token_env] = token
    command = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_repo,
        "--host",
        args.server_host,
        "--port",
        str(args.server_port),
        "--trust-remote-code",
    ]
    if not args.disable_reasoning_parser:
        command.extend(["--reasoning-parser", "qwen3"])
    if args.tokenizer_repo:
        command.extend(["--tokenizer-path", args.tokenizer_repo])
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
    )
    process._sglang_log_handle = log_handle  # type: ignore[attr-defined]
    return process


def stop_sglang_server(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    log_handle = getattr(process, "_sglang_log_handle", None)
    if log_handle is not None:
        log_handle.close()


def upload_checkpoint_files(api: Any, repo_id: str, files: list[Path]) -> None:
    for path in files:
        if path.exists():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=repo_id,
                repo_type="dataset",
            )


def write_chunk(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        "role": args.run_role,
        "benchmark_equivalent_settings": bool(args.benchmark_equivalent_settings),
        "model_repo": args.model_repo,
        "tokenizer_repo": args.tokenizer_repo,
        "input_csv": str(args.input_csv),
        "prompt_path": str(prompt_path),
        "schema_path": str(schema_path),
        "prompt_sha256": sha256_file(prompt_path),
        "schema_sha256": sha256_file(schema_path),
        "text_column": args.text_column,
        "id_column": args.id_column,
        "binary_only": bool(args.binary_only),
        "enable_thinking": bool(args.enable_thinking),
        "disable_reasoning_parser": bool(args.disable_reasoning_parser),
        "scientific_use_note": (
            "This run uses the configured stage-1 relevance gate settings. "
            "For longitudinal prevalence use, verify the model, prompt, schema, thinking mode, "
            "and input construction against the locked G0 pipeline before interpreting outputs."
        ),
        "checkpoint_every": int(args.checkpoint_every),
        "max_concurrency": int(args.max_concurrency),
        "n_input_rows": int(n_input_rows),
        "n_success": int(success_count),
        "n_errors": int(error_count),
        "last_processed_index": int(last_processed_index),
        "checkpoint_index": int(checkpoint_index),
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "retry_max_new_tokens": parse_retry_budgets(args.retry_max_new_tokens, args.max_new_tokens)[1:],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "presence_penalty": args.presence_penalty,
            "mode": "local_sglang_openai_chat",
            "reasoning_parser": None if args.disable_reasoning_parser else "qwen3",
            "selector_mode": (
                "thinking"
                if args.enable_thinking
                else "non_thinking_acquisition"
            ),
            "stored_uncertainty_signals": [
                "raw_output",
                "used_max_new_tokens",
                "attempt_number",
                "n_attempts",
                "n_retries",
                "finish_reason",
                "response_id",
                "response_model",
                "call_latency_seconds",
                "row_latency_seconds",
                "usage_prompt_tokens",
                "usage_completion_tokens",
                "usage_total_tokens",
            ],
        },
        "uploaded_prediction_chunks": uploaded_prediction_chunks,
        "uploaded_error_chunks": uploaded_error_chunks,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def process_row(
    *,
    idx: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    endpoint_url: str,
    prompt_template: str,
    budgets: list[int],
) -> dict[str, Any]:
    doc_id = str(row[args.id_column])
    article_text = str(row[args.text_column])
    prompt = build_prompt(prompt_template, article_text)
    raw_text = ""
    row_started = time.time()
    last_meta: dict[str, Any] | None = None
    last_exc: Exception | None = None

    for attempt_number, budget in enumerate(budgets, start=1):
        try:
            response_meta = call_local_openai_chat(
                endpoint_url=endpoint_url,
                model_id=args.model_repo,
                prompt=prompt,
                max_new_tokens=budget,
                temperature=args.temperature,
                top_p=args.top_p,
                presence_penalty=args.presence_penalty,
                enable_thinking=args.enable_thinking,
            )
            last_meta = response_meta
            raw_text = str(response_meta["raw_text"])
            if not raw_text.strip():
                reasoning_text = str(response_meta.get("reasoning_text") or "").strip()
                if reasoning_text:
                    raise ValueError("Empty content after reasoning; likely token budget too small for final JSON.")
                raise ValueError("Empty content returned by local OpenAI-compatible server.")
            normalized = normalize_prediction(parse_prediction(raw_text), binary_only=args.binary_only)
            normalized["doc_id"] = doc_id
            normalized["used_max_new_tokens"] = budget
            normalized["attempt_number"] = attempt_number
            normalized["n_attempts"] = attempt_number
            normalized["n_retries"] = attempt_number - 1
            normalized["finish_reason"] = response_meta.get("finish_reason")
            normalized["response_id"] = response_meta.get("response_id")
            normalized["response_model"] = response_meta.get("model")
            normalized["call_latency_seconds"] = response_meta.get("latency_seconds")
            normalized["row_latency_seconds"] = round(time.time() - row_started, 4)
            usage = response_meta.get("usage") or {}
            if isinstance(usage, dict):
                normalized["usage_prompt_tokens"] = usage.get("prompt_tokens")
                normalized["usage_completion_tokens"] = usage.get("completion_tokens")
                normalized["usage_total_tokens"] = usage.get("total_tokens")
            normalized["raw_output"] = raw_text
            return {
                "status": "ok",
                "index": idx,
                "doc_id": doc_id,
                "payload": normalized,
            }
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

    error_row = {
        "doc_id": doc_id,
        "error": str(last_exc) if last_exc else "Unknown error",
        "raw_text_excerpt": raw_text[:500] if raw_text else "",
        "used_retry_budgets": budgets,
        "n_attempts": len(budgets),
        "n_retries": max(len(budgets) - 1, 0),
        "row_latency_seconds": round(time.time() - row_started, 4),
        "last_finish_reason": (last_meta or {}).get("finish_reason"),
        "last_response_id": (last_meta or {}).get("response_id"),
        "last_call_latency_seconds": (last_meta or {}).get("latency_seconds"),
        "last_reasoning_excerpt": str((last_meta or {}).get("reasoning_text") or "")[:500],
        "last_usage": (last_meta or {}).get("usage"),
    }
    return {
        "status": "err",
        "index": idx,
        "doc_id": doc_id,
        "payload": error_row,
    }


def main() -> None:
    args = parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"Missing Hugging Face token in environment variable: {args.token_env}")

    from huggingface_hub import HfApi

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{args.run_name}.jsonl"
    errors_path = output_dir / f"{args.run_name}_errors.jsonl"
    manifest_path = output_dir / f"{args.run_name}_manifest.json"
    progress_manifest_path = output_dir / f"{args.run_name}_progress.json"
    server_log_path = output_dir / "sglang_server.log"

    prompt_path = Path(args.prompt_path).resolve()
    schema_path = Path(args.schema_path).resolve()
    prompt_template = prompt_path.read_text(encoding="utf-8")
    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    endpoint_url = f"http://{args.server_host}:{args.server_port}"
    server = start_sglang_server(args, token, server_log_path)

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

    try:
        wait_for_server(endpoint_url, timeout_seconds=args.server_start_timeout, poll_seconds=args.poll_seconds)
        budgets = parse_retry_budgets(args.retry_max_new_tokens, args.max_new_tokens)

        predictions_path.write_text("", encoding="utf-8")
        errors_path.write_text("", encoding="utf-8")
        rows = df.to_dict(orient="records")
        total_rows = len(rows)
        max_workers = max(1, int(args.max_concurrency))
        completed_count = 0
        last_processed_index = 0

        with predictions_path.open("w", encoding="utf-8") as ok_handle, errors_path.open("w", encoding="utf-8") as err_handle:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                next_submit = 0
                futures: dict[Any, int] = {}

                while next_submit < min(max_workers, total_rows):
                    row = rows[next_submit]
                    idx = next_submit + 1
                    future = executor.submit(
                        process_row,
                        idx=idx,
                        row=row,
                        args=args,
                        endpoint_url=endpoint_url,
                        prompt_template=prompt_template,
                        budgets=budgets,
                    )
                    futures[future] = idx
                    next_submit += 1

                while futures:
                    future = next(as_completed(list(futures.keys())))
                    idx = futures.pop(future)
                    result = future.result()
                    last_processed_index = max(last_processed_index, idx)
                    completed_count += 1

                    if result["status"] == "ok":
                        normalized = result["payload"]
                        ok_handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                        ok_handle.flush()
                        chunk_predictions.append(normalized)
                        success_count += 1
                        print(f"Saved prediction {completed_count}/{total_rows} for doc_id={result['doc_id']}", flush=True)
                    else:
                        error_row = result["payload"]
                        err_handle.write(json.dumps(error_row, ensure_ascii=False) + "\n")
                        err_handle.flush()
                        chunk_errors.append(error_row)
                        error_count += 1
                        print(f"Failed prediction {completed_count}/{total_rows} for doc_id={result['doc_id']}: {error_row['error']}", flush=True)

                    if next_submit < total_rows:
                        row = rows[next_submit]
                        next_idx = next_submit + 1
                        new_future = executor.submit(
                            process_row,
                            idx=next_idx,
                            row=row,
                            args=args,
                            endpoint_url=endpoint_url,
                            prompt_template=prompt_template,
                            budgets=budgets,
                        )
                        futures[new_future] = next_idx
                        next_submit += 1

                    should_checkpoint = bool(args.checkpoint_every) and (
                        completed_count % args.checkpoint_every == 0 or completed_count == total_rows
                    )
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
                            prompt_path=prompt_path,
                            schema_path=schema_path,
                            n_input_rows=total_rows,
                            success_count=success_count,
                            error_count=error_count,
                            last_processed_index=last_processed_index,
                            checkpoint_index=checkpoint_index,
                            uploaded_prediction_chunks=uploaded_prediction_chunks,
                            uploaded_error_chunks=uploaded_error_chunks,
                            started=started,
                        )
                        uploaded_files.append(progress_manifest_path)
                        upload_checkpoint_files(api, args.private_output_repo, uploaded_files)
    finally:
        stop_sglang_server(server)
    write_progress_manifest(
        manifest_path,
        args=args,
        prompt_path=prompt_path,
        schema_path=schema_path,
        n_input_rows=len(df),
        success_count=success_count,
        error_count=error_count,
        last_processed_index=success_count + error_count,
        checkpoint_index=checkpoint_index,
        uploaded_prediction_chunks=uploaded_prediction_chunks,
        uploaded_error_chunks=uploaded_error_chunks,
        started=started,
    )
    upload_checkpoint_files(api, args.private_output_repo, [manifest_path, server_log_path])
    print(f"Uploaded chunked pool relevance outputs to private dataset repo: {args.private_output_repo}", flush=True)


if __name__ == "__main__":
    main()
