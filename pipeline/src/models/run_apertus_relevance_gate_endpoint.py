import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from huggingface_hub import InferenceClient, get_token


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = ROOT / "configs" / "apertus_relevance_gate_prompt_v2_broad_vaccination.md"
DEFAULT_SCHEMA_PATH = ROOT / "configs" / "apertus_relevance_gate_schema_v2_broad_vaccination.json"
DEFAULT_MODEL_ID = "swiss-ai/Apertus-8B-Instruct-2509"
DEFAULT_PROMPT_VERSION = "relevance-gate-v2-broad-vaccination-2026-05-14"
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
VACCINE_SIGNAL_RE = re.compile(
    r"(?i)\b(cepiv\w*|ceplj\w*|cepljen\w*|cepljenje\w*|vakcin\w*|vaccin\w*|vaccine\w*|imuniz\w*|imunis\w*|immuni\w*)\b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a relevance-only vaccination gate against a Hugging Face Inference Endpoint."
    )
    parser.add_argument("--endpoint-url", required=True, help="Private Hugging Face endpoint URL.")
    parser.add_argument("--input-text", help="Single article text to classify.")
    parser.add_argument("--input-file", help="Path to a text file containing one article.")
    parser.add_argument("--input-csv", help="CSV file for batch inference.")
    parser.add_argument("--text-column", default="text", help="Text column in --input-csv.")
    parser.add_argument("--id-column", default="doc_id", help="ID column in --input-csv.")
    parser.add_argument("--output", help="Output path. Prints to stdout if omitted.")
    parser.add_argument("--prompt-path", default=str(DEFAULT_PROMPT_PATH), help="Prompt template path.")
    parser.add_argument("--schema-path", default=str(DEFAULT_SCHEMA_PATH), help="JSON schema path.")
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION, help="Prompt version label.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Model name sent to the endpoint.")
    parser.add_argument(
        "--backend",
        choices=["openai-chat", "hf-text-generation"],
        default="openai-chat",
        help="Endpoint API backend.",
    )
    parser.add_argument("--run-manifest", help="Optional path for a run manifest JSON.")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable holding the HF token.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument("--presence-penalty", type=float, default=0.0, help="Presence penalty for supported backends.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max generated tokens.")
    parser.add_argument(
        "--retry-max-new-tokens",
        default="",
        help="Comma-separated larger token budgets to retry on parse/validation failure, e.g. 512,1024,2048.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a batch run by loading any existing JSONL output and skipping completed doc_ids.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds.")
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Request non-thinking mode for backends that support chat_template_kwargs.enable_thinking.",
    )
    parser.add_argument(
        "--disable-grammar",
        action="store_true",
        help="Disable JSON grammar constraints if the endpoint backend rejects them.",
    )
    parser.add_argument(
        "--prefilter",
        choices=["none", "vaccination-keywords"],
        default="none",
        help="Optional development prefilter that forces not_relevant when no vaccine lexical signal is present.",
    )
    args = parser.parse_args()

    provided_inputs = [bool(args.input_text), bool(args.input_file), bool(args.input_csv)]
    if sum(provided_inputs) != 1:
        parser.error("Provide exactly one of --input-text, --input-file, or --input-csv.")

    return args


def load_text(args: argparse.Namespace) -> str:
    if args.input_text:
        return args.input_text
    if args.input_file:
        return Path(args.input_file).read_text(encoding="utf-8")
    raise ValueError("load_text should only be called for single-input runs.")


def load_prompt_template(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_schema(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_prompt(prompt_template: str, article_text: str) -> str:
    return prompt_template.replace("{{ARTICLE_TEXT}}", article_text.strip())


def get_client(endpoint_url: str, token_env: str, timeout: float) -> InferenceClient:
    token = os.environ.get(token_env) or get_token()
    if not token:
        raise RuntimeError(f"Missing token in environment variable {token_env}.")
    return InferenceClient(model=endpoint_url, token=token, timeout=timeout)


def call_openai_chat_endpoint(
    endpoint_url: str,
    token: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    disable_thinking: bool,
    json_mode: bool,
    timeout: float,
) -> str:
    url = endpoint_url.rstrip("/") + "/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_new_tokens,
        "presence_penalty": presence_penalty,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    response = httpx.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected OpenAI-compatible response: {data}") from exc


def call_endpoint(
    client: InferenceClient,
    endpoint_url: str,
    token: str,
    model_id: str,
    prompt: str,
    schema: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    disable_thinking: bool,
    disable_grammar: bool,
    backend: str,
    timeout: float,
) -> str:
    if backend == "openai-chat":
        return call_openai_chat_endpoint(
            endpoint_url=endpoint_url,
            token=token,
            model_id=model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            disable_thinking=disable_thinking,
            json_mode=not disable_grammar,
            timeout=timeout,
        )

    request_kwargs: dict[str, Any] = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if not disable_grammar:
        request_kwargs["grammar"] = {"type": "json", "value": schema}
    return client.text_generation(**request_kwargs)


def parse_prediction(raw_text: str) -> dict[str, Any]:
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError as exc:
        text = raw_text.strip()
        start = text.rfind("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Endpoint did not return valid JSON: {raw_text}") from exc


def normalize_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    relevance = prediction.get("relevance")
    reason_code = prediction.get("reason_code")
    if relevance not in {"relevant", "not_relevant"}:
        raise ValueError(f"Invalid relevance value: {relevance}")
    if reason_code not in ALLOWED_REASON_CODES:
        raise ValueError(f"Invalid reason_code value: {reason_code}")
    return {"relevance": relevance, "reason_code": reason_code}


def has_vaccination_lexical_signal(text: str) -> bool:
    return bool(VACCINE_SIGNAL_RE.search(text or ""))


def prefilter_prediction(text: str, mode: str) -> dict[str, Any] | None:
    if mode == "vaccination-keywords" and not has_vaccination_lexical_signal(text):
        return {
            "relevance": "not_relevant",
            "reason_code": "no_vaccination_lexical_signal",
            "_prefilter_forced": True,
        }
    return None


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


def predict_one(
    client: InferenceClient,
    token: str,
    prompt_template: str,
    schema: dict[str, Any],
    article_text: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str | None]:
    forced = prefilter_prediction(article_text, args.prefilter)
    if forced is not None:
        normalized = normalize_prediction(forced)
        normalized["_prefilter_forced"] = True
        return normalized, None

    prompt = build_prompt(prompt_template, article_text)
    raw_output = call_endpoint(
        client=client,
        endpoint_url=args.endpoint_url,
        token=token,
        model_id=args.model_id,
        prompt=prompt,
        schema=schema,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        disable_thinking=args.disable_thinking,
        disable_grammar=args.disable_grammar,
        backend=args.backend,
        timeout=args.timeout,
    )
    prediction = parse_prediction(raw_output)
    normalized = normalize_prediction(prediction)
    normalized["_prefilter_forced"] = False
    return normalized, raw_output


def predict_one_with_retries(
    client: InferenceClient,
    token: str,
    prompt_template: str,
    schema: dict[str, Any],
    article_text: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str | None, int]:
    budgets = parse_retry_budgets(args.retry_max_new_tokens, args.max_new_tokens)
    last_error: Exception | None = None
    for budget in budgets:
        original_budget = args.max_new_tokens
        try:
            args.max_new_tokens = budget
            prediction, raw_output = predict_one(client, token, prompt_template, schema, article_text, args)
            return prediction, raw_output, budget
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        finally:
            args.max_new_tokens = original_budget
    raise RuntimeError(f"All retry budgets failed: {budgets}") from last_error


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_existing_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def run_batch(args: argparse.Namespace) -> None:
    prompt_template = load_prompt_template(args.prompt_path)
    schema = load_schema(args.schema_path)
    token = os.environ.get(args.token_env) or get_token()
    if not token:
        raise RuntimeError(f"Missing token in environment variable {args.token_env}.")
    client = get_client(args.endpoint_url, args.token_env, args.timeout)

    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
    output_path = Path(args.output) if args.output else None
    records: list[dict[str, Any]] = load_existing_jsonl(output_path) if (args.resume and output_path) else []
    completed_doc_ids = {str(record["doc_id"]) for record in records}
    original_total = len(df)
    if completed_doc_ids:
        df = df[~df[args.id_column].astype(str).isin(completed_doc_ids)].copy()
    for batch_pos, (_, row) in enumerate(df.iterrows(), start=1):
        doc_id = str(row[args.id_column])
        article_text = str(row[args.text_column])
        prediction, raw_output, used_max_new_tokens = predict_one_with_retries(
            client, token, prompt_template, schema, article_text, args
        )
        record = {
            "doc_id": doc_id,
            "relevance": prediction["relevance"],
            "reason_code": prediction["reason_code"],
            "prefilter_forced": bool(prediction["_prefilter_forced"]),
            "raw_output": raw_output,
            "used_max_new_tokens": used_max_new_tokens,
        }
        records.append(record)
        if output_path:
            write_jsonl(output_path, records)
        print(f"Saved prediction {len(completed_doc_ids) + batch_pos}/{original_total} for doc_id={doc_id}")

    if args.run_manifest:
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint_url": args.endpoint_url,
            "model_id": args.model_id,
            "prompt_path": str(Path(args.prompt_path).resolve()),
            "prompt_sha256": sha256_file(args.prompt_path),
            "prompt_version": args.prompt_version,
            "schema_path": str(Path(args.schema_path).resolve()),
            "schema_sha256": sha256_file(args.schema_path),
            "input_kind": "csv",
            "input_path": args.input_csv,
            "output_path": args.output,
            "n_records": int(original_total),
            "resumed": bool(args.resume),
            "completed_doc_ids_loaded": int(len(completed_doc_ids)),
            "prefilter": args.prefilter,
            "generation": {
                "backend": args.backend,
                "grammar_enabled": not args.disable_grammar,
                "max_new_tokens": args.max_new_tokens,
                "retry_max_new_tokens": parse_retry_budgets(args.retry_max_new_tokens, args.max_new_tokens)[1:],
                "temperature": args.temperature,
                "top_p": args.top_p,
                "presence_penalty": args.presence_penalty,
                "disable_thinking": args.disable_thinking,
            },
        }
        Path(args.run_manifest).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def run_single(args: argparse.Namespace) -> None:
    prompt_template = load_prompt_template(args.prompt_path)
    schema = load_schema(args.schema_path)
    token = os.environ.get(args.token_env) or get_token()
    if not token:
        raise RuntimeError(f"Missing token in environment variable {args.token_env}.")
    client = get_client(args.endpoint_url, args.token_env, args.timeout)
    article_text = load_text(args)
    prediction, raw_output = predict_one(client, token, prompt_template, schema, article_text, args)
    output = {
        "relevance": prediction["relevance"],
        "reason_code": prediction["reason_code"],
        "prefilter_forced": bool(prediction["_prefilter_forced"]),
        "raw_output": raw_output,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.input_csv:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
