import argparse
import ast
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from huggingface_hub import InferenceClient


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = ROOT / "configs" / "apertus_5c_zero_shot_prompt.md"
DEFAULT_SCHEMA_PATH = ROOT / "configs" / "apertus_5c_zero_shot_schema.json"
DEFAULT_MODEL_ID = "swiss-ai/Apertus-8B-Instruct-2509"
DEFAULT_PROMPT_VERSION = "apertus-5c-zero-shot-v1.2-locked-2026-04-17"

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
        description="Run zero-shot 5C article classification against a Hugging Face Inference Endpoint."
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
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION, help="Locked prompt version.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Base model deployed behind the endpoint.")
    parser.add_argument(
        "--backend",
        choices=["openai-chat", "hf-text-generation"],
        default="openai-chat",
        help="Endpoint API backend. HF catalog SGLang endpoints use openai-chat.",
    )
    parser.add_argument("--run-manifest", help="Optional path for a run manifest JSON.")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable holding the HF token.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generated tokens.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds.")
    parser.add_argument(
        "--disable-grammar",
        action="store_true",
        help="Disable JSON grammar constraints if the endpoint backend rejects them.",
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
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(f"Missing token in environment variable {token_env}.")
    return InferenceClient(model=endpoint_url, token=token, timeout=timeout)


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


def call_openai_chat_endpoint(
    endpoint_url: str,
    token: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
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
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

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


def parse_prediction(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Endpoint did not return valid JSON: {raw_text}") from exc


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
    raise ValueError(f"c5_presence_d1 must be a list or list-like string, got {type(value).__name__}")


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
        }
        if text in mapping:
            return mapping[text]
    raise ValueError(f"Invalid valence value: {value}")


def normalize_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(prediction)
    normalized["c5_presence_d1"] = normalize_presence_value(normalized.get("c5_presence_d1", []))

    relevance = normalized.get("relevance")
    disease_1 = normalized.get("disease_1")
    has_relevant_5c = normalized.get("has_relevant_5c")
    presence = set(normalized["c5_presence_d1"])

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

    if disease_1 not in ALLOWED_DISEASES:
        raise ValueError(f"Invalid disease_1 value for relevant article: {disease_1}")

    if has_relevant_5c not in {"yes", "no"}:
        raise ValueError(f"Invalid has_relevant_5c value for relevant article: {has_relevant_5c}")

    if has_relevant_5c == "no":
        normalized["c5_presence_d1"] = []
        for field in CONSTRUCT_TO_VALENCE_FIELD.values():
            normalized[field] = None
        return normalized

    if not normalized["c5_presence_d1"]:
        raise ValueError("has_relevant_5c is 'yes' but c5_presence_d1 is empty.")

    for construct, field in CONSTRUCT_TO_VALENCE_FIELD.items():
        value = normalize_valence_value(normalized.get(field))
        if construct in presence:
            if value not in ALLOWED_VALENCES:
                raise ValueError(f"{field} must be one of -1, 0, 1 when {construct} is present.")
            normalized[field] = value
        else:
            normalized[field] = None

    return normalized


def predict_one(
    client: InferenceClient,
    token: str,
    prompt_template: str,
    schema: dict[str, Any],
    article_text: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
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
        disable_grammar=args.disable_grammar,
        backend=args.backend,
        timeout=args.timeout,
    )
    return normalize_prediction(parse_prediction(raw_output))


def manifest_path_for_output(args: argparse.Namespace) -> Path | None:
    if args.run_manifest:
        return Path(args.run_manifest)
    if args.output:
        return Path(f"{args.output}.manifest.json")
    return None


def write_run_manifest(
    args: argparse.Namespace,
    input_kind: str,
    input_path: str | None,
    output_path: str | None,
    n_records: int,
) -> None:
    manifest_path = manifest_path_for_output(args)
    if not manifest_path:
        return

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "endpoint_url": args.endpoint_url,
        "prompt_version": args.prompt_version,
        "prompt_path": str(Path(args.prompt_path)),
        "schema_path": str(Path(args.schema_path)),
        "prompt_sha256": sha256_file(args.prompt_path),
        "schema_sha256": sha256_file(args.schema_path),
        "input_kind": input_kind,
        "input_path": input_path,
        "output_path": output_path,
        "n_records": n_records,
        "generation": {
            "backend": args.backend,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "grammar_enabled": not args.disable_grammar,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_single(args: argparse.Namespace) -> None:
    prompt_template = load_prompt_template(args.prompt_path)
    schema = load_schema(args.schema_path)
    token = os.environ.get(args.token_env)
    client = get_client(args.endpoint_url, args.token_env, args.timeout)
    article_text = load_text(args)
    prediction = predict_one(client, token, prompt_template, schema, article_text, args)

    output_text = json.dumps(prediction, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text + "\n", encoding="utf-8")
    else:
        print(output_text)
    write_run_manifest(
        args=args,
        input_kind="text" if args.input_text else "file",
        input_path=args.input_file,
        output_path=args.output,
        n_records=1,
    )


def run_batch(args: argparse.Namespace) -> None:
    prompt_template = load_prompt_template(args.prompt_path)
    schema = load_schema(args.schema_path)
    token = os.environ.get(args.token_env)
    client = get_client(args.endpoint_url, args.token_env, args.timeout)
    df = pd.read_csv(args.input_csv)

    if args.text_column not in df.columns:
        raise ValueError(f"Missing text column: {args.text_column}")
    if args.id_column not in df.columns:
        raise ValueError(f"Missing id column: {args.id_column}")

    output_path = Path(args.output) if args.output else None
    stream_jsonl = output_path is not None and output_path.suffix.lower() == ".jsonl"
    records: list[dict[str, Any]] = []

    if stream_jsonl:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for idx, row in enumerate(df.itertuples(index=False), start=1):
                article_id = getattr(row, args.id_column)
                article_text = getattr(row, args.text_column)
                prediction = predict_one(client, token, prompt_template, schema, article_text, args)
                prediction["doc_id"] = article_id
                records.append(prediction)
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"Saved prediction {idx}/{len(df)} for doc_id={article_id}", flush=True)
    else:
        for idx, row in enumerate(df.itertuples(index=False), start=1):
            article_id = getattr(row, args.id_column)
            article_text = getattr(row, args.text_column)
            prediction = predict_one(client, token, prompt_template, schema, article_text, args)
            prediction["doc_id"] = article_id
            records.append(prediction)
            print(f"Completed prediction {idx}/{len(df)} for doc_id={article_id}", flush=True)

    if output_path is not None and not stream_jsonl:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(output_path, index=False)
    elif output_path is None:
        result_df = pd.DataFrame(records)
        print(result_df.to_json(orient="records", force_ascii=False, indent=2))

    write_run_manifest(
        args=args,
        input_kind="csv",
        input_path=args.input_csv,
        output_path=args.output,
        n_records=len(records),
    )


def main() -> None:
    args = parse_args()
    if args.input_csv:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
