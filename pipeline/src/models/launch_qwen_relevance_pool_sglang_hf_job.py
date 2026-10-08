import argparse
import json
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, Volume, get_token


ROOT = Path(__file__).resolve().parents[2]

VALIDATED_G0_SGLANG_NOTE = (
    "Validated G0 SGLang defaults from the 2026-05-26 longitudinal n=200 "
    "calibration: 200/200 parsed with Qwen/Qwen3-32B, broad v2 gate, thinking "
    "enabled, qwen3 reasoning parser, deterministic decoding, max_new_tokens "
    "512 and retry ladder 1024,2048,4096. Do not lower these defaults for "
    "reportable G0-style runs without documenting the run as a cheaper "
    "acquisition-only variant."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a private HF Job to run stage-1 Qwen relevance via a local "
            f"sglang server. {VALIDATED_G0_SGLANG_NOTE}"
        )
    )
    parser.add_argument("--job-script", default=str(ROOT / "src" / "models" / "run_qwen_relevance_pool_sglang_hf_job.py"))
    parser.add_argument("--pool-csv", default=str(ROOT / "data" / "processed" / "active_learning" / "media_active_learning_stage1_tranche_v1.csv"))
    parser.add_argument("--prompt-path", default=str(ROOT / "configs" / "apertus_relevance_gate_prompt_binary_v1.md"))
    parser.add_argument("--schema-path", default=str(ROOT / "configs" / "apertus_relevance_gate_schema_binary.json"))
    parser.add_argument("--assets-repo-id", default="Pucciami/qwen3-32b-relevance-tranche-assets-v2")
    parser.add_argument("--output-repo-id", default="Pucciami/qwen3-32b-relevance-tranche-output-v2")
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--flavor", default="h200")
    parser.add_argument("--timeout", default="18h")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--model-repo", default="Qwen/Qwen3-32B-FP8")
    parser.add_argument("--tokenizer-repo", default="Qwen/Qwen3-32B-FP8")
    parser.add_argument("--text-column", default="model_text")
    parser.add_argument("--id-column", default="doc_id")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--retry-max-new-tokens", default="1024,2048,4096")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--binary-only", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--disable-reasoning-parser", action="store_true")
    parser.add_argument("--run-name", default="qwen_stage1_relevance_tranche_v2")
    parser.add_argument("--run-role", default="active_learning_selector")
    parser.add_argument("--benchmark-equivalent-settings", action="store_true")
    return parser.parse_args()


def upload_job_assets(api: HfApi, assets_repo_id: str, uploads: list[tuple[Path, str]]) -> None:
    for local_path, remote_name in uploads:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_name,
            repo_id=assets_repo_id,
            repo_type="dataset",
        )


def main() -> None:
    args = parse_args()

    token = os.environ.get(args.token_env) or get_token()
    if not token:
        raise RuntimeError(
            f"Missing Hugging Face token in environment variable {args.token_env} and no cached login was found."
        )

    api = HfApi(token=token)
    whoami = api.whoami(token=token)
    namespace = args.namespace or whoami["name"]

    pool_df = pd.read_csv(args.pool_csv, dtype=str, keep_default_na=False)
    slim_df = pool_df[[args.id_column, "source_doc_id", args.text_column]].copy()
    if args.limit and args.limit > 0:
        slim_df = slim_df.head(args.limit).copy()
    slim_input_path = Path(args.pool_csv).resolve().parent / f"{Path(args.pool_csv).stem}_model_text_only.csv"
    slim_df.to_csv(slim_input_path, index=False, encoding="utf-8")

    api.create_repo(repo_id=args.assets_repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)
    api.create_repo(repo_id=args.output_repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)

    uploads = [
        (Path(args.job_script).resolve(), "run_qwen_relevance_pool_sglang_hf_job.py"),
        (slim_input_path.resolve(), slim_input_path.name),
        (Path(args.prompt_path).resolve(), Path(args.prompt_path).name),
        (Path(args.schema_path).resolve(), Path(args.schema_path).name),
    ]
    upload_job_assets(api, args.assets_repo_id, uploads)

    assets_volume = Volume(
        type="dataset",
        source=args.assets_repo_id,
        mount_path="/job-assets",
    )
    install_cmd = "python -m pip install --no-input pandas httpx huggingface_hub"
    run_cmd = (
        "python /job-assets/run_qwen_relevance_pool_sglang_hf_job.py "
        f"--model-repo {args.model_repo} "
        f"--tokenizer-repo {args.tokenizer_repo} "
        f"--input-csv /job-assets/{slim_input_path.name} "
        f"--prompt-path /job-assets/{Path(args.prompt_path).name} "
        f"--schema-path /job-assets/{Path(args.schema_path).name} "
        f"--private-output-repo {args.output_repo_id} "
        f"--text-column {args.text_column} "
        f"--id-column {args.id_column} "
        f"--max-new-tokens {args.max_new_tokens} "
        f"--retry-max-new-tokens {args.retry_max_new_tokens} "
        f"--temperature {args.temperature} "
        f"--top-p {args.top_p} "
        f"--presence-penalty {args.presence_penalty} "
        f"--limit {args.limit} "
        f"--checkpoint-every {args.checkpoint_every} "
        f"--max-concurrency {args.max_concurrency} "
        f"--run-name {args.run_name} "
        f"--run-role {args.run_role}"
    )
    if args.benchmark_equivalent_settings:
        run_cmd += " --benchmark-equivalent-settings"
    if args.binary_only:
        run_cmd += " --binary-only"
    if args.enable_thinking:
        run_cmd += " --enable-thinking"
    if args.disable_reasoning_parser:
        run_cmd += " --disable-reasoning-parser"
    job = api.run_job(
        image="lmsysorg/sglang:v0.5.8",
        command=["bash", "-lc", f"{install_cmd} && {run_cmd}"],
        secrets={args.token_env: token},
        flavor=args.flavor,
        timeout=args.timeout,
        labels={
            "project": "phd-5c-media",
            "phase": "17",
            "privacy": "private-only",
            "purpose": "qwen-stage1-al-acquisition",
        },
        volumes=[assets_volume],
        namespace=namespace,
        token=token,
    )

    print(
        json.dumps(
            {
                "job_id": job.id,
                "job_status": str(job.status),
                "namespace": namespace,
                "assets_repo_id": args.assets_repo_id,
                "output_repo_id": args.output_repo_id,
                "model_repo": args.model_repo,
                "tokenizer_repo": args.tokenizer_repo,
                "flavor": args.flavor,
                "timeout": args.timeout,
                "n_rows": int(len(slim_df)),
                "input_csv": str(slim_input_path),
                "binary_only": bool(args.binary_only),
                "checkpoint_every": int(args.checkpoint_every),
                "max_concurrency": int(args.max_concurrency),
                "enable_thinking": bool(args.enable_thinking),
                "disable_reasoning_parser": bool(args.disable_reasoning_parser),
                "role": args.run_role,
                "benchmark_equivalent_settings": bool(args.benchmark_equivalent_settings),
                "run_name": args.run_name,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
