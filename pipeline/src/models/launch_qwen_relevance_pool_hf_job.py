import argparse
import json
import os
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, Volume, get_token


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a private HF Job to run stage-1 Qwen relevance over the post-gold active-learning pool."
    )
    parser.add_argument("--job-script", default=str(ROOT / "src" / "models" / "run_qwen_relevance_pool_hf_job.py"))
    parser.add_argument("--pool-csv", default=str(ROOT / "data" / "processed" / "active_learning" / "media_active_learning_pool_v1.csv"))
    parser.add_argument("--prompt-path", default=str(ROOT / "configs" / "apertus_relevance_gate_prompt_v2_broad_vaccination.md"))
    parser.add_argument("--schema-path", default=str(ROOT / "configs" / "apertus_relevance_gate_schema_v2_broad_vaccination.json"))
    parser.add_argument("--assets-repo-id", default="Pucciami/qwen3-32b-relevance-pool-assets-v2-broad")
    parser.add_argument("--output-repo-id", default="Pucciami/qwen3-32b-relevance-pool-output-v2-broad")
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--flavor", default="h200")
    parser.add_argument("--timeout", default="24h")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--model-repo", default="Qwen/Qwen3-32B")
    parser.add_argument("--text-column", default="model_text")
    parser.add_argument("--id-column", default="doc_id")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--max-errors", type=int, default=0)
    parser.add_argument("--run-name", default="qwen_stage1_relevance_pool_v1")
    parser.add_argument("--disable-thinking", action="store_true")
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
        (Path(args.job_script).resolve(), "run_qwen_relevance_pool_hf_job.py"),
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
    install_cmd = "python -m pip install --no-input accelerate transformers huggingface_hub pandas"
    run_cmd = (
        "python /job-assets/run_qwen_relevance_pool_hf_job.py "
        f"--model-repo {args.model_repo} "
        f"--input-csv /job-assets/{slim_input_path.name} "
        f"--prompt-path /job-assets/{Path(args.prompt_path).name} "
        f"--schema-path /job-assets/{Path(args.schema_path).name} "
        f"--private-output-repo {args.output_repo_id} "
        f"--text-column {args.text_column} "
        f"--id-column {args.id_column} "
        f"--max-new-tokens {args.max_new_tokens} "
        f"--temperature {args.temperature} "
        f"--top-p {args.top_p} "
        f"--limit {args.limit} "
        f"--checkpoint-every {args.checkpoint_every} "
        f"--max-errors {args.max_errors} "
        f"--run-name {args.run_name}"
    )
    if args.disable_thinking:
        run_cmd += " --disable-thinking"
    job = api.run_job(
        image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        command=["bash", "-lc", f"{install_cmd} && {run_cmd}"],
        secrets={args.token_env: token},
        flavor=args.flavor,
        timeout=args.timeout,
        labels={
            "project": "phd-5c-media",
            "phase": "17",
            "privacy": "private-only",
            "purpose": "qwen-stage1-relevance-pool",
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
                "flavor": args.flavor,
                "timeout": args.timeout,
                "n_rows": int(len(slim_df)),
                "input_csv": str(slim_input_path),
                "checkpoint_every": int(args.checkpoint_every),
                "max_errors": int(args.max_errors),
                "run_name": args.run_name,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
