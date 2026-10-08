from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a private HF Job to run checkpointed stage-2 active-learning inference with the merged Qwen model."
    )
    parser.add_argument("--job-script", default=str(ROOT / "src" / "models" / "run_qwen_active_learning_stage2_hf_job.py"))
    parser.add_argument(
        "--input-csv",
        default=str(ROOT / "data" / "processed" / "active_learning" / "media_active_learning_stage2_subset_v1.csv"),
    )
    parser.add_argument(
        "--system-prompt",
        default=str(ROOT / "configs" / "apertus_5c_sft_system_prompt_v7_stage2_forced_relevant.txt"),
    )
    parser.add_argument(
        "--merged-model-repo",
        default="Pucciami/qwen3-32b-5c-media-merged-h200-first-run",
    )
    parser.add_argument("--tokenizer-repo", default="Qwen/Qwen3-32B")
    parser.add_argument("--assets-repo-id", default="Pucciami/qwen3-32b-stage2-assets-v1")
    parser.add_argument("--output-repo-id", default="Pucciami/qwen3-32b-stage2-output-v1")
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--flavor", default="h200")
    parser.add_argument("--timeout", default="16h")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--text-column", default="model_text")
    parser.add_argument("--id-column", default="doc_id")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--retry-max-new-tokens", default="512")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-name", default="qwen_active_learning_stage2_v1")
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

    api.create_repo(repo_id=args.assets_repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)
    api.create_repo(repo_id=args.output_repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)

    input_path = Path(args.input_csv).resolve()
    prompt_path = Path(args.system_prompt).resolve()
    uploads = [
        (Path(args.job_script).resolve(), "run_qwen_active_learning_stage2_hf_job.py"),
        (input_path, input_path.name),
        (prompt_path, prompt_path.name),
    ]
    upload_job_assets(api, args.assets_repo_id, uploads)

    assets_volume = Volume(type="dataset", source=args.assets_repo_id, mount_path="/job-assets")
    install_cmd = "python -m pip install --no-input accelerate transformers huggingface_hub pandas"
    run_cmd = (
        "python /job-assets/run_qwen_active_learning_stage2_hf_job.py "
        f"--model-repo {args.merged_model_repo} "
        f"--tokenizer-repo {args.tokenizer_repo} "
        f"--input-csv /job-assets/{input_path.name} "
        f"--system-prompt-path /job-assets/{prompt_path.name} "
        f"--private-output-repo {args.output_repo_id} "
        f"--text-column {args.text_column} "
        f"--id-column {args.id_column} "
        f"--checkpoint-every {args.checkpoint_every} "
        f"--max-new-tokens {args.max_new_tokens} "
        f"--retry-max-new-tokens {args.retry_max_new_tokens} "
        f"--limit {args.limit} "
        f"--run-name {args.run_name}"
    )

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
            "purpose": "qwen-stage2-al-acquisition",
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
                "merged_model_repo": args.merged_model_repo,
                "tokenizer_repo": args.tokenizer_repo,
                "n_rows": sum(1 for _ in open(input_path, "r", encoding="utf-8")) - 1,
                "flavor": args.flavor,
                "timeout": args.timeout,
                "role": "active_learning_hierarchy_selector",
                "benchmark_equivalent": False,
                "run_name": args.run_name,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
