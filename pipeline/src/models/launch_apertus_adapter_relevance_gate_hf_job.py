import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a private HF Job for adapted-Apertus relevance-only gating."
    )
    parser.add_argument("--job-script", default=str(ROOT / "src" / "models" / "run_apertus_adapter_relevance_gate_hf_job.py"))
    parser.add_argument("--base-model", default="swiss-ai/Apertus-8B-Instruct-2509")
    parser.add_argument("--adapter-repo", default=None)
    parser.add_argument("--tokenizer-repo", default=None)
    parser.add_argument("--input-csv", default=str(ROOT / "data" / "locked" / "gold100" / "gold100_test_apertus_input_clean_v2.csv"))
    parser.add_argument("--prompt-path", default=str(ROOT / "configs" / "apertus_relevance_gate_prompt_v2_broad_vaccination.md"))
    parser.add_argument("--assets-repo-id", default="Pucciami/apertus-adapter-relevance-gold100-clean-v2-al-v1-assets-v2-broad")
    parser.add_argument("--output-repo-id", default="Pucciami/apertus-adapter-relevance-gold100-clean-v2-al-v1-output-v2-broad")
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--flavor", default="a10g-small")
    parser.add_argument("--timeout", default="4h")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="doc_id")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--retry-max-new-tokens", default="1024,2048")
    parser.add_argument("--run-name", default="apertus_adapter_relevance_gold100_clean_v2_al_v1_v2_broad")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass through to the gate runner so Apertus deliberates before the relevance JSON.",
    )
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

    job_script_path = Path(args.job_script).resolve()
    input_csv_path = Path(args.input_csv).resolve()
    prompt_path = Path(args.prompt_path).resolve()

    uploads = [
        (job_script_path, job_script_path.name),
        (input_csv_path, input_csv_path.name),
        (prompt_path, prompt_path.name),
    ]
    upload_job_assets(api, args.assets_repo_id, uploads)

    assets_volume = Volume(
        type="dataset",
        source=args.assets_repo_id,
        mount_path="/job-assets",
    )
    install_cmd = "python -m pip install --no-input accelerate peft transformers huggingface_hub pandas"
    tokenizer_arg = f"--tokenizer-repo {args.tokenizer_repo} " if args.tokenizer_repo else ""
    adapter_arg = f"--adapter-repo {args.adapter_repo} " if args.adapter_repo else ""
    run_cmd = (
        f"python /job-assets/{job_script_path.name} "
        f"--base-model {args.base_model} "
        f"{adapter_arg}"
        f"{tokenizer_arg}"
        f"--input-csv /job-assets/{input_csv_path.name} "
        f"--prompt-path /job-assets/{prompt_path.name} "
        f"--private-output-repo {args.output_repo_id} "
        f"--text-column {args.text_column} "
        f"--id-column {args.id_column} "
        f"--max-new-tokens {args.max_new_tokens} "
        f"--retry-max-new-tokens {args.retry_max_new_tokens} "
        f"--run-name {args.run_name}"
    )
    if args.enable_thinking:
        run_cmd += " --enable-thinking"

    job = api.run_job(
        image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        command=["bash", "-lc", f"{install_cmd} && {run_cmd}"],
        secrets={args.token_env: token},
        flavor=args.flavor,
        timeout=args.timeout,
        labels={
            "project": "phd-5c-media",
            "phase": "19",
            "privacy": "private-only",
            "purpose": "apertus-adapter-relevance-gate",
        },
        volumes=[assets_volume],
        namespace=namespace,
        token=token,
    )

    print(json.dumps({
        "job_id": job.id,
        "job_status": str(job.status),
        "namespace": namespace,
        "assets_repo_id": args.assets_repo_id,
        "output_repo_id": args.output_repo_id,
        "base_model": args.base_model,
        "adapter_repo": args.adapter_repo,
        "generation_mode": "base_plus_adapter" if args.adapter_repo else "base_only",
        "flavor": args.flavor,
        "timeout": args.timeout,
        "run_name": args.run_name,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
