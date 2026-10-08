import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOB_SCRIPT = ROOT / "src" / "models" / "run_apertus_lora_merge_hf_job.py"
DEFAULT_SYSTEM_PROMPT = ROOT / "configs" / "apertus_5c_sft_system_prompt_v7_broad_relevance.txt"
DEFAULT_SFT_MANIFEST = ROOT / "data" / "processed" / "apertus_sft_media" / "media_sft_manifest.json"
DEFAULT_BASE_MODEL = "swiss-ai/Apertus-8B-Instruct-2509"
DEFAULT_ADAPTER_REPO = "Pucciami/apertus-5c-media-lora-first-run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a private Hugging Face Job that merges the first Apertus media "
            "LoRA adapter into the base model and uploads the merged model privately."
        )
    )
    parser.add_argument("--job-script", default=str(DEFAULT_JOB_SCRIPT))
    parser.add_argument("--system-prompt", default=str(DEFAULT_SYSTEM_PROMPT))
    parser.add_argument("--sft-manifest", default=str(DEFAULT_SFT_MANIFEST))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-repo", default=DEFAULT_ADAPTER_REPO)
    parser.add_argument("--assets-repo-id", default=None)
    parser.add_argument("--merged-repo-id", default=None)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--flavor", default="a10g-small")
    parser.add_argument("--timeout", default="4h")
    parser.add_argument("--token-env", default="HF_TOKEN")
    return parser.parse_args()


def upload_job_assets(
    *,
    api: HfApi,
    assets_repo_id: str,
    job_script: Path,
    system_prompt: Path,
    sft_manifest: Path,
) -> None:
    uploads = [
        (job_script, "run_apertus_lora_merge_hf_job.py"),
        (system_prompt, "apertus_5c_sft_system_prompt.txt"),
        (sft_manifest, "media_sft_manifest.json"),
    ]
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

    job_script = Path(args.job_script).resolve()
    system_prompt = Path(args.system_prompt).resolve()
    sft_manifest = Path(args.sft_manifest).resolve()

    api = HfApi(token=token)
    whoami = api.whoami(token=token)
    namespace = args.namespace or whoami["name"]

    assets_repo_id = args.assets_repo_id or f"{namespace}/apertus-5c-media-merge-assets-first-run"
    merged_repo_id = args.merged_repo_id or f"{namespace}/apertus-5c-media-merged-first-run"

    api.create_repo(
        repo_id=assets_repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
        token=token,
    )
    api.create_repo(
        repo_id=merged_repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
        token=token,
    )

    upload_job_assets(
        api=api,
        assets_repo_id=assets_repo_id,
        job_script=job_script,
        system_prompt=system_prompt,
        sft_manifest=sft_manifest,
    )

    assets_volume = Volume(
        type="dataset",
        source=assets_repo_id,
        mount_path="/job-assets",
    )
    install_cmd = "python -m pip install --no-input accelerate peft transformers huggingface_hub safetensors"
    merge_cmd = (
        "python /job-assets/run_apertus_lora_merge_hf_job.py "
        f"--base-model {args.base_model} "
        f"--adapter-repo {args.adapter_repo} "
        f"--merged-repo-id {merged_repo_id} "
        "--system-prompt-path /job-assets/apertus_5c_sft_system_prompt.txt "
        "--manifest-path /job-assets/media_sft_manifest.json"
    )
    job = api.run_job(
        image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        command=["bash", "-lc", f"{install_cmd} && {merge_cmd}"],
        secrets={args.token_env: token},
        flavor=args.flavor,
        timeout=args.timeout,
        labels={
            "project": "phd-5c-media",
            "phase": "12",
            "privacy": "private-only",
            "purpose": "merge-adapter-for-endpoint",
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
                "assets_repo_id": assets_repo_id,
                "adapter_repo": args.adapter_repo,
                "merged_repo_id": merged_repo_id,
                "flavor": args.flavor,
                "timeout": args.timeout,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
