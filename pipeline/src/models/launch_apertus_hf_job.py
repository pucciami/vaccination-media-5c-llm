import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data" / "processed" / "apertus_sft_media" / "media_sft_manifest.json"
DEFAULT_CONFIG = ROOT / "configs" / "apertus_5c_lora_media_first_run.json"
DEFAULT_TRAIN_SCRIPT = ROOT / "src" / "models" / "train_apertus_lora_media.py"
DEFAULT_JOB_SCRIPT = ROOT / "src" / "models" / "run_apertus_hf_job.py"
DEFAULT_TRAIN_JSONL = ROOT / "data" / "processed" / "apertus_sft_media" / "media_sft_train.jsonl"
DEFAULT_DEV_JSONL = ROOT / "data" / "processed" / "apertus_sft_media" / "media_sft_dev.jsonl"
DEFAULT_JOB_CONFIG = ROOT / "data" / "processed" / "apertus_sft_media" / "apertus_5c_lora_media_hf_job.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the private Apertus LoRA training job on Hugging Face using the "
            "adjudicated 300-article train split."
        )
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--train-script", default=str(DEFAULT_TRAIN_SCRIPT))
    parser.add_argument("--job-script", default=str(DEFAULT_JOB_SCRIPT))
    parser.add_argument("--train-jsonl", default=str(DEFAULT_TRAIN_JSONL))
    parser.add_argument("--dev-jsonl", default=str(DEFAULT_DEV_JSONL))
    parser.add_argument("--job-config-out", default=str(DEFAULT_JOB_CONFIG))
    parser.add_argument("--dataset-repo-id", default=None)
    parser.add_argument("--model-repo-id", default=None)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--flavor", default="a10g-small")
    parser.add_argument("--timeout", default="24h")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--expected-input-rows", type=int, default=300)
    parser.add_argument("--job-output-dir-remote", default="/job-output/apertus_5c_media_lora_first_run")
    return parser.parse_args()


def require_training_ready(manifest_path: Path, expected_input_rows: int | None = 300) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("training_ready", False):
        raise ValueError(
            f"SFT manifest is not training-ready: {manifest_path}. "
            "Resolve train-side issues before launching HF training."
        )
    observed_rows = manifest.get("n_input_rows", manifest.get("n_records", 0))
    if expected_input_rows is not None and int(observed_rows) != expected_input_rows:
        raise ValueError(
            f"Expected a {expected_input_rows}-row train set, found {observed_rows}."
        )
    return manifest


def write_job_config(
    *,
    base_config_path: Path,
    output_path: Path,
    train_jsonl_remote: str,
    dev_jsonl_remote: str,
    output_dir_remote: str,
) -> dict:
    config = json.loads(base_config_path.read_text(encoding="utf-8"))
    config["dataset"]["train_jsonl"] = train_jsonl_remote
    config["dataset"]["dev_jsonl"] = dev_jsonl_remote
    config["output_dir"] = output_dir_remote
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config


def upload_job_assets(
    *,
    api: HfApi,
    dataset_repo_id: str,
    train_script: Path,
    job_script: Path,
    train_jsonl: Path,
    dev_jsonl: Path,
    job_config: Path,
    manifest: Path,
) -> None:
    uploads = [
        (train_script, "train_apertus_lora_media.py"),
        (job_script, "run_apertus_hf_job.py"),
        (train_jsonl, "media_sft_train.jsonl"),
        (dev_jsonl, "media_sft_dev.jsonl"),
        (job_config, "apertus_5c_lora_media_hf_job.json"),
        (manifest, "media_sft_manifest.json"),
    ]
    for local_path, remote_name in uploads:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_name,
            repo_id=dataset_repo_id,
            repo_type="dataset",
        )


def main() -> None:
    args = parse_args()

    token = os.environ.get(args.token_env) or get_token()
    if not token:
        raise RuntimeError(
            f"Missing Hugging Face token in environment variable {args.token_env} and no cached login was found."
        )

    manifest_path = Path(args.manifest).resolve()
    config_path = Path(args.config).resolve()
    train_script = Path(args.train_script).resolve()
    job_script = Path(args.job_script).resolve()
    train_jsonl = Path(args.train_jsonl).resolve()
    dev_jsonl = Path(args.dev_jsonl).resolve()
    job_config_out = Path(args.job_config_out).resolve()

    require_training_ready(manifest_path, expected_input_rows=args.expected_input_rows)

    api = HfApi(token=token)
    whoami = api.whoami(token=token)
    namespace = args.namespace or whoami["name"]

    dataset_repo_id = args.dataset_repo_id or f"{namespace}/apertus-5c-media-sft-assets-first-run"
    model_repo_id = args.model_repo_id or f"{namespace}/apertus-5c-media-lora-first-run"

    api.create_repo(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
        token=token,
    )
    api.create_repo(
        repo_id=model_repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
        token=token,
    )

    write_job_config(
        base_config_path=config_path,
        output_path=job_config_out,
        train_jsonl_remote="/job-assets/media_sft_train.jsonl",
        dev_jsonl_remote="/job-assets/media_sft_dev.jsonl",
        output_dir_remote=args.job_output_dir_remote,
    )
    upload_job_assets(
        api=api,
        dataset_repo_id=dataset_repo_id,
        train_script=train_script,
        job_script=job_script,
        train_jsonl=train_jsonl,
        dev_jsonl=dev_jsonl,
        job_config=job_config_out,
        manifest=manifest_path,
    )

    dataset_volume = Volume(
        type="dataset",
        source=dataset_repo_id,
        mount_path="/job-assets",
    )
    install_cmd = (
        "python -m pip install --no-input accelerate bitsandbytes datasets peft "
        "transformers huggingface_hub"
    )
    train_cmd = (
        "python /job-assets/run_apertus_hf_job.py "
        "--config /job-assets/apertus_5c_lora_media_hf_job.json "
        "--train-script /job-assets/train_apertus_lora_media.py "
        f"--model-repo-id {model_repo_id} "
        "--manifest /job-assets/media_sft_manifest.json"
    )
    job = api.run_job(
        image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        command=["bash", "-lc", f"{install_cmd} && {train_cmd}"],
        secrets={args.token_env: token},
        flavor=args.flavor,
        timeout=args.timeout,
        labels={
            "project": "phd-5c-media",
            "phase": "10",
            "privacy": "private-only",
            "source_tier": "mainstream_media",
        },
        volumes=[dataset_volume],
        namespace=namespace,
        token=token,
    )

    print(json.dumps({
        "job_id": job.id,
        "job_status": str(job.status),
        "namespace": namespace,
        "dataset_repo_id": dataset_repo_id,
        "model_repo_id": model_repo_id,
        "flavor": args.flavor,
        "timeout": args.timeout,
        "expected_input_rows": args.expected_input_rows,
        "job_output_dir_remote": args.job_output_dir_remote,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
