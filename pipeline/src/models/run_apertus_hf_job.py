import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Apertus LoRA training inside a Hugging Face Job and upload the final "
            "adapter to a private Hub model repository."
        )
    )
    parser.add_argument("--config", required=True, help="Job-local training config JSON.")
    parser.add_argument("--train-script", required=True, help="Job-local training script path.")
    parser.add_argument("--model-repo-id", required=True, help="Private HF model repo for the adapter.")
    parser.add_argument("--manifest", default=None, help="Optional SFT manifest to upload to the model repo.")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable containing HF token.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    train_script = Path(args.train_script).resolve()
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(
            f"Missing Hugging Face token in environment variable: {args.token_env}"
        )

    subprocess.run(
        [sys.executable, str(train_script), "--config", str(config_path)],
        check=True,
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"]).resolve()
    adapter_dir = output_dir / "final_adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Training finished but adapter directory is missing: {adapter_dir}")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.model_repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=args.model_repo_id,
        repo_type="model",
        commit_message="Upload first media LoRA adapter",
    )

    extra_files = [
        output_dir / "training_config_used.json",
        output_dir / "trainer_state.json",
        output_dir / "tokenization_audit.csv",
        output_dir / "tokenization_audit_summary.json",
    ]
    if args.manifest:
        extra_files.append(Path(args.manifest).resolve())

    for file_path in extra_files:
        if file_path.exists():
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=args.model_repo_id,
                repo_type="model",
            )

    print(f"Uploaded final adapter to private model repo: {args.model_repo_id}")


if __name__ == "__main__":
    main()
