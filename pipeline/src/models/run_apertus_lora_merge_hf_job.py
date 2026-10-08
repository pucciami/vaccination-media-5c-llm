import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the first private Apertus media LoRA adapter into the base model "
            "inside a Hugging Face Job and upload the merged model to a private repo."
        )
    )
    parser.add_argument("--base-model", required=True, help="Base model repo id.")
    parser.add_argument("--adapter-repo", required=True, help="Private adapter repo id.")
    parser.add_argument("--merged-repo-id", required=True, help="Private repo id for the merged model.")
    parser.add_argument("--output-dir", default="/job-output/apertus_5c_media_merged_first_run")
    parser.add_argument("--system-prompt-path", default=None, help="Optional prompt file to upload with the model.")
    parser.add_argument("--manifest-path", default=None, help="Optional training manifest to upload with the model.")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable containing the HF token.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"Missing Hugging Face token in environment variable: {args.token_env}")

    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, token=token)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        token=token,
    )
    peft_model = PeftModel.from_pretrained(
        base_model,
        args.adapter_repo,
        token=token,
    )
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(
        str(output_dir),
        safe_serialization=True,
        max_shard_size="5GB",
    )
    tokenizer.save_pretrained(str(output_dir))

    merge_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "adapter_repo": args.adapter_repo,
        "merged_repo_id": args.merged_repo_id,
        "output_dir": str(output_dir),
        "dtype": "bfloat16",
        "safe_serialization": True,
        "max_shard_size": "5GB",
    }
    merge_manifest_path = output_dir / "merge_manifest.json"
    merge_manifest_path.write_text(
        json.dumps(merge_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.merged_repo_id,
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=args.merged_repo_id,
        repo_type="model",
        commit_message="Upload merged Apertus media model",
    )

    extra_files = []
    if args.system_prompt_path:
        extra_files.append(Path(args.system_prompt_path).resolve())
    if args.manifest_path:
        extra_files.append(Path(args.manifest_path).resolve())

    for file_path in extra_files:
        if file_path.exists():
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_path.name,
                repo_id=args.merged_repo_id,
                repo_type="model",
            )

    print(f"Uploaded merged model to private repo: {args.merged_repo_id}")


if __name__ == "__main__":
    main()
