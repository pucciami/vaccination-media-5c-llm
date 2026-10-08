#!/usr/bin/env python3
"""Launch the NLLB-200 translation of gold100 SL->EN as an HF Job.

Uploads the Slovenian source CSV + the job runner to a private assets dataset,
runs NLLB on a small GPU, and the job uploads the English model-input CSV to a
private output dataset. Decoupled from the Qwen classifier under evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token

HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--job-script", default=str(HERE / "run_nllb_translation_hf_job.py"))
    p.add_argument("--input-csv", default=str(HERE.parent / "outputs" / "gold100_sl_source_for_translation.csv"))
    p.add_argument("--model", default="facebook/nllb-200-distilled-1.3B")
    p.add_argument("--assets-repo-id", default="Pucciami/gold100-en-translation-assets-v1")
    p.add_argument("--output-repo-id", default="Pucciami/gold100-en-translation-output-v1")
    p.add_argument("--namespace", default=None)
    p.add_argument("--flavor", default="t4-small")
    p.add_argument("--timeout", default="2h")
    p.add_argument("--token-env", default="HF_TOKEN")
    p.add_argument("--image", default="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get(args.token_env) or get_token()
    if not token:
        raise RuntimeError("No HF token (HF_TOKEN env or cached login).")
    api = HfApi(token=token)
    namespace = args.namespace or api.whoami(token=token)["name"]

    input_path = Path(args.input_csv).resolve()
    script_path = Path(args.job_script).resolve()

    install_cmd = "python -m pip install --no-input --quiet transformers sentencepiece huggingface_hub pandas"
    run_cmd = (
        "python /job-assets/run_nllb_translation_hf_job.py "
        f"--input-csv /job-assets/{input_path.name} "
        f"--private-output-repo {args.output_repo_id} "
        f"--model {args.model} "
        "--text-column text_sl --id-column doc_id"
    )
    full_cmd = f"{install_cmd} && {run_cmd}"

    print(json.dumps({
        "namespace": namespace, "assets_repo": args.assets_repo_id, "output_repo": args.output_repo_id,
        "flavor": args.flavor, "image": args.image, "model": args.model,
        "input_csv": str(input_path), "command": full_cmd,
    }, indent=2))
    if args.dry_run:
        print("\n[dry-run] not launching.")
        return

    api.create_repo(repo_id=args.assets_repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)
    api.create_repo(repo_id=args.output_repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)
    for local, remote in [(script_path, script_path.name), (input_path, input_path.name)]:
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=args.assets_repo_id, repo_type="dataset", token=token)

    job = api.run_job(
        image=args.image,
        command=["bash", "-lc", full_cmd],
        secrets={args.token_env: token},
        flavor=args.flavor,
        timeout=args.timeout,
        labels={"project": "phd-5c-media", "phase": "english-sensitivity", "purpose": "nllb-translation"},
        volumes=[Volume(type="dataset", source=args.assets_repo_id, mount_path="/job-assets")],
        namespace=namespace,
        token=token,
    )
    print(json.dumps({"job_id": job.id, "job_status": str(job.status), "output_repo": args.output_repo_id}, indent=2))


if __name__ == "__main__":
    main()
