import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a private HF Job to run locked gold100 evaluation against the merged Qwen media model."
    )
    parser.add_argument("--job-script", default=str(ROOT / "src" / "models" / "run_qwen_merged_eval_hf_job.py"))
    parser.add_argument("--evaluator-script", default=str(ROOT / "src" / "eval" / "evaluate_apertus_zero_shot.py"))
    parser.add_argument("--system-prompt", default=str(ROOT / "configs" / "apertus_5c_sft_system_prompt_v7_stage2_forced_relevant.txt"))
    parser.add_argument("--input-csv", default=str(ROOT / "data" / "locked" / "gold100" / "gold100_test_apertus_input_clean.csv"))
    parser.add_argument("--gold-csv", default=str(ROOT / "data" / "locked" / "gold100" / "gold100_test_gold_labels.csv"))
    parser.add_argument("--merged-model-repo", default="Pucciami/qwen3-32b-5c-media-merged-h200-first-run")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter-repo", default=None)
    parser.add_argument("--assets-repo-id", default="Pucciami/qwen3-32b-5c-media-eval-assets-first-run")
    parser.add_argument("--output-repo-id", default="Pucciami/qwen3-32b-5c-media-eval-first-run")
    parser.add_argument("--output-prefix", default="gold100_qwen_lora_adapted")
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--flavor", default="h200")
    parser.add_argument("--timeout", default="6h")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--retry-max-new-tokens", default="")
    parser.add_argument(
        "--force-relevant-output",
        action="store_true",
        help="Pass through to the runner for forced-relevant stage-2 extractor evaluation.",
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

    use_base_plus_adapter = bool(args.base_model and args.adapter_repo)
    use_merged_repo = (not use_base_plus_adapter) and bool(args.merged_model_repo)
    if not use_merged_repo and not use_base_plus_adapter:
        raise ValueError("Provide either --merged-model-repo or both --base-model and --adapter-repo.")

    job_script_path = Path(args.job_script).resolve()
    evaluator_script_path = Path(args.evaluator_script).resolve()
    system_prompt_path = Path(args.system_prompt).resolve()
    input_csv_path = Path(args.input_csv).resolve()
    gold_csv_path = Path(args.gold_csv).resolve()

    remote_job_script = job_script_path.name
    remote_evaluator_script = evaluator_script_path.name
    remote_system_prompt = system_prompt_path.name
    remote_input_csv = input_csv_path.name
    remote_gold_csv = gold_csv_path.name

    uploads = [
        (job_script_path, remote_job_script),
        (evaluator_script_path, remote_evaluator_script),
        (system_prompt_path, remote_system_prompt),
        (input_csv_path, remote_input_csv),
        (gold_csv_path, remote_gold_csv),
    ]
    upload_job_assets(api, args.assets_repo_id, uploads)

    assets_volume = Volume(
        type="dataset",
        source=args.assets_repo_id,
        mount_path="/job-assets",
    )
    install_cmd = "python -m pip install --no-input accelerate peft transformers huggingface_hub pandas"
    model_arg = f"--model-repo {args.merged_model_repo}" if use_merged_repo else (
        f"--base-model {args.base_model} --adapter-repo {args.adapter_repo}"
    )
    eval_cmd = (
        f"python /job-assets/{remote_job_script} "
        f"{model_arg} "
        f"--input-csv /job-assets/{remote_input_csv} "
        f"--gold-csv /job-assets/{remote_gold_csv} "
        f"--system-prompt-path /job-assets/{remote_system_prompt} "
        f"--evaluator-script /job-assets/{remote_evaluator_script} "
        f"--private-output-repo {args.output_repo_id} "
        f"--output-prefix {args.output_prefix} "
        f"--max-new-tokens {args.max_new_tokens} "
        f"--retry-max-new-tokens {args.retry_max_new_tokens}"
    )
    if args.force_relevant_output:
        eval_cmd += " --force-relevant-output"
    job = api.run_job(
        image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        command=["bash", "-lc", f"{install_cmd} && {eval_cmd}"],
        secrets={args.token_env: token},
        flavor=args.flavor,
        timeout=args.timeout,
        labels={
            "project": "phd-5c-media",
            "phase": "16",
            "privacy": "private-only",
            "purpose": "qwen-merged-locked-eval",
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
        "merged_model_repo": args.merged_model_repo,
        "base_model": args.base_model,
        "adapter_repo": args.adapter_repo,
        "flavor": args.flavor,
        "timeout": args.timeout,
        "max_new_tokens": args.max_new_tokens,
        "retry_max_new_tokens": args.retry_max_new_tokens,
        "force_relevant_output": bool(args.force_relevant_output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
