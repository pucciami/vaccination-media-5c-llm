import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a private HF Job to run locked gold100 evaluation for adapted Apertus."
    )
    parser.add_argument("--job-script", default=str(ROOT / "src" / "models" / "run_apertus_adapter_eval_hf_job.py"))
    parser.add_argument("--evaluator-script", default=str(ROOT / "src" / "eval" / "evaluate_apertus_zero_shot.py"))
    parser.add_argument("--system-prompt", default=str(ROOT / "configs" / "apertus_5c_sft_system_prompt_v7_stage2_forced_relevant.txt"))
    parser.add_argument("--input-csv", default=str(ROOT / "data" / "locked" / "gold100" / "gold100_test_apertus_input_clean_v2.csv"))
    parser.add_argument("--gold-csv", default=str(ROOT / "data" / "locked" / "gold100" / "gold100_test_gold_labels.csv"))
    parser.add_argument("--base-model", default="swiss-ai/Apertus-8B-Instruct-2509")
    parser.add_argument("--adapter-repo", default=None)
    parser.add_argument("--tokenizer-repo", default=None)
    parser.add_argument("--assets-repo-id", default="Pucciami/apertus-5c-media-eval-clean-v2-al-v1-stage2-forced-assets")
    parser.add_argument("--output-repo-id", default="Pucciami/apertus-5c-media-eval-clean-v2-al-v1-stage2-forced-output")
    parser.add_argument("--output-prefix", default="gold100_apertus_lora_clean_v2_al_v1_stage2_forced")
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--flavor", default="a10g-small")
    parser.add_argument("--timeout", default="4h")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--retry-max-new-tokens", default="1024,2048")
    parser.add_argument("--generation-max-time", type=float, default=0.0)
    parser.add_argument(
        "--assistant-json-prefill-mode",
        choices=["none", "relevant_hierarchy_prefix"],
        default="none",
    )
    parser.add_argument(
        "--force-relevant-output",
        action="store_true",
        help="Pass through to the runner for forced-relevant stage-2 extractor evaluation.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass through to the runner so Apertus reasons before emitting JSON (reasoning-parity side experiment).",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Pass through to the runner to use vendor-recommended sampling instead of greedy decoding.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-eval", action="store_true")
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
    evaluator_script_path = Path(args.evaluator_script).resolve()
    system_prompt_path = Path(args.system_prompt).resolve()
    input_csv_path = Path(args.input_csv).resolve()
    gold_csv_path = Path(args.gold_csv).resolve()

    uploads = [
        (job_script_path, job_script_path.name),
        (evaluator_script_path, evaluator_script_path.name),
        (system_prompt_path, system_prompt_path.name),
        (input_csv_path, input_csv_path.name),
        (gold_csv_path, gold_csv_path.name),
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
    eval_cmd = (
        f"python /job-assets/{job_script_path.name} "
        f"--base-model {args.base_model} "
        f"{adapter_arg}"
        f"{tokenizer_arg}"
        f"--input-csv /job-assets/{input_csv_path.name} "
        f"--gold-csv /job-assets/{gold_csv_path.name} "
        f"--system-prompt-path /job-assets/{system_prompt_path.name} "
        f"--evaluator-script /job-assets/{evaluator_script_path.name} "
        f"--private-output-repo {args.output_repo_id} "
        f"--output-prefix {args.output_prefix} "
        f"--max-new-tokens {args.max_new_tokens} "
        f"--retry-max-new-tokens {args.retry_max_new_tokens} "
        f"--generation-max-time {args.generation_max_time} "
        f"--assistant-json-prefill-mode {args.assistant_json_prefill_mode}"
    )
    if args.force_relevant_output:
        eval_cmd += " --force-relevant-output"
    if args.enable_thinking:
        eval_cmd += " --enable-thinking"
    if args.do_sample:
        eval_cmd += f" --do-sample --temperature {args.temperature} --top-p {args.top_p} --seed {args.seed}"
    if args.skip_eval:
        eval_cmd += " --skip-eval"

    job = api.run_job(
        image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
        command=["bash", "-lc", f"{install_cmd} && {eval_cmd}"],
        secrets={args.token_env: token},
        flavor=args.flavor,
        timeout=args.timeout,
        labels={
            "project": "phd-5c-media",
            "phase": "19",
            "privacy": "private-only",
            "purpose": "apertus-adapter-stage2-forced-eval",
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
        "force_relevant_output": bool(args.force_relevant_output),
        "enable_thinking": bool(args.enable_thinking),
        "skip_eval": bool(args.skip_eval),
        "generation_max_time": args.generation_max_time,
        "assistant_json_prefill_mode": args.assistant_json_prefill_mode,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
