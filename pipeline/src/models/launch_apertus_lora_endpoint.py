import argparse
import json
import os

import httpx
from huggingface_hub import HfApi, get_token
from huggingface_hub.errors import HfHubHTTPError


DEFAULT_ENDPOINT_NAME = "apertus-5c-media-lora-first-run"
DEFAULT_BASE_REPO = "swiss-ai/Apertus-8B-Instruct-2509"
DEFAULT_ADAPTER_REPO = "Pucciami/apertus-5c-media-lora-first-run"
DEFAULT_ADAPTER_NAME = "apertus-5c-media-lora-first-run"
DEFAULT_REGION = "us-east-1"
DEFAULT_VENDOR = "aws"
DEFAULT_INSTANCE_TYPE = "nvidia-l4"
DEFAULT_INSTANCE_SIZE = "x1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a private Hugging Face Inference Endpoint for the adapted "
            "Apertus media model."
        )
    )
    parser.add_argument("--endpoint-name", default=DEFAULT_ENDPOINT_NAME)
    parser.add_argument(
        "--deployment-mode",
        choices=["sglang-lora", "merged-repo"],
        default="sglang-lora",
        help="Use base+LoRA serving or serve a merged private model repo directly.",
    )
    parser.add_argument("--base-repo", default=DEFAULT_BASE_REPO)
    parser.add_argument(
        "--tokenizer-repo",
        default=None,
        help="Optional tokenizer repo/path override, useful when serving a merged model repo.",
    )
    parser.add_argument("--adapter-repo", default=DEFAULT_ADAPTER_REPO)
    parser.add_argument("--adapter-name", default=DEFAULT_ADAPTER_NAME)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--vendor", default=DEFAULT_VENDOR)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--instance-size", default=DEFAULT_INSTANCE_SIZE)
    parser.add_argument("--min-replica", type=int, default=0)
    parser.add_argument("--max-replica", type=int, default=1)
    parser.add_argument("--scale-to-zero-timeout", type=int, default=15)
    parser.add_argument("--type", default="private")
    parser.add_argument("--token-env", default="HF_TOKEN")
    return parser.parse_args()


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

    try:
        api.delete_inference_endpoint(args.endpoint_name, namespace=namespace, token=token)
    except HfHubHTTPError:
        pass

    model_payload = {
        "repository": args.base_repo,
        "task": "text-generation",
        "framework": "pytorch",
        "image": {
            "sGLang": {
                "healthRoute": "/health",
                "port": 30000,
                "url": "lmsysorg/sglang:v0.5.8",
                "tensorParallelSize": 1,
            }
        },
        "env": {},
    }
    if args.deployment_mode == "sglang-lora":
        model_payload["args"] = [
            "--trust-remote-code",
            "--lora-paths",
            f"{args.adapter_name}={args.adapter_repo}",
            "--max-loras-per-batch",
            "1",
        ]
        model_payload["secrets"] = {
            "HF_TOKEN": token,
        }
    else:
        tokenizer_repo = args.tokenizer_repo or DEFAULT_BASE_REPO
        model_payload["args"] = [
            "--trust-remote-code",
            "--tokenizer-path",
            tokenizer_repo,
        ]
        model_payload["secrets"] = {
            "HF_TOKEN": token,
        }

    payload = {
        "compute": {
            "accelerator": "gpu",
            "instanceSize": args.instance_size,
            "instanceType": args.instance_type,
            "scaling": {
                "minReplica": args.min_replica,
                "maxReplica": args.max_replica,
                "scaleToZeroTimeout": args.scale_to_zero_timeout,
            },
        },
        "model": model_payload,
        "name": args.endpoint_name,
        "provider": {
            "region": args.region,
            "vendor": args.vendor,
        },
        "type": args.type,
        "tags": [
            "phd-5c-media",
            "phase-12",
            "private-only",
            "lora-eval",
        ],
    }

    response = httpx.post(
        f"https://api.endpoints.huggingface.cloud/v2/endpoint/{namespace}",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    endpoint = response.json()

    print(
        json.dumps(
            {
                "endpoint_name": endpoint["name"],
                "namespace": namespace,
                "deployment_mode": args.deployment_mode,
                "repository": args.base_repo,
                "tokenizer_repo": args.tokenizer_repo,
                "adapter_repo": args.adapter_repo,
                "adapter_name": args.adapter_name,
                "status": endpoint["status"]["state"],
                "url": endpoint["status"]["url"],
                "vendor": args.vendor,
                "region": args.region,
                "instance_type": args.instance_type,
                "instance_size": args.instance_size,
                "type": args.type,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
