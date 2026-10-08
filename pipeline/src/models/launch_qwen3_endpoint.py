import argparse
import json
import os

import httpx
from huggingface_hub import HfApi, get_token
from huggingface_hub.errors import HfHubHTTPError


DEFAULT_ENDPOINT_NAME = "qwen3-32b-relevance-gate"
DEFAULT_MODEL_REPO = "Qwen/Qwen3-32B-FP8"
DEFAULT_TOKENIZER_REPO = "Qwen/Qwen3-32B"
DEFAULT_REGION = "us-east-1"
DEFAULT_VENDOR = "aws"
DEFAULT_INSTANCE_TYPE = "nvidia-l4"
DEFAULT_INSTANCE_SIZE = "x2"
DEFAULT_CONTAINER_URL = "lmsysorg/sglang:v0.5.8"
DEFAULT_REASONING_PARSER = "qwen3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a private Hugging Face Inference Endpoint for Qwen3 relevance-gate evaluation."
    )
    parser.add_argument("--endpoint-name", default=DEFAULT_ENDPOINT_NAME)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--tokenizer-repo", default=DEFAULT_TOKENIZER_REPO)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--vendor", default=DEFAULT_VENDOR)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--instance-size", default=DEFAULT_INSTANCE_SIZE)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--min-replica", type=int, default=0)
    parser.add_argument("--max-replica", type=int, default=1)
    parser.add_argument("--scale-to-zero-timeout", type=int, default=15)
    parser.add_argument("--type", default="private")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--reasoning-parser", default=DEFAULT_REASONING_PARSER)
    parser.add_argument("--container-url", default=DEFAULT_CONTAINER_URL)
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
        "repository": args.model_repo,
        "task": "text-generation",
        "framework": "pytorch",
        "image": {
            "sGLang": {
                "healthRoute": "/health",
                "port": 30000,
                "url": args.container_url,
                "tensorParallelSize": args.tensor_parallel_size,
            }
        },
        "args": [
            "--trust-remote-code",
            "--reasoning-parser",
            args.reasoning_parser,
            "--tokenizer-path",
            args.tokenizer_repo,
        ],
        "env": {},
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
            "phase-14",
            "private-only",
            "qwen3",
            "relevance-gate",
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
                "repository": args.model_repo,
                "tokenizer_repo": args.tokenizer_repo,
                "status": endpoint["status"]["state"],
                "url": endpoint["status"]["url"],
                "vendor": args.vendor,
                "region": args.region,
                "instance_type": args.instance_type,
                "instance_size": args.instance_size,
                "tensor_parallel_size": args.tensor_parallel_size,
                "type": args.type,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
