import argparse
import json
import os

from huggingface_hub import HfApi, get_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pause an existing Hugging Face Inference Endpoint and print its resulting state."
    )
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--namespace", default=None)
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
    endpoint = api.pause_inference_endpoint(args.endpoint_name, namespace=namespace, token=token)

    print(
        json.dumps(
            {
                "endpoint_name": endpoint.name,
                "namespace": namespace,
                "repository": getattr(endpoint, "repository", None),
                "status": str(endpoint.status),
                "url": getattr(endpoint, "url", None),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
