import argparse
import json
import os
import time

from huggingface_hub import HfApi, get_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume an existing Hugging Face Inference Endpoint and print its current state."
    )
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--namespace", default=None)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--running-ok", action="store_true", help="Treat an already-running endpoint as success.")
    parser.add_argument("--wait-running", action="store_true", help="Poll until the endpoint reaches running state.")
    parser.add_argument("--wait-timeout", type=int, default=1800, help="Max seconds to wait when --wait-running is used.")
    parser.add_argument("--poll-seconds", type=int, default=15, help="Polling interval in seconds.")
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

    endpoint = api.get_inference_endpoint(args.endpoint_name, namespace=namespace, token=token)
    state = str(endpoint.status)

    if "running" in state.lower() and args.running_ok:
        final_endpoint = endpoint
    else:
        final_endpoint = api.resume_inference_endpoint(args.endpoint_name, namespace=namespace, token=token)

    if args.wait_running:
        deadline = time.time() + args.wait_timeout
        while time.time() < deadline:
            current = api.get_inference_endpoint(args.endpoint_name, namespace=namespace, token=token)
            if "running" in str(current.status).lower():
                final_endpoint = current
                break
            time.sleep(args.poll_seconds)
        else:
            final_endpoint = api.get_inference_endpoint(args.endpoint_name, namespace=namespace, token=token)

    print(
        json.dumps(
            {
                "endpoint_name": final_endpoint.name,
                "namespace": namespace,
                "repository": getattr(final_endpoint, "repository", None),
                "status": str(final_endpoint.status),
                "url": getattr(final_endpoint, "url", None),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
