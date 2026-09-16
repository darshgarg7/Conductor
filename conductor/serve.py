"""Launch one bounded model worker; scale replicas with one GPU per process."""
from __future__ import annotations

import argparse
from conductor.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/serving/development.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
    service = config.get("serving", {})
    if not service.get("require_api_key", True) and args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("unauthenticated development mode binds only to localhost")
    import uvicorn
    from conductor.serving.api import create_app
    uvicorn.run(create_app(config), host=args.host, port=args.port, workers=1,
                limit_concurrency=int(service.get("max_pending", 128)) + 16,
                timeout_keep_alive=5, timeout_graceful_shutdown=int(service.get("shutdown_timeout_seconds", 10)))


if __name__ == "__main__":
    main()
