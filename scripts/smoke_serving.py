"""Exercise a real trained checkpoint through authenticated localhost HTTP."""
from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from conductor.schema import ExecutionState
from conductor.controller.artifacts import resolve_checkpoint
from conductor.utils.config import load_config
from conductor.utils.runs import Run


async def requests(url: str, key: str, count: int) -> dict:
    async with httpx.AsyncClient(base_url=url, timeout=60) as client:
        assert (await client.post("/v1/route", json={"state": ExecutionState("2+3", "math").to_dict(), "k": 2})).status_code == 401
        async def route(index: int) -> dict:
            body = {"state": ExecutionState(f"Calculate {index}+3", "math").to_dict(), "k": 2}
            response = await client.post("/v1/route", headers={"X-API-Key": key}, json=body)
            response.raise_for_status()
            result = response.json()
            assert result["decision_source"] == "model"
            assert len(result["decision"]["selected_agents"]) <= 2
            return result
        values = await asyncio.gather(*(route(index) for index in range(count)))
        assert len({value["request_id"] for value in values}) == count
        metrics = await client.get("/metrics", headers={"X-API-Key": key})
        metrics.raise_for_status()
        return {"status": "measured", "request_count": count, "responses": values,
                "prometheus_metrics": metrics.text, "unauthorized_http_status": 401,
                "scope": "Real checkpoint HTTP routing on localhost; no downstream execution, task-success or CUDA claim."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/serving/granite_pilot.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--output", default="outputs/research/granite-pilot/serving")
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("requests must be positive")
    config = load_config(args.config)
    checkpoint = args.checkpoint or config["checkpoint"]
    import json
    stage = json.loads((resolve_checkpoint(checkpoint) / "controller.json").read_text())["stage"]
    if stage not in {"sft", "preference"}:
        raise ValueError("Serving smoke requires an actual post-trained checkpoint")
    run = Run(args.output, config, checkpoint)
    key = secrets.token_hex(24)  # ephemeral authentication secret is never saved or printed
    environment = {**os.environ, config["serving"].get("api_key_env", "CONDUCTOR_ROUTING_API_KEY"): key}
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with (Path(args.output) / "server.log").open("w") as log:
        server = subprocess.Popen([sys.executable, "-m", "conductor.serve", "--config", args.config,
                                    "--checkpoint", checkpoint, "--port", str(port)], env=environment,
                                   stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError("Server exited during startup; inspect server.log")
                try:
                    if httpx.get(url + "/health/ready", timeout=1).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.2)
            else:
                raise TimeoutError("Checkpoint server startup deadline exceeded")
            result = asyncio.run(requests(url, key, args.requests))
            if any(value["checkpoint_stage"] != stage for value in result["responses"]):
                raise RuntimeError("HTTP checkpoint stage differs from the requested artifact")
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
                raise RuntimeError("Server failed graceful shutdown")
    result["graceful_shutdown_returncode"] = server.returncode
    if server.returncode not in {0, -15} or "Application shutdown complete" not in (Path(args.output) / "server.log").read_text():
        raise RuntimeError("Server did not finish application shutdown successfully")
    run.finish(result)


if __name__ == "__main__":
    main()
