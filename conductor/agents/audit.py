"""Immutable specialist identity evidence, without credentials or grading labels."""
from __future__ import annotations
import hashlib
import inspect
from typing import Any
from conductor.agents.base import CAPABILITIES
from conductor.datasets.integrity import digest


def specialist_audit(agents: dict[str, Any]) -> dict[str, Any]:
    records = []
    for name, agent in sorted(agents.items()):
        settings = getattr(agent, "config", {})
        safe = {key: value for key, value in settings.items()
                if key not in {"api_key", "password", "token", "authorization", "overrides"}}
        backend = safe.get("backend", "deterministic")
        resolved = getattr(getattr(getattr(agent, "model", None), "config", None), "_commit_hash", None)
        immutable_api = safe.get("immutable_model_version") if backend == "api" else None
        immutable = bool(resolved or immutable_api) if backend not in {"deterministic", "workflow"} else True
        try:
            source_hash = hashlib.sha256(inspect.getsource(type(agent)).encode()).hexdigest()
        except (OSError, TypeError):
            source_hash = None
        records.append({"agent": name, "backend": backend, "frozen": bool(agent.frozen),
                        "model_name": safe.get("model_name"), "resolved_model_revision": resolved,
                        "immutable_model_version": immutable_api, "identity_verifiable": immutable,
                        "prompt_sha256": digest({"capability": getattr(agent, "capability", CAPABILITIES.get(name)),
                                                 "template": "public_state_json_work_answer_v1"}),
                        "corpus_sha256": safe.get("corpus_sha256"), "configuration_sha256": digest(safe),
                        "implementation_sha256": source_hash})
    return {"specialists": records, "fingerprint": digest(records),
            "all_frozen": all(record["frozen"] for record in records),
            "all_frozen_llm": all(record["backend"] in {"hf", "api"} and record["frozen"] for record in records),
            "all_identities_verifiable": all(record["identity_verifiable"] for record in records)}
