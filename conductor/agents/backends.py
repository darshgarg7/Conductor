"""Optional frozen Hugging Face and OpenAI-compatible specialist backends."""
from __future__ import annotations

import asyncio
import json
import os
import random
import hashlib
import urllib.error
import threading
import time
import urllib.request
from typing import Any

from conductor.agents.base import CAPABILITIES, estimated_tokens
from conductor.schema import AgentOutput, ExecutionState


def prompt(name: str, state: ExecutionState) -> str:
    return (f"You are the {name} specialist. Capability: {CAPABILITIES[name]}\n"
            "Use only the supplied public task and state. Return JSON with work (string) and answer (string or null).\n"
            + json.dumps(state.to_dict(), sort_keys=True))


def unpack(content: str) -> tuple[str, str | None, bool]:
    """Return (work, answer, invalid_output), distinguishing unfinished from invalid."""
    try:
        value = json.loads(content)
        if not isinstance(value, dict) or not {"work", "answer"} <= value.keys():
            return content, None, True
        if not isinstance(value["work"], str) or (value["answer"] is not None and not isinstance(value["answer"], str)):
            return content, None, True
        return value["work"], value["answer"], False
    except (ValueError, TypeError):
        return content, None, True


class HFAgent:
    frozen = True

    def __init__(self, name: str, config: dict[str, Any], shared: dict[str, Any]) -> None:
        self.name, self.capability, self.config = name, CAPABILITIES[name], config
        if int(config.get("max_new_tokens", 128)) < 1 or int(config.get("max_prompt_tokens", 2048)) < 1:
            raise ValueError("HF specialist prompt/output token limits must be positive")
        model_name = config["model_name"]
        self.revision = config.get("revision", "main")
        key = json.dumps([model_name, self.revision, config.get("device", "cpu"), config.get("dtype", "float32")])
        if key not in shared:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=self.revision, trust_remote_code=False)
            tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
            dtype = getattr(torch, config.get("dtype", "float32"))
            model = AutoModelForCausalLM.from_pretrained(model_name, revision=self.revision, torch_dtype=dtype, trust_remote_code=False)
            model.to(config.get("device", "cpu")).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            shared[key] = (model, tokenizer, threading.Lock())
        self.model, self.tokenizer, self.lock = shared[key]

    async def execute(self, state: ExecutionState) -> AgentOutput:
        return await asyncio.to_thread(self._execute, state)

    def _execute(self, state: ExecutionState) -> AgentOutput:
        import torch
        started = time.perf_counter()
        with self.lock, torch.inference_mode():
            text = prompt(self.name, state)
            if self.tokenizer.chat_template:
                text = self.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                          tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            context = getattr(self.model.config, "max_position_embeddings", 2048)
            context = context if isinstance(context, int) and context > 0 else 2048
            maximum = min(int(self.config.get("max_prompt_tokens", 2048)), context - int(self.config.get("max_new_tokens", 128)))
            if inputs["input_ids"].shape[1] > maximum:
                raise ValueError("HF specialist input exceeds its bounded prompt/context budget before generation")
            generation = self.model.generate(**inputs, max_new_tokens=self.config.get("max_new_tokens", 128),
                                             do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
            if self.model.device.type == "cuda":
                torch.cuda.synchronize(self.model.device)
            produced = generation[0, inputs["input_ids"].shape[1]:]
            content = self.tokenizer.decode(produced, skip_special_tokens=True)
            tokens = int(inputs["input_ids"].numel() + produced.numel())
        work, answer, invalid_output = unpack(content)
        return AgentOutput(self.name, work, tokens, time.perf_counter() - started,
                           tokens * self.config.get("token_price_per_million", 0.0) / 1e6,
                           {"answer": answer, "invalid_output": invalid_output, "backend": "hf",
                            "token_accounting": "model_tokenizer", "model_name": self.config["model_name"],
                            "model_revision": self.revision,
                            "resolved_model_revision": getattr(self.model.config, "_commit_hash", None),
                            "max_new_tokens": self.config.get("max_new_tokens", 128),
                            "execution_semantics": "shared frozen model; generation serialized by per-model lock"})


class APIAgent:
    frozen = True

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name, self.capability, self.config = name, CAPABILITIES[name], config

    async def execute(self, state: ExecutionState) -> AgentOutput:
        return await asyncio.to_thread(self._execute, state)

    def _execute(self, state: ExecutionState) -> AgentOutput:
        started = time.perf_counter()
        text = prompt(self.name, state)
        body = {"model": self.config["model_name"], "messages": [{"role": "user", "content": text}],
                "temperature": 0, "max_tokens": self.config.get("max_new_tokens", 128)}
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.config.get("api_key_env", "OPENAI_API_KEY"))
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(self.config["base_url"].rstrip("/") + "/chat/completions",
                                         data=json.dumps(body).encode(), headers=headers)
        retries = int(self.config.get("max_retries", 2))
        if retries < 0:
            raise ValueError("max_retries must be nonnegative")
        randomizer = random.Random(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16))
        retry_events = []
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.get("timeout_seconds", 60)) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == retries:
                    raise
                delay = min(float(self.config.get("retry_max_delay_seconds", 4)), 0.25 * 2 ** attempt) * randomizer.uniform(0.5, 1.5)
                retry_events.append({"attempt": attempt + 1, "status": error.code, "delay_seconds": delay,
                                     "charged_tokens_unknown": True})
                time.sleep(delay)
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            content = json.dumps(content)
        usage = result.get("usage", {})
        tokens = usage.get("total_tokens", estimated_tokens(text) + estimated_tokens(content))
        work, answer, invalid_output = unpack(content)
        return AgentOutput(self.name, work, int(tokens), time.perf_counter() - started,
                           tokens * self.config.get("token_price_per_million", 0.0) / 1e6,
                           {"answer": answer, "invalid_output": invalid_output, "backend": "api",
                            "token_accounting": "provider_usage" if "total_tokens" in usage else "estimated_whitespace",
                            "model_name": result.get("model", self.config["model_name"]),
                            "max_new_tokens": self.config.get("max_new_tokens", 128),
                            "provider_system_fingerprint": result.get("system_fingerprint"),
                            "retry_events": retry_events, "http_attempts": len(retry_events) + 1,
                            "token_usage_unknown": bool(retry_events), "cost_usage_unknown": bool(retry_events),
                            "immutable_model_version": self.config.get("immutable_model_version"),
                            "identity_verifiable": bool(self.config.get("immutable_model_version"))})
