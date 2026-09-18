"""Tokenizer-only context audit over every original pilot training state.

No model weights or held-out task inventory are loaded. Legacy key coverage is
literal coverage in decoded inputs, not a claim that truncated JSON is valid or
that the language model understands a retained field. Priority coverage records
the production tokenizer's complete priority-field audit as well.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from conductor.controller import hf
from conductor.controller.hf import HFController
from conductor.datasets.integrity import digest, file_digest
from conductor.schema import ExecutionState
from conductor.utils.runs import write_json


MODEL = "ibm-granite/granite-3.1-1b-a400m-base"
REVISION = "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29"
MANDATORY_KEYS = ("agents_already_called", "remaining_budget", "current_step", "previous_routing_decisions")


def has_private_key(value: Any) -> bool:
    if isinstance(value, dict):
        return "expected_answer" in value or any(has_private_key(item) for item in value.values())
    if isinstance(value, list):
        return any(has_private_key(item) for item in value)
    return False


def bare_controller(tokenizer: PreTrainedTokenizerBase, mode: str, maximum: int) -> HFController:
    controller = HFController.__new__(HFController)
    controller.tokenizer = tokenizer
    controller.state_serialization = mode
    controller.max_length = maximum
    controller._token_cache = OrderedDict()
    controller._token_cache_details = OrderedDict()
    controller._token_cache_size = 0
    controller._token_cache_hits = controller._token_cache_misses = 0
    return controller


def audit(source: Path) -> dict[str, Any]:
    source_hash = file_digest(source)
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if not records or any(record.get("split") != "train" for record in records):
        raise ValueError("audit requires original training SFT records only")
    if any(has_private_key(record["state"]) for record in records):
        raise ValueError("private expected_answer key found in public execution state")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    controller_results = {}
    for mode, maximum in (("legacy", 128), ("priority_v1", 128), ("priority_v1", 256)):
        controller = bare_controller(tokenizer, mode, maximum)
        rows = []
        for index, record in enumerate(records):
            state = ExecutionState(**record["state"])
            serialized = controller.serialize(state)
            row = {"source_record_index": index, "task_id": record["task_id"], "current_step": state.current_step,
                   "task_type": state.task_type, "state_sha256": digest(record["state"]), "max_length": maximum}
            if mode == "legacy":
                header, tail = serialized.rsplit("\0CONDUCTOR_STATE\0", 1)
                full_header = tokenizer.encode(header, add_special_tokens=False)
                budget = maximum - tokenizer.num_special_tokens_to_add(pair=False)
                kept_header = full_header[:max(1, budget // 3)]
                full_tail = tokenizer.encode(tail, add_special_tokens=False)
                row.update(header_tokens_total=len(full_header), header_tokens_kept=len(kept_header),
                           header_truncated=len(kept_header) < len(full_header),
                           tail_tokens_total=len(full_tail), tail_tokens_kept=min(len(full_tail), budget - len(kept_header)))
                decoded_header = tokenizer.decode(kept_header, skip_special_tokens=True)
                row["task_type_value_available"] = f"Task type: {state.task_type}" in decoded_header
                row["complete_task_available"] = state.user_task in decoded_header
                # Legacy never reported formal truncation details. Keep these
                # exact-text/token measurements separate rather than inferring
                # semantic history/task preservation from field-key sightings.
                row["task_truncated"] = None
                row["history_truncated"] = None
            try:
                encoded = controller.tokenize_states([state])
            except ValueError as error:
                row.update(status="does_not_fit", error=str(error), effective_tokens=None)
                rows.append(row)
                continue
            tokens = encoded["input_ids"][0][encoded["attention_mask"][0].bool()].tolist()
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            if "expected_answer" in decoded:
                raise ValueError("private expected_answer text found in tokenized input")
            row.update(status="tokenized", effective_tokens=len(tokens), token_cap_respected=len(tokens) <= maximum,
                       input_token_ids_sha256=hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
                       decoded_input=decoded)
            if mode == "legacy":
                row["mandatory_key_coverage"] = {key: '"' + key + '"' in decoded for key in MANDATORY_KEYS}
            else:
                detail = controller.last_tokenization_details[0]
                row.update(detail)
                # Complete prefix preservation is guaranteed by production
                # token budgeting; additionally expose independent decoded
                # literal key coverage for direct inspection.
                row["priority_key_coverage"] = {key: '"' + key + '"' in decoded for key in detail["priority_fields"]}
                row["task_type_value_available"] = f"Task type: {state.task_type}" in decoded
            rows.append(row)
        later = [row for row in rows if row["current_step"] > 0]
        summary = {"states": len(rows), "later_states": len(later),
                   "tokenized_states": sum(row["status"] == "tokenized" for row in rows),
                   "failed_to_fit_states": sum(row["status"] != "tokenized" for row in rows),
                   "all_token_caps_respected": all(row.get("token_cap_respected", False) for row in rows),
                   "task_type_value_available_states": sum(row.get("task_type_value_available", False) for row in rows)}
        if mode == "legacy":
            summary.update(header_truncated_states=sum(row["header_truncated"] for row in rows),
                           later_states_missing_all_mandatory_keys=sum(not any(row.get("mandatory_key_coverage", {}).values()) for row in later),
                           later_key_missing_counts={key: sum(not row.get("mandatory_key_coverage", {}).get(key, False) for row in later)
                                                     for key in MANDATORY_KEYS})
        else:
            summary.update(task_truncated_states=sum(row.get("task_truncated", False) for row in rows),
                           history_truncated_states=sum(row.get("history_truncated", False) for row in rows),
                           complete_priority_prefix_states=sum(row["status"] == "tokenized" and all(row["priority_key_coverage"].values()) for row in rows))
        controller_results[f"{mode}_{maximum}"] = {"summary": summary, "states": rows}
    if file_digest(source) != source_hash:
        raise ValueError("original pilot source changed during audit")
    return {"source": str(source), "source_sha256": source_hash, "tokenizer_model": MODEL,
            "tokenizer_revision": REVISION, "source_hf_sha256": file_digest(Path(hf.__file__)),
            "verification_script_sha256": file_digest(Path(__file__)),
            "libraries": {package: version(package) for package in ("torch", "transformers", "tokenizers")},
            "scope": "All original pilot training states; tokenizer-only production paths. Literal legacy coverage is not semantic comprehension. No held-out tasks or model weights loaded.",
            "controllers": controller_results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/granite-pilot/data/sft.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/research/routing-repair/context_audit.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable audit output exists: {args.output}")
    result = audit(args.source)
    write_json(args.output, result)
    print(json.dumps({name: value["summary"] for name, value in result["controllers"].items()}, indent=2))


if __name__ == "__main__":
    main()
