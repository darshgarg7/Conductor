"""Convert measured counterfactual winners into unambiguous routing supervision.

A successful trajectory can contain wasted calls. These labels instead describe
the best measured first action from a shared state and continuation policy. This
is an offline teacher, not an oracle supplied to the controller at inference.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from conductor.controller.actions import ActionCatalog
from conductor.datasets.integrity import digest, durable_jsonl, file_digest, record_checksum, seal
from conductor.routing.serialization import serialize_state
from conductor.schema import ExecutionState, RoutingDecision
from conductor.training.data import read_records
from conductor.utils.config import load_config
from conductor.utils.runs import Run


def curate_preferences(records: list[dict[str, Any]], k: int = 2,
                       max_rejected_per_state: int = 2
                       ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Keep one successful winner per exact state; retain bounded hard negatives.

    Reward comparisons require the same continuation configuration. Ties between
    measured winners use catalog order, independent of held-out performance.
    Confidence is not part of a categorical action's identity.
    """
    if max_rejected_per_state < 1:
        raise ValueError("max_rejected_per_state must be positive")
    catalog = ActionCatalog(k)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("split") != "train":
            raise ValueError("curation accepts training records only")
        checksum = record.get("metadata", {}).get("record_checksum_sha256")
        if checksum is not None and checksum != record_checksum(record):
            raise ValueError("preference source checksum mismatch")
        if record.get("comparison") != "identical_state_counterfactual_first_action":
            raise ValueError("curation requires exact-state counterfactual comparisons")
        if record.get("chosen_success") is not True:
            raise ValueError("SFT winners must have observed task success")
        if not isinstance(record.get("task_id"), str) or not record["task_id"]:
            raise ValueError("task_id is required")
        state = ExecutionState(**record["state"])
        if record.get("state_canonical") != serialize_state(state):
            raise ValueError("counterfactual state identity mismatch")
        for field in ("chosen_reward", "rejected_reward"):
            value = record[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("preference rewards must be finite numbers")
        if record["chosen_reward"] <= record["rejected_reward"]:
            raise ValueError("chosen action must improve measured reward")
        for field in ("chosen", "rejected"):
            RoutingDecision(**record[field]).validate(k)
            catalog.index(record[field])
        if catalog.index(record["chosen"]) == catalog.index(record["rejected"]):
            raise ValueError("chosen and rejected categorical actions must differ")
        chosen_grade = record.get("chosen_metrics", {}).get("grader_score")
        rejected_grade = record.get("rejected_metrics", {}).get("grader_score")
        if (not isinstance(chosen_grade, (int, float)) or not isinstance(rejected_grade, (int, float))
                or not math.isfinite(chosen_grade) or not math.isfinite(rejected_grade)
                or chosen_grade < rejected_grade):
            raise ValueError("curation cannot sacrifice observed task quality")
        groups[(record["task_id"], digest(record["state"]))].append(record)
    if not groups:
        raise ValueError("no preference winners to curate")
    sft, preferences = [], []
    conflicts = 0
    for (task_id, state_hash), pairs in sorted(groups.items()):
        continuation = {pair.get("continuation_configuration_sha256") for pair in pairs}
        if len(continuation) != 1 or None in continuation:
            raise ValueError("cannot compare rewards across different continuation configurations")
        sources = set()
        for pair in pairs:
            chosen_source = pair["chosen_metrics"].get("source_identity")
            rejected_source = pair["rejected_metrics"].get("source_identity")
            if chosen_source != rejected_source:
                raise ValueError("chosen/rejected source configuration identities differ")
            sources.add(digest(chosen_source))
        if len(sources) != 1:
            raise ValueError("cannot compare rewards across source configuration identities")
        conflicts += len({catalog.index(pair["chosen"]) for pair in pairs}) > 1
        winner = min(pairs, key=lambda pair: (-pair["chosen_reward"], catalog.index(pair["chosen"]), digest(pair)))
        selected = [pair for pair in pairs if catalog.index(pair["chosen"]) == catalog.index(winner["chosen"])]
        # Small reward gaps are harder comparisons; prefer one unsuccessful route
        # as well when available, so cost-only pairs do not erase correctness.
        unique: dict[int, dict[str, Any]] = {}
        for pair in sorted(selected, key=lambda pair: (-pair["rejected_reward"], digest(pair))):
            unique.setdefault(catalog.index(pair["rejected"]), pair)
        negatives = list(unique.values())[:max_rejected_per_state]
        failures = [pair for pair in unique.values() if pair.get("rejected_success") is False]
        if max_rejected_per_state > 1 and failures and not any(pair.get("rejected_success") is False for pair in negatives):
            negatives[-1] = failures[0]
        provenance = {"curation": "counterfactual_winner_v1", "source_state_sha256": state_hash,
                      "source_pair_sha256": digest(winner), "measured_chosen_reward": winner["chosen_reward"],
                      "continuation_configuration_sha256": next(iter(continuation))}
        decision = RoutingDecision(**winner["chosen"]).to_dict()
        sft.append(seal({"task_id": task_id, "split": "train", "state": winner["state"],
                         "decision": decision, "trajectory_success": True,
                         "source_policy": "counterfactual_winner", "metadata": provenance},
                        digest({"task": task_id, "state": state_hash, "method": "counterfactual_winner_v1"})))
        preferences.extend(negatives)
    audit = {"method": "counterfactual_winner_v1", "source_pairs": len(records),
             "exact_states": len(groups), "sft_examples": len(sft), "preference_examples": len(preferences),
             "training_tasks": len({record["task_id"] for record in sft}),
             "states_with_conflicting_source_winners": conflicts,
             "step_distribution": dict(Counter(str(record["state"]["current_step"]) for record in sft)),
             "target_distribution": dict(Counter(str(catalog.index(record["decision"])) for record in sft)),
             "selection_uses_heldout_labels": False,
             "scope": "Best measured candidate with a fixed rule continuation; not a globally optimal multi-step policy."}
    return sft, preferences, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(config["curation"]["output"])
    if output.exists():
        raise FileExistsError(f"use a fresh curation directory: {output}")
    source = Path(config["curation"]["source"])
    source_hash = file_digest(source)
    records = read_records(source, "dpo")
    sft, preferences, audit = curate_preferences(
        records, int(config.get("routing", {}).get("k", 2)),
        int(config["curation"].get("max_rejected_per_state", 2)))
    if file_digest(source) != source_hash:
        raise ValueError("preference source changed during curation")
    run = Run(output, config)
    durable_jsonl(output / "sft.jsonl", sft)
    durable_jsonl(output / "preferences.jsonl", preferences)
    run.finish({**audit, "source_sha256": source_hash,
                "sft_sha256": file_digest(output / "sft.jsonl"),
                "preferences_sha256": file_digest(output / "preferences.jsonl")})


if __name__ == "__main__":
    main()
