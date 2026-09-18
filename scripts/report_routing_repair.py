"""Archive and summarize the locked routing-repair experiment without weights."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from conductor.analyze import markdown_table
from conductor.datasets.integrity import file_digest, record_checksum
from conductor.evaluation.provenance import canonical_hash
from conductor.evaluation.diagnostics import routing_diagnostics
from conductor.metrics.aggregate import aggregate_metrics, paired_differences, trajectory_metrics
from conductor.schema import AGENT_NAMES, ExecutionState, RoutingDecision
from conductor.utils.runs import write_json


PHASES = ("generation", "probe", "sft", "preference", "reference", "frozen", "evaluation", "dense-sequential")



def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)
    return left == right


def checked_inventory(data: Path) -> dict[str, dict[str, Any]]:
    tasks = read_jsonl(data / "tasks.jsonl")
    inventory = {task["id"]: task for task in tasks}
    require(len(inventory) == len(tasks), "duplicate task IDs in inventory")
    require(len({task["user_task"] for task in tasks}) == len(tasks), "duplicate public prompts in inventory")
    train = [task for task in tasks if task["split"] == "train"]
    heldout = [task for task in tasks if task["split"] == "eval"]
    require(len(train) == 36 and len(heldout) == 48 and len(tasks) == 84, "inventory must contain 36 train and 48 held-out tasks")
    require(sorted(Counter(task["metadata"]["template_family"] for task in train).values()) == [9] * 4,
            "training inventory must contain four concrete templates with nine tasks each")
    require(sorted(Counter(task["metadata"]["template_family"] for task in heldout).values()) == [8] * 6,
            "held-out inventory must contain six concrete templates with eight tasks each")
    manifest = read_json(data / "manifest.json")
    require(file_digest(data / "tasks.jsonl") == manifest["task_inventory_sha256"], "inventory checksum differs from locked manifest")
    require(set(manifest["completed_task_ids"]) == set(inventory), "generation manifest is incomplete")
    for name in ("trajectories.jsonl", "preferences.jsonl", "sft.jsonl", "curated/preferences.jsonl", "curated/sft.jsonl"):
        with (data / name).open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                require(item.get("metadata", {}).get("record_checksum_sha256") == record_checksum(item),
                        f"record checksum mismatch in {name}")
    curator = read_json(data / "curated/metrics.json")
    for name, field in (("preferences.jsonl", "source_sha256"), ("curated/sft.jsonl", "sft_sha256"),
                        ("curated/preferences.jsonl", "preferences_sha256")):
        require(file_digest(data / name) == curator[field], f"curation data identity mismatch in {name}")
    return inventory



def checked_training_partitions(source: Path, data: Path, inventory: dict[str, dict[str, Any]]) -> None:
    expected = {identity for identity, task in inventory.items() if task["split"] == "train"}
    first = None
    for phase in ("probe", "sft", "preference"):
        partition = read_json(source / phase / "data_partition.json")
        train, validation = partition["train_task_ids"], partition["validation_task_ids"]
        require(len(train) == len(set(train)) == 27 and len(validation) == len(set(validation)) == 9
                and not set(train) & set(validation) and set(train) | set(validation) == expected,
                "fitting and validation partitions must cover disjoint 27/9 training tasks")
        require(first is None or first == partition, "representation/SFT/DPO task partitions differ")
        first = partition
    rows = read_jsonl(data / "curated/sft.jsonl")
    require(len(rows) == 180 and {row["task_id"] for row in rows} == expected, "curated SFT must cover 180 states across all 36 training tasks")
    for name in ("sft.jsonl", "preferences.jsonl"):
        for row in read_jsonl(data / "curated" / name):
            require(row["task_id"] in expected and row["split"] == "train", "held-out tasks entered curated training")
            state = ExecutionState(**row["state"])
            require(state.user_task == inventory[row["task_id"]]["user_task"]
                    and state.task_type == inventory[row["task_id"]]["task_type"], "curated public state identity differs")
    validation_rows = [row for row in rows if row["task_id"] in first["validation_task_ids"]]
    templates = Counter(inventory[identity]["metadata"]["template_family"] for identity in first["validation_task_ids"])
    require(templates == {"inline_lookup": 3, "reverse_string": 3, "count_vowels": 3}
            and len(validation_rows) == 45
            and sum(row["decision"]["terminate"] for row in validation_rows) == 32
            and not any("math" in row["decision"]["selected_agents"] for row in validation_rows),
            "report's fixed validation coverage disclosure differs from actual partition")
    probe = read_json(source / "probe/metrics.json")
    require(probe["training_tasks"] == 27 and probe["validation_tasks"] == 9
            and probe["dataset_sha256"] == file_digest(data / "curated/sft.jsonl"), "probe source identity or task counts differ")
    require(set(probe["ablations"]) == {"last/none", "last/layer_norm", "mean/none", "mean/layer_norm"}, "probe ablations differ from fixed protocol")
    selected = min(probe["ablations"], key=lambda name: (-probe["ablations"][name]["validation_accuracy"],
        probe["ablations"][name]["validation_loss"], name))
    require(probe["selected_representation"] == selected, "head representation selection differs from preregistered rule")
    for phase, name, counts in (("sft", "sft.jsonl", (135, 45)), ("preference", "preferences.jsonl", (270, 90))):
        metrics = read_json(source / phase / "metrics.json")
        require(metrics["dataset_sha256"] == file_digest(data / "curated" / name)
                and (metrics["train_examples"], metrics["validation_examples"]) == counts,
                "training data identity or fixed label partition differs")


def checked_expert_stats(stats: dict[str, Any], architecture: dict[str, Any]) -> None:
    layers = stats.get("layers", {})
    require(set(layers) == {str(index) for index in range(architecture["num_hidden_layers"])},
            "expert trace lacks expected pretrained layers")
    categories = stats.get("task_type_activation_counts", {})
    require(bool(categories), "expert trace lacks task-type observations")
    for layer, row in layers.items():
        counts = row["activation_counts"]
        observations = row["observations"]
        require(len(counts) == row["num_experts"] == architecture["num_local_experts"]
                and type(observations) is int and observations > 0
                and all(type(value) is int and value >= 0 for value in counts)
                and sum(counts) == observations * architecture["num_experts_per_tok"],
                "expert counts disagree with nonpadding observations and internal top-k")
        require(row["utilized_experts"] == sum(value > 0 for value in counts), "utilized-expert count differs")
        require(len(row["activation_frequency"]) == len(counts)
                and all(math.isclose(value, count / sum(counts), rel_tol=1e-5, abs_tol=1e-7)
                        for value, count in zip(row["activation_frequency"], counts)), "expert frequency differs from counts")
        require(math.isfinite(row["routing_entropy"]) and 0 <= row["routing_entropy"] <= math.log(len(counts)) + 1e-6,
                "expert entropy is outside finite probability bounds")
        category_counts = [category.get(layer) for category in categories.values()]
        require(all(isinstance(values, list) and len(values) == len(counts)
                    and all(type(value) is int and value >= 0 for value in values) for values in category_counts)
                and [sum(values[index] for values in category_counts) for index in range(len(counts))] == counts,
                "expert task-type counts do not sum to layer observations")


def checked_expert_probes(directory: Path, policies: set[str], inventory: dict[str, dict[str, Any]],
                          architecture: dict[str, Any]) -> None:
    expected = policies & {"base_moe", "conductor_sft", "conductor_preference"}
    probes = read_json(directory / "initial_state_probe.json")
    rollout = read_json(directory / "expert_stats_rollout.json")
    require(set(probes) == set(rollout) == expected, "expert probe policy coverage differs")
    ids = {identity for identity, task in inventory.items() if task["split"] == "eval"}
    for policy in expected:
        states = probes[policy]["states"]
        require(len(states) == 48 and {row["task_id"] for row in states} == ids, "initial expert probes must cover the same 48 held-out tasks")
        for row in states:
            require(row["task_type"] == inventory[row["task_id"]]["task_type"], "expert probe task type differs")
            RoutingDecision(**row["decision"]).validate(2)
        checked_expert_stats(probes[policy]["expert_stats"], architecture)
        checked_expert_stats(rollout[policy], architecture)


def checked_comparison(directory: Path, expected_policies: set[str], inventory: dict[str, dict[str, Any]],
                       shared: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Verify raw coverage, exact grades, summaries and shared rollout conditions."""
    heldout_ids = {identity for identity, task in inventory.items() if task["split"] == "eval"}
    audit = read_json(directory / "split_audit.json")
    require(set(audit["heldout_task_ids"]) == heldout_ids, "split audit differs from locked inventory")
    metrics = read_json(directory / "metrics.json")
    statuses = metrics["policy_status"]
    require(len(statuses) == len(expected_policies) and {row["policy"] for row in statuses} == expected_policies,
            "comparison must contain exactly the required policy statuses")
    require(all(row.get("status") == "measured" and row.get("task_count") == 48 for row in statuses),
            "every comparison policy must measure 48 tasks")
    published = metrics["policies"]
    require(len(published) == len(expected_policies) and {row["policy"] for row in published} == expected_policies,
            "comparison must contain exactly the required policy summaries")
    require(metrics.get("task_count") == 48 and metrics.get("specialists_unchanged") is True,
            "comparison lacks complete held-out and frozen specialist confirmation")
    run = read_json(directory / "run.json")
    config = run["configuration"]
    budgets = {key: config.get(key, default) for key, default in {
        "k": 2, "max_rounds": 3, "token_budget": 8192, "agent_call_budget": 12, "routing_interval": 1}.items()}
    require(metrics["fixed_budgets"] == budgets, "reported budgets differ from resolved run configuration")
    provenance = read_json(directory / "evaluation_provenance.json")
    require(metrics["provenance"] == provenance, "metrics and provenance disagree")
    specialists = read_json(directory / "specialists.json")
    require(specialists["sha256"] == canonical_hash(specialists["identities"]) == provenance["specialist_sha256"],
            "specialist identity checksum is invalid")
    frozen = read_json(directory / "specialist_audit.json")
    require(frozen.get("all_frozen") is True and frozen.get("all_identities_verifiable") is True,
            "specialist audit is not frozen and verifiable")
    require({row["agent"] for row in frozen["specialists"]} == set(AGENT_NAMES)
            and len(frozen["specialists"]) == len(AGENT_NAMES)
            and all(row.get("frozen") is True and row.get("identity_verifiable") is True for row in frozen["specialists"]),
            "individual specialist audit is incomplete or unfrozen")
    require(all(row.get("frozen") is True for row in specialists["identities"].values())
            and set(specialists["identities"]) == set(AGENT_NAMES), "individual specialist identities are not frozen")
    expected_experiment = canonical_hash({"agents": config.get("agents", {}), "budgets": budgets,
        "seed": config.get("seed", 42), "heldout_task_ids": audit["heldout_task_ids"], "inference": config.get("inference", {})})
    require(provenance["experiment_sha256"] == expected_experiment and provenance["config_sha256"] == canonical_hash(config),
            "evaluation configuration fingerprints are invalid")
    conditions = {"budgets": budgets, "data_sha256": provenance["data_sha256"],
                  "specialist_sha256": provenance["specialist_sha256"], "specialist_audit_fingerprint": frozen["fingerprint"],
                  "experiment_sha256": provenance["experiment_sha256"], "seed": run["seed"]}
    require(shared is None or shared == conditions, "comparison conditions or specialist identities differ")
    trajectories = read_jsonl(directory / "trajectories.jsonl")
    require({item["policy"] for item in trajectories} == expected_policies, "raw policy coverage differs from required policies")
    for policy in expected_policies:
        selected = [item for item in trajectories if item["policy"] == policy]
        require(len(selected) == 48 and {item["task"]["id"] for item in selected} == heldout_ids,
                "raw policy must cover every locked task exactly once")
        status = next(row for row in statuses if row["policy"] == policy)
        for item in selected:
            task = item["task"]
            require(task == inventory[task["id"]], "raw task identity differs from locked inventory")
            exact = isinstance(item.get("final_answer"), str) and item["final_answer"].strip() == task["expected_answer"].strip()
            require(type(item.get("task_success")) is bool and item["task_success"] is exact
                    and equivalent(item.get("grader_score"), float(exact)), "raw success or grader differs from exact answer")
            metadata = item["metadata"]
            require(metadata.get("evaluation_budgets") == budgets
                    and metadata.get("specialist_fingerprint") == frozen["fingerprint"], "raw budgets or specialist audit differ")
            for field in ("data_sha256", "specialist_sha256", "experiment_sha256", "config_sha256"):
                require(metadata.get(field) == provenance[field] == status.get(field), "raw trajectory fingerprint differs")
            require(metadata.get("checkpoint_sha256") == status.get("checkpoint_sha256"), "checkpoint identity differs within policy")
            for step in item["steps"]:
                state = ExecutionState(**step["state"])
                require(state.user_task == task["user_task"] and state.task_type == task["task_type"],
                        "public execution state identity differs from task")
    summaries = aggregate_metrics([trajectory_metrics(item) for item in trajectories])
    for row in summaries:
        claimed = next(item for item in published if item["policy"] == row["policy"])
        require(set(claimed) == set(row) and all(equivalent(value, claimed[key]) for key, value in row.items()),
                "reported aggregate differs from recomputed raw trajectory metrics")
    # Recompute category and concrete-template exports as well, retaining their observed scope.
    rows = [trajectory_metrics(item) for item in trajectories]
    for key, groups in (("categories", ("policy", "category", "split", "generalization")),
                        ("templates", ("policy", "template_family", "generalization"))):
        actual = aggregate_metrics(rows, groups)
        claimed = metrics[key]
        require(len(actual) == len(claimed), f"reported {key} coverage differs")
        for row in actual:
            matches = [item for item in claimed if all(item.get(field) == row[field] for field in groups)]
            require(len(matches) == 1 and set(matches[0]) == set(row)
                    and all(equivalent(value, matches[0][field]) for field, value in row.items()),
                    f"reported {key} differs from raw trajectory metrics")
    return trajectories, summaries, conditions


def checked_gzip(path: Path, expected_sha256: str, expected_bytes: int) -> None:
    """CRC, decompression, byte count and original digest are checked by streaming."""
    digest, size = hashlib.sha256(), 0
    with gzip.open(path, "rb") as handle:
        while block := handle.read(1024 * 1024):
            size += len(block)
            digest.update(block)
    require(size == expected_bytes and digest.hexdigest() == expected_sha256, "gzip round-trip differs from original trace bytes")


def checked_adapter_proof(source: Path) -> dict[str, Any]:
    proof = read_json(source / "adapter-verification/post_training_verification.json")
    require(proof.get("nvidia_execution_verified") is False and proof.get("frozen_base_weight_audit_performed") is False,
            "tensor verification scope must not imply CUDA or a complete frozen-base audit")
    require(proof.get("git_clean_at_verification") is True, "tensor proof must come from clean verification source")
    require(proof["verification_script_sha256"] == file_digest(Path("scripts/verify_pilot_adapters.py")),
            "saved-tensor verification script identity differs")
    for phase in ("sft", "preference"):
        metrics = read_json(source / phase / "metrics.json")
        require(proof["stage_metrics"][phase] == metrics, "tensor proof metrics differ from actual training stage")
        summary = proof["stage_weight_summaries"][phase]
        require(summary["router_lora_B_tensor_count"] > 0
                and summary["nonzero_router_lora_B_tensor_count"] == summary["router_lora_B_tensor_count"]
                and summary["nonzero_lora_B_tensor_count"] == summary["lora_B_tensor_count"], "saved router/attention LoRA updates are not verified")
        pointer = read_json(source / phase / "checkpoint_pointer.json")
        checkpoint = source / phase / pointer["path"]
        require(checkpoint.resolve().is_relative_to((source / phase).resolve()), "checkpoint pointer escapes its stage")
        for relative, digest in proof["artifact_files_sha256"][phase].items():
            require((checkpoint / relative).resolve().is_relative_to(checkpoint.resolve()), "proof artifact escapes checkpoint")
            require(file_digest(checkpoint / relative) == digest, "saved-tensor proof artifact hash differs")
    deltas = proof["sft_to_preference_weight_deltas"]
    require(deltas["adapters"]["changed_tensor_count"] > 0 and deltas["head"]["changed_tensor_count"] > 0
            and proof["changed_router_adapter_tensor_count"] > 0, "DPO saved adapter/head/router changes are not verified")
    reference = read_json(source / "preference/reference_log_probabilities.json")
    require(reference["sft_checkpoint_sha256"] == proof["exact_sft_reference_sha256"]
            == proof["stage_metrics"]["preference"]["reference_checkpoint_sha256"], "DPO exact SFT reference proof differs")
    return proof


def checked_context_audit(path: Path) -> dict[str, Any]:
    audit = read_json(path)
    require(audit["source_sha256"] == file_digest(Path("results/granite-pilot/data/sft.jsonl")),
            "context audit must cover the unchanged original pilot training states")
    require(audit["source_hf_sha256"] == file_digest(Path("conductor/controller/hf.py"))
            and audit["verification_script_sha256"] == file_digest(Path("scripts/audit_controller_context.py")),
            "context audit implementation identity differs")
    require(audit["tokenizer_model"] == "ibm-granite/granite-3.1-1b-a400m-base"
            and audit["tokenizer_revision"] == "408b6e90baab8cf24f4aa9f8e19703ffa0a53b29",
            "context audit tokenizer identity differs")
    for name in ("legacy_128", "priority_v1_128", "priority_v1_256"):
        value = audit["controllers"][name]
        require(len(value["states"]) == value["summary"]["states"] == 72
                and value["summary"]["later_states"] == 48
                and value["summary"]["all_token_caps_respected"] is True,
                "context audit must cover all original training states within token caps")
    return audit


def archive(source: Path, data: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"use a fresh archive: {output}")
    renderer_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    renderer_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    inventory = checked_inventory(data)
    checked_training_partitions(source, data, inventory)
    proof = checked_adapter_proof(source)
    context_path = source / "context_audit_public.json"
    context_audit = checked_context_audit(context_path)
    provenance: dict[str, Any] = {}
    for phase in PHASES:
        run = json.loads((source / phase / "run.json").read_text())
        runtime = run.get("runtime_seconds")
        if (run.get("git_dirty") is not False or isinstance(runtime, bool)
                or not isinstance(runtime, (int, float)) or not math.isfinite(runtime) or runtime <= 0):
            raise ValueError(f"phase must complete from a clean source tree: {phase}")
        provenance[phase] = {key: run[key] for key in ("git_commit", "git_dirty", "seed", "runtime_seconds")}
    for phase in ("sft", "preference"):
        metrics = json.loads((source / phase / "metrics.json").read_text())
        if not metrics.get("completed") or not metrics.get("pretrained") or metrics.get("specialists_updated") is not False:
            raise ValueError(f"incomplete pretrained coordinator-only training: {phase}")
    shared_conditions = None
    merged, display_rows, omissions, compressed = [], [], [], []
    labels = {"reference": {"conductor_sft": "Original SFT", "conductor_preference": "Original DPO"},
              "frozen": {"conductor_sft": "Frozen-backbone head"},
              "evaluation": {"conductor_sft": "Repaired LoRA SFT", "conductor_preference": "Repaired DPO",
                             "all_agent": "All-Agent parallel", "rule_based": "Rules",
                             "random_top_k": "Random top-k", "base_moe": "Pretrained + random head"},
              "dense-sequential": {"all_agent": "All-Agent sequential"}}
    for phase in PHASES:
        directory = source / phase
        if phase in labels:
            trajectories, summaries, shared_conditions = checked_comparison(
                directory, set(labels[phase]), inventory, shared_conditions)
            checked_expert_probes(directory, set(labels[phase]), inventory, proof["base_architecture"])
            require(shared_conditions["data_sha256"] == file_digest(data / "tasks.jsonl"),
                    "comparison corpus fingerprint differs from archived inventory")
            for row in summaries:
                label = labels[phase][row["policy"]]
                display_rows.append({**row, "policy": label,
                                     "success": f"{round(row['success_rate'] * row['task_count'])}/{row['task_count']}"})
            for item in trajectories:
                item["policy"] = labels[phase][item["policy"]]
                merged.append(item)
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(directory)
            if (len(relative.parts) == 1 and path.suffix in {".json", ".jsonl", ".csv"}
                    and path.name not in {"checkpoint_pointer.json", "latest_resume.json"}):
                target = output / "phases" / phase / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            elif path.name in {"controller.json", "adapter_config.json"} and relative.parts[0] in {"selected", "adapter"}:
                target = output / "phases" / phase / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            elif len(relative.parts) == 1:
                omissions.append({"phase": phase, "path": str(relative), "sha256": file_digest(path),
                                  "reason": "Weights, features, optimizer state, local pointers and logs remain local."})
    for path in sorted(data.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        target = output / "data" / path.relative_to(data)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "preference_trials.jsonl":
            with target.with_suffix(".jsonl.gz").open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, filename="", mode="wb", mtime=0) as packed:
                    with path.open("rb") as handle:
                        shutil.copyfileobj(handle, packed)
            descriptor = {"archive": str(target.with_suffix(".jsonl.gz").relative_to(output)),
                          "uncompressed_sha256": file_digest(path), "uncompressed_bytes": path.stat().st_size}
            checked_gzip(target.with_suffix(".jsonl.gz"), descriptor["uncompressed_sha256"], descriptor["uncompressed_bytes"])
            compressed.append({**descriptor, "round_trip_verified": True})
        else:
            shutil.copyfile(path, target)
    verification = source / "adapter-verification"
    for path in sorted(verification.glob("*.json")):
        target = output / "verification" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(context_path, output / "verification/context_audit.json")
    provenance["archive"] = {"script_sha256": file_digest(Path(__file__)), "archive_source_commit": renderer_commit,
        "archive_source_dirty": renderer_dirty, "task_inventory_sha256": file_digest(data / "tasks.jsonl"),
        "adapter_proof_sha256": file_digest(verification / "post_training_verification.json"),
        "context_audit_sha256": file_digest(context_path),
        "comparison_conditions": shared_conditions,
        "scope": "Renderer provenance is separate from unchanged experiment phase commits. CUDA and full frozen-base tensor verification are not claimed."}
    write_json(output / "provenance.json", provenance)
    write_json(output / "omitted_artifacts.json", omissions)
    write_json(output / "compressed_artifacts.json", compressed)
    rows = [trajectory_metrics(item) for item in merged]
    paired = [paired_differences(rows, baseline, candidate, bootstrap_samples=2000, seed=42)
              for baseline in ("Original SFT", "Original DPO", "Frozen-backbone head", "Rules", "All-Agent sequential")
              for candidate in ("Repaired LoRA SFT", "Repaired DPO") if baseline != candidate]
    write_json(output / "paired_repair_comparisons.json", paired)
    routing = routing_diagnostics(merged, seed=42)
    write_json(output / "routing_diagnostics.json", routing)
    probe = json.loads((output / "phases/probe/metrics.json").read_text())
    curator = json.loads((output / "data/curated/metrics.json").read_text())
    sft = json.loads((output / "phases/sft/metrics.json").read_text())
    dpo = json.loads((output / "phases/preference/metrics.json").read_text())
    ablations = [{"representation": name, **result} for name, result in probe["ablations"].items()]
    summary = {row["policy"]: row for row in display_rows}
    known_templates = {task["metadata"]["template_family"] for task in inventory.values() if task["split"] == "train"}
    transfer = []
    for policy in ("Repaired LoRA SFT", "Repaired DPO"):
        records = [item for item in merged if item["policy"] == policy]
        seen = [item for item in records if item["task"]["metadata"]["template_family"] in known_templates]
        unseen = [item for item in records if item not in seen]
        transfer.append(f"{policy}: {sum(item['task_success'] for item in seen)}/{len(seen)} seen-template tasks, "
                        f"{sum(item['task_success'] for item in unseen)}/{len(unseen)} unseen compositions")
    decisions = [row["decision"] for row in read_jsonl(data / "curated/sft.jsonl")]
    label_counts = Counter("stop" if decision["terminate"] else ",".join(decision["selected_agents"]) for decision in decisions)
    legacy_audit = context_audit["controllers"]["legacy_128"]["summary"]
    priority_audit = context_audit["controllers"]["priority_v1_256"]["summary"]
    text = f"""# Routing repair: measured CPU results

The original Granite pilot collapsed to coder → stop. This follow-up repairs
state truncation and teacher labels, then evaluates all candidates on a newly
locked 48-task inventory. It tests correctness on synthetic templates, with
the same eight frozen deterministic specialists and execution budgets.

**The repair does not establish a superior coordination policy.** Repaired SFT
solves {summary['Repaired LoRA SFT']['success']} tasks and repaired DPO solves
{summary['Repaired DPO']['success']}. Rules solve {summary['Rules']['success']}
and random sparse routing solves {summary['Random top-k']['success']}.
The original SFT and DPO checkpoints solve {summary['Original SFT']['success']}
and {summary['Original DPO']['success']} on this same inventory.

{'; '.join(transfer)}. SFT and DPO average
{summary['Repaired LoRA SFT']['mean_agent_calls']:.3f} and
{summary['Repaired DPO']['mean_agent_calls']:.3f} calls per task, respectively.

The curated label counts are `{dict(sorted(label_counts.items()))}`. Required
multi-agent handoffs and execution modes are absent from this training inventory.
The inspected inventory is now development evidence; the next
experiment needs a fresh locked test with unseen dependency structures.

## Training and selection

Generation uses 36 training tasks and 48 held-out tasks, seed 314159. Evaluation
has eight examples per concrete template. Two composition categories are absent
from training. Inputs, labels, source commits and hardware remain in the raw
phase records; [the committed protocol](../../docs/routing_repair_protocol.md)
describes the selection rule and limitations.

[The tokenizer-only context audit](verification/context_audit.json) covers all
72 original pilot training states. Under the legacy 128-token format,
{legacy_audit['later_states_missing_all_mandatory_keys']}/{legacy_audit['later_states']}
later states lack all four critical progress keys in decoded input. Priority
serialization at 256 tokens preserves the complete priority prefix in
{priority_audit['complete_priority_prefix_states']}/{priority_audit['states']}
states, truncates {priority_audit['task_truncated_states']} task texts and
{priority_audit['history_truncated_states']} histories. Literal field preservation
does not establish model comprehension. Reproduce with
`python scripts/audit_controller_context.py --output FRESH_AUDIT.json`.

The curator retains {curator['sft_examples']} distinct-state SFT labels and
{curator['preference_examples']} preference pairs. It selects successful measured
counterfactual winners under a shared rule continuation. That is useful offline
supervision, not evidence of a globally optimal controller.

All four probes share frozen pretrained weights and a 256-token priority input.
The selected representation is **{probe['selected_representation']}**. Epoch and
representation selection use internal validation only; no held-out score
selects the model. The unchanged partition fits 27 tasks and reserves nine,
with 135 fitting labels and 45 validation labels. Validation contains three
lookup, three reverse-string and three vowel-count tasks, with no arithmetic
addition task or math-agent target. Of its 45 labels, 32 are stopping actions;
perfect head accuracy describes this limited, correlated validation set.

{markdown_table(ablations, [('representation','Head input'),('epoch','Selected epoch'),('validation_accuracy','Global action accuracy'),('validation_loss','Cross entropy'),('feature_rms','Feature RMS'),('initial_logit_std','Initial logit standard deviation')])}

The fitted head warms up {sft['epochs']}-epoch attention/router LoRA SFT, then
{dpo['epochs']}-epoch categorical DPO against the exact frozen new SFT checkpoint.
SFT fits {sft['train_examples']} labels and reserves {sft['validation_examples']};
DPO fits {dpo['train_examples']} pairs and reserves {dpo['validation_examples']}.
The frozen-head candidate is a distinct ablation and does not update the backbone.
DPO validation pair-ranking accuracy is {dpo['final_validation']['accuracy']:.1%};
it is neither full-catalog argmax accuracy nor online task success.

## Online execution

{markdown_table(display_rows, [('policy','Policy'),('success','Exact success'),('mean_agent_calls','Mean agent calls'),('mean_controller_tokens','Controller tokens'),('mean_downstream_tokens','Tool token estimates'),('p95_wall_clock_seconds','System p95 seconds')])}

[Paired differences](paired_repair_comparisons.json) compare identical task IDs.
The old checkpoints keep their original 128-token format. The repaired pipeline
bundles input format, 256-token cap, teacher curation, more data and head warmup;
old/new differences cannot identify a single causal change. Probe comparisons
isolate pooling and normalization within the new representation.

Per-template/category summaries, exact output traces, requested stopping actions,
expert probes and task-weighted routing diagnostics are archived.
[Saved-tensor verification](verification/post_training_verification.json)
checks finite FP32 attention/router LoRA updates, SFT-to-DPO adapter/head changes
and the exact SFT reference identity. It loads adapter/head tensors only;
no complete frozen-base tensor audit or NVIDIA run is claimed. Gzip traces
are round-trip checked against original uncompressed hashes and byte counts.
The permutation
null guards against interpreting a unique random route per task as specialization;
association remains descriptive and does not demonstrate semantic MoE experts.

## What this does and does not establish

This is a single-seed repair experiment on related arithmetic, lookup and string
templates. The held-out composition labels support a narrow transfer test. IDs
are distinct, but template siblings are correlated; bootstrap intervals describe
this inventory rather than broad language-task populations. Head selection uses
several internal-validation comparisons and can overfit that partition.

The rule baseline already understands the public task taxonomy, so this corpus
does not establish a need for a 1.3B MoE or superiority over smaller routers.
Dense sequential execution is reported alongside the weaker parallel control.
Fewer calls are not monetary savings: controller tokens are real HF tokenizer
counts, tool tokens are estimates, and configured zero prices leave costs unknown.
CPU timings do not establish NVIDIA performance or an optimization speedup.
CUDA/NCCL, GPU profiling, SLURM and the NVIDIA container remain unvalidated.

Both favorable and unfavorable results remain published. Deployment and support
ticket acceptance require their own gates; this experiment does not validate
the unrelated support fixture or make the learned controller production ready.
"""
    (output / "report.md").write_text(text)
    plot(display_rows, output)
    write_json(output / "checksums.json", {str(path.relative_to(output)): file_digest(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "checksums.json"})


def plot(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [row for row in rows if row["policy"] in {"Original SFT", "Frozen-backbone head", "Repaired LoRA SFT", "Repaired DPO", "Rules", "All-Agent sequential"}]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    names = [row["policy"].replace(" ", "\n", 1) for row in selected]
    for axis, field, label in zip(axes, ("success_rate", "mean_agent_calls"), ("Exact task success fraction (48 tasks)", "Mean downstream agent calls")):
        axis.bar(names, [row[field] for row in selected], color="#475569")
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=25, labelsize=8)
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylim(0, 1.05)
    fig.suptitle("Routing repair: CPU + frozen deterministic specialists; one seed")
    fig.tight_layout()
    fig.savefig(output / "repair.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/research/routing-repair"))
    parser.add_argument("--data", type=Path, default=Path("data/generated/routing-repair"))
    parser.add_argument("--output", type=Path, default=Path("results/routing-repair"))
    args = parser.parse_args()
    archive(args.source, args.data, args.output)


if __name__ == "__main__":
    main()
