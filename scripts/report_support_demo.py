"""Archive completed support demos and render a report from their measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any


PUBLIC_FILES = (
    "run.json", "metrics.json", "tasks.jsonl", "trajectories.jsonl", "tickets.csv",
    "routing_requests.jsonl", "load_requests.jsonl", "load_requests.csv",
    "shutdown.json", "service_acceptance.json", "specialist_audit.json",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_raw(source: Path, metrics: dict[str, Any]) -> None:
    """Reject inconsistent summaries; this is not experiment authentication."""
    def rows(name: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (source / name).read_text().splitlines() if line]

    tasks = rows("tasks.jsonl")
    identities = {task["id"]: task for task in tasks}
    if not tasks or len(identities) != len(tasks):
        raise ValueError("Invalid support task inventory")
    trajectories = rows("trajectories.jsonl")
    audit = read_json(source / "specialist_audit.json")
    if audit["fingerprint"] != metrics["specialist_fingerprint"] or not audit["all_frozen"]:
        raise ValueError("Invalid fixed-specialist identity evidence")
    expected_fields = {"user_task", "task_type", "conversation_state", "previous_agent_outputs",
                       "agents_already_called", "tool_results", "remaining_budget", "current_step",
                       "previous_routing_decisions"}
    for policy, summary in metrics["diagnostics"].items():
        cases = [item for item in trajectories if item["policy"] == policy]
        if len(cases) != len(tasks) or {item["task"]["id"] for item in cases} != set(identities):
            raise ValueError("Policies must cover the same unique support inventory")
        successes, contracts = 0, 0
        for item in cases:
            task = identities[item["task"]["id"]]
            if item["task"] != task:
                raise ValueError("Trajectory task differs from the recorded inventory")
            correct = item["final_answer"].strip() == task["expected_answer"].strip()
            if type(item["task_success"]) is not bool or item["task_success"] != correct:
                raise ValueError("Exact diagnostic score does not match the raw answer")
            checks = item["metadata"]["diagnostic_contract"]
            if item["metadata"]["specialist_fingerprint"] != audit["fingerprint"]:
                raise ValueError("Specialists differ between diagnostic trajectories")
            if not checks or any(type(value) is not bool for value in checks.values()):
                raise ValueError("Invalid raw diagnostic contract checks")
            successes += correct
            contracts += all(checks.values())
            for step in item["steps"]:
                if set(step["state"]) != expected_fields or step["state"]["user_task"] != task["user_task"]:
                    raise ValueError("Invalid public execution-state boundary")
        if (summary["tasks"], summary["successes"], summary["contract_valid"]) != (len(cases), successes, contracts):
            raise ValueError("Diagnostic summary differs from raw trajectories")
    load = rows("load_requests.jsonl")
    for phase in metrics["load"]:
        samples = [item for item in load if item["concurrency"] == phase["concurrency"]]
        successes = sum(item["status"] == "measured" for item in samples)
        if len(samples) != phase["requests"] or successes != phase["successful_requests"]:
            raise ValueError("HTTP load summary differs from raw requests")
        if phase["failures"] != len(samples) - successes:
            raise ValueError("HTTP failure count differs from raw requests")
        latencies = [item["client_latency_seconds"] for item in samples if item["client_latency_seconds"] is not None]
        from conductor.metrics.aggregate import percentile
        for q in (50, 95):
            observed = percentile(latencies, q)
            reported = phase[f"p{q}_latency_seconds"]
            if observed != reported:
                raise ValueError("HTTP percentile differs from raw timings")
    acceptance = read_json(source / "service_acceptance.json")
    if acceptance != metrics["acceptance"] or acceptance["passed"] != all(acceptance["checks"].values()):
        raise ValueError("Service acceptance summary is inconsistent")


def number(value: Any, scale: float = 1.0) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Report measurements must be finite numbers or null")
    return f"{value * scale:.3f}"


def render(output: Path, records: dict[str, dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    repository = Path(__file__).resolve().parents[1]
    def documentation(name: str) -> str:
        return Path(os.path.relpath(repository / "docs" / name, output.resolve())).as_posix()

    lines = [
        "# Recorded support demonstration", "",
        "This CPU demonstration connects actual trained-controller HTTP decisions to fixed,",
        "read-only support specialists. Tickets and observations are synthetic. It tests",
        "the execution loop and service boundary; it is not a customer deployment, an",
        "independent diagnostic accuracy benchmark, or NVIDIA performance validation.", "",
        "The rules, runbook, and expected responses were designed together. Diagnostic",
        "success is an exact canonical-response match; contract validity separately",
        "checks public evidence and read-only advice. Repeated HTTP samples do not add",
        "independent diagnostic tasks.", "",
        "## Diagnostic outcomes", "",
        "| Run | Policy | Exact successes | Grounded contracts |",
        "| --- | --- | --- | --- |",
    ]
    for label, record in records.items():
        for policy, values in record["metrics"]["diagnostics"].items():
            count = values["tasks"]
            lines.append(f"| {label} | {policy} | {values['successes']}/{count} | "
                         f"{values['contract_valid']}/{count} |")
    lines.extend([
        "", "A valid routing response does not imply a correct diagnostic response.",
        "The learned policy is a shadow candidate if diagnostic acceptance fails; the",
        "rule control remains the recommendation for this fixed demonstration corpus.", "",
        "## HTTP load samples", "",
        "Each phase replays positive-budget public states through the warmed service.",
        "Client p50/p95 include local HTTP and server queue wait, and exclude waiting",
        "for a load-generator concurrency slot. Dispatch attempts/second includes",
        "failed/cancelled attempts; successful completion throughput is retained",
        "separately in raw metrics. These short samples are not a sustained capacity test.", "",
        "| Run | Concurrency | Successful / attempted | Failures | Dispatch attempts/s | Client p50 (ms) | Client p95 (ms) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for label, record in records.items():
        for phase in record["metrics"]["load"]:
            lines.append(
                f"| {label} | {phase['concurrency']} | "
                f"{phase['successful_requests']}/{phase['requests']} | {phase['failures']} | "
                f"{number(phase['requests_per_second'])} | "
                f"{number(phase['p50_latency_seconds'], 1000)} | "
                f"{number(phase['p95_latency_seconds'], 1000)} |"
            )
    lines.extend(["", "## Acceptance and recommendation", ""])
    for label, record in records.items():
        metrics = record["metrics"]
        checks = metrics["illustrative_acceptance"]
        lines.extend([
            f"### {label}", "",
            f"- Service-boundary checks: **{'pass' if metrics['acceptance']['passed'] else 'fail'}**.",
            f"- Model diagnostic gate: **{'pass' if checks['diagnostic_quality_passed'] else 'fail'}**.",
            f"- Configured HTTP load gates: **{'pass' if checks['all_load_checks_passed'] else 'fail'}**.",
            f"- Recorded recommendation: `{checks['recommendation']}`.",
            f"- Illustrative targets: `{json.dumps(checks['targets'], sort_keys=True)}`.", "",
            f"Inspect [{label}/service_acceptance.json]({label}/service_acceptance.json) and "
            f"[{label}/metrics.json]({label}/metrics.json) for the individual checks.", "",
        ])
    lines.extend([
        "The targets are configured examples, not an achieved customer SLA. The service",
        "checks cover the recorded local boundary. They do not demonstrate trained-GPU",
        "overload recovery, kernel cancellation, external ingress, or all operational failures.", "",
        "## Latency figure", "",
        "![Measured client p50/p95 HTTP latency by concurrency](latency.png)", "",
        "The panels use separate y-axis scales so the tiny controller's millisecond timings remain visible.",
        "The figure compares different checkpoints and repeated public states. It does",
        "not isolate a batching improvement: no matched individual-inference ablation",
        "is present in this demonstration. Use the separate research inference benchmark",
        "for that comparison.", "",
        "## Provenance", "",
    ])
    for label, record in records.items():
        run = record["run"]
        checkpoint = record["metrics"]["checkpoint"]
        commit = run["git_commit"]
        lines.extend([
            f"- **{label}:** [{commit[:12]}](https://github.com/darshgarg7/Conductor/commit/{commit}), "
            f"seed {run['seed']}, clean source `{not run['git_dirty']}`, "
            f"backend `{checkpoint['backend']}`, stage `{checkpoint['stage']}`, "
            f"pretrained `{checkpoint['pretrained']}`.",
        ])
    lines.extend([
        "", "Each raw run records task-inventory, public-runbook, specialist, combined",
        "workload, and checkpoint hashes. `specialist_audit.json` preserves the frozen",
        "tool identities. The archive validator checks raw scores, task coverage, public",
        "state fields, frozen identities, load percentiles, and service-check consistency.",
        "These checks establish artifact consistency, not independent authentication.",
    ])
    lines.extend([
        "", "Raw JSON/JSONL/CSV are copied byte-for-byte. `checksums.json` inventories the",
        "published files; `provenance.json` records hashes of omitted local server logs",
        "and resolved serving files. Secrets and model weights are not published.", "",
        "Regenerate a fresh archive with:", "",
        "```bash",
        "python scripts/report_support_demo.py \\",
        "  --run tiny=outputs/demos/support-tiny \\",
        "  --run granite=outputs/demos/support-granite \\",
        "  --output outputs/reports/support-demo",
        "```", "",
        f"See the [customer scenario]({documentation('customer_case_study.md')}),",
        f"[walkthrough]({documentation('demo_walkthrough.md')}), and",
        f"[operations guide]({documentation('support_operations.md')}) for the architecture and limits.", "",
    ])
    (output / "report.md").write_text("\n".join(lines))

    figure, axes = plt.subplots(1, len(records), figsize=(5 * len(records), 4.5), squeeze=False)
    for axis, (label, record) in zip(axes.flat, records.items()):
        labels, medians, tails = [], [], []
        for phase in record["metrics"]["load"]:
            if phase["p50_latency_seconds"] is None or phase["p95_latency_seconds"] is None:
                continue
            labels.append(f"concurrency {phase['concurrency']}")
            medians.append(phase["p50_latency_seconds"] * 1000)
            tails.append(phase["p95_latency_seconds"] * 1000)
        locations = list(range(len(labels)))
        first = axis.bar([value - .18 for value in locations], medians, .36, label="client p50")
        second = axis.bar([value + .18 for value in locations], tails, .36, label="client p95")
        axis.bar_label(first, fmt="%.1f", padding=3)
        axis.bar_label(second, fmt="%.1f", padding=3)
        axis.set_ylim(0, max([1.0, *tails]) * 1.25)
        axis.set_xticks(locations, labels)
        axis.set_ylabel("Local HTTP elapsed time (ms)")
        axis.set_title(f"{label}: trained CPU service")
        axis.legend(loc="upper left")
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("Short replay samples · separate y-axis scales")
    figure.tight_layout()
    figure.savefig(output / "latency.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=completed demo directory")
    parser.add_argument("--output", required=True, type=Path, help="Fresh archive directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Archive output must be a fresh directory")
    sources: dict[str, Path] = {}
    records: dict[str, dict[str, Any]] = {}
    for spec in args.run:
        label, separator, source = spec.partition("=")
        if not separator or not re.fullmatch(r"[a-z][a-z0-9_-]*", label) or label in sources:
            parser.error("Use unique safe labels: LABEL=directory")
        sources[label] = Path(source)
        run = read_json(sources[label] / "run.json")
        metrics = read_json(sources[label] / "metrics.json")
        if run.get("metrics") != metrics or not run.get("runtime_seconds"):
            raise ValueError(f"{label}: incomplete or inconsistent run")
        if not re.fullmatch(r"[0-9a-f]{40}", run.get("git_commit", "")) or run.get("git_dirty") is not False:
            raise ValueError(f"{label}: publication requires a recorded clean source commit")
        if metrics["checkpoint"]["stage"] not in {"sft", "preference"}:
            raise ValueError(f"{label}: trained checkpoint required")
        if run["configuration"]["resolved_serving_config"]["model"]["device"] != "cpu":
            raise ValueError(f"{label}: this report renders the CPU demonstration only")
        for name in PUBLIC_FILES:
            path = sources[label] / name
            if not path.is_file():
                raise ValueError(f"{label}: missing raw artifact {name}")
            # Refuse accidental publication of personal local paths. Logs/config
            # descriptors retain hashes without copying their contents.
            if re.search(rb"/(?:Users|home)/[^/\s\"']+", path.read_bytes()):
                raise ValueError(f"{label}: raw artifact contains a personal absolute path: {name}")
        validate_raw(sources[label], metrics)
        records[label] = {"run": run, "metrics": metrics}
    if len({record["metrics"]["support_workload_sha256"] for record in records.values()}) != 1:
        raise ValueError("All published runs must share the support workload and frozen specialists")
    args.output.mkdir(parents=True)
    provenance: dict[str, Any] = {
        "scope": "CPU support fixtures and localhost HTTP, not NVIDIA validation",
        "renderer_sha256": sha256(Path(__file__)), "runs": {},
    }
    for label, source in sources.items():
        destination = args.output / label
        destination.mkdir()
        for name in PUBLIC_FILES:
            shutil.copyfile(source / name, destination / name)
        provenance["runs"][label] = {
            "git_commit": records[label]["run"]["git_commit"],
            "checkpoint": records[label]["metrics"]["checkpoint"],
            "omitted": {name: {"sha256": sha256(source / name), "bytes": (source / name).stat().st_size}
                        for name in ("server.log", "serving_config.yaml") if (source / name).is_file()},
        }
    render(args.output, records)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    checksums = {str(path.relative_to(args.output)): sha256(path)
                 for path in sorted(args.output.rglob("*")) if path.is_file()}
    (args.output / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")


if __name__ == "__main__":
    main()
