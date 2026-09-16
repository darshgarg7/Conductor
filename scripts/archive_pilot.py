"""Archive completed pilot measurements without model weights or optimizer state."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from conductor.datasets.integrity import file_digest
from conductor.utils.runs import write_json


PHASES = (
    "generation", "sft", "preference", "evaluation", "inference", "analysis",
    "dense-sequential-evaluation", "strong-dense-evaluation", "serving", "export-verification",
    "export-copy-attempt", "export-numerical-attempt",
)
OMITTED = {"controller_trace.json", "latest_resume.json"}
DENSE_ARTIFACTS = {"run.json", "metrics.json", "evaluation_provenance.json", "specialist_audit.json",
                   "trajectories.jsonl", "task_metrics.csv"}


def archive(source: Path, data: Path, output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Archive destination must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, object] = {}
    omitted: list[dict[str, object]] = []

    def copy(path: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    def omit(path: Path, reason: str) -> None:
        omitted.append({"source": str(path.relative_to(source)), "bytes": path.stat().st_size,
                        "sha256": file_digest(path), "reason": reason})

    for phase in PHASES:
        directory = source / phase
        if not directory.is_dir():
            if phase in {"export-copy-attempt", "export-numerical-attempt"}:
                continue
            raise FileNotFoundError(directory)
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if phase == "export-verification" and path.name == "manifest.json":
                omit(path, "The raw export manifest contains a host-specific path; validation.json retains unchanged numerical fields and file hashes.")
            elif path.name in OMITTED or path.suffix == ".log":
                omit(path, "Console logs, profiler traces and local pointers stay in the working output.")
            elif phase in {"dense-sequential-evaluation", "strong-dense-evaluation"} and path.name not in DENSE_ARTIFACTS:
                omit(path, "Supplementary dense controls retain raw trajectories, task metrics, configuration and identities; redundant summaries are omitted.")
            elif phase in {"sft", "preference"} and path.name in {"checkpoint_pointer.json", "expert_statistics.json", "expert_statistics_before.json"}:
                omit(path, "Local checkpoint pointers and training-state expert diagnostics are omitted; matched held-out expert probes are archived.")
            elif path.name == "report.md":
                omit(path, "The combined pilot report replaces separate generated phase reports.")
            elif path.suffix in {".json", ".jsonl", ".csv", ".md", ".txt"}:
                copy(path, output / "phases" / phase / path.name)
        run_path = directory / "run.json"
        if run_path.exists():
            run = json.loads(run_path.read_text())
            provenance[phase] = {key: run.get(key) for key in ("git_commit", "git_dirty", "seed", "runtime_seconds")}
        if phase in {"sft", "preference"}:
            copy(directory / "adapter" / "adapter_config.json", output / "phases" / phase / "adapter_config.json")
            pointer = json.loads((directory / "latest_resume.json").read_text())
            resume = Path(pointer["path"])
            if not resume.is_absolute():
                resume = directory / resume
            copy(resume / "resume_manifest.json", output / "phases" / phase / "resume_manifest.json")
        if phase == "analysis":
            for path in sorted((directory / "plots").iterdir()):
                if path.suffix in {".png", ".pdf"}:
                    copy(path, output / "plots" / path.name)
    for path in sorted(data.iterdir()):
        if path.is_file() and path.suffix in {".json", ".jsonl"}:
            copy(path, output / "data" / path.name)
    for name in ("post_training_verification.json", "supervision_diagnostics.json", "doctor.json"):
        original = source / name
        if not original.exists():
            original = source / "adapter-verification" / name
        copy(original, output / "verification" / name)
    for name in ("post_training_sft_verification.json", "verify_post_training.py", "validate_merged_export.py"):
        if (source / name).exists():
            copy(source / name, output / "verification" / name)
    manifest_path = source / "export" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    validation_fields = ("original_base_model", "original_resolved_revision", "source_checkpoint", "preserve_model",
                         "in_place_merge", "validation_probe_count", "validation_tolerances", "validation_k_values",
                         "maximum_logit_difference", "maximum_probability_difference", "probability_temperature",
                         "routing_validation", "action_catalog_size", "token_accounting", "context_strategy", "max_length")
    write_json(output / "phases" / "export-verification" / "validation.json",
               {"source_manifest_sha256": file_digest(manifest_path), "files_sha256": manifest["files_sha256"],
                "validation": {key: manifest["validation"][key] for key in validation_fields},
                "scope": "Derived subset of the original export manifest; local configuration paths are omitted."})
    omit(manifest_path, "The original manifest contains a host-specific model path; unchanged validation fields and file hashes are published separately.")
    for path in sorted((source / "validation").iterdir()):
        if path.is_file() and "before_probability_gates" not in path.name:
            if path.suffix == ".xml":
                omit(path, "Pytest captured host-specific paths; parsed test counts are retained in checks.json.")
            else:
                copy(path, output / "verification" / path.name)
    write_json(output / "provenance.json", provenance)
    write_json(output / "omitted_artifacts.json", omitted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/research/granite-pilot"))
    parser.add_argument("--data", type=Path, default=Path("data/generated/granite-pilot"))
    parser.add_argument("--output", type=Path, default=Path("results/granite-pilot"))
    args = parser.parse_args()
    archive(args.source, args.data, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
