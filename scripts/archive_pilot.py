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
            raise FileNotFoundError(directory)
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if path.name in OMITTED or path.suffix == ".log":
                omit(path, "Console logs, profiler traces and local pointers stay in the working output.")
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
    for name in ("post_training_verification.json", "post_training_sft_verification.json",
                 "supervision_diagnostics.json", "doctor.json", "verify_post_training.py"):
        copy(source / name, output / "verification" / name)
    copy(source / "export" / "manifest.json", output / "phases" / "export-verification" / "manifest.json")
    for path in sorted((source / "validation").iterdir()):
        if path.is_file() and "before_probability_gates" not in path.name:
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
