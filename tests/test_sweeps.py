"""Sweep variants must change the field used by their actual CLI consumer."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from conductor.utils.config import load_config
from conductor.utils.sweeps import materialize_sweep

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module,config_path,field", [
    ("conductor.train", "configs/training/tiny_sft.yaml", "training"),
    ("conductor.evaluate", "configs/evaluation/heldout.yaml", "k"),
    ("conductor.benchmark", "configs/inference/development.yaml", "k_values"),
])
def test_sweep_changes_consumer_k_and_isolates_artifacts(tmp_path, module, config_path, field) -> None:
    before = load_config(ROOT / config_path)
    paths = materialize_sweep(ROOT / config_path, tmp_path / "variants", module=module,
                              seeds=[42, 43], top_k=[1, 3], run_root=tmp_path / "runs")
    assert len(paths) == 4
    outputs = []
    for path in paths:
        config = load_config(path)
        k = config["sweep"]["agent_top_k"]
        assert config["sweep"]["module"] == module
        if field == "training":
            assert config["routing"]["k"] == k
            outputs.append(config["training"]["output"])
        else:
            assert config[field] == ([k] if field == "k_values" else k)
            outputs.append(config["output"])
        assert "include" not in config
    assert len(set(outputs)) == 4
    assert load_config(ROOT / config_path) == before
    assert json.loads((tmp_path / "variants/manifest.json").read_text())["module"] == module


@pytest.mark.parametrize("options", [{"seeds": [42, 42]}, {"top_k": [0]}, {"top_k": [4]},
    {"top_k": [1, 1]}, {"seeds": [-1]}, {"module": "conductor.generate"}])
def test_invalid_sweep_does_not_publish_partial_configs(tmp_path, options) -> None:
    values = {"seeds": [42], "top_k": [1], **options}
    with pytest.raises(ValueError):
        materialize_sweep(ROOT / "configs/training/tiny_sft.yaml", tmp_path / "invalid", **values)
    assert not (tmp_path / "invalid").exists()


def test_wrong_default_module_and_existing_sweep_are_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="select --module"):
        materialize_sweep(ROOT / "configs/evaluation/heldout.yaml", tmp_path / "wrong", seeds=[42], top_k=[1])
    config = ROOT / "configs/training/tiny_sft.yaml"
    paths = materialize_sweep(config, tmp_path / "existing", seeds=[42], top_k=[1])
    before = paths[0].read_bytes()
    with pytest.raises(FileExistsError):
        materialize_sweep(config, tmp_path / "existing", seeds=[42], top_k=[1])
    assert paths[0].read_bytes() == before


def launcher(tmp_path: Path, manifest: Path, *, index: str = "0", override: str = ""):
    """Execute the real shell selector; replace only the final model invocation."""
    interpreter = tmp_path / "python with spaces"
    record = tmp_path / "invocation.json"
    interpreter.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
if "-m" in sys.argv:
    with open(os.environ["CONDUCTOR_TEST_INVOCATION"], "w") as handle:
        json.dump(sys.argv[1:], handle)
else:
    os.execv(os.environ["CONDUCTOR_TEST_REAL_PYTHON"], [os.environ["CONDUCTOR_TEST_REAL_PYTHON"], *sys.argv[1:]])
''')
    interpreter.chmod(0o755)
    environment = {**os.environ, "CONDUCTOR_ROOT": str(ROOT), "CONDUCTOR_PYTHON": str(interpreter),
                   "SLURM_ARRAY_TASK_ID": index, "CONDUCTOR_MODULE": override,
                   "CONDUCTOR_TEST_REAL_PYTHON": sys.executable, "CONDUCTOR_TEST_INVOCATION": str(record)}
    result = subprocess.run(["bash", str(ROOT / "scripts/slurm/sweep.slurm"), str(manifest)], env=environment,
                            text=True, capture_output=True, timeout=30)
    return result, record


def test_launcher_automatically_uses_declared_benchmark_module(tmp_path) -> None:
    paths = materialize_sweep(ROOT / "configs/inference/development.yaml", tmp_path / "bench",
                              module="conductor.benchmark", seeds=[42], top_k=[1, 3])
    result, record = launcher(tmp_path, tmp_path / "bench/manifest.txt", index="1")
    assert result.returncode == 0, result.stderr
    assert json.loads(record.read_text()) == ["-m", "conductor.benchmark", "--config", str(paths[1])]
    assert load_config(paths[1])["k_values"] == [3]


@pytest.mark.parametrize("index,override,detail", [
    ("0", "conductor.train", "conflicts"), ("-1", "", "outside"), ("9", "", "outside"),
])
def test_launcher_rejects_wrong_module_and_invalid_array_index(tmp_path, index, override, detail) -> None:
    materialize_sweep(ROOT / "configs/evaluation/heldout.yaml", tmp_path / "eval",
                      module="conductor.evaluate", seeds=[42], top_k=[1])
    result, record = launcher(tmp_path, tmp_path / "eval/manifest.txt", index=index, override=override)
    assert result.returncode != 0 and detail in result.stderr
    assert not record.exists()
