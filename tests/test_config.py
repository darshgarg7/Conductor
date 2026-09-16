from pathlib import Path
import pytest
from conductor.utils.config import load_config


def test_relative_includes_merge_without_mutating_parent(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("model: {backend: tiny, hidden_dim: 64}\nseed: 1\n")
    (tmp_path / "child.yaml").write_text("include: base.yaml\nmodel: {hidden_dim: 32}\n")
    assert load_config(tmp_path / "child.yaml") == {"model": {"backend": "tiny", "hidden_dim": 32}, "seed": 1}
    assert load_config(tmp_path / "base.yaml")["model"]["hidden_dim"] == 64


def test_include_cycles_fail(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("include: b.yaml\n")
    (tmp_path / "b.yaml").write_text("include: a.yaml\n")
    with pytest.raises(ValueError, match="cyclic"):
        load_config(tmp_path / "a.yaml")
