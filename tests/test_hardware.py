import pytest
import torch
from conductor.doctor import inspect
from conductor.utils.hardware import validate_device


def test_no_cpu_result_can_satisfy_cuda_gate() -> None:
    with pytest.raises(ValueError, match="real NVIDIA"):
        validate_device("cpu", "float32", require_cuda=True)


def test_unavailable_cuda_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        validate_device("cuda:0", "float32")


def test_doctor_measures_cpu_probe() -> None:
    report = inspect("cpu", "float32", probe=True)
    assert report["status"] == "compatible" and report["probe"]["finite"]
    assert report["validation"]["nvidia_cuda"] is False


def test_unsupported_precision_fails() -> None:
    with pytest.raises(ValueError, match="float16 CPU"):
        validate_device("cpu", "float16")
