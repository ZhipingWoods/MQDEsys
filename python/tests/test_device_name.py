"""The device name in a report is a label: failing to read it must never fail the job."""

from __future__ import annotations

import pytest
import torch

from mqde.pipeline.envinfo import device_name


def test_cpu_and_non_cuda_devices_are_called_cpu() -> None:
    assert device_name("cpu") == "cpu"
    assert device_name("mps") == "cpu"


def test_a_cuda_device_is_named_by_the_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    seen: list[object] = []
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d: seen.append(d) or "NVIDIA Test GPU")
    assert device_name("cuda:1") == "NVIDIA Test GPU"
    assert seen == [torch.device("cuda:1")], "the device that was asked for, not always device 0"


def test_without_cuda_the_name_is_cpu_even_for_a_cuda_device_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert device_name("cuda:0") == "cpu"


def test_a_driver_that_cannot_say_does_not_fail_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CPU-only torch with is_available patched to True (as the serving tests do) raises AssertionError in get_device_name."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def boom(d: object) -> str:
        raise AssertionError("Torch not compiled with CUDA enabled")

    monkeypatch.setattr(torch.cuda, "get_device_name", boom)
    assert device_name("cuda:0") == "cuda:0", "falls back to the device string"
