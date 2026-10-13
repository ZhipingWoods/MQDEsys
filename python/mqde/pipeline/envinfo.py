"""Facts about the machine a job ran on, for its report."""

from __future__ import annotations

import torch


def device_name(device: str) -> str:
    """The driver's name for a CUDA device ("NVIDIA GeForce RTX 4090"), "cpu" for anything else. It is a label in the report, so a
    driver that cannot answer (or a CPU-only torch asked about a CUDA device) falls back to the device string instead of failing the job."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return "cpu"
    try:
        return torch.cuda.get_device_name(torch.device(device))
    except Exception:
        return device
