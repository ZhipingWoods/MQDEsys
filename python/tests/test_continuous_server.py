"""In-process serving for VLA actions and WM/EWM video."""

from __future__ import annotations

import numpy as np
import torch

from mqde.backend.continuous_server import smoke_and_load


class FakeVLA:
    def predict(self, sample, seed: int = 0):
        return {"vl_embs": torch.zeros(1, 2, 3), "actions": torch.full((4, 7), float(seed))}


class FakeWorld:
    def generate(self, sample, seed: int = 0, inputs=None, steps=None):
        return torch.full((4, 3, 8, 8), 0.5)


def test_vla_native_server_smoke_and_load() -> None:
    r = smoke_and_load(FakeVLA(), "vla", {"requests": 2, "concurrency": 1, "action_dim": 7})
    assert r["status"] == "succeeded" and r["protocol"] == "vla/actions-v1"
    assert r["load"]["ok"] == 2 and r["load"]["failed"] == 0
    assert r["load"]["latency_ms"]["p50"] is not None
    assert r["smoke"]["ok"] is True


def test_world_and_ewm_native_server_return_frames() -> None:
    for domain, conditioned in (("world", False), ("ewm", True)):
        r = smoke_and_load(FakeWorld(), domain, {"requests": 1, "concurrency": 1}, action_conditioned=conditioned, frames=4)
        assert r["status"] == "succeeded" and r["protocol"] == "world/video-v1"
        assert r["load"]["ok"] == 1 and r["smoke"]["ok"] is True
