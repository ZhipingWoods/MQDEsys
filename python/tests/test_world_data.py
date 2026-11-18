"""World-model data sources: shape, ground truth, action conditioning, disjoint splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mqde.data.world import load_samples, load_split, synthetic


def test_synthetic_clips_have_the_requested_shape_and_ground_truth() -> None:
    clips = synthetic(5, seed=3, frames=4, size=32)
    assert len(clips) == 5
    for i, c in enumerate(clips):
        assert c.frames.shape == (4, 32, 32, 3)
        assert c.frames.dtype == np.float32
        assert c.frames.min() >= 0.0 and c.frames.max() <= 1.0
        assert c.prompt and c.meta["id"] == i
        assert c.actions is None


def test_action_conditioned_samples_carry_one_action_per_frame() -> None:
    clips = synthetic(3, seed=1, frames=5, action_conditioned=True, action_dim=128)
    assert all(c.actions is not None and c.actions.shape == (5, 128) for c in clips)
    assert all(np.isfinite(c.actions).all() for c in clips if c.actions is not None)


def test_calibration_and_evaluation_are_disjoint_for_the_synthetic_source() -> None:
    calib, ev = load_split("synthetic-video", 4, "synthetic-video", 3, seed=7, frames=4)
    assert [c.meta["id"] for c in calib] == [0, 1, 2, 3]
    assert [c.meta["id"] for c in ev] == [0, 1, 2]
    assert [c.prompt for c in calib] != [c.prompt for c in ev]


def test_npz_directory_source_round_trips(tmp_path: Path) -> None:
    clips = synthetic(2, seed=4, frames=4, action_conditioned=True)
    for i, c in enumerate(clips):
        np.savez(tmp_path / f"clip{i}.npz", frames=c.frames, actions=c.actions, prompt=c.prompt)
    loaded = load_samples(str(tmp_path), 2, seed=0, frames=4, action_conditioned=True)
    assert len(loaded) == 2
    assert loaded[0].frames.shape == (4, 32, 32, 3)
    assert loaded[0].actions is not None and loaded[0].actions.shape == (4, 128)
    assert loaded[0].prompt == clips[0].prompt


def test_ewm_rejects_a_clip_without_actions(tmp_path: Path) -> None:
    c = synthetic(1, seed=0, frames=3)[0]
    np.savez(tmp_path / "clip.npz", frames=c.frames, prompt=c.prompt)
    with pytest.raises(ValueError, match="actions"):
        load_samples(str(tmp_path), 1, action_conditioned=True)
