"""Video / world-model calibration and evaluation samples.

The built-in ``synthetic-video`` source needs no download and gives a ground-truth future, so a
quantized DiT can be scored both against the unquantized model (``PSNR(q, fp)``) and against the real
frames (``PSNR(q, gt)``). Each sample is a short clip plus the text prompt that describes its
motion. EWM samples additionally carry a per-frame action vector.

External sources are directories (or single files) of ``.npz`` clips:

``frames``  float array ``[T, H, W, 3]`` in ``[0, 1]`` (required)
``actions`` float array ``[T, A]`` (optional; required by the EWM pipeline)
``prompt``  string (optional)

``.mp4`` decoding is deliberately not a dependency; export frames with the project's own tooling or
use the synthetic source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SIZE = 32
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (225, 55, 55),
    "green": (45, 175, 75),
    "blue": (55, 85, 225),
    "yellow": (230, 200, 45),
}
DIRECTIONS = ["right", "left", "up", "down", "diagonal"]
SHAPES = ["square", "circle"]


@dataclass
class WorldSample:
    """One short clip: ``frames`` is ``[T, H, W, 3]`` float32 in ``[0, 1]``."""

    frames: np.ndarray
    prompt: str
    actions: np.ndarray | None = None  # [T, A] float32 for action-conditioned (EWM) samples
    meta: dict[str, Any] = field(default_factory=dict)


def _shape_xy(direction: str, t: float, size: int) -> tuple[int, int]:
    """Position of the moving object at normalised time ``t`` in ``[0, 1]``."""
    lo, hi = int(size * 0.22), int(size * 0.78)
    span = hi - lo
    if direction == "right":
        return lo + int(span * t), size // 2
    if direction == "left":
        return hi - int(span * t), size // 2
    if direction == "up":
        return size // 2, hi - int(span * t)
    if direction == "down":
        return size // 2, lo + int(span * t)
    return lo + int(span * t), lo + int(span * t)


def _draw_clip(shape: str, color: tuple[int, int, int], direction: str, frames: int, size: int,
               action_conditioned: bool, rng: np.random.Generator, action_dim: int = 8) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((frames, size, size, 3), dtype=np.float32)
    actions = np.zeros((frames, action_dim), dtype=np.float32)
    radius = max(3, size // 10)
    dx, dy = _direction_vector(direction)
    for i in range(frames):
        t = i / max(1, frames - 1)
        img = Image.new("RGB", (size, size), (238, 238, 238))
        d = ImageDraw.Draw(img)
        # a static landmark makes the clip's global motion visible to PSNR, not only the moving object
        d.rectangle([2, size - 7, 7, size - 2], fill=(120, 120, 120))
        cx, cy = _shape_xy(direction, t, size)
        box = [cx - radius, cy - radius, cx + radius, cy + radius]
        if shape == "circle":
            d.ellipse(box, fill=color)
        else:
            d.rectangle(box, fill=color)
        out[i] = np.asarray(img, dtype=np.float32) / 255.0
        # normalised action: direction * speed. The first slot is the frame index so EWM can tell
        # where in the clip it is even when the motion is static.
        actions[i, 0] = t
        if action_dim > 1:
            actions[i, 1] = dx
        if action_dim > 2:
            actions[i, 2] = dy
        if action_dim > 3:
            actions[i, 3:] = rng.standard_normal(action_dim - 3).astype(np.float32) * 0.05
    return out, actions


def _direction_vector(direction: str) -> tuple[float, float]:
    if direction == "right":
        return 1.0, 0.0
    if direction == "left":
        return -1.0, 0.0
    if direction == "up":
        return 0.0, -1.0
    if direction == "down":
        return 0.0, 1.0
    return 0.7071, 0.7071


def synthetic(n: int, seed: int = 0, frames: int = 4, size: int = SIZE,
              action_conditioned: bool = False, action_dim: int = 128) -> list[WorldSample]:
    """Procedurally generated clips with a known future and a prompt that describes the motion."""
    if frames < 2:
        raise ValueError("frames must be >= 2")
    rng = np.random.default_rng(seed)
    names = list(COLORS)
    out: list[WorldSample] = []
    for i in range(n):
        color_name = names[int(rng.integers(len(names)))]
        shape = SHAPES[int(rng.integers(len(SHAPES)))]
        direction = DIRECTIONS[int(rng.integers(len(DIRECTIONS)))]
        clip, actions = _draw_clip(shape, COLORS[color_name], direction, frames, size, action_conditioned, rng, action_dim)
        prompt = f"a {color_name} {shape} moves {direction} across a pale background"
        out.append(
            WorldSample(
                frames=clip,
                prompt=prompt,
                actions=actions if action_conditioned else None,
                meta={"source": "synthetic-video", "id": i, "direction": direction, "color": color_name, "shape": shape},
            )
        )
    return out


def _load_npz(path: Path, action_conditioned: bool) -> WorldSample:
    with np.load(path) as z:
        if "frames" not in z:
            raise ValueError(f"{path}: missing 'frames' array ([T, H, W, 3] in [0, 1])")
        frames = np.asarray(z["frames"], dtype=np.float32)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"{path}: frames must be [T, H, W, 3], got {frames.shape}")
        if frames.shape[0] < 2:
            raise ValueError(f"{path}: a clip needs at least 2 frames, got {frames.shape[0]}")
        if frames.max(initial=0.0) > 1.0 + 1e-6:
            frames = frames / 255.0  # accept uint8-style arrays too
        frames = np.clip(frames, 0.0, 1.0)
        actions = np.asarray(z["actions"], dtype=np.float32) if "actions" in z else None
        if action_conditioned and actions is None:
            raise ValueError(f"{path}: EWM samples need an 'actions' array [T, A]")
        prompt = str(z["prompt"]) if "prompt" in z else f"video clip from {path.name}"
    return WorldSample(frames=frames, prompt=prompt, actions=actions, meta={"source": str(path)})


def _npz_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".npz":
            raise FileNotFoundError(f"world dataset '{path}' must be an .npz clip or a directory of them")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"world dataset '{path}' not found")
    files = sorted(path.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"{path}: no .npz clips found")
    return files


def load_samples(source: str, n: int, seed: int = 0, offset: int = 0, frames: int = 4,
                 action_conditioned: bool = False, action_dim: int = 128) -> list[WorldSample]:
    if source in ("synthetic", "synthetic-video"):
        return synthetic(n, seed + offset, frames=frames, action_conditioned=action_conditioned, action_dim=action_dim)
    files = _npz_files(Path(source))
    if offset >= len(files):
        return []
    chosen = files[offset : offset + n]
    if len(chosen) < n:
        raise ValueError(f"{source}: asked for {n} clips at offset {offset}, found {len(files)}")
    return [_load_npz(p, action_conditioned) for p in chosen]


def load_split(calib_source: str, n_calib: int, eval_source: str, n_eval: int, seed: int = 0,
               frames: int = 4, action_conditioned: bool = False, action_dim: int = 128) -> tuple[list[WorldSample], list[WorldSample]]:
    """Disjoint calibration / evaluation sets; the same synthetic source is split by seed."""
    if calib_source == eval_source:
        if calib_source in ("synthetic", "synthetic-video"):
            return (
                synthetic(n_calib, seed, frames, action_conditioned=action_conditioned, action_dim=action_dim),
                synthetic(n_eval, seed + 10_000, frames, action_conditioned=action_conditioned, action_dim=action_dim),
            )
        both = load_samples(calib_source, n_calib + n_eval, seed, 0, frames, action_conditioned, action_dim)
        return both[:n_calib], both[n_calib:]
    return (
        load_samples(calib_source, n_calib, seed, 0, frames, action_conditioned, action_dim),
        load_samples(eval_source, n_eval, seed + 1, 0, frames, action_conditioned, action_dim),
    )
