"""VLM calibration / evaluation samples: an image (or a few), a question, and sometimes the right answer.

Sources:
- synthetic: procedurally drawn 448x448 scenes (coloured shapes) with three kinds of question whose answer is read off the
  drawing itself -- the colour of a named shape, how many shapes there are, whether a coloured shape is present. Needs no
  download, and having a ground truth lets the job report real accuracy next to the quantization error.
- a LeRobot v3 dataset directory (``mqde.data.vla.load_lerobot``): real camera images with one fixed captioning question and no
  reference text; the unquantized model's own greedy answer is the reference then.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SIZE = 448
SHAPES = ["square", "circle", "triangle"]
PALETTE: dict[str, tuple[int, int, int]] = {"red": (220, 40, 40), "green": (40, 170, 60), "blue": (40, 80, 220), "yellow": (230, 200, 40)}
CAPTION_QUESTION = "Describe the image in one sentence."


@dataclass
class VLMSample:
    images: list[np.ndarray]  # HxWx3 uint8
    question: str
    answer: str | None = None  # ground truth text when the source has one
    meta: dict[str, Any] = field(default_factory=dict)


def _draw(d: ImageDraw.ImageDraw, shape: str, cx: int, cy: int, r: int, fill: tuple[int, int, int]) -> None:
    if shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)


def _scene(rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
    """2-3 shapes of distinct kinds on a pale background, centres spread over a grid so none overlap."""
    img = Image.new("RGB", (SIZE, SIZE), (235, 235, 235))
    d = ImageDraw.Draw(img)
    n = int(rng.integers(2, 4))
    kinds = list(rng.permutation(SHAPES)[:n])
    colors = list(rng.permutation(list(PALETTE))[:n])
    slots = list(rng.permutation(9)[:n])  # 3x3 grid cells
    centers: dict[str, tuple[int, int]] = {}
    for kind, color, slot in zip(kinds, colors, slots, strict=True):
        cy, cx = (int(slot) // 3) * 150 + 75, (int(slot) % 3) * 150 + 75
        _draw(d, kind, cx, cy, 48, PALETTE[color])
        centers[kind] = (cy, cx)
    meta = {"present": list(kinds), "colors": {k: c for k, c in zip(kinds, colors, strict=True)}, "centers": centers,
            "palette": {c: PALETTE[c] for c in PALETTE}}
    return np.asarray(img), meta


def synthetic(n: int, seed: int = 0) -> list[VLMSample]:
    rng = np.random.default_rng(seed)
    out: list[VLMSample] = []
    for i in range(n):
        img, meta = _scene(rng)
        kind = ("color", "count", "exists")[i % 3]
        if kind == "color":
            target = str(rng.choice(meta["present"]))
            q, a = f"What color is the {target}? Answer with one word.", meta["colors"][target]
            meta["target"] = target
        elif kind == "count":
            q, a = "How many shapes are in the image? Answer with a number.", str(len(meta["present"]))
        else:
            asked = str(rng.choice(SHAPES))
            q, a = f"Is there a {asked} in the image? Answer yes or no.", "yes" if asked in meta["present"] else "no"
            meta["asked"] = asked
        out.append(VLMSample(images=[img], question=q, answer=a, meta={**meta, "kind": kind, "source": "synthetic", "id": i}))
    return out


def load_lerobot_vlm(root: str | Path, n: int, seed: int = 0, offset: int = 0) -> list[VLMSample]:
    from mqde.data.vla import load_lerobot

    return [VLMSample(images=s.images[:1], question=CAPTION_QUESTION, answer=None, meta={"source": "lerobot", **(s.meta or {})})
            for s in load_lerobot(root, n, seed, offset)]


def load_samples(source: str, n: int, seed: int = 0, offset: int = 0) -> list[VLMSample]:
    if source == "synthetic":
        return synthetic(n, seed + offset)
    if not Path(source).exists():
        raise FileNotFoundError(f"VLM dataset '{source}' not found (use 'synthetic' or a LeRobot v3 directory)")
    return load_lerobot_vlm(source, n, seed, offset)


def load_split(calib_source: str, n_calib: int, eval_source: str, n_eval: int, seed: int = 0) -> tuple[list[VLMSample], list[VLMSample]]:
    """Disjoint calibration / evaluation sets. Same source: loaded once and split, so no frame appears in both."""
    if calib_source == eval_source:
        if calib_source == "synthetic":
            return synthetic(n_calib, seed), synthetic(n_eval, seed + 10_000)
        both = load_lerobot_vlm(calib_source, n_calib + n_eval, seed)
        return both[:n_calib], both[n_calib:]
    return load_samples(calib_source, n_calib, seed), load_samples(eval_source, n_eval, seed + 1)
