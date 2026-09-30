"""VLA calibration / evaluation samples.

Sources:
- lerobot: a LeRobot v3 dataset directory (meta/info.json, meta/tasks.parquet, data/chunk-*/file-*.parquet)
  e.g. lerobot/libero_10_image. Frames are sampled evenly across episodes.
- synthetic: procedurally drawn table-top scenes (colored blocks) with manipulation instructions.
  Needs no download; useful for smoke tests, not for accuracy claims.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from mqde.models.minicpm_vla import VLASample


def _decode_image(v: Any) -> np.ndarray:
    if isinstance(v, dict) and v.get("bytes") is not None:
        return np.asarray(Image.open(io.BytesIO(v["bytes"])).convert("RGB"))
    if isinstance(v, (bytes, bytearray)):
        return np.asarray(Image.open(io.BytesIO(v)).convert("RGB"))
    arr = np.asarray(v)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return arr


def load_lerobot(root: str | Path, n: int, seed: int = 0, offset: int = 0) -> list[VLASample]:
    """Evenly samples `n` frames (distinct episodes where possible) from a LeRobot v3 dataset."""
    import pyarrow.parquet as pq

    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    features = info.get("features", {})
    image_keys = [k for k, v in features.items() if v.get("dtype") in ("image", "video") and not k.endswith("wrist_image")]
    image_keys += [k for k, v in features.items() if v.get("dtype") in ("image", "video") and k.endswith("wrist_image")]
    if not image_keys:
        raise ValueError(f"{root}: no image features (video datasets need decoding support)")
    tasks: dict[int, str] = {}
    tp = root / "meta" / "tasks.parquet"
    if tp.exists():
        t = pq.read_table(tp).to_pandas()
        if "task_index" in t.columns:
            text_col = "task" if "task" in t.columns else t.index.name
            for idx, row in t.iterrows():
                tasks[int(row["task_index"])] = str(row[text_col] if text_col in t.columns else idx)
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"{root}: no data/chunk-*/file-*.parquet")

    # Frames are drawn evenly along each episode (skipping the static start and keeping 10 future
    # actions), episodes are interleaved, and the list is deterministic for a given seed so that
    # [0, offset) (calibration) and [offset, offset + n) (evaluation) never share a frame.
    rng = np.random.default_rng(seed)
    cols = [*image_keys, "observation.state", "action", "episode_index", "frame_index", "task_index"]
    per_episode: list[list[VLASample]] = []
    total = 0
    for f in files:
        table = pq.read_table(f, columns=[c for c in cols if c in pq.read_schema(f).names]).to_pandas()
        for ep in sorted(table["episode_index"].unique()):
            rows = table[table["episode_index"] == ep].sort_values("frame_index")
            lo, hi = len(rows) // 5, len(rows) - 10
            if hi <= lo:
                continue
            k = 8
            idxs = np.unique(np.linspace(lo, hi - 1, k).astype(int) + rng.integers(0, max(1, (hi - lo) // k), k)).clip(lo, hi - 1)
            ep_samples = []
            for i in idxs:
                row = rows.iloc[int(i)]
                future = np.stack(rows["action"].iloc[int(i) : int(i) + 10].to_numpy()).astype(np.float32)
                ep_samples.append(
                    VLASample(
                        images=[_decode_image(row[k]) for k in image_keys],
                        text=tasks.get(int(row["task_index"]), "complete the task"),
                        state=np.asarray(row["observation.state"], dtype=np.float32),
                        gt_action=future,
                        meta={"source": f.name, "episode": int(ep), "frame": int(row["frame_index"])},
                    )
                )
            per_episode.append(ep_samples)
            total += len(ep_samples)
        if total >= n + offset:
            break
    # round-robin across episodes so every slice covers as many episodes/tasks as possible
    samples: list[VLASample] = []
    for j in range(max((len(e) for e in per_episode), default=0)):
        samples.extend(e[j] for e in per_episode if j < len(e))
    return samples[offset : offset + n]


COLORS = {"red": (220, 40, 40), "green": (40, 170, 60), "blue": (40, 80, 220), "yellow": (230, 200, 40)}
VERBS = ["pick up the {c} block", "push the {c} block to the left", "put the {c} block in the bowl", "grasp the {c} cube"]


def synthetic(n: int, seed: int = 0, views: int = 2) -> list[VLASample]:
    rng = np.random.default_rng(seed)
    out: list[VLASample] = []
    for i in range(n):
        imgs = []
        names = list(COLORS)
        rng.shuffle(names)
        for v in range(views):
            img = Image.new("RGB", (256, 256), tuple(int(x) for x in rng.integers(150, 220, 3)))
            d = ImageDraw.Draw(img)
            d.rectangle([0, 170, 256, 256], fill=(120, 90, 60))  # table
            for c in names[:3]:
                x, y = rng.integers(20, 200), rng.integers(120, 200)
                s = int(rng.integers(18, 36))
                d.rectangle([x, y, x + s, y + s], fill=COLORS[c])
            d.ellipse([90 + v * 20, 20, 160 + v * 20, 90], fill=(60, 60, 60))  # gripper
            imgs.append(np.asarray(img))
        verb = VERBS[int(rng.integers(len(VERBS)))].format(c=names[0])
        state = rng.standard_normal(8).astype(np.float32) * 0.1
        out.append(VLASample(images=imgs, text=verb, state=state, meta={"source": "synthetic", "id": i}))
    return out


def load_samples(source: str, n: int, seed: int = 0, offset: int = 0) -> list[VLASample]:
    if source == "synthetic":
        return synthetic(n, seed + offset)
    return load_lerobot(source, n, seed, offset)


def load_split(calib_source: str, n_calib: int, eval_source: str, n_eval: int, seed: int = 0) -> tuple[list[VLASample], list[VLASample]]:
    """Disjoint calibration / evaluation sets. Same source: loaded once and split (no shared frame)."""
    if calib_source == eval_source:
        if calib_source == "synthetic":
            return synthetic(n_calib, seed), synthetic(n_eval, seed + 10_000)
        both = load_lerobot(calib_source, n_calib + n_eval, seed)
        return both[:n_calib], both[n_calib:]
    return load_samples(calib_source, n_calib, seed), load_samples(eval_source, n_eval, seed + 1)
