"""Adapter for MiniCPM-RobotManip (MiniCPM-V 4.6 VLM + flow-matching DiT action head).

Inference follows the model card: images resized to 448x448, MiniCPM-V chat template, VLM last
hidden state -> action head with 4 denoising steps. For evaluation we make the flow-matching
noise deterministic (seeded per sample) so BF16 and quantized runs see identical noise and the
action difference is attributable to quantization alone.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from mqde.models.base import QuantModel

IMAGE_SIZE = (448, 448)
STATE_DIM = 80


@dataclass
class VLASample:
    """One decision step: camera views + instruction + proprio state."""

    images: list[np.ndarray]  # HxWx3 uint8
    text: str
    state: np.ndarray  # (<=80,) float32, zero-padded to STATE_DIM
    embodiment_id: int = 0
    gt_action: np.ndarray | None = None  # (horizon_gt, action_dim_gt) if available
    meta: dict[str, Any] | None = None


class MiniCPMVLA:
    family_name = "minicpm_vla"

    def __init__(self, path: str | Path, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
        from transformers import AutoModel, AutoProcessor

        self.path = str(path)
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(self.path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.path, trust_remote_code=True)
        self.model.to(self.device).eval()
        self.qmodel = QuantModel(self.model, QuantModel.load_family(self.family_name))
        self.horizon = int(self.model.config.action_horizon)
        self.action_dim = int(self.model.config.action_dim)

    # ---------------- preprocessing ----------------
    def preprocess(self, sample: VLASample) -> dict[str, torch.Tensor]:
        content: list[dict[str, Any]] = []
        for img in sample.images:
            arr = np.asarray(Image.fromarray(img).convert("RGB").resize(IMAGE_SIZE)).copy()
            content.append({"type": "image", "image": arr})
        content.append({"type": "text", "text": sample.text})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": False},
        )
        return {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    def _state(self, sample: VLASample) -> torch.Tensor:
        s = np.zeros(STATE_DIM, dtype=np.float32)
        v = np.asarray(sample.state, dtype=np.float32).ravel()[:STATE_DIM]
        s[: v.size] = v
        return torch.from_numpy(s).to(self.device).view(1, 1, STATE_DIM)

    # ---------------- inference ----------------
    @torch.inference_mode()
    def vl_embeddings(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.model._vlm_forward(inputs)
        return out.hidden_states[-1].to(self.model.action_head_dtype)

    @torch.inference_mode()
    def predict(self, sample: VLASample, seed: int = 0, inputs: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """Returns {'vl_embs': [1, T, 1024] fp32, 'actions': [horizon, action_dim] fp32}."""
        inputs = inputs if inputs is not None else self.preprocess(sample)
        vl = self.vl_embeddings(inputs)
        emb = torch.tensor([sample.embodiment_id], dtype=torch.long, device=self.device)
        with self._seeded(seed):
            actions = self.model.action_head.predict_action(vl, self._state(sample), emb)
        return {"vl_embs": vl.float(), "actions": actions[0].float()}

    @contextlib.contextmanager
    def _seeded(self, seed: int) -> Iterator[None]:
        # Seed only the noise draw inside predict_action; restore RNG state afterwards.
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            yield

    def layer_inputs_hook_targets(self) -> Sequence[str]:
        return [u.name for u in self.qmodel.units]
