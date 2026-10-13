"""A tiny stand-in for MiniCPM-V with the *same module paths* (model.vision_tower / model.merger / model.language_model /
lm_head), so the minicpmv family template, the real FakeQuantEngine and the real probe run on CPU in a second."""

from __future__ import annotations

import zlib
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from mqde.models.base import QuantModel

V, D = 64, 128  # D is a multiple of the recipes' group size (128), so per-group recipes quantize every layer
EOS = 0


class TinyVLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        lin = nn.Linear
        blk = lambda a, b: nn.ModuleDict({"fc1": lin(a, b), "fc2": lin(b, a)})  # noqa: E731
        self.model = nn.ModuleDict(
            {
                "vision_tower": nn.ModuleDict(
                    {"encoder": nn.ModuleDict({"layers": nn.ModuleList([nn.ModuleDict({"mlp": blk(D, D)}) for _ in range(2)])})}
                ),
                "merger": nn.ModuleDict({"mlp": nn.ModuleList([nn.ModuleDict({"linear_1": lin(D, D)})])}),
                "language_model": nn.ModuleDict(
                    {
                        "embed_tokens": nn.Embedding(V, D),
                        "layers": nn.ModuleList(
                            [
                                nn.ModuleDict({"mlp": nn.ModuleDict({"gate_proj": lin(D, 2 * D), "down_proj": lin(2 * D, D)})})
                                for _ in range(2)
                            ]
                        ),
                    }
                ),
            }
        )
        self.lm_head = lin(D, V)
        with torch.no_grad():
            self.lm_head.weight.mul_(6.0)  # peaky next-token distributions: top-1 agreement means something

    def forward(self, feat: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        v = feat
        for layer in self.model["vision_tower"]["encoder"]["layers"]:
            v = v + layer["mlp"]["fc2"](torch.relu(layer["mlp"]["fc1"](v)))
        v = self.model["merger"]["mlp"][0]["linear_1"](v)
        lm = self.model["language_model"]
        h = lm["embed_tokens"](ids) + v[:, None, :]
        for layer in lm["layers"]:
            h = h + layer["mlp"]["down_proj"](F.silu(layer["mlp"]["gate_proj"](h)))
        return self.lm_head(h)


class FakeAdapter:
    """The four calls the VLM job makes (see mqde.models.minicpmv)."""

    family_name = "minicpmv"

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device("cpu")
        self.model = TinyVLM().eval()
        self.qmodel = QuantModel(self.model, QuantModel.load_family(self.family_name))

    def preprocess(self, sample: Any) -> dict[str, torch.Tensor]:
        img = torch.from_numpy(sample.images[0][::56, ::56].astype("float32")).mean(-1).reshape(-1) / 255.0  # 64 values
        feat = torch.cat([img, img.flip(0)])[None]
        ids = [1 + zlib.crc32(w.encode()) % (V - 1) for w in sample.question.lower().split()]
        return {"feat": feat, "ids": torch.tensor([ids])}

    @torch.inference_mode()
    def generate(self, inputs: dict[str, torch.Tensor], max_new_tokens: int) -> list[int]:
        ids, out = inputs["ids"], []
        for _ in range(max_new_tokens):
            nxt = int(self.model(inputs["feat"], ids)[0, -1].argmax())
            if nxt == EOS and out:
                break
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
        return out

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"t{i}" for i in ids)

    @torch.inference_mode()
    def answer_logits(self, inputs: dict[str, torch.Tensor], answer: list[int]) -> torch.Tensor:
        ids = torch.cat([inputs["ids"], torch.tensor([answer])], dim=1)
        return self.model(inputs["feat"], ids)[:, -(len(answer) + 1) :]

    @torch.inference_mode()
    def prompt_forward(self, inputs: dict[str, torch.Tensor]) -> None:
        self.model(inputs["feat"], inputs["ids"])
