"""Module wrappers used by the blockwise engine.

FakeQuantLinear keeps the original weight and applies weight QDQ once (cached) and activation QDQ
per forward. The fake-quantized weight is computed with mqde.quant.numerics, which is bit-exact
with the C++ core, so fake-quant evaluation and real export share one numerical definition.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mqde.quant.numerics import QuantSpec, fake_quant, fake_quant_act


class FakeQuantLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        w_spec: QuantSpec,
        a_spec: QuantSpec,
        name: str = "",
        w_override: torch.Tensor | None = None,
        in_scale: torch.Tensor | None = None,
        final: bool = False,
    ) -> None:
        """w_override: transformed weight from an algorithm (AWQ/SmoothQuant scaled, or GPTQ-compensated).
        in_scale: per-input-channel divisor for activations paired with a scaled weight.
        final: w_override is already on the quantization grid (GPTQ); use it without re-quantizing."""
        super().__init__()
        self.name = name
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.w_spec = w_spec
        self.a_spec = a_spec
        self.algo = ""
        self.orig_dtype = linear.weight.dtype
        w = (linear.weight.detach() if w_override is None else w_override).to(torch.float32)
        if final:
            dq = w
        else:
            dq, _, _ = fake_quant(w, w_spec)
        self.register_buffer("weight", dq.to(self.orig_dtype), persistent=False)
        self.register_buffer(
            "in_scale", None if in_scale is None else in_scale.to(linear.weight.device, torch.float32), persistent=False
        )
        self.bias = linear.bias

    @classmethod
    def replace(cls, linear: nn.Linear, w_spec: QuantSpec, a_spec: QuantSpec, name: str = "") -> FakeQuantLinear:
        return cls(linear, w_spec, a_spec, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_scale is not None:
            x = (x.to(torch.float32) / self.in_scale).to(x.dtype)
        if self.a_spec.enabled:
            x = fake_quant_act(x, self.a_spec)
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))

    def extra_repr(self) -> str:
        w = f"{self.w_spec.format}{self.w_spec.bits if self.w_spec.format == 'int' else ''}"
        a = f"{self.a_spec.format}{self.a_spec.bits if self.a_spec.format == 'int' else ''}" if self.a_spec.enabled else "none"
        return f"in={self.in_features}, out={self.out_features}, w={w}/{self.w_spec.granularity}, a={a}, algo={self.algo or 'rtn'}"


def get_submodule_parent(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    if "." not in name:
        return root, name
    parent, child = name.rsplit(".", 1)
    return root.get_submodule(parent), child


def replace_module(root: nn.Module, name: str, new: nn.Module) -> None:
    parent, child = get_submodule_parent(root, name)
    setattr(parent, child, new)
