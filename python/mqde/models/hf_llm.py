"""Generic Hugging Face causal-LM adapter (LLaMA / Qwen2 / Qwen3 / Mistral / Gemma ...).

Exposes the uniform block view the blockwise engine needs (docs/原始设想 "BaseModel"):
``get_blocks()`` returns the decoder layers, ``first_block_inputs()`` captures the hidden states that
enter block 0 (Catcher trick) together with the kwargs every layer needs (position embeddings,
attention mask), and ``QuantModel`` provides the per-linear ``Unit`` classification used everywhere else.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from mqde.models.base import QuantModel


class _Caught(Exception):
    pass


class HFCausalLM:
    family_name = "generic_decoder"

    def __init__(self, path: str | Path, device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.path = str(path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        hf = AutoModelForCausalLM.from_pretrained(self.path, dtype=dtype, trust_remote_code=False)
        self.model: nn.Module = cast(Any, hf).to(self.device).eval()
        cast(Any, self.model).config.use_cache = False
        self.qmodel = QuantModel(self.model, QuantModel.load_family(self.family_name))

    def to(self, device: str | torch.device) -> HFCausalLM:
        """Move the live model to `device` (T-1304/T-1306: quant_device/evaluation.device split).

        Only ever called when the caller already knows the target differs from the current device --
        everything downstream (blockwise capture, forward passes, FakeQuantLinear) reads the device off
        the module/tensor it is handed rather than a value cached at construction time, so moving the
        container here is enough; detached originals held by a quantizer's own restore() bookkeeping are
        untouched by this (they are no longer part of the module tree `.to()` recurses through), which is
        exactly what keeps them pinned at the quant device they were captured on.
        """
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    @classmethod
    def from_model(cls, model: nn.Module, tokenizer: Any = None) -> HFCausalLM:
        """Wrap an already-built model (tests: tiny random configs, no weights on disk)."""
        self = cls.__new__(cls)
        p = next(model.parameters())
        self.path, self.device, self.dtype, self.tokenizer = "<memory>", p.device, p.dtype, tokenizer
        self.model = model.eval()
        cast(Any, self.model).config.use_cache = False
        self.qmodel = QuantModel(self.model, QuantModel.load_family(self.family_name))
        return self

    # ------------------------------------------------------------ block view
    def decoder(self) -> nn.Module:
        m = self.model
        return cast(nn.Module, m.model) if hasattr(m, "model") else m

    def get_blocks(self) -> nn.ModuleList:
        return cast(nn.ModuleList, cast(Any, self.decoder()).layers)

    def block_prefix(self, i: int) -> str:
        for name, mod in self.model.named_modules():
            if mod is self.get_blocks()[i]:
                return name
        raise KeyError(i)

    @torch.inference_mode()
    def first_block_inputs(self, batches: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        """Run each batch until block 0 and record its positional input and kwargs."""
        blocks = self.get_blocks()
        hs: list[torch.Tensor] = []
        kws: list[dict[str, Any]] = []
        orig = blocks[0]

        class Catcher(nn.Module):
            def __init__(self, inner: nn.Module) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
                hs.append(hidden_states)
                kws.append({k: v for k, v in kwargs.items() if k not in ("past_key_values", "past_key_value", "use_cache")})
                raise _Caught

        blocks[0] = Catcher(orig)
        try:
            for ids in batches:
                with contextlib.suppress(_Caught):
                    self.model(input_ids=ids.to(self.device))
        finally:
            blocks[0] = orig
        return hs, kws

    @staticmethod
    def run_block(block: nn.Module, h: torch.Tensor, kw: dict[str, Any]) -> torch.Tensor:
        out = block(h, **kw)
        return out[0] if isinstance(out, (tuple, list)) else out

    @torch.inference_mode()
    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=ids.to(self.device)).logits
