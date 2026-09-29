"""KV-cache quantization (T-0607; docs/原始设想 "KV_REGISTRY").

Fake-quantizes the key/value tensors that would be written to the cache (post-RoPE keys, values)
right before attention is computed, so evaluation sees exactly the precision the cache would hold.
Implemented as a Hugging Face attention interface wrapper: it works for every model that dispatches
through ``ALL_ATTENTION_FUNCTIONS`` (LLaMA, Qwen2/3, Mistral, Gemma, ...), prefill and decode alike.

Tensors are [batch, heads, seq, head_dim]. Granularity:
  per_token  one scale per (batch, head, token) over head_dim  -- KIVI-style for values
  per_head   one scale per (batch, head) over (seq, head_dim)
  per_channel one scale per (batch, head, channel) over seq    -- KIVI-style for keys
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from mqde.quant.numerics import QuantSpec, fake_quant

KV_REGISTRY: dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    def deco(cls: type) -> type:
        KV_REGISTRY[name] = cls
        return cls

    return deco


def _fq(x: torch.Tensor, spec: QuantSpec, gran: str) -> torch.Tensor:
    b, h, s, d = x.shape
    xf = x.to(torch.float32)
    if gran == "per_token":
        m = xf.reshape(-1, d)
    elif gran == "per_head":
        m = xf.reshape(b * h, s * d)
    elif gran == "per_channel":
        m = xf.permute(0, 1, 3, 2).reshape(b * h * d, s)
    else:
        raise ValueError(f"unknown KV granularity '{gran}'")
    dq, _, _ = fake_quant(m, spec.with_granularity("per_channel"))
    if gran == "per_channel":
        dq = dq.reshape(b, h, d, s).permute(0, 1, 3, 2)
    return dq.reshape(b, h, s, d).to(x.dtype)


@register("naive")
@dataclass
class NaiveKVQuant:
    spec: QuantSpec
    k_gran: str = "per_token"
    v_gran: str = "per_token"

    def quant_k(self, k: torch.Tensor) -> torch.Tensor:
        return _fq(k, self.spec, self.k_gran)

    def quant_v(self, v: torch.Tensor) -> torch.Tensor:
        return _fq(v, self.spec, self.v_gran)


@register("kivi")
@dataclass
class KiviKVQuant(NaiveKVQuant):
    """KIVI: keys per-channel, values per-token (Liu et al. 2024)."""

    k_gran: str = "per_channel"
    v_gran: str = "per_token"


def from_spec(kv: QuantSpec, method: str = "naive", granularity: str | None = None) -> NaiveKVQuant | None:
    if not kv.enabled:
        return None
    cls = KV_REGISTRY[method]
    if granularity and method == "naive":
        return cls(kv, granularity, granularity)
    return cls(kv)


@contextmanager
def kv_quant(model: torch.nn.Module, q: NaiveKVQuant | None) -> Iterator[None]:
    """Route attention through a KV-quantizing wrapper for the duration of the block."""
    if q is None:
        yield
        return
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.llama.modeling_llama import eager_attention_forward

    cfg = model.config
    base_impl = cfg._attn_implementation or "eager"
    base = ALL_ATTENTION_FUNCTIONS.get(base_impl, eager_attention_forward) if base_impl != "eager" else eager_attention_forward
    name = "mqde_kvq"

    def wrapped(module: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, **kw: Any) -> Any:
        return base(module, query, q.quant_k(key), q.quant_v(value), *args, **kw)

    ALL_ATTENTION_FUNCTIONS[name] = wrapped
    # The mask builder is also chosen by _attn_implementation; an unknown name gets no mask at all,
    # which sdpa happens to treat as causal but eager does not. Reuse the base implementation's.
    # (masking_utils looks the name up in the *global* mapping, hence register()).
    ALL_MASK_ATTENTION_FUNCTIONS.register(name, ALL_MASK_ATTENTION_FUNCTIONS[base_impl])
    cfgs = [c for c in [cfg, getattr(cfg, "text_config", None)] if c is not None]
    saved = [c._attn_implementation for c in cfgs]
    for c in cfgs:
        c._attn_implementation = name
    try:
        yield
    finally:
        for c, s in zip(cfgs, saved):
            c._attn_implementation = s
        ALL_ATTENTION_FUNCTIONS.pop(name, None)
        ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.pop(name, None)
