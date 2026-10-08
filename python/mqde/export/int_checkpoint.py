"""GPTQ (v1, GPTQModel/AutoGPTQ) and AWQ (llm-awq GEMM) checkpoints for vLLM / SGLang / transformers.

Codes are recovered from the fake-quantized weight with the layer's own qparams, so the checkpoint holds
exactly the integers the fake-quant evaluation used; only the scales are narrowed to fp16 for storage
(``real_check`` reports the effect). Symmetric specs are stored the usual way: codes + 2^(b-1), zero 2^(b-1).
Per-layer bit overrides and FP-kept layers (mixed precision) go into GPTQModel's ``dynamic`` config, which
vLLM's gptq / gptq_marlin loaders understand; AWQ lists FP layers in ``modules_to_not_convert``.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from mqde import core
from mqde.export.base import Exporter, ExportResult, fake_linears, register
from mqde.quant.modules import FakeQuantLinear

_COPY = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
         "generation_config.json", "chat_template.jinja", "tokenizer.model", "added_tokens.json")


def layer_codes(m: FakeQuantLinear) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """(codes [out, in] u8, zeros [out, groups] u8, scales [out, groups] f32, group_size) in the unsigned
    convention w = scale * (code - zero)."""
    s = m.w_spec
    if m.w_scale is None or m.w_zero is None:
        raise ValueError(f"{m.name}: no weight qparams (see Exporter.check)")
    out, inn = m.out_features, m.in_features
    gs = s.group_size if s.granularity == "per_group" else inn
    groups = inn // gs
    scale = m.w_scale.detach().to(torch.float32).reshape(out, groups)
    zero = m.w_zero.detach().to(torch.float32).reshape(out, groups)
    w = m.weight.detach().to(torch.float32).reshape(out, groups, gs)
    q = torch.round(w / scale[..., None]) + zero[..., None]
    q = q.clamp(s.qmin, s.qmax).reshape(out, inn)
    off = (1 << (s.bits - 1)) if s.symmetric else 0
    codes = (q + off).to(torch.int64)
    zeros = (zero + off).to(torch.int64)
    return codes.cpu().numpy().astype(np.uint8), zeros.cpu().numpy().astype(np.uint8), scale.cpu().numpy(), gs


class _IntExporter(Exporter):
    layout = ""
    bits_ok: tuple[int, ...] = ()

    def __init__(self, dtype: torch.dtype = torch.float16) -> None:
        self.dtype = dtype

    def check_layer(self, name: str, m: FakeQuantLinear) -> str | None:
        s = m.w_spec
        if s.format != "int":
            return f"weight format {s.format} is not an integer format"
        if s.bits not in self.bits_ok:
            return f"{s.bits}-bit weights are not supported by {self.name} (supported: {self.bits_ok})"
        if s.granularity == "per_tensor":
            return "per-tensor weight scales are not representable"
        if m.w_scale is None or m.w_zero is None:
            return "layer has no weight qparams (quantized before qparams were tracked)"
        if m.in_scale is not None:
            return "activation smoothing scale (AWQ/SmoothQuant transform) must be folded into the previous op first"
        gs = s.group_size if s.granularity == "per_group" else m.in_features
        if (m.out_features * s.bits) % 32 or (self.layout == "gptq" and (m.in_features * s.bits) % 32):
            return f"shape [{m.out_features}, {m.in_features}] does not pack into 32-bit words at {s.bits} bit"
        if self.layout == "awq" and m.in_features % gs:
            return "in_features not divisible by group_size"
        return None

    # ---------------------------------------------------------------- config
    def _config(self, fq: dict[str, FakeQuantLinear], kept: list[str]) -> dict[str, Any]:
        specs = {n: m.w_spec for n, m in fq.items()}
        key = {n: (s.bits, s.group_size if s.granularity == "per_group" else -1, s.symmetric) for n, s in specs.items()}
        base = Counter(key.values()).most_common(1)[0][0]  # the recipe's spec; overrides go into `dynamic`
        bits, gs, sym = base
        if self.layout == "awq":
            return {"quant_method": "awq", "bits": bits, "group_size": gs, "zero_point": True, "version": "gemm",
                    "modules_to_not_convert": sorted({n.split(".")[-1] for n in kept}) or None}
        dynamic: dict[str, dict[str, Any]] = {}
        for n, k in key.items():
            if k != base:
                dynamic[f"+:^{re.escape(n)}$"] = {"bits": k[0], "group_size": k[1], "sym": k[2]}
        for n in kept:
            dynamic[f"-:^{re.escape(n)}$"] = {}
        cfg = {"quant_method": "gptq", "bits": bits, "group_size": gs, "desc_act": False, "sym": sym,
               "checkpoint_format": "gptq", "true_sequential": True, "static_groups": False, "pack_dtype": "int32"}
        if dynamic:
            cfg["dynamic"] = dynamic
        return cfg

    # ---------------------------------------------------------------- export
    @torch.inference_mode()
    def export(self, model: nn.Module, out_dir: str | Path, source_dir: str | Path | None = None) -> ExportResult:
        from safetensors.torch import save_file

        problems = self.check(model)
        if problems:
            head = "; ".join(f"{n}: {r}" for n, r in problems[:5])
            raise ValueError(f"{self.name}: {len(problems)} layer(s) cannot be exported ({head})")
        fq = fake_linears(model)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        sd: dict[str, torch.Tensor] = {}
        for n, m in fq.items():
            codes, zeros, scales, gs = layer_codes(m)
            v1 = self.layout == "gptq"
            qw, qz, sc = core.pack_int(self.layout, codes, zeros, scales, m.w_spec.bits, v1=v1)
            sd[f"{n}.qweight"] = torch.from_numpy(qw)
            sd[f"{n}.qzeros"] = torch.from_numpy(qz)
            sd[f"{n}.scales"] = torch.from_numpy(sc)
            if self.layout == "gptq":
                sd[f"{n}.g_idx"] = torch.arange(m.in_features, dtype=torch.int32) // gs
            if m.bias is not None:
                sd[f"{n}.bias"] = m.bias.detach().to(self.dtype).cpu()
        prefixes = tuple(f"{n}." for n in fq)
        tied = bool(getattr(cast(Any, model).config, "tie_word_embeddings", False))
        kept = [n for n, mod in model.named_modules() if isinstance(mod, nn.Linear) and not (tied and n == "lm_head")]
        for k, v in model.state_dict().items():
            if k.startswith(prefixes) or (tied and k == "lm_head.weight"):
                continue
            sd[k] = (v.to(self.dtype) if v.is_floating_point() else v).detach().cpu().contiguous()
        save_file(sd, str(out / "model.safetensors"), metadata={"format": "pt", "producer": "mqde"})
        # FP linears other than lm_head (mixed precision "skip" rules) must be excluded from quantization
        kept_q = [n for n in kept if n != "lm_head" and re.search(r"layers\.\d+\.", n)]
        qcfg = self._config(fq, kept_q)
        cfg = dict(cast(Any, model.config).to_dict())
        cfg["torch_dtype"] = str(self.dtype).replace("torch.", "")
        cfg["quantization_config"] = qcfg
        cfg.pop("_attn_implementation_autoset", None)
        (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
        (out / "quantize_config.json").write_text(json.dumps({k: v for k, v in qcfg.items() if k != "quant_method"}, indent=2))
        if source_dir:
            for f in _COPY:
                if (Path(source_dir) / f).is_file():
                    shutil.copy2(Path(source_dir) / f, out / f)
        files = sorted(p.name for p in out.iterdir())
        notes = []
        acts = {m.a_spec.format for m in fq.values() if m.a_spec.enabled}
        if acts:
            notes.append("activation quantization is not part of GPTQ/AWQ checkpoints; the backend runs activations in "
                         f"{self.dtype} (fake-quant evaluated {sorted(acts)} activations)")
        return ExportResult(format=self.name, out_dir=str(out), files=files, quantized=sorted(fq), kept_fp=kept_q,
                            quantization_config=qcfg, bytes=sum(p.stat().st_size for p in out.iterdir()), notes=notes)


@register("gptq")
class GPTQExporter(_IntExporter):
    """GPTQ v1 checkpoint (qweight / qzeros / scales / g_idx, desc_act=False)."""

    layout = "gptq"
    bits_ok = (2, 3, 4, 8)
    backends = ("vllm", "sglang", "transformers")


@register("awq")
class AWQExporter(_IntExporter):
    """AWQ GEMM checkpoint (zero_point=True)."""

    layout = "awq"
    bits_ok = (4,)
    backends = ("vllm", "sglang", "transformers")

    def check_layer(self, name: str, m: FakeQuantLinear) -> str | None:
        r = super().check_layer(name, m)
        if r is None and m.w_spec.granularity != "per_group":
            return "AWQ checkpoints need per-group scales"
        return r
