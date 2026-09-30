"""Blockwise sequential quantization (T-0508; docs/原始设想 "BlockwiseOpt / run_block_loop").

Unlike the per-layer engine (which calibrates every layer on the *FP* model's activations), this
engine walks the decoder blocks in order and feeds block i with the outputs of the *already
quantized* blocks 0..i-1 — the GPTQModel / LLMC convention. Errors made early are therefore seen
(and partly compensated) by later layers.

Inside a block the layers are processed in dependency groups (attention inputs, o_proj, MLP inputs,
down_proj). After each group is quantized the block is re-run so the next group is calibrated on the
activations it will actually receive.

State machine per block:  capture(group) -> algorithm(layer) -> replace(FakeQuantLinear) -> next group
                          -> forward the block once more to produce inputs for the next block.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from mqde import core
from mqde.quant import algorithms
from mqde.quant.engine import ActivationCache, LayerTrace, RecipeSpec, compare
from mqde.quant.modules import FakeQuantLinear, replace_module, static_act_params
from mqde.quant.numerics import QuantSpec, fake_quant, fake_quant_act, qdq

# Order in which linears inside a decoder block depend on each other.
GROUPS = [
    ("attn_in", re.compile(r"(q_proj|k_proj|v_proj|qkv_proj|in_proj)$")),
    ("attn_out", re.compile(r"(o_proj|out_proj|dense)$")),
    ("mlp_in", re.compile(r"(gate_proj|up_proj|gate_up_proj|fc1|w1|w3)$")),
    ("mlp_out", re.compile(r"(down_proj|fc2|w2)$")),
]

# Where a group's shared input scale s is folded (T-0603): the op producing that input is divided by s
# (norm weight, or the output rows of a linear), the consumers' weights are multiplied by s. FP-exact.
FOLD_PREV = {"attn_in": "input_layernorm", "attn_out": "v_proj", "mlp_in": "post_attention_layernorm", "mlp_out": "up_proj"}


@dataclass
class LayerPolicy:
    """Per-layer override (mixed precision, T-0608). bits=None keeps the recipe; skip keeps FP."""

    bits: int | None = None
    skip: bool = False


@dataclass
class MixedPrecisionPolicy:
    """Rules are (regex on module path, LayerPolicy); first match wins. Block rules use 'block:<i>' or
    'block:<lo>-<hi>'; negative indices count from the end ('block:-1' = last block) once n_blocks is set."""

    rules: list[tuple[str, LayerPolicy]] = field(default_factory=list)
    n_blocks: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> MixedPrecisionPolicy:
        rules = []
        for r in (d or {}).get("rules", []):
            rules.append((r["match"], LayerPolicy(bits=r.get("bits"), skip=bool(r.get("skip", False)))))
        return cls(rules)

    def lookup(self, name: str, block: int) -> LayerPolicy:
        for pat, pol in self.rules:
            if pat.startswith("block:"):
                m = re.fullmatch(r"(-?\d+)(?:-(-?\d+))?", pat[6:])
                if not m:
                    raise ValueError(f"bad block rule '{pat}'")
                lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
                if self.n_blocks is not None:
                    lo, hi = (lo + self.n_blocks if lo < 0 else lo), (hi + self.n_blocks if hi < 0 else hi)
                if lo <= block <= hi:
                    return pol
            elif re.search(pat, name):
                return pol
        return LayerPolicy()

    @staticmethod
    def apply_bits(spec: QuantSpec, bits: int | None) -> QuantSpec:
        """Change bit width; qmin/qmax are properties of QuantSpec, so they follow automatically."""
        if bits is None or not spec.enabled or spec.format != "int":
            return spec
        return QuantSpec(**{**spec.__dict__, "bits": int(bits)})


@dataclass
class BlockwiseResult:
    traces: list[LayerTrace]
    seconds: float
    skipped: list[str]


class BlockwiseQuantizer:
    def __init__(
        self,
        adapter: Any,  # HFCausalLM-like: get_blocks(), block_prefix(i), first_block_inputs(), run_block()
        recipe: RecipeSpec,
        algorithm: str = "rtn",
        policy: MixedPrecisionPolicy | None = None,
        max_rows: int = 2048,
        act_static: str | None = None,  # None = dynamic per-token; minmax | ema | percentile = static per-tensor
        progress: Callable[[dict[str, Any]], None] | None = None,
        fold_scales: bool = True,  # AWQ/SmoothQuant: one scale per input group, folded into the previous op
    ) -> None:
        self.ad = adapter
        self.recipe = recipe
        self.algorithm = algorithm
        algorithms.resolve(algorithm)
        self.policy = policy or MixedPrecisionPolicy()
        self.max_rows = max_rows
        self.act_static = act_static
        self.progress = progress or (lambda e: None)
        self.originals: dict[str, nn.Linear] = {}
        steps = algorithms.resolve(algorithm)
        self.transform = next((t for t in steps if t in ("awq", "smoothquant")), None)
        self.fold = fold_scales and self.transform is not None
        # with folding the per-layer algorithm is just the quantizer step; the transform is applied per group
        self.layer_algorithm = "+".join(t for t in steps if t != self.transform) if self.fold else algorithm
        self._saved: list[tuple[torch.Tensor, torch.Tensor]] = []  # (tensor modified in place, original copy)
        self._fold_info: dict[str, dict[str, Any]] = {}

    def _linears(self, block: nn.Module) -> dict[str, nn.Linear]:
        return {n: m for n, m in block.named_modules() if isinstance(m, nn.Linear)}

    def _groups(self, block: nn.Module) -> list[list[str]]:
        return [g for _, g in self._keyed_groups(block)]

    def _keyed_groups(self, block: nn.Module) -> list[tuple[str, list[str]]]:
        lin = self._linears(block)
        out, seen = [], set()
        for key, pat in GROUPS:
            g = [n for n in lin if pat.search(n) and n not in seen]
            seen.update(g)
            if g:
                out.append((key, g))
        rest = [n for n in lin if n not in seen]
        if rest:
            out.append(("other", rest))
        return out

    def _scale_(self, t: torch.Tensor, s: torch.Tensor, dim: int, divide: bool) -> None:
        self._saved.append((t, t.detach().clone()))
        shape = [1] * t.dim()
        shape[dim] = -1
        f = s.to(t.device, torch.float32).reshape(shape)
        t.data.copy_((t.data.to(torch.float32) / f if divide else t.data.to(torch.float32) * f).to(t.dtype))

    @torch.inference_mode()
    def _fold_block(self, block: nn.Module, bi: int, prefix: str, hs: list[torch.Tensor], kws: list[dict[str, Any]]) -> None:
        """Search one smoothing scale per input group on the FP block and fold it (llm-awq's apply_scale)."""
        groups = [(k, g) for k, g in self._keyed_groups(block) if k in FOLD_PREV]
        cache = self._capture(block, [g[0] for _, g in groups], hs, kws)
        mods = dict(block.named_modules())
        fn, params = algorithms.ALGO_REGISTRY[self.transform], algorithms.algorithm_params(self.transform)
        for key, names in groups:
            prev_name = next((n for n in mods if n.endswith(FOLD_PREV[key])), None)
            prev = mods.get(prev_name) if prev_name else None
            consumers = [mods[n] for n in names]
            cin = consumers[0].in_features
            if isinstance(prev, nn.Linear):
                ok = prev.out_features == cin  # e.g. v_proj -> o_proj breaks under GQA (kv heads < heads)
            else:
                w = getattr(prev, "weight", None)
                ok = prev is not None and w is not None and w.dim() == 1 and w.numel() == cin and "gemma" not in type(prev).__name__.lower()
            x = cache.get(names[0])
            note = {"transform": self.transform, "group": key}
            if not ok or x is None:
                note["folded"] = f"no ({prev_name or 'no producer'} not foldable)"
            else:
                ws = MixedPrecisionPolicy.apply_bits(self.recipe.weight, self.policy.lookup(f"{prefix}.{names[0]}", bi).bits)
                w_cat = torch.cat([m.weight.detach().to(torch.float32) for m in consumers], dim=0)
                r = fn(w_cat, x.to(w_cat.device), ws, self.recipe.act, params)
                if r.in_scale is None:
                    note["folded"] = "no (identity scale was best)"
                else:
                    sc = r.in_scale
                    for m in consumers:
                        self._scale_(m.weight, sc, dim=1, divide=False)
                    self._scale_(prev.weight, sc, dim=0, divide=True)  # norm weight, or a linear's output rows
                    if getattr(prev, "bias", None) is not None:
                        self._scale_(prev.bias, sc, dim=0, divide=True)
                    note.update({"folded": f"into {prev_name}", "s_min": float(sc.min()), "s_max": float(sc.max()), **(r.info or {})})
            for n in names:
                self._fold_info[f"{prefix}.{n}"] = note
        del cache

    @torch.inference_mode()
    def _capture(self, block: nn.Module, names: list[str], hs: list[torch.Tensor], kws: list[dict[str, Any]]) -> ActivationCache:
        cache = ActivationCache(max_rows=self.max_rows)
        handles = [block.get_submodule(n).register_forward_hook(lambda m, i, o, n=n: cache.add(n, i[0])) for n in names]
        try:
            for h, kw in zip(hs, kws):
                self.ad.run_block(block, h, kw)
        finally:
            for hd in handles:
                hd.remove()
        return cache

    @torch.inference_mode()
    def _quantize_layer(self, block: nn.Module, bi: int, prefix: str, name: str, x: torch.Tensor | None) -> LayerTrace | None:
        full = f"{prefix}.{name}"
        lin = block.get_submodule(name)
        pol = self.policy.lookup(full, bi)
        if pol.skip:
            return None
        ws = MixedPrecisionPolicy.apply_bits(self.recipe.weight, pol.bits)
        as_ = self.recipe.act
        if ws.granularity == "per_group" and lin.in_features % ws.group_size != 0:
            return None
        t0 = time.perf_counter()
        w = lin.weight.detach().to(torch.float32)
        xc = x.to(w.device) if x is not None else None
        ar = algorithms.run(self.layer_algorithm, w, xc, ws, as_)
        wq, w_scale, w_zero = algorithms.materialize(ar, ws)
        eff = wq / ar.in_scale if ar.in_scale is not None else wq
        wm = compare(w, eff)
        tr = LayerTrace(
            name=full, component="language", modality="language", block=bi, shape=(lin.out_features, lin.in_features),
            w_sqnr_db=wm["sqnr_db"] if wm["sqnr_db"] is not None else float("inf"),
            w_rel_l2=wm["rel_l2"] or 0.0, w_cosine=wm["cosine"] or 1.0,
            w_kurtosis=core.tensor_stats(w.cpu().numpy())["kurtosis"],
        )
        tr.algorithm, tr.algo_info = self.algorithm, {**(ar.info or {}), "bits": ws.bits}
        if full in self._fold_info:
            tr.algo_info["smooth"] = self._fold_info[full]
        act_p = None
        if xc is not None and as_.enabled and self.act_static:
            xs0 = xc / ar.in_scale if ar.in_scale is not None else xc
            act_p = static_act_params(xs0, as_.with_granularity("per_tensor"), self.act_static)
        if xc is not None:
            xs = xc / ar.in_scale if ar.in_scale is not None else xc
            if act_p is not None:
                xq = qdq(xs, act_p[0], act_p[1], as_)
            else:
                xq = fake_quant_act(xs, as_) if as_.enabled else xs
            om = compare(xc @ w.T, xq @ wq.T)
            tr.out_sqnr_db, tr.out_rel_l2, tr.out_cosine = om["sqnr_db"], om["rel_l2"], om["cosine"]
            st = core.tensor_stats(xc.cpu().numpy())
            tr.act_absmax, tr.act_kurtosis, tr.act_outlier_ratio = st["absmax"], st["kurtosis"], st["outlier_ratio"]
        fq = FakeQuantLinear(lin, ws, as_, full, w_override=wq, in_scale=ar.in_scale, final=True,
                             w_qparams=(w_scale, w_zero))
        fq.algo = self.algorithm
        if act_p is not None:
            fq.act_scale, fq.act_zero = act_p[0].to(w.device), act_p[1].to(w.device)
            tr.algo_info["act"] = f"static-{self.act_static}"
        self.originals[full] = lin
        replace_module(block, name, fq)
        tr.seconds = time.perf_counter() - t0
        return tr

    @torch.inference_mode()
    def run(self, calib: list[torch.Tensor]) -> BlockwiseResult:
        t0 = time.perf_counter()
        hs, kws = self.ad.first_block_inputs(calib)
        traces: list[LayerTrace] = []
        skipped: list[str] = []
        blocks = self.ad.get_blocks()
        n_lin = sum(len(self._linears(b)) for b in blocks)
        self.policy.n_blocks = len(blocks)
        for bi, block in enumerate(blocks):
            prefix = self.ad.block_prefix(bi)
            if self.fold:
                self._fold_block(block, bi, prefix, hs, kws)
            for group in self._groups(block):
                needs_x = self.layer_algorithm != "rtn" or self.recipe.act.enabled
                cache = self._capture(block, group, hs, kws) if needs_x else None
                for name in group:
                    tr = self._quantize_layer(block, bi, prefix, name, cache.get(name) if cache else None)
                    if tr is None:
                        skipped.append(f"{prefix}.{name}")
                    else:
                        traces.append(tr)
                        self.progress({"kind": "layer", "index": len(traces) - 1, "total": n_lin, "trace": tr.to_dict()})
                del cache
            # inputs for the next block come from this (now quantized) block
            hs = [self.ad.run_block(block, h, kw) for h, kw in zip(hs, kws)]
            self.progress({"kind": "block", "index": bi, "total": len(blocks)})
        return BlockwiseResult(traces=traces, seconds=time.perf_counter() - t0, skipped=skipped)

    def restore(self) -> None:
        root = self.ad.model
        for full, lin in self.originals.items():
            replace_module(root, full, lin)
        self.originals.clear()
        for t, orig in reversed(self._saved):  # undo scale folding (norms / producer rows / consumer columns)
            t.data.copy_(orig)
        self._saved.clear()
        self._fold_info.clear()
