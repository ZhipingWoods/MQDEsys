"""Logits probe (T-0510): how much a quantized model's next-token *distribution* moved, which perplexity cannot show.

Perplexity looks only at the probability of the token that actually came next; two models can share a perplexity and
still disagree on what to say. This module compares full output distributions position by position:

* ``kl``: mean KL(reference || quantized) in nats;
* ``top1_agreement``: share of positions where both models pick the same token;
* ``ref_top1_prob_drop``: mean decrease of the probability given to the reference's top-1 token.

The reference model is gone once layers are replaced in place, so the reference is captured **before** quantization as a
``Reference`` and the quantized model is compared against it afterwards. Two reference forms:

* ``exact=True``: all logits (float16). Exact KL, memory = positions x vocab x 2 bytes (about 5.8 GiB for 40 x 512
  positions and a 152k vocabulary);
* ``exact=False`` (default): the top-``k`` logits per position plus the log-sum-exp over the whole vocabulary. Top-k token
  probabilities are exact; the probability mass outside the reference's top-k is carried as a single "rest" bucket, so KL
  is a lower bound of the true KL (merging outcomes can only lower a KL divergence). ``probe_error`` measures the gap.

    ref = capture_reference(ad.logits, windows)        # before quantization
    ...quantize...
    m = compare_to_reference(ad.logits, windows, ref)  # {"kl", "top1_agreement", "ref_top1_prob_drop", ...}
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

LogitsFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class Reference:
    """Reference distributions of every position of every window, captured once."""

    seq_len: int
    n_windows: int
    exact: bool
    k: int
    logits: torch.Tensor | None = None  # [n_windows, seq_len, vocab] float16, only when exact
    topk_val: torch.Tensor | None = None  # [n_windows, seq_len, k] float32 logits
    topk_idx: torch.Tensor | None = None  # [n_windows, seq_len, k] int64
    lse: torch.Tensor | None = None  # [n_windows, seq_len] float32 log-sum-exp over the full vocabulary

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.logits, self.topk_val, self.topk_idx, self.lse) if t is not None)


def split_windows(ids: torch.Tensor, seq_len: int, max_windows: int | None = None) -> torch.Tensor:
    """Non-overlapping windows [n, seq_len] (same protocol as eval.ppl.perplexity)."""
    n = ids.numel() // seq_len
    if max_windows:
        n = min(n, max_windows)
    if n == 0:
        raise ValueError(f"need at least {seq_len} token ids, got {ids.numel()}")
    return ids[: n * seq_len].reshape(n, seq_len)


@torch.inference_mode()
def capture_reference(logits_fn: LogitsFn, windows: torch.Tensor, k: int = 256, exact: bool = False) -> Reference:
    n, s = windows.shape
    ref = Reference(seq_len=s, n_windows=n, exact=exact, k=k)
    full, vals, idxs, lses = [], [], [], []
    for i in range(n):
        lg = logits_fn(windows[i : i + 1]).float()[0]  # [seq, vocab]
        if exact:
            full.append(lg.to(torch.float16).cpu())
        v, ix = lg.topk(min(k, lg.shape[-1]), dim=-1)
        vals.append(v.cpu())
        idxs.append(ix.cpu())
        lses.append(torch.logsumexp(lg, dim=-1).cpu())
    ref.topk_val, ref.topk_idx, ref.lse = torch.stack(vals), torch.stack(idxs), torch.stack(lses)
    if exact:
        ref.logits = torch.stack(full)
    return ref


def _kl_exact(ref_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    lp = F.log_softmax(ref_logits.float(), dim=-1)
    lq = F.log_softmax(q_logits.float(), dim=-1)
    return (lp.exp() * (lp - lq)).sum(dim=-1)  # [seq]


def _kl_topk(ref_val: torch.Tensor, ref_idx: torch.Tensor, ref_lse: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL over the reference's top-k tokens plus one bucket for everything else (a lower bound of the true KL)."""
    lq = F.log_softmax(q_logits.float(), dim=-1)  # [seq, vocab]
    lp_top = ref_val - ref_lse[:, None]  # exact log p of the top-k tokens
    lq_top = lq.gather(-1, ref_idx.to(lq.device))
    p_top = lp_top.exp()
    p_rest = (1.0 - p_top.sum(-1)).clamp_min(0.0)
    q_rest = (1.0 - lq_top.exp().sum(-1)).clamp_min(1e-30)
    kl = (p_top * (lp_top - lq_top)).sum(-1)
    kl = kl + torch.where(p_rest > 0, p_rest * (p_rest.clamp_min(1e-30).log() - q_rest.log()), torch.zeros_like(p_rest))
    return kl


@torch.inference_mode()
def compare_to_reference(logits_fn: LogitsFn, windows: torch.Tensor, ref: Reference,
                         progress: Callable[[int, int], None] | None = None) -> dict[str, float]:
    if tuple(windows.shape) != (ref.n_windows, ref.seq_len):
        raise ValueError(f"windows {tuple(windows.shape)} differ from the reference ({ref.n_windows}, {ref.seq_len})")
    assert ref.topk_val is not None and ref.topk_idx is not None and ref.lse is not None
    kl_sum = agree = drop = 0.0
    n_pos = 0
    for i in range(ref.n_windows):
        q = logits_fn(windows[i : i + 1]).float()[0]
        if ref.exact:
            assert ref.logits is not None
            kl = _kl_exact(ref.logits[i].to(q.device), q)
        else:
            kl = _kl_topk(ref.topk_val[i].to(q.device), ref.topk_idx[i], ref.lse[i].to(q.device), q)
        ref_top1 = ref.topk_idx[i][:, 0].to(q.device)
        agree += float((q.argmax(-1) == ref_top1).sum())
        p_ref_top1 = (ref.topk_val[i][:, 0] - ref.lse[i]).exp().to(q.device)
        p_q_top1 = F.softmax(q, dim=-1).gather(-1, ref_top1[:, None])[:, 0]
        drop += float((p_ref_top1 - p_q_top1).sum())
        kl_sum += float(kl.sum())
        n_pos += q.shape[0]
        if progress:
            progress(i + 1, ref.n_windows)
    return {"kl": kl_sum / n_pos, "top1_agreement": agree / n_pos, "ref_top1_prob_drop": drop / n_pos, "positions": n_pos,
            "windows": ref.n_windows, "seq_len": ref.seq_len, "exact": ref.exact, "k": ref.k, "reference_mib": ref.nbytes() / 2**20}


# ---------------------------------------------------------------------------------------------------------------
# Per-sample probe (T-1002): inputs that are not a flat window of token ids -- an image plus a question, whose length
# differs from sample to sample. Same measures and the same _kl_exact / _kl_topk as above; the model is a function of the
# sample *index* returning [1, len_i, vocab], and a boolean mask per sample picks the positions that count (the answer
# tokens, not the image / question positions the model is only conditioned on).

SampleLogitsFn = Callable[[int], torch.Tensor]


@dataclass
class SampleReference:
    exact: bool
    k: int
    masks: list[torch.Tensor]  # [len_i] bool, on cpu
    logits: list[torch.Tensor] | None = None  # exact: [n_i, vocab] float16, only the masked positions
    topk_val: list[torch.Tensor] | None = None  # [n_i, k] float32
    topk_idx: list[torch.Tensor] | None = None  # [n_i, k] int64
    lse: list[torch.Tensor] | None = None  # [n_i] float32

    @property
    def n_samples(self) -> int:
        return len(self.masks)

    def nbytes(self) -> int:
        parts = [*(self.logits or []), *(self.topk_val or []), *(self.topk_idx or []), *(self.lse or [])]
        return sum(t.numel() * t.element_size() for t in parts)


@torch.inference_mode()
def capture_sample_reference(fn: SampleLogitsFn, n_samples: int, masks: list[torch.Tensor], k: int = 256,
                             exact: bool = False) -> SampleReference:
    if len(masks) != n_samples:
        raise ValueError(f"{len(masks)} masks for {n_samples} samples")
    ref = SampleReference(exact=exact, k=k, masks=[m.cpu() for m in masks], logits=[] if exact else None, topk_val=[], topk_idx=[], lse=[])
    assert ref.topk_val is not None and ref.topk_idx is not None and ref.lse is not None
    for i in range(n_samples):
        if not bool(masks[i].any()):
            raise ValueError(f"sample {i} has no positions to compare (empty mask)")
        lg = fn(i).float()[0]
        if lg.shape[0] != masks[i].shape[0]:
            raise ValueError(f"sample {i}: the model returned {lg.shape[0]} positions, the mask has {masks[i].shape[0]}")
        lg = lg[masks[i].to(lg.device)]
        if exact:
            assert ref.logits is not None
            ref.logits.append(lg.to(torch.float16).cpu())
        v, ix = lg.topk(min(k, lg.shape[-1]), dim=-1)
        ref.topk_val.append(v.cpu())
        ref.topk_idx.append(ix.cpu())
        ref.lse.append(torch.logsumexp(lg, dim=-1).cpu())
    return ref


@torch.inference_mode()
def compare_to_sample_reference(fn: SampleLogitsFn, ref: SampleReference,
                                progress: Callable[[int, int], None] | None = None) -> dict[str, float]:
    assert ref.topk_val is not None and ref.topk_idx is not None and ref.lse is not None
    kl_sum = agree = drop = 0.0
    n_pos = 0
    for i in range(ref.n_samples):
        full = fn(i).float()[0]
        if full.shape[0] != ref.masks[i].shape[0]:
            raise ValueError(f"sample {i}: {full.shape[0]} positions, the reference has {ref.masks[i].shape[0]}")
        q = full[ref.masks[i].to(full.device)]
        if ref.exact:
            assert ref.logits is not None
            kl = _kl_exact(ref.logits[i].to(q.device), q)
        else:
            kl = _kl_topk(ref.topk_val[i].to(q.device), ref.topk_idx[i], ref.lse[i].to(q.device), q)
        ref_top1 = ref.topk_idx[i][:, 0].to(q.device)
        agree += float((q.argmax(-1) == ref_top1).sum())
        p_ref_top1 = (ref.topk_val[i][:, 0] - ref.lse[i]).exp().to(q.device)
        drop += float((p_ref_top1 - F.softmax(q, dim=-1).gather(-1, ref_top1[:, None])[:, 0]).sum())
        kl_sum += float(kl.sum())
        n_pos += q.shape[0]
        if progress:
            progress(i + 1, ref.n_samples)
    return {"kl": kl_sum / n_pos, "top1_agreement": agree / n_pos, "ref_top1_prob_drop": drop / n_pos, "positions": n_pos,
            "samples": ref.n_samples, "exact": ref.exact, "k": ref.k, "reference_mib": ref.nbytes() / 2**20}


@torch.inference_mode()
def probe_error(logits_fn_ref: LogitsFn, logits_fn_q: LogitsFn, windows: torch.Tensor, k: int = 256) -> dict[str, float]:
    """Gap between the top-k KL and the exact KL for two *live* models (tests / validation; needs both at once)."""
    exact = topk = 0.0
    n_pos = 0
    for i in range(windows.shape[0]):
        r = logits_fn_ref(windows[i : i + 1]).float()[0]
        q = logits_fn_q(windows[i : i + 1]).float()[0]
        v, ix = r.topk(min(k, r.shape[-1]), dim=-1)
        exact += float(_kl_exact(r, q).sum())
        topk += float(_kl_topk(v, ix, torch.logsumexp(r, -1), q).sum())
        n_pos += r.shape[0]
    return {"exact_kl": exact / n_pos, "topk_kl": topk / n_pos, "gap": (exact - topk) / n_pos,
            "relative_gap": (exact - topk) / exact if exact else 0.0}
