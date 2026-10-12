"""Blockwise engine / KV / mixed precision / static activation on a tiny random Llama (offline)."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")

from mqde.eval.ppl import perplexity
from mqde.models.hf_llm import HFCausalLM
from mqde.quant import kv as kvq
from mqde.quant.blockwise import BlockwiseQuantizer, MixedPrecisionPolicy
from mqde.quant.engine import RecipeSpec
from mqde.quant.modules import FakeQuantLinear, static_act_params
from mqde.quant.numerics import QuantSpec, fake_quant, params_from_range


def _tiny() -> HFCausalLM:
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=128, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    cfg._attn_implementation = "eager"
    return HFCausalLM.from_model(transformers.LlamaForCausalLM(cfg).float())


def _calib(n: int = 4, s: int = 32) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(1)
    return [torch.randint(0, 128, (1, s), generator=g) for _ in range(n)]


def _fq_layers(model: nn.Module) -> dict[str, FakeQuantLinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, FakeQuantLinear)}


def test_blockwise_rtn_matches_per_layer_fake_quant() -> None:
    ad = _tiny()
    orig = {n: m.weight.detach().clone() for n, m in ad.model.named_modules() if isinstance(m, nn.Linear) and "layers." in n}
    r = RecipeSpec.load("w4a16")
    res = BlockwiseQuantizer(ad, r, "rtn").run(_calib())
    fq = _fq_layers(ad.model)
    assert len(fq) == len(orig) == 14 and len(res.traces) == 14 and not res.skipped
    for n, m in fq.items():
        assert torch.equal(m.weight, fake_quant(orig[n], r.weight)[0]), n


def test_restore_and_requantize_are_idempotent() -> None:
    ad = _tiny()
    ids = _calib(1, 48)[0]
    ref = ad.logits(ids).clone()
    r = RecipeSpec.load("w4a8")
    q = BlockwiseQuantizer(ad, r, "gptq")
    q.run(_calib())
    w1 = {n: m.weight.clone() for n, m in _fq_layers(ad.model).items()}
    q_logits = ad.logits(ids).clone()
    assert not torch.equal(q_logits, ref)
    q.restore()
    assert not _fq_layers(ad.model) and torch.equal(ad.logits(ids), ref)
    q2 = BlockwiseQuantizer(ad, r, "gptq")
    q2.run(_calib())
    w2 = {n: m.weight for n, m in _fq_layers(ad.model).items()}
    assert w1.keys() == w2.keys() and all(torch.equal(w1[k], w2[k]) for k in w1)
    assert torch.equal(ad.logits(ids), q_logits)
    q2.restore()


def test_to_moves_the_model_and_restore_stays_consistent_after_moving_back() -> None:
    """T-1304/T-1306 quant_device/eval_device split: llm_job.py quantizes on one device, moves the model
    to another for eval, then moves back before calling restore() -- the detached originals restore()
    swaps back in were never touched by either move (see HFCausalLM.to()'s docstring), so this only works
    if the live model is back on the quant device by the time restore() runs. Runs the real cross-device
    round trip when CUDA is available; degrades to a same-device no-op otherwise, which still exercises
    the sequencing (just not a real device change)."""
    ad = _tiny()
    assert ad.device == torch.device("cpu")
    ids = _calib(1, 48)[0]
    ref = ad.logits(ids).clone()
    r = RecipeSpec.load("w4a16")
    q = BlockwiseQuantizer(ad, r, "rtn")
    q.run(_calib())  # quantize on the "quant device" (cpu)
    target = "cuda:0" if torch.cuda.is_available() else "cpu"
    ad.to(target)
    assert ad.device == torch.device(target) and next(ad.model.parameters()).device.type == torch.device(target).type
    q_logits = ad.logits(ids).clone()  # eval on the "eval device"; .logits() moves ids there itself
    assert not torch.equal(q_logits.cpu(), ref)
    ad.to("cpu")  # back to the quant device before restore(), same order llm_job.py uses
    assert next(ad.model.parameters()).device.type == "cpu"
    q.restore()
    assert not _fq_layers(ad.model) and torch.equal(ad.logits(ids), ref)


def test_gptq_blockwise_beats_rtn_on_output() -> None:
    errs = {}
    for algo in ("rtn", "gptq"):
        ad = _tiny()
        ids = _calib(1, 48)[0]
        ref = ad.logits(ids)
        BlockwiseQuantizer(ad, RecipeSpec.load("w2a16"), algo).run(_calib(8))
        errs[algo] = float((ad.logits(ids) - ref).norm() / ref.norm())
    assert errs["gptq"] < errs["rtn"], errs


def test_mixed_precision_bits_and_skip() -> None:
    ad = _tiny()
    pol = MixedPrecisionPolicy.from_dict({"rules": [
        {"match": r"down_proj$", "skip": True},
        {"match": "block:0", "bits": 8},
    ]})
    res = BlockwiseQuantizer(ad, RecipeSpec.load("w4a16"), "rtn", policy=pol).run(_calib())
    fq = _fq_layers(ad.model)
    assert sorted(res.skipped) == ["model.layers.0.mlp.down_proj", "model.layers.1.mlp.down_proj"]
    assert isinstance(ad.model.get_submodule("model.layers.0.mlp.down_proj"), nn.Linear)
    for n, m in fq.items():
        bits = 8 if n.startswith("model.layers.0.") else 4
        assert m.w_spec.bits == bits, n
        assert (m.w_spec.qmin, m.w_spec.qmax) == ((0, 2**bits - 1) if not m.w_spec.symmetric else (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1))
        g = m.weight.reshape(-1, m.w_spec.group_size) if m.w_spec.group_size else m.weight
        assert max(len(torch.unique(row)) for row in g) <= 2**bits


def test_block_rules_negative_and_ranges() -> None:
    p = MixedPrecisionPolicy.from_dict({"rules": [{"match": "block:-1", "bits": 8}, {"match": "block:1-2", "bits": 3}]})
    p.n_blocks = 4
    assert [p.lookup("x", b).bits for b in range(4)] == [None, 3, 3, 8]


def test_static_activation_params_are_per_tensor_and_used() -> None:
    ad = _tiny()
    r = RecipeSpec.load("w4a8")
    BlockwiseQuantizer(ad, r, "rtn", act_static="minmax").run(_calib())
    for m in _fq_layers(ad.model).values():
        assert m.act_scale is not None and m.act_scale.numel() == 1
    x = torch.randn(256, 256) * 3
    spec = r.act.with_granularity("per_tensor")
    s, z = static_act_params(x, spec, "minmax")
    s2, z2 = params_from_range(x.min().reshape(1, 1), x.max().reshape(1, 1), spec)
    assert torch.equal(s, s2) and torch.equal(z, z2)
    x[0, 0] = 1e3  # one outlier: percentile ignores it, minmax does not
    sp, _ = static_act_params(x, spec, "percentile")
    sm, _ = static_act_params(x, spec, "minmax")
    assert float(sp) < 0.1 * float(sm)


def test_kv_quant_wrapper_hits_every_layer_and_restores() -> None:
    ad = _tiny()
    ids = _calib(1, 48)[0]
    ref = ad.logits(ids)
    calls = []

    class Identity(kvq.NaiveKVQuant):
        def quant_k(self, k: torch.Tensor) -> torch.Tensor:
            return k

        def quant_v(self, v: torch.Tensor) -> torch.Tensor:
            return v

    with kvq.kv_quant(ad.model, Identity(QuantSpec())):  # routing alone must not change the model (mask!)
        assert torch.allclose(ad.logits(ids), ref, atol=1e-5)

    class Spy(kvq.NaiveKVQuant):
        def quant_k(self, k: torch.Tensor) -> torch.Tensor:
            calls.append(k.shape)
            return super().quant_k(k)

    spec = QuantSpec.from_dict({"format": "int", "bits": 8, "symmetric": False})
    with kvq.kv_quant(ad.model, Spy(spec)):
        out8 = ad.logits(ids)
    assert len(calls) == 2 and calls[0] == (1, 2, 48, 64)
    assert ad.model.config._attn_implementation == "eager"
    assert torch.equal(ad.logits(ids), ref)
    rel8 = float((out8 - ref).norm() / ref.norm())
    with kvq.kv_quant(ad.model, kvq.from_spec(QuantSpec.from_dict({"format": "int", "bits": 2, "symmetric": False}), "kivi")):
        rel2 = float((ad.logits(ids) - ref).norm() / ref.norm())
    assert 0 < rel8 < 0.02 and rel8 < rel2


def test_kivi_granularity() -> None:
    q = kvq.from_spec(QuantSpec.from_dict({"format": "int", "bits": 4}), "kivi")
    k = torch.randn(1, 2, 16, 8)
    k[..., 3] *= 100  # outlier channel: per-channel keys isolate it
    naive = kvq.from_spec(QuantSpec.from_dict({"format": "int", "bits": 4}), "naive", "per_token")
    e_kivi = float((q.quant_k(k) - k)[..., :3].norm())
    e_naive = float((naive.quant_k(k) - k)[..., :3].norm())
    assert e_kivi < 0.2 * e_naive


def test_perplexity_uniform_logits_equals_vocab() -> None:
    V = 50
    r = perplexity(lambda ids: torch.zeros(1, ids.shape[1], V), torch.randint(0, V, (1000,)), seq_len=100)
    assert r["windows"] == 10 and math.isclose(r["ppl"], V, rel_tol=1e-5)
