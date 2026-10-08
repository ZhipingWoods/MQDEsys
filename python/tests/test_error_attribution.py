"""Error attribution acceptance (T-0906): a layer that is made worse on purpose must be the one the report points at.

The layer is "injected" through the product's own mixed-precision policy (that layer at 2 bit, every other at 4 bit),
so the damage goes through the real quantize / trace path instead of being patched in afterwards. Two checks:

1. per-layer localisation: the damaged layer has the lowest output SQNR of all layers (GPTQ captures activations),
   by a clear margin, and the same model without damage has no such outlier;
2. fake -> real: a corrupted packed checkpoint is the largest error between the fake-quant and the exported model
   (test_real_check.py::test_corrupted_packing_is_detected covers the integer-code level).

The FP-fallback sensitivity ranking is deliberately NOT asserted here: on a random tiny model its metric is dominated by
noise (the undamaged control ranks block 0 first as well), so a pass would prove nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("safetensors")
sys.path.insert(0, str(Path(__file__).parent / "export"))

import _tiny

from mqde.export import get_exporter
from mqde.export.real_check import real_check
from mqde.quant.blockwise import BlockwiseQuantizer, MixedPrecisionPolicy
from mqde.quant.engine import RecipeSpec

VICTIMS = [
    "model.layers.1.mlp.up_proj",
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.1.self_attn.o_proj",
]


def quantize(victim: str | None):  # type: ignore[no-untyped-def]
    ad = _tiny.tiny()
    rules = [{"match": victim.replace(".", r"\."), "bits": 2}] if victim else []
    q = BlockwiseQuantizer(ad, RecipeSpec.load("w4a16"), "gptq", policy=MixedPrecisionPolicy.from_dict({"rules": rules}))
    return ad, q.run(_tiny.calib(8, 64))


def ranked(traces) -> list[tuple[str, float]]:  # type: ignore[no-untyped-def]
    return sorted(((t.name, t.out_sqnr_db) for t in traces if t.out_sqnr_db is not None), key=lambda x: x[1])


def control_spread() -> float:
    """Gap in dB between the worst and second-worst layer of the undamaged model: the natural spread."""
    _, res = quantize(None)
    r = ranked(res.traces)
    assert len(r) == 14
    return r[1][1] - r[0][1]


def test_control_has_no_outlier() -> None:
    """Without damage the worst layer is not far from the next ones; this is what makes the injected case meaningful."""
    assert control_spread() < 2.0


@pytest.mark.parametrize("victim", VICTIMS)
def test_the_damaged_layer_is_ranked_worst_by_a_clear_margin(victim: str) -> None:
    _, res = quantize(victim)
    r = ranked(res.traces)
    assert r[0][0] == victim, r[:3]
    # The margin is judged against the undamaged model's own spread, not a magic number: o_proj sees flatter activations
    # than up_proj, so the same 2-bit damage costs it fewer dB (about 4 versus about 9 here), but never as few as the
    # 0.1 to 0.3 dB between the worst layers of the control.
    assert r[1][1] - r[0][1] > 5 * control_spread(), r[:3]
    by = {t.name: t for t in res.traces}
    assert by[victim].algo_info["bits"] == 2 and all(t.algo_info["bits"] == 4 for n, t in by.items() if n != victim)
    # the same layer also has the lowest weight SQNR, so both views of the trace agree
    assert min(res.traces, key=lambda t: t.w_sqnr_db).name == victim


def test_the_report_summary_points_at_the_damaged_layer() -> None:
    """The job report's worst_layers (what the UI and the HTML report list) starts with it."""
    victim = VICTIMS[0]
    _, res = quantize(victim)
    worst = sorted(({"name": t.name, "out_sqnr_db": t.out_sqnr_db, "w_sqnr_db": t.w_sqnr_db} for t in res.traces),
                   key=lambda d: d["out_sqnr_db"] if d["out_sqnr_db"] is not None else d["w_sqnr_db"])[:10]
    assert worst[0]["name"] == victim


def test_damage_that_is_not_in_the_layer_is_not_blamed_on_it() -> None:
    """Negative control: damage one layer, and no *other* layer may fall into the damaged range."""
    victim = VICTIMS[1]
    _, res = quantize(victim)
    r = ranked(res.traces)
    others = [s for n, s in r if n != victim]
    assert min(others) > r[0][1] + 5 * control_spread()


def test_corrupted_packing_is_the_largest_fake_to_real_error(tmp_path: Path) -> None:
    from safetensors.torch import load_file, save_file

    ad, _ = quantize(None)
    get_exporter("gptq").export(ad.model, tmp_path)
    clean = real_check(ad.model, tmp_path)
    assert clean["codes_identical"] and clean["w_max_abs"] < 1e-2  # the honest export differs only by fp16 scale storage
    f = tmp_path / "model.safetensors"
    sd = load_file(str(f))
    k = "model.layers.0.mlp.gate_proj.qweight"
    sd[k][2, 5] ^= 0b1111 << 8  # change one 4-bit code to another value
    save_file(sd, str(f))
    bad = real_check(ad.model, tmp_path)
    assert not bad["codes_identical"] and bad["code_mismatch"] >= 1
    worst = max((x for x in bad["per_layer"] if not x.get("missing")), key=lambda x: x["w_max_abs"])
    assert worst["name"] == "model.layers.0.mlp.gate_proj"
    assert bad["w_max_abs"] > 10 * clean["w_max_abs"]  # far above the storage-rounding noise
    assert torch.isfinite(torch.tensor(bad["w_rel_l2_mean"]))
