"""T-R10 dynamic quantization: a skip list generated from per-layer quantization error (no evaluation needed)."""

from __future__ import annotations

import re

import pytest

from mqde.quant.blockwise import MixedPrecisionPolicy
from mqde.quant.suggest import suggest_skips


def layer(name: str, db: float, comp: str = "language", shape: tuple[int, int] = (256, 128)) -> dict:
    return {"name": name, "component": comp, "out_sqnr_db": db, "shape": list(shape)}


def lang(n: int, bad: dict[int, float]) -> list[dict]:
    return [layer(f"model.language_model.layers.{i}.mlp.up_proj", bad.get(i, 38.0 + 0.1 * (i % 5))) for i in range(n)]


def test_clear_outliers_are_skipped_and_the_rest_is_left_alone() -> None:
    s = suggest_skips(lang(24, {3: 27.0, 11: 28.5}), bits=4)
    assert {x["name"] for x in s["layers"]} == {"model.language_model.layers.3.mlp.up_proj", "model.language_model.layers.11.mlp.up_proj"}
    assert all(r["skip"] is True for r in s["rules"]) and len(s["rules"]) == 2


def test_rules_are_anchored_so_layer_1_does_not_catch_layer_11() -> None:
    s = suggest_skips(lang(24, {1: 26.0}), bits=4)
    pol = MixedPrecisionPolicy.from_dict({"rules": s["rules"]})
    assert pol.lookup("model.language_model.layers.1.mlp.up_proj", 1).skip
    for other in (11, 10, 21, 2):
        assert not pol.lookup(f"model.language_model.layers.{other}.mlp.up_proj", other).skip
    assert re.search(s["rules"][0]["match"], "model.language_model.layers.1.mlp.up_proj")


def test_nothing_is_flagged_when_all_layers_are_alike() -> None:
    s = suggest_skips(lang(24, {}), bits=4)
    assert s["rules"] == [] and s["layers"] == [] and s["extra_weight_bytes"] == 0


def test_a_component_that_is_uniformly_worse_is_not_flagged_wholesale() -> None:
    vision = [layer(f"model.vision_tower.encoder.layers.{i}.mlp.fc1", 30.0 + 0.1 * (i % 3), "vision") for i in range(20)]
    s = suggest_skips(vision + lang(20, {}), bits=4)
    assert s["layers"] == [], "vision is 8 dB worse than language everywhere: that is the component, not an outlier layer"


def test_an_outlier_is_judged_against_its_own_component() -> None:
    vision = [layer(f"model.vision_tower.encoder.layers.{i}.mlp.fc1", 30.0 + 0.1 * (i % 3), "vision") for i in range(20)]
    vision[4] = layer("model.vision_tower.encoder.layers.4.mlp.fc1", 21.0, "vision")
    s = suggest_skips(vision + lang(20, {}), bits=4)
    assert [x["name"] for x in s["layers"]] == ["model.vision_tower.encoder.layers.4.mlp.fc1"]
    assert s["layers"][0]["component"] == "vision"


def test_at_most_max_fraction_of_the_layers_are_skipped_worst_first() -> None:
    bad = {i: 20.0 + i for i in range(8)}  # 8 of 40 layers far below the rest
    s = suggest_skips(lang(40, bad), bits=4, max_fraction=0.1)
    assert [x["name"].split(".")[3] for x in s["layers"]] == ["0", "1", "2", "3"], "4 of 40, lowest SQNR first"


def test_extra_weight_bytes_is_what_keeping_those_layers_in_16_bit_costs() -> None:
    layers = lang(24, {3: 27.0})
    layers[3]["shape"] = [512, 256]
    s = suggest_skips(layers, bits=4)
    assert s["extra_weight_bytes"] == 512 * 256 * (16 - 4) // 8
    assert suggest_skips(layers, bits=8)["extra_weight_bytes"] == 512 * 256 * (16 - 8) // 8


def test_layers_without_a_measurement_are_ignored() -> None:
    layers = lang(24, {3: 27.0}) + [{"name": "model.x", "component": "language", "out_sqnr_db": None, "shape": [8, 8]}]
    assert [x["name"].split(".")[3] for x in suggest_skips(layers, bits=4)["layers"]] == ["3"]


@pytest.mark.parametrize("n", [0, 1, 3])
def test_too_few_layers_to_tell_an_outlier_from_noise(n: int) -> None:
    assert suggest_skips(lang(n, {0: 10.0}), bits=4)["rules"] == []


def test_a_layer_must_be_both_a_statistical_outlier_and_a_real_gap_below_the_median() -> None:
    # near-identical layers (MAD ~ 0): a 1 dB dip is many sigmas but not worth 12/4 more bytes per weight
    assert suggest_skips(lang(24, {7: 36.5}), bits=4)["rules"] == []
    # noisy layers (MAD ~ 4 dB): a 5 dB dip is a big gap but within the spread
    noisy = [layer(f"model.language_model.layers.{i}.mlp.up_proj", 38.0 + (8.0 if i % 2 else 0.0)) for i in range(24)]
    noisy[6] = layer("model.language_model.layers.6.mlp.up_proj", 33.0)
    assert suggest_skips(noisy, bits=4)["rules"] == []
