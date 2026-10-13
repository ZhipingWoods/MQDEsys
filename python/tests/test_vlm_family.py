"""The minicpmv family template against the real MiniCPM-V 4.6 module layout (a tiny model with the same module paths)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mqde.models.base import QuantModel


class Tiny(nn.Module):
    """Same module paths as MiniCPMV4_6ForConditionalGeneration: model.{vision_tower,merger,language_model} + lm_head."""

    def __init__(self) -> None:
        super().__init__()
        lin = lambda: nn.Linear(8, 8)  # noqa: E731
        enc = nn.ModuleList([nn.ModuleDict({"self_attn": nn.ModuleDict({"q_proj": lin(), "k_proj": lin(), "out_proj": lin()}),
                                            "mlp": nn.ModuleDict({"fc1": lin(), "fc2": lin()})}) for _ in range(2)])
        vit_merger = nn.ModuleDict({"self_attn": nn.ModuleDict({"q_proj": lin()}), "linear_1": lin(), "linear_2": lin()})
        tower = nn.ModuleDict({"embeddings": nn.ModuleDict({"patch_embedding": nn.Conv2d(3, 8, 2)}),
                               "encoder": nn.ModuleDict({"layers": enc}), "vit_merger": vit_merger})
        layers = nn.ModuleList([nn.ModuleDict({"mlp": nn.ModuleDict({"gate_proj": lin(), "down_proj": lin()}),
                                               "linear_attn": nn.ModuleDict({"in_proj_qkv": lin(), "in_proj_a": lin(), "out_proj": lin()})})
                                for _ in range(3)])
        merger = nn.ModuleDict({"mlp": nn.ModuleList([nn.ModuleDict({"linear_1": lin(), "linear_2": lin()})])})
        self.model = nn.ModuleDict({"vision_tower": tower, "merger": merger,
                                    "language_model": nn.ModuleDict({"embed_tokens": nn.Embedding(10, 8), "layers": layers})})
        self.lm_head = lin()


@pytest.fixture(scope="module")
def qm() -> QuantModel:
    return QuantModel(Tiny(), QuantModel.load_family("minicpmv"))


def by_component(qm: QuantModel) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for u in qm.units:
        out.setdefault(u.component, []).append(u.name)
    return out


def test_every_linear_is_assigned_to_a_component_nothing_falls_into_other(qm: QuantModel) -> None:
    comps = by_component(qm)
    assert "other" not in comps, comps.get("other")
    assert set(comps) == {"vision", "vision_merger", "merger", "language", "lm_head"}


def test_the_vit_merger_is_not_swallowed_by_the_vision_tower(qm: QuantModel) -> None:
    comps = by_component(qm)
    assert all("vit_merger" in n for n in comps["vision_merger"]) and len(comps["vision_merger"]) == 3
    assert not any("vit_merger" in n for n in comps["vision"])
    assert len(comps["vision"]) == 2 * 5  # 5 linears per ViT layer, 2 layers


def test_blocks_are_indexed_per_component(qm: QuantModel) -> None:
    assert sorted({b.index for b in qm.get_blocks("vision")}) == [0, 1]
    assert sorted({b.index for b in qm.get_blocks("language")}) == [0, 1, 2]
    assert all(b.modality == "vision" for b in qm.get_blocks("vision_merger"))
    assert {u.modality for u in qm.units if u.component == "language"} == {"language"}


def test_precision_sensitive_layers_stay_fp_by_default(qm: QuantModel) -> None:
    keep = {u.name for u in qm.units if u.keep_fp}
    assert "lm_head" in keep, "the (tied) output head is not quantized"
    assert any(n.endswith("linear_attn.in_proj_a") for n in keep), "the tiny gate projections of the linear attention stay FP"
    assert not any("mlp.gate_proj" in n for n in keep) and not any("q_proj" in n for n in keep)


def test_forward_still_works_and_units_have_the_right_shapes(qm: QuantModel) -> None:
    assert all(u.shape == (8, 8) for u in qm.units)
    assert torch.nn.functional.linear(torch.ones(1, 8), qm.get_module("lm_head").weight).shape == (1, 8)
