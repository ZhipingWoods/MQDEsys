"""Model abstraction layer (docs/ARCHITECTURE.md §6.7).

A QuantModel exposes a uniform block view over any architecture so algorithms never hard-code
module paths. Grouping comes from the data-driven family template (data/model_families/*.yaml),
the same file the C++ ModelSpec uses, so unit names line up between estimator, trace and UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from mqde.quant.modules import FakeQuantLinear


def data_dir() -> Path:
    import os

    env = os.environ.get("MQDE_DATA_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[3] / "data"


@dataclass
class Unit:
    """One quantizable linear layer."""

    name: str  # module path, e.g. vlm.model.language_model.layers.3.mlp.up_proj
    component: str
    modality: str
    block: int
    shape: tuple[int, int]
    keep_fp: bool


@dataclass
class Block:
    component: str
    modality: str
    index: int
    prefix: str  # module path of the block
    units: list[Unit] = field(default_factory=list)


class QuantModel:
    """Uniform access to blocks / linears of a loaded torch model."""

    def __init__(self, model: nn.Module, family: dict[str, Any]) -> None:
        self.model = model
        self.family = family
        self._keep_fp = [re.compile(p) for p in family.get("default_keep_fp", [])]
        self.units: list[Unit] = []
        self.blocks: list[Block] = []
        self._index()

    @staticmethod
    def load_family(name: str) -> dict[str, Any]:
        path = data_dir() / "model_families" / f"{name}.yaml"
        with open(path) as f:
            return yaml.safe_load(f)

    def _classify(self, name: str) -> tuple[str, str, int, str]:
        for comp in self.family["components"]:
            prefix = comp["prefix"]
            weight_name = name + ".weight"
            if not weight_name.startswith(prefix):
                continue
            block, block_prefix = -1, prefix.rstrip(".")
            if comp.get("block_regex"):
                m = re.search(comp["block_regex"], weight_name[len(prefix) :])
                if m:
                    block = int(m.group(1))
                    block_prefix = prefix + weight_name[len(prefix) : len(prefix) + m.end()].rstrip(".")
            return comp["name"], comp.get("modality", "language"), block, block_prefix
        return "other", "other", -1, ""

    def _index(self) -> None:
        blocks: dict[tuple[str, int], Block] = {}
        for name, mod in self.model.named_modules():
            if not isinstance(mod, (nn.Linear, FakeQuantLinear)):
                continue
            comp, modality, block, block_prefix = self._classify(name)
            keep = any(p.search(name + ".weight") for p in self._keep_fp)
            u = Unit(name, comp, modality, block, (mod.out_features, mod.in_features), keep)
            self.units.append(u)
            key = (comp, block)
            if key not in blocks:
                blocks[key] = Block(comp, modality, block, block_prefix)
            blocks[key].units.append(u)
        self.blocks = sorted(blocks.values(), key=lambda b: (b.component, b.index))

    def get_blocks(self, component: str | None = None, modality: str | None = None) -> list[Block]:
        return [
            b
            for b in self.blocks
            if (component is None or b.component == component) and (modality is None or b.modality == modality)
        ]

    def get_module(self, name: str) -> nn.Module:
        return self.model.get_submodule(name)

    def components(self) -> list[str]:
        seen: list[str] = []
        for u in self.units:
            if u.component not in seen:
                seen.append(u.component)
        return seen

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for c in self.components():
            us = [u for u in self.units if u.component == c]
            out[c] = {
                "modality": us[0].modality,
                "linears": len(us),
                "quantizable": sum(not u.keep_fp for u in us),
                "blocks": len({u.block for u in us if u.block >= 0}),
            }
        return out


def module_device(m: nn.Module) -> torch.device:
    for p in m.parameters():
        return p.device
    for b in m.buffers():
        return b.device
    return torch.device("cpu")
