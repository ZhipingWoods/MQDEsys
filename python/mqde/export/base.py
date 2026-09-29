"""Exporter interface + registry (docs/原始设想 "EXPORTER_REGISTRY").

An exporter turns a model whose quantized layers are ``FakeQuantLinear`` (weights on the integer grid,
with the exact qparams kept in ``w_scale`` / ``w_zero``) into a checkpoint a serving backend loads. Before
writing anything it runs ``check()`` so unsupported layers (wrong format, bit width, shape, or an
un-foldable activation transform) are reported instead of silently exported wrong.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import nn

from mqde.quant.modules import FakeQuantLinear

EXPORTER_REGISTRY: dict[str, type[Exporter]] = {}


def register(name: str) -> Callable[[type[Exporter]], type[Exporter]]:
    def deco(cls: type[Exporter]) -> type[Exporter]:
        cls.name = name
        EXPORTER_REGISTRY[name] = cls
        return cls

    return deco


def get_exporter(name: str, **kw: Any) -> Exporter:
    if name not in EXPORTER_REGISTRY:
        raise KeyError(f"unknown export format '{name}' (known: {', '.join(sorted(EXPORTER_REGISTRY))})")
    return EXPORTER_REGISTRY[name](**kw)


@dataclass
class ExportResult:
    format: str
    out_dir: str
    files: list[str]
    quantized: list[str]
    kept_fp: list[str]
    quantization_config: dict[str, Any]
    bytes: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class Exporter:
    name = ""
    backends: tuple[str, ...] = ()

    def check_layer(self, name: str, m: FakeQuantLinear) -> str | None:
        """None if the layer can be exported, else the reason it cannot."""
        raise NotImplementedError

    def check(self, model: nn.Module) -> list[tuple[str, str]]:
        return [(n, r) for n, m in fake_linears(model).items() if (r := self.check_layer(n, m)) is not None]

    def export(self, model: nn.Module, out_dir: str | Path, source_dir: str | Path | None = None) -> ExportResult:
        raise NotImplementedError


def fake_linears(model: nn.Module) -> dict[str, FakeQuantLinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, FakeQuantLinear)}
