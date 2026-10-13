"""AGENTS.md section 6: `trust_remote_code=false` by default. Loading a model directory with it on runs whatever Python the directory
ships, so a model adapter must not turn it on by itself. MiniCPM-V 4.6 is a native transformers architecture (no auto_map, no .py
files in its directory) and does not need it."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

from mqde.models import minicpmv


class _Cfg:
    eos_token_id = 1


class _Model(torch.nn.Module):
    generation_config = _Cfg()

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.ModuleDict()
        self.lm_head = torch.nn.Linear(2, 2)


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []

    class Proc:
        tokenizer = types.SimpleNamespace(eos_token_id=1)

        @classmethod
        def from_pretrained(cls, path: str, **kw: object) -> "Proc":
            calls.append(("processor", kw))
            return cls()

    class Auto:
        @classmethod
        def from_pretrained(cls, path: str, **kw: object) -> _Model:
            calls.append(("model", kw))
            return _Model()

    fake = types.ModuleType("transformers")
    fake.AutoProcessor = Proc  # type: ignore[attr-defined]
    fake.AutoModelForImageTextToText = Auto  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake)

    class FakeQuantModel:
        units: list = []

        def __init__(self, model: object, family: object) -> None:
            pass

        @staticmethod
        def load_family(name: str) -> dict:
            return {"components": []}

    monkeypatch.setattr(minicpmv, "QuantModel", FakeQuantModel)
    return calls


def test_the_adapter_does_not_run_code_from_the_model_directory_by_default(seen: list, tmp_path: Path) -> None:
    minicpmv.MiniCPMV(tmp_path, device="cpu")
    assert [k for k, _ in seen] == ["processor", "model"]
    assert all(kw.get("trust_remote_code") is False for _, kw in seen), seen


def test_it_is_opt_in_and_reaches_both_loaders(seen: list, tmp_path: Path) -> None:
    minicpmv.MiniCPMV(tmp_path, device="cpu", trust_remote_code=True)
    assert all(kw["trust_remote_code"] is True for _, kw in seen)
