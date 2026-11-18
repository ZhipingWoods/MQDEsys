"""Benchmark registry and suite loading (evalscope-style ``BenchmarkMeta``).

The registry is deliberately independent of any evaluation backend: a benchmark is a name, a
human-readable label, a task kind (``ppl`` / ``mcq`` / ``qa``) and, for file-backed tasks, a JSONL
path.  ``offline.py`` knows how to score each kind, while external runners (lm-eval, EvalScope)
can use the same names and metadata.

Suites live in ``data/datasets/eval_suites.yaml``.  A suite is just an ordered list of benchmark
names; ``quick`` is the smoke set used by the wizard, ``standard`` is the wider default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BenchmarkMeta:
    name: str
    label: str
    kind: str  # ppl | mcq | qa
    description: str = ""
    path: str | None = None
    metrics: tuple[str, ...] = ()
    tags: tuple[str, ...] = field(default_factory=tuple)
    group: str = "其他"
    language: tuple[str, ...] = ()
    runner: str = "offline"
    offline: bool = True
    source: str = ""
    sample_count: int | None = None
    dataset: str = ""
    subset: str = ""
    split: str = ""
    template: str = ""
    max_new_tokens: int = 64

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "label": self.label, "kind": self.kind, "description": self.description,
                "path": self.path, "metrics": list(self.metrics), "tags": list(self.tags),
                "group": self.group, "language": list(self.language), "runner": self.runner,
                "offline": self.offline, "source": self.source, "sample_count": self.sample_count,
                "dataset": self.dataset, "subset": self.subset, "split": self.split,
                "template": self.template, "max_new_tokens": self.max_new_tokens}


BENCHMARK_REGISTRY: dict[str, BenchmarkMeta] = {}


def register(meta: BenchmarkMeta) -> BenchmarkMeta:
    if meta.kind not in ("ppl", "mcq", "qa"):
        raise ValueError(f"benchmark '{meta.name}': unknown kind '{meta.kind}' (known: ppl, mcq, qa)")
    if meta.name in BENCHMARK_REGISTRY:
        raise ValueError(f"benchmark '{meta.name}' is already registered")
    BENCHMARK_REGISTRY[meta.name] = meta
    return meta


def data_dir() -> Path:
    from mqde.models.base import data_dir as _data_dir

    return _data_dir()


def _register_builtins() -> None:
    d = data_dir() / "datasets"
    doc = yaml.safe_load((d / "benchmarks.yaml").read_text(encoding="utf-8")) or {}
    for row in doc.get("benchmarks", []):
        path = row.get("path")
        if path and not Path(path).is_absolute():
            path = str(d / path)
        register(BenchmarkMeta(
            name=str(row["name"]), label=str(row.get("label", row["name"])), kind=str(row["kind"]),
            description=str(row.get("description", "")), path=path,
            metrics=tuple(row.get("metrics", [])), tags=tuple(row.get("tags", [])),
            group=str(row.get("group", "其他")), language=tuple(row.get("language", [])),
            runner=str(row.get("runner", "offline")), offline=bool(row.get("offline", True)),
            source=str(row.get("source", "")),
            sample_count=(int(row["sample_count"]) if row.get("sample_count") is not None else None)))


def _register_native() -> None:
    p = data_dir() / "datasets" / "native_benchmarks.yaml"
    if not p.is_file():
        return
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    for row in doc.get("benchmarks", []):
        register(BenchmarkMeta(
            name=str(row["name"]), label=str(row.get("label", row["name"])), kind=str(row["kind"]),
            description=str(row.get("description", "")), metrics=tuple(row.get("metrics", [])),
            tags=tuple(row.get("tags", [])), group=str(row.get("group", "其他")),
            language=tuple(row.get("language", [])), runner=str(row.get("runner", "native")),
            offline=bool(row.get("offline", False)), source=str(row.get("source", "")),
            dataset=str(row.get("dataset", "")), subset=str(row.get("subset", "")),
            split=str(row.get("split", "")), template=str(row.get("template", "")),
            max_new_tokens=int(row.get("max_new_tokens", 64))))


_register_builtins()
_register_native()


def get(name: str) -> BenchmarkMeta:
    if name in BENCHMARK_REGISTRY:
        return BENCHMARK_REGISTRY[name]
    p = Path(name)
    if p.is_file() and p.suffix.lower() == ".jsonl":
        # A user-supplied JSONL file: kind is inferred from the first non-comment line.
        import json

        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            row = json.loads(line)
            kind = "mcq" if "choices" in row and "answer" in row else "qa"
            return BenchmarkMeta(str(p), p.stem, kind, f"Custom JSONL ({kind})", path=str(p),
                                 metrics=("accuracy",), tags=("custom",))
        raise ValueError(f"{p}: no cases")
    raise KeyError(f"unknown benchmark '{name}' (known: {', '.join(sorted(BENCHMARK_REGISTRY))})")


def suite(name: str | None) -> list[str]:
    """Names in a suite, or an explicit comma/space separated list, or ``[]``."""
    if not name:
        return []
    p = Path(name)
    if p.is_file() and p.suffix.lower() in (".yaml", ".yml"):
        doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    else:
        doc = yaml.safe_load((data_dir() / "datasets" / "eval_suites.yaml").read_text(encoding="utf-8")) or {}
    suites = doc.get("suites") or doc
    if name in suites and isinstance(suites[name], list):
        return [str(x) for x in suites[name]]
    # Not a suite name: treat it as an explicit list ("smoke_qa,smoke_mcq") or a single benchmark.
    return [x for x in name.replace(",", " ").split() if x]


def catalog() -> list[dict[str, Any]]:
    return [m.to_dict() for m in sorted(BENCHMARK_REGISTRY.values(), key=lambda m: m.name)]
