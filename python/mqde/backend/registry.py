"""Deployment backend registry (T-E).

A lightweight discovery layer: each backend (llamacpp / vllm / sglang) registers the metadata a job
needs to decide whether it can deploy — which JobSpec field enables it, which export formats it
serves, whether it needs CUDA, and how to check its binary is installed. The registry does NOT
replace the serve methods in llm_job.py; it records what exists so the job can query backends
without hardcoding imports, and so a new backend can be added without touching llm_job.py's
dispatch.

Mirrors the @register idiom in python/mqde/quant/algorithms.py (ALGO_REGISTRY). The sync check
scripts/check_backend_registry_sync.py keeps this registry and data/domains/*.yaml's
deploy_backends lists in agreement, the same way check_algo_registry_sync.py keeps
ALGO_REGISTRY and data/algorithms.yaml in sync.

Why a registry now, before VLM export exists: the previous session (2026-10-16) confirmed the
architecture gap is real (llm_job.py hardcodes backend imports) but held off because the serve
methods mix generic process lifecycle with LLM-specific scoring (PPL via logprobs). This registry
captures only the generic metadata — the serve methods stay where they are until a second
domain (VLM) needs to deploy and forces the scoring abstraction. Adding the registry now means
that when VLM export lands, the wizard and the job can already ask "which backends can this
domain deploy to?" through one query instead of a new hardcoded check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# A callable that raises FileNotFoundError if the backend's binary/env is missing.
# Mirrors vllm_bin() / sglang_python() / server_bin() in the server modules.
BinCheck = Callable[[], None]


@dataclass
class BackendEntry:
    """Static metadata about a deployment backend, registered at import time."""

    id: str  # "vllm", "sglang", "llamacpp" — matches data/domains/*.yaml deploy_backends
    server_class: str  # "VllmServer" / "SglangServer" / "LlamaServer" — for docs/errors
    spec_field: str  # which JobSpec field enables this backend (e.g. "vllm", "llamacpp_serve")
    export_formats: list[str]  # which export formats this backend can serve (e.g. ["awq", "gptq"])
    needs_cuda: bool  # whether the backend requires a CUDA device
    bin_check: BinCheck  # raises FileNotFoundError if the binary/env is missing
    # Where the server module lives, so a caller can import the server class by string id
    # (kept as a string to avoid importing torch-heavy modules at registry query time)
    module: str = ""


BACKEND_REGISTRY: dict[str, BackendEntry] = {}


def register_backend(entry: BackendEntry) -> BackendEntry:
    if entry.id in BACKEND_REGISTRY:
        raise ValueError(f"backend '{entry.id}' registered twice")
    BACKEND_REGISTRY[entry.id] = entry
    return entry


def list_backends() -> list[str]:
    return sorted(BACKEND_REGISTRY)


def get_backend(backend_id: str) -> BackendEntry:
    if backend_id not in BACKEND_REGISTRY:
        raise KeyError(f"unknown backend '{backend_id}' (known: {', '.join(list_backends())})")
    return BACKEND_REGISTRY[backend_id]


def backends_for_domain(deploy_backends: list[str]) -> list[BackendEntry]:
    """Resolve a domain's deploy_backends list to registered entries; unknown ones are skipped
    with the caller left to surface them (the sync check catches drift)."""
    return [BACKEND_REGISTRY[b] for b in deploy_backends if b in BACKEND_REGISTRY]
