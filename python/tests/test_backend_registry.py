"""Unit tests for python/mqde/backend/registry.py (T-E).

These tests import only the registry module itself — not torch, not the server modules
(which import heavy deps) — to verify the metadata layer in isolation. The server modules'
_register() calls are tested separately by importing those modules explicitly (see bottom).
"""

from __future__ import annotations

import sys
import pathlib

import pytest

# Make the package importable without an install when running from the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))


def _fresh_registry():
    """Return a clean registry dict and the module-level helpers, without touching the global
    state that the server modules populate on import."""
    from mqde.backend import registry as _reg
    return _reg


def test_register_and_lookup():
    from mqde.backend.registry import BackendEntry, BACKEND_REGISTRY, register_backend

    # Use a unique id so this test is independent of the server modules' registrations
    test_id = "_lookup_test_only_" + str(id(object()))
    entry = BackendEntry(
        id=test_id,
        server_class="TestServer",
        spec_field="test_field",
        export_formats=["awq"],
        needs_cuda=True,
        bin_check=lambda: None,
        module="mqde.backend.test_backend",
    )
    register_backend(entry)
    assert BACKEND_REGISTRY[test_id] is entry
    assert entry.spec_field == "test_field"
    assert entry.needs_cuda is True
    assert "awq" in entry.export_formats


def test_register_duplicate_raises():
    from mqde.backend.registry import BackendEntry, BACKEND_REGISTRY, register_backend
    # Use a unique id that is not already in the global registry to test the duplicate guard
    entry = BackendEntry(
        id="_dup_test_only_" + str(id(object())),  # unique per run
        server_class="X",
        spec_field="x",
        export_formats=[],
        needs_cuda=False,
        bin_check=lambda: None,
    )
    register_backend(entry)
    with pytest.raises(ValueError, match="registered twice"):
        register_backend(entry)


def test_get_backend_unknown_raises():
    from mqde.backend.registry import get_backend
    with pytest.raises(KeyError, match="unknown backend"):
        get_backend("__no_such_backend__")


def test_backends_for_domain_filters_unknown():
    from mqde.backend.registry import BackendEntry, BACKEND_REGISTRY, backends_for_domain
    # Ask for a mix of registered and unregistered ids — unregistered ones are silently skipped
    known = list(BACKEND_REGISTRY.keys())[:1]
    result = backends_for_domain(known + ["__no_such__"])
    assert len(result) == len(known)
    if known:
        assert result[0].id == known[0]


def test_three_server_modules_register_on_import():
    """Importing each server module triggers its _register() call; all three must appear in the
    global BACKEND_REGISTRY with the right metadata."""
    # Import order doesn't matter — each module guards against duplicate registration
    import mqde.backend.vllm_server      # noqa: F401
    import mqde.backend.sglang_server    # noqa: F401
    import mqde.backend.llamacpp_server  # noqa: F401

    from mqde.backend.registry import BACKEND_REGISTRY

    for backend_id in ("vllm", "sglang", "llamacpp"):
        assert backend_id in BACKEND_REGISTRY, f"'{backend_id}' not in BACKEND_REGISTRY"

    vllm = BACKEND_REGISTRY["vllm"]
    assert vllm.needs_cuda is True
    assert "awq" in vllm.export_formats
    assert "compressed-tensors" in vllm.export_formats
    assert vllm.spec_field == "vllm"

    sglang = BACKEND_REGISTRY["sglang"]
    assert sglang.needs_cuda is True
    assert "gptq" in sglang.export_formats
    assert "compressed-tensors" in sglang.export_formats

    llamacpp = BACKEND_REGISTRY["llamacpp"]
    assert llamacpp.needs_cuda is False
    assert llamacpp.export_formats == ["gguf"]
    assert llamacpp.spec_field == "llamacpp_serve"


def test_llm_domain_deploy_backends_all_registered():
    """data/domains/llm.yaml declares deploy_backends; all of them must be in the registry."""
    import yaml
    import mqde.backend.vllm_server      # noqa: F401
    import mqde.backend.sglang_server    # noqa: F401
    import mqde.backend.llamacpp_server  # noqa: F401
    from mqde.backend.registry import BACKEND_REGISTRY

    llm_yaml = pathlib.Path(__file__).parents[2] / "data" / "domains" / "llm.yaml"
    d = yaml.safe_load(llm_yaml.read_text())
    for b in d.get("deploy_backends", []):
        assert b in BACKEND_REGISTRY, f"llm.yaml declares '{b}' but it's not in BACKEND_REGISTRY"
