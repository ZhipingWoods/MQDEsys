"""Every data/model_families template loads, and mainstream checkpoints resolve to the intended family."""

from __future__ import annotations

import re

import yaml

from mqde.models.base import data_dir, family_for_model_type


def test_every_family_template_has_valid_regexes_and_components() -> None:
    for path in sorted((data_dir() / "model_families").glob("*.yaml")):
        fam = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert fam["family"], path
        assert fam.get("components"), f"{path.name}: no components"
        for comp in fam["components"]:
            assert "prefix" in comp, f"{path.name}: component without prefix key"
            if comp.get("block_regex"):
                re.compile(comp["block_regex"])
        for pattern in fam.get("default_keep_fp", []):
            re.compile(pattern)


def test_mainstream_vlm_and_llm_model_types_resolve_to_a_specific_family() -> None:
    cases = {
        "qwen2_vl": ("qwen_vl", "vlm"),
        "qwen2_5_vl": ("qwen_vl", "vlm"),
        "qwen3_vl": ("qwen_vl", "vlm"),
        "llava": ("llava", "vlm"),
        "llava_next": ("llava", "vlm"),
        "llava_onevision": ("llava", "vlm"),
        "internvl_chat": ("internvl", "vlm"),
        "glm4v": ("glm4v", "vlm"),
        "llama4": ("llama4", "vlm"),
        "idefics3": ("idefics3", "vlm"),
        "minicpmv4_6": ("minicpmv", "vlm"),
        "llama": ("decoder_llm", "llm"),
        "qwen2": ("decoder_llm", "llm"),
        "qwen3_moe": ("decoder_llm", "llm"),
        "mixtral": ("decoder_llm", "llm"),
        "deepseek_v3": ("decoder_llm", "llm"),
        "gpt_oss": ("decoder_llm", "llm"),
    }
    for model_type, (family, modality) in cases.items():
        fam = family_for_model_type(model_type)
        assert fam["family"] == family, model_type
        assert fam["modality"] == modality, model_type


def test_unknown_model_type_falls_back_to_generic_decoder() -> None:
    fam = family_for_model_type("definitely_not_a_real_architecture")
    assert fam["family"] == "generic_decoder"
