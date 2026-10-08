"""Self-contained HTML report (T-0806).

The fixtures in tests/fixtures/reports are real job reports of Qwen2.5-0.5B (vLLM-served exports, a llama.cpp GGUF
run, and a full-featured run with sensitivity analysis) with paths removed and per-layer data trimmed to the fields the
report reads.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from mqde.report.html import bar_chart, fmt, nbytes, pct, render_html

FIX = Path(__file__).parent / "fixtures" / "reports"


def load(name: str) -> dict:
    return json.loads((FIX / f"llm_{name}.json").read_text())


class Tags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.tags: list[tuple[str, dict]] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self.tags.append((tag, dict(attrs)))
        if tag not in ("meta", "br", "hr", "img", "input", "link", "line", "rect"):
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("line", "rect"):
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> with {self.stack[-3:]}")
        else:
            self.stack.pop()


@pytest.mark.parametrize("name", ["vllm", "gguf", "full"])
@pytest.mark.parametrize("lang", ["en", "zh"])
def test_page_is_well_formed_and_offline(name: str, lang: str) -> None:
    html = render_html(load(name), lang)
    p = Tags()
    p.feed(html)
    assert not p.errors and not p.stack, (p.errors[:3], p.stack)
    assert html.startswith("<!doctype html>") and f'<html lang="{lang}">' in html
    # nothing that would fetch or run anything
    assert not [t for t, _ in p.tags if t in ("script", "iframe", "object", "embed", "img", "link")]
    assert not re.search(r"https?://|url\(|@import|src=|href=", html)


def test_vllm_report_shows_the_measured_numbers() -> None:
    r = load("vllm")
    html = render_html(r, "en")
    rec = r["recipes"]["W4A16"]
    assert f'{r["baseline"]["ppl"]:.3f}' in html and f'{rec["summary"]["ppl"]:.3f}' in html
    for fmt_name, ex in rec["export"].items():
        v = ex["vllm"]
        assert fmt_name in html
        assert f'{v["served_ppl"]:.3f}' in html and f'{v["served_vs_real_ppl_pct"]:+.2f}%' in html
        assert v["engine"]["quant_kernel"] in html
    assert "vLLM deployment" in html and "Export and fake/real consistency" in html
    assert render_html(r, "zh").count("vLLM 部署") == 1


def test_gguf_report_has_the_llamacpp_table_and_server_numbers() -> None:
    r = load("gguf")
    html = render_html(r, "en")
    for t, x in r["llamacpp"]["results"].items():
        assert f"<b>{t.upper()}</b>" in html and f'{x["ppl"]:.3f}' in html
    q4 = r["llamacpp"]["results"]["q4_k"]
    assert f'{q4["serve"]["load"]["output_tokens_per_s"]:,.0f}' in html
    assert "llama.cpp deployment" in html


def test_sensitivity_section_and_worst_layers() -> None:
    r = load("full")
    html = render_html(r, "en")
    assert html.count("Sensitivity and mixed-precision suggestion") == len([x for x in r["recipes"].values() if x.get("sensitivity")])
    worst = min((x for x in r["recipes"]["W4A16"]["layers"] if x.get("out_sqnr_db") is not None), key=lambda x: x["out_sqnr_db"])
    assert worst["name"] in html


def test_absent_sections_are_left_out_not_faked() -> None:
    html = render_html(load("gguf"), "en")
    assert "vLLM deployment" not in html and "Sensitivity" not in html
    r = load("vllm")
    for rec in r["recipes"].values():
        rec.pop("export", None)
    html = render_html(r, "en")
    assert "Export and fake/real" not in html and "vLLM deployment" not in html


def test_failed_and_skipped_vllm_stages_say_why() -> None:
    r = load("vllm")
    ex = r["recipes"]["W4A16"]["export"]
    ex["awq"]["vllm"] = {"status": "failed", "error": "boom", "diagnosis": "GPU out of memory while loading"}
    ex["compressed-tensors"]["vllm"] = {"status": "skipped", "reason": "no CUDA device"}
    html = render_html(r, "en")
    assert "GPU out of memory while loading" in html and "no CUDA device" in html


def test_text_from_the_report_is_escaped() -> None:
    r = load("vllm")
    r["recipes"]["W4A16<img src=x>"] = r["recipes"].pop("W4A16")
    html = render_html(r, "en")
    assert "<img" not in html and "&lt;img src=x&gt;" in html


def test_only_llm_reports_and_known_languages() -> None:
    with pytest.raises(ValueError, match="LLM"):
        render_html({"kind": "vla"}, "en")
    with pytest.raises(ValueError, match="lang"):
        render_html(load("vllm"), "fr")


def test_helpers() -> None:
    assert fmt(None) == "–" and fmt(float("nan")) == "–" and fmt(1234.5, 1) == "1,234.5" and fmt(7) == "7"
    assert pct(None) == "–" and pct(0.0123) == "+0.01%" and pct(-1.5) == "-1.50%"
    assert nbytes(None) == "–" and nbytes(512) == "512 B" and nbytes(2048) == "2.0 KiB" and nbytes(5 * 2**30) == "5.0 GiB"
    assert bar_chart(["a"], [None], "t") == ""  # nothing to draw
    svg = bar_chart(["a", "b"], [1.0, 3.0], "t")
    assert svg.startswith("<svg") and svg.count("<rect") == 2 and "a: 1.00" in svg


def test_command_line(tmp_path: Path) -> None:
    import sys

    from mqde.report.html import main

    src = tmp_path / "report.json"
    src.write_text((FIX / "llm_vllm.json").read_text())
    old = sys.argv
    sys.argv = ["html", str(src), "--lang", "zh"]
    try:
        main()
    finally:
        sys.argv = old
    out = tmp_path / "report.zh.html"
    assert out.is_file() and "量化评估报告" in out.read_text(encoding="utf-8")
