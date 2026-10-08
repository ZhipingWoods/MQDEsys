"""Project skills (.claude/skills): well-formed front matter, and every repository path they mention exists."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SKILLS = sorted(p for p in (ROOT / ".claude" / "skills").glob("*/SKILL.md"))
PATH_RE = re.compile(r"`((?:python|data|docs|scripts|examples|tests|cpp|web|\.claude)/[A-Za-z0-9_./-]+)")


def front_matter(p: Path) -> tuple[dict, str]:
    text = p.read_text()
    assert text.startswith("---\n"), f"{p}: no front matter"
    head, body = text[4:].split("\n---\n", 1)
    return yaml.safe_load(head), body


def test_there_are_skills() -> None:
    assert {p.parent.name for p in SKILLS} >= {"mqde-quantize-eval", "mqde-deploy-benchmark", "mqde-extend", "mqde-dev-gate"}


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.parent.name)
def test_front_matter(skill: Path) -> None:
    fm, body = front_matter(skill)
    assert fm["name"] == skill.parent.name
    desc = " ".join(str(fm["description"]).split())
    assert 80 < len(desc) < 1024
    assert re.search(r"\bUse (when|before)\b", desc), "the description must say when to use the skill"
    assert "Do NOT use" in desc, "the description must say when NOT to use it"
    assert fm["user-invocable"] is True
    assert str(fm["allowed-tools"]).strip()
    assert len(body.splitlines()) < 200, "keep SKILL.md short; move detail to references/"


@pytest.mark.parametrize("md", sorted((ROOT / ".claude" / "skills").rglob("*.md")), ids=lambda p: str(p.relative_to(ROOT)))
def test_mentioned_repository_paths_exist(md: Path) -> None:
    missing = []
    for m in PATH_RE.finditer(md.read_text()):
        rel = m.group(1).rstrip(".,:;)/")
        if "<" in rel or "*" in rel:
            continue
        if not (ROOT / rel).exists():
            missing.append(rel)
    assert not missing, f"{md.name} mentions paths that do not exist: {missing}"


@pytest.mark.parametrize("skill", SKILLS, ids=lambda p: p.parent.name)
def test_no_machine_specific_paths_or_addresses(skill: Path) -> None:
    text = "\n".join(p.read_text() for p in skill.parent.rglob("*.md"))
    assert not re.search(r"/cache/|/Users/|/home/[a-z]|\b(10|192\.168)(\.\d{1,3}){2,3}\b", text)
