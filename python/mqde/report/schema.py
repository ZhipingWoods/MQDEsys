"""Validate a job's report.json against ``report.schema.json`` (T-0806).

A small validator for the subset of JSON Schema the report schema uses (type, required, properties, items,
additionalProperties, enum, minimum, minLength, minItems, maxItems, allOf / if / then / not), so no extra dependency is
needed. The schema file is plain draft-07 and also works with the ``jsonschema`` package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("report.schema.json")
_TYPES: dict[str, tuple[type, ...]] = {"object": (dict,), "array": (list,), "string": (str,), "boolean": (bool,),
                                       "integer": (int,), "number": (int, float), "null": (type(None),)}


def load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _is(v: Any, t: str) -> bool:
    if isinstance(v, bool) and t in ("integer", "number"):
        return False
    return isinstance(v, _TYPES[t])


def check(v: Any, s: dict[str, Any], path: str = "$") -> list[str]:
    """Error strings (empty when valid); each starts with the JSON path of the offending value."""
    errs: list[str] = []
    want = s.get("type")
    if want is not None and not any(_is(v, t) for t in ([want] if isinstance(want, str) else want)):
        return [f"{path}: expected {want}, got {type(v).__name__}"]
    if "enum" in s and v not in s["enum"]:
        errs.append(f"{path}: {v!r} is not one of {s['enum']}")
    if isinstance(v, (int, float)) and not isinstance(v, bool) and "minimum" in s and v < s["minimum"]:
        errs.append(f"{path}: {v} is below the minimum {s['minimum']}")
    if isinstance(v, str) and len(v) < s.get("minLength", 0):
        errs.append(f"{path}: empty string")
    if isinstance(v, list):
        if len(v) < s.get("minItems", 0) or len(v) > s.get("maxItems", len(v)):
            errs.append(f"{path}: {len(v)} items, expected {s.get('minItems', 0)}..{s.get('maxItems', 'any')}")
        for i, x in enumerate(v):
            errs += check(x, s["items"], f"{path}[{i}]") if "items" in s else []
    if isinstance(v, dict):
        errs += [f"{path}: missing required key '{k}'" for k in s.get("required", []) if k not in v]
        props = s.get("properties", {})
        extra = s.get("additionalProperties")
        for k, x in v.items():
            sub = props.get(k, extra if isinstance(extra, dict) else None)
            if sub is not None:
                errs += check(x, sub, f"{path}.{k}")
    for sub in s.get("allOf", []):
        cond = sub.get("if")
        if cond is not None and not check(v, cond, path) and "then" in sub:
            errs += check(v, sub["then"], path)
    if "not" in s and not check(v, s["not"], path):
        errs.append(f"{path}: matches a forbidden schema")
    return errs


def validate_report(report: Any) -> list[str]:
    return check(report, load_schema())
