"""VLM calibration / evaluation samples: deterministic, disjoint, and the ground-truth answers are really in the picture."""

from __future__ import annotations

import numpy as np
import pytest

from mqde.data.vlm import SHAPES, VLMSample, load_split, synthetic


def test_synthetic_is_deterministic_and_has_question_answer_and_an_image() -> None:
    a, b = synthetic(6, seed=3), synthetic(6, seed=3)
    assert [s.question for s in a] == [s.question for s in b] and [s.answer for s in a] == [s.answer for s in b]
    assert all(np.array_equal(x.images[0], y.images[0]) for x, y in zip(a, b, strict=True))
    for s in a:
        assert isinstance(s, VLMSample) and "?" in s.question and s.answer
        assert len(s.images) == 1 and s.images[0].shape == (448, 448, 3) and s.images[0].dtype == np.uint8


def test_different_seeds_give_different_samples() -> None:
    assert [s.question + s.answer for s in synthetic(8, seed=0)] != [s.question + s.answer for s in synthetic(8, seed=1)]


def color_of(s: VLMSample, where: tuple[int, int]) -> tuple[int, ...]:
    return tuple(int(v) for v in s.images[0][where[0], where[1]])


def test_the_answer_is_read_off_the_picture_not_made_up() -> None:
    """Colour questions: the named shape's centre pixel has that colour. Count questions: that many shapes were drawn."""
    kinds = set()
    for s in synthetic(60, seed=5):
        kinds.add(s.meta["kind"])
        m = s.meta
        if m["kind"] == "color":
            cy, cx = m["centers"][m["target"]]
            got = color_of(s, (cy, cx))
            assert got == m["palette"][s.answer], (s.question, got, s.answer)
        elif m["kind"] == "count":
            assert int(s.answer) == len(m["centers"]), s.question
        elif m["kind"] == "exists":
            assert (s.answer == "yes") == (m["asked"] in m["present"]), s.question
    assert kinds == {"color", "count", "exists"}, "all three question kinds are produced"


def test_shapes_are_distinguishable_in_the_question() -> None:
    assert len(set(SHAPES)) == len(SHAPES) >= 3
    qs = [s.question for s in synthetic(40, seed=1) if s.meta["kind"] == "color"]
    assert qs and all(any(sh in q for sh in SHAPES) for q in qs)


def test_calibration_and_evaluation_samples_never_share_an_image() -> None:
    cal, ev = load_split("synthetic", 12, "synthetic", 12, seed=0)
    assert len(cal) == len(ev) == 12
    cal_keys = {s.images[0].tobytes() for s in cal}
    assert not [s for s in ev if s.images[0].tobytes() in cal_keys]


def test_unknown_source_is_a_clear_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises((FileNotFoundError, ValueError)):
        load_split(str(tmp_path / "no-such-dataset"), 2, "synthetic", 2)
