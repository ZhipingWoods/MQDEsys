"""Calibration text sources (mqde.data.text): the built-in wikitext-2, and a local file or directory the user names."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from mqde.data import text as T


class CharTok:
    """One token per character: a window's ids say exactly which text it was cut from."""

    def __call__(self, text: str, return_tensors: str = "pt"):
        return SimpleNamespace(input_ids=torch.tensor([[ord(c) for c in text]]))


def windows_text(batches: list[torch.Tensor]) -> list[str]:
    return ["".join(chr(int(i)) for i in b[0]) for b in batches]


def corpus(tmp_path, name="a.txt", body="abcdefghij" * 40):
    p = tmp_path / name
    p.write_text(body)
    return p


def test_the_default_source_is_wikitext_and_unchanged(monkeypatch):
    monkeypatch.setattr(T, "wikitext_text", lambda split: "wiki " * 400 if split == "train" else "no")
    got = T.calib_batches(CharTok(), n_samples=3, seq_len=50, seed=0)
    assert len(got) == 3 and all(b.shape == (1, 50) for b in got)
    assert all(w in "wiki " * 400 for w in windows_text(got))
    # same seed, same windows as before this feature existed: the default must not drift
    ids = torch.tensor([ord(c) for c in "wiki " * 400])
    starts = torch.randint(0, ids.numel() - 50 - 1, (3,), generator=torch.Generator().manual_seed(0))
    assert [b[0].tolist() for b in got] == [ids[s : s + 50].tolist() for s in starts.tolist()]


def test_a_text_file_is_the_calibration_data(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "wikitext_text", lambda split: pytest.fail("wikitext must not be read for a file"))
    p = corpus(tmp_path, body="0123456789" * 30)
    info: dict = {}
    got = T.calib_batches(CharTok(), n_samples=4, seq_len=40, seed=1, source=str(p), info=info)
    assert all(w in "0123456789" * 30 for w in windows_text(got))
    assert info["files"] == 1 and info["chars"] == 300 and info["truncated"] is False and info["tokens"] == 300
    assert info["dataset"] == str(p)


def test_same_seed_same_windows_other_seed_other_windows(tmp_path):
    p = corpus(tmp_path, body="".join(chr(65 + i % 26) * (1 + i % 7) for i in range(400)))
    a = T.calib_batches(CharTok(), 5, 30, seed=0, source=str(p))
    b = T.calib_batches(CharTok(), 5, 30, seed=0, source=str(p))
    c = T.calib_batches(CharTok(), 5, 30, seed=1, source=str(p))
    assert windows_text(a) == windows_text(b)
    assert windows_text(a) != windows_text(c)


def test_jsonl_takes_the_text_field_or_a_bare_string_and_skips_records_without_text(tmp_path):
    p = tmp_path / "d.jsonl"
    lines = [json.dumps({"text": "alpha " * 30}), json.dumps({"id": 1}), json.dumps("beta " * 30), "", json.dumps({"text": ""})]
    p.write_text("\n".join(lines))
    info: dict = {}
    T.calib_batches(CharTok(), 2, 20, source=str(p), info=info)
    assert info["chars"] == len("alpha " * 30) + len("beta " * 30) + len("\n\n")


def test_a_bad_jsonl_line_names_the_line(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"text": "ok " * 40}) + "\n{not json\n")
    with pytest.raises(ValueError, match=r"d\.jsonl:2"):
        T.calib_batches(CharTok(), 1, 10, source=str(p))


def test_json_list_of_strings_or_objects(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(["x" * 50, "y" * 50]))
    b.write_text(json.dumps([{"text": "p" * 50}, {"text": "q" * 50}, {"other": 1}]))
    for p, n in ((a, 100), (b, 100)):
        info: dict = {}
        T.calib_batches(CharTok(), 1, 10, source=str(p), info=info)
        assert info["chars"] == n + 2


def test_a_json_that_is_not_a_list_is_refused(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps({"text": "x" * 100}))
    with pytest.raises(ValueError, match="expected a JSON list"):  # not just "list": the pytest tmp path contains that word
        T.calib_batches(CharTok(), 1, 10, source=str(p))


def test_parquet_text_column(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    p = tmp_path / "d.parquet"
    pq.write_table(pa.table({"text": ["row one " * 20, "", "row two " * 20], "n": [1, 2, 3]}), p)
    info: dict = {}
    T.calib_batches(CharTok(), 1, 10, source=str(p), info=info)
    assert info["chars"] == len("row one " * 20) + len("row two " * 20) + 2


def test_parquet_without_a_text_column_says_which_columns_it_has(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    p = tmp_path / "d.parquet"
    pq.write_table(pa.table({"content": ["x" * 100]}), p)
    with pytest.raises(ValueError, match="content"):
        T.calib_batches(CharTok(), 1, 10, source=str(p))


def test_a_directory_is_read_recursively_in_a_fixed_order_and_other_files_are_ignored(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.txt").write_text("B" * 60)
    (tmp_path / "a.md").write_text("A" * 60)
    (tmp_path / "sub" / "c.jsonl").write_text(json.dumps({"text": "C" * 60}))
    (tmp_path / "weights.bin").write_bytes(b"\x00" * 1000)
    (tmp_path / "pic.png").write_bytes(b"\x89PNG")
    loaded = T.load_calib_text(str(tmp_path))
    assert loaded.files == 3
    assert loaded.text == "A" * 60 + "\n\n" + "B" * 60 + "\n\n" + "C" * 60  # sorted by relative path: a.md, b.txt, sub/c.jsonl


def test_a_missing_path_is_not_found():
    with pytest.raises(FileNotFoundError, match="nope"):
        T.calib_batches(CharTok(), 1, 10, source="/nonexistent/nope")


def test_a_directory_or_file_with_nothing_readable_lists_what_is_supported(tmp_path):
    (tmp_path / "x.bin").write_bytes(b"123")
    with pytest.raises(ValueError, match=r"\.txt.*\.parquet"):
        T.calib_batches(CharTok(), 1, 10, source=str(tmp_path))
    with pytest.raises(ValueError, match=r"\.txt.*\.parquet"):
        T.calib_batches(CharTok(), 1, 10, source=str(tmp_path / "x.bin"))


def test_too_little_text_for_one_window_is_an_error_that_says_how_much_there_is(tmp_path):
    p = corpus(tmp_path, body="z" * 100)
    with pytest.raises(ValueError, match=r"100 tokens.*512"):
        T.calib_batches(CharTok(), 1, 512, source=str(p))
    T.calib_batches(CharTok(), 1, 98, source=str(p))  # needs seq_len + 2 tokens: 100 is exactly enough
    with pytest.raises(ValueError):
        T.calib_batches(CharTok(), 1, 99, source=str(p))


def test_a_huge_corpus_is_cut_at_the_limit_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "MAX_CALIB_CHARS", 250)
    (tmp_path / "a.txt").write_text("a" * 200)
    (tmp_path / "b.txt").write_text("b" * 200)
    (tmp_path / "c.txt").write_text("c" * 200)
    info: dict = {}
    got = T.calib_batches(CharTok(), 3, 20, source=str(tmp_path), info=info)
    assert info["truncated"] is True and info["chars"] == 250
    assert info["files"] == 2, "a.txt whole, b.txt cut; c.txt contributed nothing and is not counted"
    assert not any("c" in w for w in windows_text(got)), "the files after the limit are not read"


def test_undecodable_bytes_do_not_fail_the_job(tmp_path):
    p = tmp_path / "d.txt"
    p.write_bytes(b"good text " * 20 + b"\xff\xfe" + b" more text" * 20)
    got = T.calib_batches(CharTok(), 2, 30, source=str(p))
    assert len(got) == 2


def test_builtin_name_is_stable():
    assert T.BUILTIN_CALIB == "wikitext-2-raw-v1:train"
    assert T.TEXT_SUFFIXES == (".txt", ".md", ".jsonl", ".json", ".parquet")


def test_suffixes_are_matched_case_insensitively(tmp_path):
    (tmp_path / "UPPER.TXT").write_text("U" * 60)
    (tmp_path / "Mixed.Jsonl").write_text(json.dumps({"text": "M" * 60}))
    loaded = T.load_calib_text(str(tmp_path))
    assert loaded.files == 2 and loaded.text == "M" * 60 + "\n\n" + "U" * 60
    assert T.load_calib_text(str(tmp_path / "UPPER.TXT")).text == "U" * 60


def test_a_file_with_nothing_in_it_is_not_counted_as_read(tmp_path):
    (tmp_path / "a_empty.txt").write_text("   \n\n  ")
    (tmp_path / "b.txt").write_text("real " * 30)
    (tmp_path / "c_blank.jsonl").write_text(json.dumps({"text": ""}) + "\n" + json.dumps({"id": 2}))
    loaded = T.load_calib_text(str(tmp_path))
    assert loaded.files == 1 and loaded.text == "real " * 30


# ---------------------------------------------------------------- text pasted into the form (calibration.custom_samples)
def test_pasted_samples_are_the_calibration_text_and_nothing_else_is_read(monkeypatch):
    monkeypatch.setattr(T, "wikitext_text", lambda split: pytest.fail("wikitext must not be read for pasted samples"))
    info: dict = {}
    got = T.calib_batches(CharTok(), 4, 40, seed=1, texts=["0123456789" * 15, "abcdefghij" * 15], info=info)
    pool = "0123456789" * 15 + "\n\n" + "abcdefghij" * 15
    assert all(w in pool for w in windows_text(got))
    assert info["dataset"] == "custom_samples" and info["samples"] == 2 and info["chars"] == len(pool) and info["tokens"] == len(pool)
    assert info["truncated"] is False


def test_pasted_samples_win_over_a_named_source(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "wikitext_text", lambda split: pytest.fail("not read"))
    p = corpus(tmp_path, body="FILE" * 100)
    got = T.calib_batches(CharTok(), 2, 20, source=str(p), texts=["pasted " * 30])
    assert not any("F" in w for w in windows_text(got)), "the file was read although samples were pasted"


def test_blank_pasted_samples_are_dropped_and_an_all_blank_paste_is_an_error():
    info: dict = {}
    T.calib_batches(CharTok(), 1, 10, texts=["", "   \n", "real text " * 10, "\t"], info=info)
    assert info["samples"] == 1
    with pytest.raises(ValueError, match="custom_samples"):
        T.calib_batches(CharTok(), 1, 10, texts=["", "  ", "\n"])
    with pytest.raises(ValueError, match="custom_samples"):
        T.calib_batches(CharTok(), 1, 10, texts=[])


def test_too_little_pasted_text_says_how_much_there_is():
    with pytest.raises(ValueError, match=r"custom_samples.*40 tokens.*512"):
        T.calib_batches(CharTok(), 1, 512, texts=["x" * 40])


def test_pasted_text_is_cut_at_the_same_limit_as_a_corpus(monkeypatch):
    monkeypatch.setattr(T, "MAX_CALIB_CHARS", 250)
    info: dict = {}
    T.calib_batches(CharTok(), 2, 20, texts=["a" * 200, "b" * 200, "c" * 200], info=info)
    assert info["truncated"] is True and info["chars"] == 250 and info["samples"] == 2


def test_pasted_samples_must_be_strings():
    with pytest.raises(ValueError, match="custom_samples"):
        T.calib_batches(CharTok(), 1, 10, texts=["ok " * 20, 5])  # type: ignore[list-item]
