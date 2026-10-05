"""Tests for mmorch.data: the benchmark order, the prompt reader, and the CSV reader and writers."""

from __future__ import annotations

import csv
import dataclasses
import gzip
import io
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from mmorch.data import BENCHMARKS, Prompt, iter_prompts, load_prompts, read_csv_gz, write_csv, write_dicts

# ---------------------------------------------------------------- benchmarks and prompts


def test_benchmarks_are_in_paper_order() -> None:
    assert BENCHMARKS == ("HumanEval", "MBPP", "GSM8K", "MATH", "TruthfulQA", "ARC", "HellaSwag", "MMLU-Pro")


def test_iter_prompts_yields_the_prompts_in_file_order(
    tiny_prompts: Path, tiny_prompt_records: list[dict[str, str]]
) -> None:
    prompts = iter_prompts(tiny_prompts)
    assert isinstance(prompts, Iterator)
    assert list(prompts) == [Prompt(r["qid"], r["benchmark"], r["question"]) for r in tiny_prompt_records]


def test_iter_prompts_decodes_utf8_text(tiny_prompts: Path, tiny_prompt_records: list[dict[str, str]]) -> None:
    question = next(p.question for p in iter_prompts(tiny_prompts) if p.qid == "HumanEval_2")
    assert question == tiny_prompt_records[1]["question"]
    assert not question.isascii()
    assert '"' in question
    assert "\n" in question


def test_iter_prompts_skips_no_line(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl.gz"
    with gzip.open(path, "wb") as f:
        f.write(b'{"qid": "MBPP_1", "benchmark": "MBPP", "question": "q"}\r\n\r\n')
    with pytest.raises(json.JSONDecodeError):
        list(iter_prompts(path))


def test_load_prompts_filters_by_benchmark_in_file_order(tiny_prompts: Path) -> None:
    assert [p.qid for p in load_prompts(tiny_prompts, "HumanEval")] == ["HumanEval_1", "HumanEval_2", "HumanEval_3"]
    assert [p.qid for p in load_prompts(tiny_prompts, "MBPP")] == ["MBPP_1"]
    assert load_prompts(tiny_prompts, "GSM8K") == []


def test_load_prompts_without_a_benchmark_returns_every_prompt(
    tiny_prompts: Path, tiny_prompt_records: list[dict[str, str]]
) -> None:
    assert load_prompts(tiny_prompts) == list(iter_prompts(tiny_prompts))
    assert [p.qid for p in load_prompts(tiny_prompts)] == [r["qid"] for r in tiny_prompt_records]


def test_prompts_are_frozen() -> None:
    prompt = Prompt("HumanEval_1", "HumanEval", "q")
    with pytest.raises(dataclasses.FrozenInstanceError):
        prompt.question = "changed"


# ---------------------------------------------------------------- reading CSV


def test_read_csv_gz_round_trips_a_crlf_csv_as_string_dicts(tmp_path: Path) -> None:
    path = tmp_path / "routing_keyword.csv.gz"
    error = 'Error code: 400 - {"a, b"}\nsecond line'
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "complexity", "latency_ms", "success", "error"])
        writer.writerow(["HumanEval_1", "LOW", "956.0410976409912", "1", ""])
        writer.writerow(["MBPP_7", "HIGH", "0.0", "0", error])
        writer.writerow(["ARC_3", "MEDIUM", "1.5", "1", "café"])
    with gzip.open(path, "rb") as f:
        assert f.read().count(b"\r\n") == 4

    assert read_csv_gz(path) == [
        {"qid": "HumanEval_1", "complexity": "LOW", "latency_ms": "956.0410976409912", "success": "1", "error": ""},
        {"qid": "MBPP_7", "complexity": "HIGH", "latency_ms": "0.0", "success": "0", "error": error},
        {"qid": "ARC_3", "complexity": "MEDIUM", "latency_ms": "1.5", "success": "1", "error": "café"},
    ]


# ---------------------------------------------------------------- writing CSV


def test_write_csv_bytes(tmp_path: Path) -> None:
    path = tmp_path / "verification.csv"
    rows = [
        ["Table 1 GSM8K: runs / success / failures", "6,595 / 5,924 / 671", "6,595 / 5,924 / 671", "yes"],
        ["Table 1 GSM8K: success (%)", 89.8, "89.8", "yes"],
        ["Résumé", 31019, 0, "NO"],
    ]
    assert write_csv(path, ["claim", "paper", "reproduced", "match"], rows) == path

    data = path.read_bytes()
    assert data == (
        b"claim,paper,reproduced,match\r\n"
        b'Table 1 GSM8K: runs / success / failures,"6,595 / 5,924 / 671","6,595 / 5,924 / 671",yes\r\n'
        b"Table 1 GSM8K: success (%),89.8,89.8,yes\r\n"
        b"R\xc3\xa9sum\xc3\xa9,31019,0,NO\r\n"
    )
    assert all(line.endswith(b"\r\n") for line in data.splitlines(keepends=True))


def test_write_dicts_renders_values_as_python_does(tmp_path: Path) -> None:
    fields = ["qid", "question", "latency_ms", "ttft_ms", "tokens_per_second", "success", "error"]
    rows: list[dict[str, Any]] = [
        {
            "qid": "HumanEval_2",
            "question": 'Prove that "x"\nholds',
            "latency_ms": 956.0410976409912,
            "ttft_ms": 0,
            "tokens_per_second": 0.0,
            "success": True,
            "error": None,
        },
        {
            "qid": "HumanEval_3",
            "question": "x",
            "latency_ms": 0,
            "ttft_ms": 0,
            "tokens_per_second": 0,
            "success": False,
            "error": "Error code: 400",
        },
    ]
    path = tmp_path / "HumanEval_keyword.csv"
    assert write_dicts(path, fields, (row for row in rows)) == path

    assert path.read_bytes() == (
        b"qid,question,latency_ms,ttft_ms,tokens_per_second,success,error\r\n"
        b'HumanEval_2,"Prove that ""x""\nholds",956.0410976409912,0,0.0,True,\r\n'
        b"HumanEval_3,x,0,0,0,False,Error code: 400\r\n"
    )
    reference = io.StringIO(newline="")
    writer = csv.DictWriter(reference, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    assert path.read_bytes() == reference.getvalue().encode("utf-8")


def test_write_dicts_follows_the_fieldnames_order(tmp_path: Path) -> None:
    path = write_dicts(tmp_path / "b.csv", ("benchmark", "qid"), [{"qid": "HE_0", "benchmark": "HumanEval"}])
    assert path.read_bytes() == b"benchmark,qid\r\nHumanEval,HE_0\r\n"


def test_write_dicts_rejects_unknown_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dict contains fields not in fieldnames"):
        write_dicts(tmp_path / "x.csv", ["a"], [{"a": 1, "b": 2}])


@pytest.mark.parametrize(
    ("write", "rows"),
    [(write_csv, [[1]]), (write_dicts, [{"a": 1}])],
    ids=["write_csv", "write_dicts"],
)
def test_writers_create_missing_parent_directories(
    tmp_path: Path, write: Callable[[Path, list[str], Any], Path], rows: Any
) -> None:
    path = tmp_path / "live" / "keyword" / "out.csv"
    assert write(path, ["a"], rows) == path
    assert path.read_bytes() == b"a\r\n1\r\n"
