"""The prompt record, the canonical benchmark order, and the project's only CSV reader and writers.

Every CSV goes through write_csv or write_dicts, so the byte format is defined in one place: utf-8, the csv
module's default excel dialect (CRLF line endings, minimal quoting) and newline='' so that nothing else
translates line endings. Values are written as Python renders them: pre-formatted strings, raw ints, True and
False, None as '', int 0 as '0' and float 0.0 as '0.0'.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Paper and figure order: per-benchmark rows, means and x ticks.
BENCHMARKS: Final[tuple[str, ...]] = (
    "HumanEval",
    "MBPP",
    "GSM8K",
    "MATH",
    "TruthfulQA",
    "ARC",
    "HellaSwag",
    "MMLU-Pro",
)


@dataclass(frozen=True, slots=True)
class Prompt:
    """One benchmark prompt from data/prompts.jsonl.gz."""

    qid: str
    benchmark: str
    question: str


def iter_prompts(path: Path) -> Iterator[Prompt]:
    """Yield the prompts of a gzip JSON-lines file in file order, decoding it as UTF-8.

    Every line must be a JSON object with qid, benchmark and question; no line is skipped.
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            yield Prompt(d["qid"], d["benchmark"], d["question"])


def load_prompts(path: Path, benchmark: str | None = None) -> list[Prompt]:
    """Return the prompts in file order, only those of one benchmark when it is given."""
    return [p for p in iter_prompts(path) if benchmark is None or p.benchmark == benchmark]


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    """Read a gzip CSV with a header row into string dicts, in file order."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> Path:
    """Write a header and rows with csv.writer, creating the parent directories, and return path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def write_dicts(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> Path:
    """Write rows with csv.DictWriter in fieldnames order, creating the parent directories, and return path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(rows)
    return path
