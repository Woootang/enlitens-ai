"""Helpers for persisting web intelligence outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def append_jsonl(file_name: str, records: Iterable[Mapping]) -> None:
    """Append mapping objects to a JSONL file under the data directory."""

    path = DATA_DIR / file_name
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def write_snapshot(file_name: str, payload: Mapping) -> None:
    """Write a full JSON snapshot to disk, overwriting previous content."""

    path = DATA_DIR / file_name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
