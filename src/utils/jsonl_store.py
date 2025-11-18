#!/usr/bin/env python3
"""
Utility helpers for safely appending records to JSON Lines ledgers.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl_record(
    record: Dict[str, Any],
    ledger_path: Path,
    *,
    mirror_path: Optional[Path] = None,
    sort_keys: bool = True,
) -> None:
    """
    Append a JSON object as a single line to the ledger while guarding against partial writes.

    Args:
        record: The dictionary payload to persist.
        ledger_path: Target JSONL file path.
        mirror_path: Optional path to copy the resulting ledger to after append.
        sort_keys: Deterministic key ordering for readability.
    """

    _ensure_parent(ledger_path)

    serialized = json.dumps(record, ensure_ascii=False, sort_keys=sort_keys)
    tmp_path = ledger_path.with_suffix(ledger_path.suffix + ".tmp")

    # Write to a temporary file first
    with open(tmp_path, "w", encoding="utf-8") as tmp_file:
        tmp_file.write(serialized)
        tmp_file.write("\n")
        tmp_file.flush()
        os.fsync(tmp_file.fileno())

    # Append atomically to the ledger
    with open(ledger_path, "ab") as ledger_file, open(tmp_path, "rb") as tmp_file:
        shutil.copyfileobj(tmp_file, ledger_file)
        ledger_file.flush()
        os.fsync(ledger_file.fileno())

    tmp_path.unlink(missing_ok=True)

    if mirror_path:
        _ensure_parent(mirror_path)
        shutil.copy2(ledger_path, mirror_path)

