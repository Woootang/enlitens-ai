"""Utilities for managing the web domain allowlist."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Set

ALLOWLIST_PATH = Path("tools/web/allowed_domains.yml")


@lru_cache(maxsize=1)
def load_allowed_hosts() -> Set[str]:
    if not ALLOWLIST_PATH.exists():
        return set()
    hosts: Set[str] = set()
    for line in ALLOWLIST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            line = line[2:]
        hosts.add(line.lower())
    return hosts


def is_host_allowed(host: str) -> bool:
    hosts = load_allowed_hosts()
    return not hosts or host.lower() in hosts
