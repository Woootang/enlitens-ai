from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

USAGE_FILE = Path("logs") / "ai_usage.json"
DEFAULT_LIMITS = {
    "gemini_cli": int(os.getenv("GEMINI_DAILY_LIMIT", "2000")),
    "deep_research": int(os.getenv("DEEP_RESEARCH_DAILY_LIMIT", "250")),
    "codex_cli": int(os.getenv("CODEX_LOCAL_LIMIT", "1500")),
    "openai_factcheck": int(os.getenv("OPENAI_FACTCHECK_LIMIT", "500")),
}

OPENAI_PRICING_USD_PER_MTOKEN = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-search-preview": {"input": 0.25, "output": 0.90},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
}

DEFAULT_OPENAI_PRICING = {"input": 1.25, "output": 5.0}  # fallback (per 1M tokens)


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _load_usage() -> Dict[str, Any]:
    if not USAGE_FILE.exists():
        return {}
    try:
        with USAGE_FILE.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        # Corrupted file – back up and start fresh
        backup = USAGE_FILE.with_suffix(".json.bak")
        USAGE_FILE.rename(backup)
        return {}
    except Exception:
        return {}


def _save_usage(payload: Dict[str, Any]) -> None:
    USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with USAGE_FILE.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def record_usage(
    tool: str,
    count: int = 1,
    metadata: Optional[Dict[str, Any]] = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
    cost_usd: float = 0.0,
    model: Optional[str] = None,
) -> None:
    """
    Increment usage counters for a given tool.

    Parameters
    ----------
    tool:
        Identifier such as "gemini_cli" or "deep_research".
    count:
        Number of requests to add (default 1).
    metadata:
        Optional payload describing the event. Recent events are capped automatically.
    """

    if not tool:
        return

    payload = _load_usage()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    tool_record = payload.setdefault(today, {}).setdefault(
        tool, {"count": 0, "events": [], "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    )
    tool_record["count"] = int(tool_record.get("count", 0)) + max(1, int(count))
    timestamp = _utc_now_iso()
    tool_record["last_used"] = timestamp
    if tokens_in:
        tool_record["tokens_in"] = int(tool_record.get("tokens_in", 0)) + int(tokens_in)
    if tokens_out:
        tool_record["tokens_out"] = int(tool_record.get("tokens_out", 0)) + int(tokens_out)
    if cost_usd:
        tool_record["cost_usd"] = round(float(tool_record.get("cost_usd", 0.0)) + float(cost_usd), 6)
    if model:
        tool_record["model"] = model
    if metadata:
        tool_record["events"].append({"timestamp": timestamp, "metadata": metadata})
        tool_record["events"] = tool_record["events"][-50:]
    _save_usage(payload)


@dataclass
class UsageSnapshot:
    count: int
    limit: int
    remaining: int
    percent: float
    last_used: Optional[str]
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "limit": self.limit,
            "remaining": self.remaining,
            "percent": self.percent,
            "last_used": self.last_used,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 4),
            "model": self.model,
        }


def _summary_for_tool(day_payload: Dict[str, Any], tool: str) -> UsageSnapshot:
    record = day_payload.get(tool, {})
    count = int(record.get("count", 0))
    limit = DEFAULT_LIMITS.get(tool, 0)
    remaining = max(0, limit - count) if limit else 0
    percent = round((count / limit) * 100, 2) if limit else 0.0
    return UsageSnapshot(
        count=count,
        limit=limit,
        remaining=remaining,
        percent=percent,
        last_used=record.get("last_used"),
        tokens_in=int(record.get("tokens_in", 0)),
        tokens_out=int(record.get("tokens_out", 0)),
        cost_usd=float(record.get("cost_usd", 0.0)),
        model=record.get("model"),
    )


def get_usage_summary() -> Dict[str, Any]:
    """
    Return today's usage summary for the dashboard/API.
    """

    payload = _load_usage()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    day_record = payload.get(today, {})

    summary = {}
    for tool in DEFAULT_LIMITS.keys():
        summary[tool] = _summary_for_tool(day_record, tool).to_dict()

    total_requests = sum(item["count"] for item in summary.values())
    total_cost = round(sum(item["cost_usd"] for item in summary.values()), 4)
    return {
        "date": today,
        "totals": {
            "requests": total_requests,
            "limits": sum(item["limit"] for item in summary.values()),
            "cost_usd": total_cost,
            "estimated_cost_ceiling": math.ceil(total_cost),
        },
        "tools": summary,
    }


def compute_openai_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    model_key = model or ""
    rates = OPENAI_PRICING_USD_PER_MTOKEN.get(model_key, DEFAULT_OPENAI_PRICING)
    input_rate = rates["input"] / 1_000_000
    output_rate = rates["output"] / 1_000_000
    return round(tokens_in * input_rate + tokens_out * output_rate, 6)


def record_openai_usage(
    tool: str,
    *,
    model: str,
    tokens_in: int,
    tokens_out: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    cost = compute_openai_cost(model, tokens_in, tokens_out)
    meta = metadata or {}
    meta["model"] = model
    record_usage(
        tool,
        metadata=meta,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        model=model,
    )

