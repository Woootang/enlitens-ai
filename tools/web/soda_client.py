"""Utility for querying Socrata (SODA) open data endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import backoff
import httpx


@backoff.on_exception(backoff.expo, httpx.RequestError, max_time=60)
def soda_query(
    base_url: str,
    dataset_id: str,
    *,
    where: Optional[str] = None,
    limit: int = 100,
    select: Optional[str] = None,
) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"$limit": limit}
    if where:
        params["$where"] = where
    if select:
        params["$select"] = select

    response = httpx.get(f"{base_url}/resource/{dataset_id}.json", params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []
