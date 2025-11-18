"""
Lightweight integration helpers for Google Maps' new AI/MCP features.

These utilities provide a thin async wrapper around the Maps Places APIs so
agents can pull neighbourhood cues, third places, and resource metadata when
curating context.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

import httpx

DEFAULT_BASE_URL = "https://maps.googleapis.com/maps/api/place"


class GoogleMapsContextClient:
    """Async helper for Google Maps Places endpoints."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set.")
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)

    async def text_search(
        self,
        query: str,
        *,
        location: Optional[str] = None,
        radius_meters: Optional[int] = None,
        types: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        """Run a text-based search for third places or resources."""

        params: Dict[str, Any] = {"query": query, "key": self.api_key}
        if location:
            params["location"] = location
        if radius_meters:
            params["radius"] = radius_meters
        if types:
            params["type"] = ",".join(types)

        response = await self.client.get(f"{self.base_url}/textsearch/json", params=params)
        response.raise_for_status()
        return response.json()

    async def nearby_search(
        self,
        *,
        location: str,
        radius_meters: int = 2000,
        keyword: Optional[str] = None,
        ranking: str = "prominence",
    ) -> Dict[str, Any]:
        """Find locations near a lat/long pair."""

        params: Dict[str, Any] = {
            "location": location,
            "radius": radius_meters,
            "rankby": ranking,
            "key": self.api_key,
        }
        if keyword:
            params["keyword"] = keyword

        response = await self.client.get(f"{self.base_url}/nearbysearch/json", params=params)
        response.raise_for_status()
        return response.json()

    async def place_details(
        self,
        place_id: str,
        *,
        fields: Optional[Iterable[str]] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return extended metadata for a place (address, website, opening hours…)."""

        params: Dict[str, Any] = {"place_id": place_id, "key": self.api_key}
        if fields:
            params["fields"] = ",".join(fields)
        if language:
            params["language"] = language

        response = await self.client.get(f"{self.base_url}/details/json", params=params)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "GoogleMapsContextClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

