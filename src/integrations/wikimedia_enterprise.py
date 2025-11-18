"""
Client utilities for the Wikimedia Enterprise API.

Supports logging in with username/password to obtain JWT tokens and running
on-demand article lookups that downstream agents can fold into topic
alignment or persona enrichment.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

AUTH_ENDPOINT = "https://auth.enterprise.wikimedia.com/v1/login"
API_BASE = "https://api.enterprise.wikimedia.com/v2"


class WikimediaEnterpriseClient:
    """Async wrapper for the Wikimedia Enterprise REST endpoints."""

    def __init__(
        self,
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        access_token: Optional[str] = None,
        refresh_token: Optional[str] = None,
        base_url: str = API_BASE,
        timeout: float = 15.0,
    ) -> None:
        self.username = username or os.environ.get("WIKIMEDIA_ENTERPRISE_USERNAME")
        self.password = password or os.environ.get("WIKIMEDIA_ENTERPRISE_PASSWORD")
        self.access_token = access_token or os.environ.get("WIKIMEDIA_ENTERPRISE_ACCESS_TOKEN")
        self.refresh_token = refresh_token or os.environ.get("WIKIMEDIA_ENTERPRISE_REFRESH_TOKEN")
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)

    async def authenticate(self) -> None:
        """Fetch new access and refresh tokens using username/password credentials."""

        if not self.username or not self.password:
            raise ValueError(
                "Wikimedia Enterprise username/password are required for authentication."
            )

        response = await self.client.post(
            AUTH_ENDPOINT,
            json={"username": self.username, "password": self.password},
        )
        response.raise_for_status()
        payload = response.json()
        self.access_token = payload.get("access_token")
        self.refresh_token = payload.get("refresh_token")

    def _auth_headers(self) -> Dict[str, str]:
        if not self.access_token:
            raise RuntimeError("Access token missing; call authenticate() first.")
        return {"Authorization": f"Bearer {self.access_token}"}

    async def get_article(
        self,
        title: str,
        *,
        project: str = "enwiki",
        limit: int = 1,
    ) -> Dict[str, Any]:
        """Fetch the live version of a single article."""

        payload = {
            "filters": [{"field": "is_part_of.identifier", "value": project}],
            "limit": limit,
        }
        response = await self.client.post(
            f"{self.base_url}/articles/{title}",
            headers=self._auth_headers(),
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def list_projects(self) -> Dict[str, Any]:
        response = await self.client.get(
            f"{self.base_url}/projects",
            headers=self._auth_headers(),
        )
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "WikimediaEnterpriseClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

