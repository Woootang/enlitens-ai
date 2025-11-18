"""Content extraction helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from bs4 import BeautifulSoup
import trafilatura


def extract_main_text(html: str, url: str | None = None) -> Optional[Dict[str, Any]]:
    """Return the main text body and metadata using Trafilatura."""

    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        output_format="txt",
    )
    if not text:
        return None

    metadata = trafilatura.extract_metadata(html, default_url=url)
    return {
        "text": text,
        "title": getattr(metadata, "title", None) if metadata else None,
        "metadata": metadata.as_dict() if metadata else {},
    }


def extract_by_selector(html: str, selector: str) -> list[str]:
    """Grab text nodes matching a CSS selector via BeautifulSoup."""

    soup = BeautifulSoup(html, "lxml")
    elements = soup.select(selector)
    return [el.get_text(separator=" ", strip=True) for el in elements if el]
