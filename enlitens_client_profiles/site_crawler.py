"""Minimal site crawler for enlitens.com to prime persona prompts."""

from __future__ import annotations

from collections import deque
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Set

import httpx
from bs4 import BeautifulSoup


class SiteCrawler:
    def __init__(self, base_url: str, *, limit: int = 50, cache_dir: Optional[Path] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.limit = limit
        self.visited: Set[str] = set()
        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def crawl(self, *, urls: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, object]]:
        pages: Dict[str, Dict[str, object]] = {}
        client = httpx.Client(timeout=12.0, headers={"User-Agent": "EnlitensPersonaCrawler/1.1"})

        try:
            if urls:
                iterable = list(urls)[: self.limit]
                for url in iterable:
                    if len(pages) >= self.limit:
                        break
                    payload = self._fetch_url(client, url)
                    if payload:
                        pages[url] = payload
                        time.sleep(0.4)
                return self._finalise_pages(pages)

            queue: deque[str] = deque([self.base_url])
            while queue and len(pages) < self.limit:
                url = queue.popleft()
                if url in self.visited:
                    continue
                payload = self._fetch_url(client, url)
                if not payload:
                    continue
                pages[url] = payload

                for link in payload.get("links", []):
                    if link not in self.visited and link.startswith(self.base_url):
                        queue.append(link)

                time.sleep(0.4)
        finally:
            client.close()

        return self._finalise_pages(pages)

    def _fetch_url(self, client: httpx.Client, url: str) -> Optional[Dict[str, object]]:
        if url in self.visited:
            return None
        self.visited.add(url)

        try:
            response = client.get(url)
            response.raise_for_status()
        except Exception:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        title_tag = soup.find("title")
        headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"]) if h.get_text(strip=True)]
        text = soup.get_text(separator=" ")
        flattened = " ".join(text.split())
        header_summary = " | ".join(headings[:5])
        summary = f"{title_tag.get_text(strip=True) if title_tag else ''}\n{header_summary}\n\n{flattened}".strip()

        if self.cache_dir:
            cache_file = self.cache_dir / f"{len(self.visited)}.txt"
            try:
                cache_file.write_text(summary, encoding="utf-8")
            except Exception:
                pass

        links: Set[str] = set()
        for link in soup.find_all("a", href=True):
            href = link["href"].split("#")[0]
            if not href:
                continue
            if href.startswith("http") and href.startswith(self.base_url):
                links.add(href)
            elif href.startswith("/"):
                links.add(self.base_url + href)

        return {"title": title_tag.get_text(strip=True) if title_tag else "", "headings": headings, "summary": summary, "links": list(links)}

    def _finalise_pages(self, pages: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
        for value in pages.values():
            value.pop("links", None)
        return pages

