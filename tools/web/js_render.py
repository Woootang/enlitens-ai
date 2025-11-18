"""Playwright-based renderer for JS-heavy pages."""

from __future__ import annotations

import asyncio
from typing import Optional

from playwright.async_api import async_playwright
from pydantic import BaseModel, HttpUrl

from .allowlist import is_host_allowed
from .extractors import extract_main_text


class RenderRequest(BaseModel):
    url: HttpUrl
    wait_ms: int = 3000


async def _render_html(req: RenderRequest) -> Optional[str]:
    host = req.url.host.lower()
    if not is_host_allowed(host):
        raise PermissionError(f"Host '{host}' is not in the JS render allowlist")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-dev-shm-usage", "--no-sandbox"],
        )
        page = await browser.new_page(
            viewport={"width": 1280, "height": 720},
            user_agent="EnlitensBrowserTool/0.1 (+https://enlitens.org)",
        )
        await page.goto(str(req.url), wait_until="networkidle")
        await page.wait_for_timeout(req.wait_ms)
        html = await page.content()
        await browser.close()
        return html


def render_js_page(req: RenderRequest) -> Optional[str]:
    """Synchronously render a JS-heavy page."""

    return asyncio.run(_render_html(req))


class RenderResult(BaseModel):
    url: HttpUrl
    title: str | None
    text: str


def render_and_extract(req: RenderRequest) -> Optional[RenderResult]:
    html = render_js_page(req)
    if not html:
        return None
    extracted = extract_main_text(html, url=str(req.url))
    if not extracted or not extracted.get("text"):
        return None
    return RenderResult(
        url=req.url,
        title=extracted.get("title"),
        text=extracted["text"],
    )
