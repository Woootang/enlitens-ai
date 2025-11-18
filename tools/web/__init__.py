"""Web tooling package."""

from __future__ import annotations

try:
    from .tool_wrappers import (
        render_js_page_tool,
        scrape_url_tool,
        web_search_ddg_tool,
    )

    __all__ = [
        "web_search_ddg_tool",
        "scrape_url_tool",
        "render_js_page_tool",
    ]
except ImportError:  # pragma: no cover - optional dependency (langchain)
    render_js_page_tool = None
    scrape_url_tool = None
    web_search_ddg_tool = None
    __all__ = []
