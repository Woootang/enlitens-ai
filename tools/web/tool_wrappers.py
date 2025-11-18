"""LangChain/LangGraph tool wrappers for web utilities."""

from __future__ import annotations

from typing import List, Optional

from langchain.tools import tool

from .js_render import RenderRequest, RenderResult, render_and_extract
from .scrape_url import ScrapeUrlRequest, ScrapeUrlResult, scrape_url
from .web_search_ddg import WebSearchRequest, WebSearchResult, ddg_text_search


@tool("web_search_ddg", args_schema=WebSearchRequest)
def web_search_ddg_tool(req: WebSearchRequest) -> List[WebSearchResult]:
    """Search DuckDuckGo/DDGS and return normalized results."""

    return ddg_text_search(req)


@tool("scrape_url", args_schema=ScrapeUrlRequest)
def scrape_url_tool(req: ScrapeUrlRequest) -> Optional[ScrapeUrlResult]:
    """Fetch a URL and return readable text if available."""

    return scrape_url(req)


@tool("render_js_page", args_schema=RenderRequest)
def render_js_page_tool(req: RenderRequest) -> Optional[RenderResult]:
    """Render a JS-heavy page (allowlisted hosts only) and extract text."""

    return render_and_extract(req)
