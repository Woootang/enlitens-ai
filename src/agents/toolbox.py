"""Central registry for reusable LangGraph tools."""

from __future__ import annotations

from typing import List

from langchain.tools import BaseTool

from tools.web import render_js_page_tool, scrape_url_tool, web_search_ddg_tool


def get_web_tools() -> List[BaseTool]:
    """Return the core web tooling suite."""

    return [
        web_search_ddg_tool,
        scrape_url_tool,
        render_js_page_tool,
    ]
