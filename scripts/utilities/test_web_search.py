#!/usr/bin/env python
"""Quick smoke test for the DDG search tool."""

from tools.web.web_search_ddg import WebSearchRequest, ddg_text_search


if __name__ == "__main__":
    req = WebSearchRequest(query="autistic burnout support st louis", max_results=3)
    results = ddg_text_search(req)
    for idx, res in enumerate(results, start=1):
        print(f"[{idx}] {res.title}\n    {res.url}\n    {res.snippet}\n")
