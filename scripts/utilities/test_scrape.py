#!/usr/bin/env python
"""Quick smoke test for the scrape_url helper."""

from tools.web.scrape_url import ScrapeUrlRequest, scrape_url


if __name__ == "__main__":
    req = ScrapeUrlRequest(url="https://health.mo.gov/")
    result = scrape_url(req)
    if not result:
        raise SystemExit("Failed to scrape test URL")
    print(f"Title: {result.title}")
    snippet = "\n".join(result.text.splitlines()[:10])
    print(snippet)
