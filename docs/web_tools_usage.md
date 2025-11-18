# Web Tools Usage Guide

This document summarizes the new web intelligence stack and how to operate it safely.

## Core Packages
- `tools/web/web_search_ddg.py` – DDG/DDGS search wrapper with retry + caching.
- `tools/web/scrape_url.py` – HTTP fetch + Trafilatura extraction.
- `tools/web/js_render.py` – Playwright renderer for JS-only pages (allowlisted hosts).
- `tools/web/feeds.py` – RSS/Atom helpers and default health feeds.
- `tools/web/openalex_client.py` – Scholarly search via OpenAlex.
- `tools/web/tool_wrappers.py` – LangGraph-compatible tool bindings.

## Cache & Rate Limiting
- Disk cache stored in `cache/http/` via `diskcache`.
- Default TTL = 24h to avoid duplicate hits; adjust via `fetch_url(..., ttl=seconds)`.
- Backoff: exponential retry for HTTP errors up to 60s total.
- Per-request delay: 200ms sleep to throttle bursts.

## Allowlist & Robots
- Approved domains stored in `tools/web/allowed_domains.yml`.
- JS rendering refuses hosts not on the allowlist.
- `tools/web/http_client.fetch_url` checks robots.txt (see `robots_guard` module) before downloading a page.

## Storage
- JSONL snapshots written to `data/*.jsonl` for auditing (e.g., `local_news.jsonl`, `policy_updates.jsonl`).
- Helpers available in `src/utils/web_data_store.py` (`append_jsonl`, `write_snapshot`).

## Background Runs
- Use `scripts/run_web_intel_snapshot.py` to collect a fresh batch outside the main document pipeline (suitable for cron).

## Safety Checklist
- Update allowlist + policy doc when adding domains.
- Respect `robots.txt` rejections (logged at DEBUG).
- Monitor `data/*.jsonl` sizes and rotate if needed.
- Use Playwright only when simple fetch fails; start browser pod separately if running at scale.

