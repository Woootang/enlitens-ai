# Web Access Policy

This document defines the guardrails for the new web tooling stack.

## Robots.txt and RSL
- Fetch `robots.txt` for every domain before scraping.
- Obey `Disallow` rules and any declared crawl-delay.
- Treat AI-licensing indicators (e.g., RSL, meta tags) as opt-out signals and skip those resources unless we have written permission.

## Allowlist
- All automated requests must target domains listed in the engineering allowlist (maintained in `tools/web/allowed_domains.yml`).
- Any new domain requires approval from the AI lead and clinical lead.

## Authentication & Paywalls
- Do not bypass paywalls, login walls, or captchas.
- Abort requests that return captcha challenges, error 403/429, or other block indicators.

## Rate Limiting
- Respect a default ceiling of 2 concurrent requests per domain and a minimum 250ms delay between requests.
- Use the shared DiskCache layer to avoid re-fetching the same resource inside 24 hours unless content is explicitly known to change faster.

## PII and Sensitive Content
- Limit scraping to public, non-user-generated pages (government, health agencies, reputable news, public program directories).
- Never collect personally identifiable information or private forum posts.

## Logging & Monitoring
- Log domain, status code, response time, and whether the response came from cache.
- Surface repeated block events to the engineering channel for manual review.

## Manual Overrides
- Any exception workflow must be documented in an incident ticket and reviewed by the AI lead and compliance officer before deployment.
