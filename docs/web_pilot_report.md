# Web Intelligence Pilot Report – Initial Dry Run

**Date:** 2025-11-13

## Summary
- Ran `scripts/run_web_intel_snapshot.py` to exercise all web-connected agents (news, policy, resources, events, research, myth, community impact, symptom trends).
- CLI smoke tests executed:
  - `scripts/test_web_search.py`
  - `scripts/test_scrape.py`
  - Playwright render against `https://health.mo.gov/`
  - RSS check (CDC feed – returned empty payload, possibly rate-limited)
  - OpenAlex scholarly query
- Validation test suite (`pytest`) extended with web safety tests covering robots.txt and allowlist enforcement.

## Outstanding Items
- RSS feeds to confirm in production (CDC feed returned zero rows during dry run; monitor when scheduling).
- Schedule `scripts/run_web_intel_snapshot.py` via cron/systemd for continuous refresh.
- Coordinate clinician review for new warning/review checklist outputs once pilot corpus is processed.

## Next Steps
1. Run full pipeline on staging subset and capture dashboard screenshots (quality, warnings, review checklist, compliance message).
2. Log any manual follow-ups in `docs/backlog_refinements.md`.
3. Prepare stakeholder summary after clinician QA sign-off.
