# External Data Enrichment Plan

## Goals

- Pull locally grounded signals (third places, resources, environmental cues)
  to tighten persona → neighbourhood alignment.
- Enrich topic alignment and health briefs with factual snippets from
  authoritative sources (Wikimedia, Google datasets).
- Keep the integrations optional and rate-limit friendly so the pipeline can
  run offline when necessary.

## Modules Added

1. `src/integrations/google_maps_context.py`
   - Async helper around Google Maps Places text/nearby/search endpoints.
   - Intended to feed `ContextCuratorAgent` with third-place inventories keyed
     by persona neighbourhoods.
   - Reads `GOOGLE_MAPS_API_KEY` from the environment.

2. `src/integrations/wikimedia_enterprise.py`
   - Async client for the Wikimedia Enterprise API (On-demand + project list).
   - Supports username/password authentication and token reuse.
   - Keys pulled from environment variables:
     `WIKIMEDIA_ENTERPRISE_USERNAME`, `WIKIMEDIA_ENTERPRISE_PASSWORD`,
     optional `WIKIMEDIA_ENTERPRISE_ACCESS_TOKEN`.

## Next Wiring Steps

### Persona Context

- When we load persona metadata, map any `locality` or ZIP into a lat/long.  
  The maps client can then surface:
  - Libraries, sensory-friendly venues, third places within 2km.
  - School district / festival data using keyword filters.
- Cache results per neighbourhood so repeated docs do not re-query.

### Topic Alignment

- Use the Wikimedia client inside `TopicAlignmentBuilder` to fetch the latest
  summary/infobox for key entities (conditions, interventions, researchers).  
  Feed a condensed version into the verifier so it can check for factual gaps.

### Health Brief

- Attach `key_statistics` discovered via Maps/Wikimedia into the health brief
  prompt (`HealthReportSynthesizerAgent`) to guarantee numeric receipts.

## Testing Checklist

1. Set the relevant API credentials and run:
   ```bash
   python -m asyncio scripts/test_kimi_client.py  # (for Moonshot)
   python - <<'PY'
   import asyncio
   from src.integrations.google_maps_context import GoogleMapsContextClient
   async def main():
       async with GoogleMapsContextClient() as client:
           result = await client.text_search("sensory-friendly cafe in St. Louis")
           print(result["results"][:2])
   asyncio.run(main())
   PY
   ```
2. Repeat for Wikimedia:
   ```bash
   python - <<'PY'
   import asyncio
   from src.integrations.wikimedia_enterprise import WikimediaEnterpriseClient
   async def main():
       async with WikimediaEnterpriseClient() as client:
           await client.authenticate()
           article = await client.get_article("Neurodiversity")
           print(article.get("results", [])[:1])
   asyncio.run(main())
   PY
   ```
3. Once verified, hook the data into `data_profiles` and log latency,
   per-request token usage, and cache hit rate.

