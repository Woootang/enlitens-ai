# Enlitens Client Profile Pipeline

This package generates deeply contextual client personas before the main PDF processing
pipeline runs. Profiles draw from:

- `enlitens_knowledge_base/intakes.txt` (verbatim enquiries)
- `enlitens_knowledge_base/transcripts.txt` (founder voice)
- `enlitens_knowledge_base/st_louis_health_report.pdf`
- Additional knowledge assets in `enlitens_knowledge_base/`
- Curated geography data for the greater St. Louis region
- A live snapshot of `https://www.enlitens.com/` (provided via manual site export)

## Layout

```
enlitens_client_profiles/
├── cache/                  # prompt/keyed cache for reproducibility
├── logs/                   # telemetry logs (also pushed to monitoring dashboard)
├── profiles/               # generated persona JSON documents
├── config.py               # runtime configuration
├── data_ingestion.py       # loads raw artifacts into structured bundles + analytics snapshot
├── analytics.py            # GA4 + Search Console integrations
├── knowledge_keeper.py     # builds knowledge graph/context index
├── foundation_builder.py   # converts graph into persona scaffolds
├── deep_research.py        # outbound search agent + cache
├── orchestrator.py         # supervisors to run the agent chain
├── llm.py                  # thin wrapper around Ollama/vLLM client
├── prompt_builder.py       # templated prompts for persona creation
├── profile_builder.py      # converts bundle + research into a persona
├── profile_pipeline.py     # batch pipeline + manifest & similarity management
├── schema.py               # Pydantic models for output validation
├── stl_geography.py        # curated municipal/neighborhood data
├── telemetry.py            # monitoring + dashboard integration
├── similarity.py           # cosine/jaccard deduplication helpers
├── matching.py             # runtime persona matching + AI context helpers
└── generate_profiles.py    # CLI entry point
```

## Usage

```bash
python3 -m enlitens_client_profiles.generate_profiles --count 10 --config-dump
```

Flags:

- `--model`: override default LLM model defined in settings.
- `--monitor-url`: push telemetry to an alternate dashboard endpoint.
- `--no-cache`: force regeneration of profiles even if prompt cache exists.
- `--allow-duplicates`: bypass similarity guardrails (use for research only).
- `--config-dump`: print the resolved configuration before running.
- `--google-credentials`: explicit path to the Google service-account JSON (defaults to
  `$GOOGLE_APPLICATION_CREDENTIALS` or `enlitens_client_profiles/credentials/service_account.json`).
- `--ga-property`: GA4 property id used for analytics lookups (numbers only).
- `--gsc-site`: Search Console site URL (must match the property you granted access to).
- `--analytics-lookback`: number of days (>=7) of analytics data to include.

Each profile is written as JSON inside `profiles/` and recorded in
`profiles_manifest.json` to avoid duplicates. Telemetry logs are written to
`logs/profile_pipeline.log` and optionally streamed to the monitoring server.

## Schema Overview

Personas conform to `ClientProfileDocument` with the following notable sections:

- **meta** – profile id, persona name/tagline, attribute tags, source documents.
- **demographics** – age range, gender/pronouns, occupation, education, locality.
- **neurodivergence_profile** – ND identities, diagnosis journey, language preferences.
- **clinical_challenges** – presenting issues, nervous-system pattern, mood/trauma notes.
- **adaptive_strengths & executive_function** – reframed strengths, friction points, coping hacks.
- **sensory_profile** – sensitivities, seeking behaviours, regulation methods.
- **goals_motivations & pain_points_barriers** – therapy/life goals, motivations, internal/systemic barriers.
- **cultural_context & local_environment** – community anchors, commute, local stressors, safe spaces.
- **support_system, tech_media_habits, therapy_preferences** – relational map, channel preferences, what works/doesn’t in therapy.
- **quotes & narrative** – struggle/hope quotes plus a Liz-tone narrative summary with highlight bullets.
- **marketing_copy & seo_brief** – website/email/social snippets and keyword plans tailored to the persona.
- **resources & analytics** – recommended offers/referrals, coverage notes, similarity fingerprint.

## Similarity & Governance

- A cosine + Jaccard similarity index (`cache/similarity_index.json`) enforces the
  `< 0.41` overlap rule. Potential duplicates are written to `cache/conflicts/` and
  surfaced as `profile_similarity_flagged` telemetry events.
- The CLI option `--allow-duplicates` records near-duplicates without blocking, but
  still logs the similarity report for review.
- Dashboard updates (`/api/stats`) now expose persona totals, top localities,
  neuro-identity counts, and conflict counts. Check the “Persona Insights” card on the
  monitoring UI for live telemetry.

## Matching & AI Context

- `matching.load_persona_library(Path)` loads the library as Pydantic models.
- `matching.match_personas(...)` returns scored matches for intake narratives + attribute tags.
- `matching.build_ai_context(persona)` provides a compact dictionary ready for AI prompts
  or assessment generators.

## Multi-Agent Flow

The orchestrator now runs a fixed chain for every persona:

1. **Knowledge Keeper** ingests intakes, transcripts, health report, knowledge assets, GA4
   and Search Console snapshots into an on-disk knowledge graph (`cache/knowledge_graph.graphml`).
2. **Foundation Builder** evaluates the graph to produce a scaffold of demographics,
   locality hypotheses, family/occupation clues, and a research to-do list.
3. **Deep Research Agent** issues outbound queries (Serper + Brave supported) to fill
   the scaffold gaps. The research cache is persisted in `cache/research/`.
4. **Writer (LLM)** receives the foundation, analytics and external research summaries to
   generate the full persona JSON.
5. **Validator & Archivist** enforce the `<0.41` similarity rule, persist JSON to
   `profiles/`, update manifests, and emit telemetry.

The pipeline refuses to write a persona if the deep research agent produced no results.
Check `logs/profile_pipeline.log` for `persona_research_missing` events if API access
fails or quotas are exhausted.

## Google Analytics / Search Console Setup

1. Create a Google service account in the Cloud Console and download the JSON key.
2. Add the service account email as **Viewer/Analyst** on your GA4 property and
   as **Full user** (or Restricted) in Search Console for the desired site.
3. Place the JSON file at `enlitens_client_profiles/credentials/service_account.json`
   (ignored by git) or set the environment variable `GOOGLE_APPLICATION_CREDENTIALS`
   to the path.
4. Provide the GA property id and GSC site via CLI flags or env vars:

```bash
export GA4_PROPERTY_ID=123456789
export GSC_SITE_URL=https://www.enlitens.com/
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
```

## External Research Keys

- `SERPER_API_KEY` – Google SERP proxy with 2,500 monthly free calls.
- `BRAVE_API_KEY` – Brave Search API (free tier available).
- `FIRECRAWL_API_KEY` (optional) – Add if you want page scraping support.

Set whichever keys you have as environment variables before running the pipeline.
The agent will prefer Serper, fall back to Brave, and log a warning if no tools are
available (personas will be skipped in that case).

## Operations & Refresh Cadence

- Generated personas append to `logs/profile_pipeline.log` with `profile_created` events.
- Similarity vectors and attribute tags are cached for reuse; run `python3 -m
  compileall enlitens_client_profiles` after schema changes if needed.
- See `/enlitens_client_profiles/cache/profile_generation_manifest.json` for the
  tracking manifest. The monitoring dashboard displays total personas and the timestamp of the last similarity index sync.

