# Persona Governance & Refresh SOP

## Review Cadence

- **Quarterly**: rotate through the top 50 “high engagement” personas and confirm
  demographics, nervous-system framing, and resource links. Update
  `meta.version` when substantive edits occur.
- **Semi-annual**: full-library sweep to reconcile telemetry (locality coverage,
  neuro-identity representation) and retire or merge low-usage personas.
- **On-Demand Triggers**:
  - telemetry `profile_similarity_flagged`
  - client feedback indicating mismatch or missing representation
  - regional data shifts (e.g., new public-health report)

## Anti-Bias Checklist (Each Review)

- Language remains neurodiversity-affirming (no deficit framing).
- Cultural/context references align with lived language from intakes.
- Quotes and narratives continue to sound like Liz Wooten’s voice.
- Personas representing priority populations (BIPOC, LGBTQ+, rural, metro-east,
  late-diagnosed adults, parents/caregivers) remain present and accurate.

## Similarity Workflow

1. Run `python3 -m enlitens_client_profiles.generate_profiles --count X`.
2. Inspect `cache/conflicts/` if any were flagged. Merge or differentiate before
   rerunning.
3. Commit updated personas with message summarising adjustments.

## Dashboard Monitoring

- Persona totals, conflict counts, and top localities surface on the monitoring
  dashboard. Investigate spikes in similarity conflicts or localization gaps.
- `similarity_index.json` timestamp ensures index refresh. Run `rm
  enlitens_client_profiles/cache/similarity_index.json` to rebuild from scratch
  if needed.

## Documentation

- Update `README.md` when schema fields change.
- Record major persona merges/retirements in `logs/persona_changes.md` (create if
  absent) for historical auditing.


