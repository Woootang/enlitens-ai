# Backlog: Post-Pilot Refinements

Collect follow-up items after each pilot run so we can schedule incremental improvements.

| Category | Observation | Proposed Action |
|----------|-------------|-----------------|
| Educational Depth | Sections with `< 0.6` coverage / depth score. | Add targeted few-shots for weak sections, retry document with additional priming. |
| Analytics Coverage | GA4/GSC snapshot missing or empty. | Verify service-account access; consider caching daily analytics JSON. |
| Persona Alignment | `persona_segments` not referenced in final copy. | Add dynamic prompt inserts showcasing segment quotes. |
| Compliance | Missing “Educational content only...” tag in creative outputs. | Validate via `review_checklist`; update prompts or template where missing. |
| Dashboard | Additional metrics desired (e.g., segment token counts). | Extend `/api/metrics` to surface desired signals. |

> Append dated rows as the team records new findings.

