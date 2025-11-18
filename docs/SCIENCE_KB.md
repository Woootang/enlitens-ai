# Science-First Knowledge Base

Enlitens now captures every science extraction run in an append-only knowledge base that lives under `data/knowledge_base/`. This tier is purpose-built for the new *science-first* pipeline mode (`PIPELINE_MODE=science_only` or `--pipeline-mode science_only`) so we can defer educational/marketing fan-out until we have a dense research corpus.

## Directory layout

- `data/knowledge_base/science_entries.jsonl` — line-delimited `ScienceExtractionRecord` objects (one per document + revision).
- `data/knowledge_base/science_manifest.json` — fast lookup index keyed by `document_id` with revision, title, hashes, and source text path.
- `data/knowledge_base/text/<document_id>.md` — full extracted markdown per paper for downstream RAG.

Both JSON files are safe to tail/stream (append-only). Each entry includes `source_sha256` + `text_sha256` so downstream jobs can confirm provenance before re-processing.

## Schema highlights

`src/models/enlitens_schemas.py` now defines the following Pydantic models:

- `ScienceStudyMetadata`: study identifiers, DOI/PMID, ingestion timestamp, source hashes, pipeline mode, and revision counter.
- `ScienceExtractionRecord`: bundles structured `ResearchContent`, optional `ClinicalContent` translation, extracted entities, quote/stat snapshots, translation summary, and health digest context.
- `ScienceQuote` / `ScienceTranslationSummary`: lightweight helpers for quote/stat capture and clinician/client-ready summaries.

Refer to the model docstrings for the full field list. Any consumer can call `ScienceExtractionRecord.model_validate(payload)` for strict validation.

## Writing & reading entries

`ScienceKnowledgeWriter` (`src/utils/knowledge_writer.py`) centralizes persistence:

```python
from src.utils.knowledge_writer import ScienceKnowledgeWriter

writer = ScienceKnowledgeWriter()
records = writer.load_records(limit=5)   # Returns validated ScienceExtractionRecord objects
manifest = writer.read_manifest()        # Quick metadata lookup for dashboards/agents
```

- `append_record(record)` updates the JSONL + manifest atomically and increments revisions if the PDF/text hash changes.
- `write_text_blob(document_id, text)` persists full-text markdown that RAG pipelines can mount later.

## Using the KB in future phases

1. **Educational/SEO batches** — Instead of re-reading PDFs, load slices of `science_entries.jsonl` filtered by manifest metadata (keywords, ingestion range, etc.) and feed those into the existing agents in batch mode.
2. **Monitoring + dashboards** — The manifest includes per-document hashes and timestamps, so we can chart ingestion velocity and detect stale entries.
3. **Search/RAG** — Hydrate the manifest into Qdrant or any vector DB by embedding the saved text files + structured fields. Because every entry keeps the same `document_id`, agents can request the corresponding education/marketing pass later.

> Tip: `view_kb.py` can be updated to read from `ScienceKnowledgeWriter.load_records()` to inspect the new JSONL directly from the CLI.

With this structure, we can continue to run the lean science pipeline 24/7, accumulate verified research/translation data, and then schedule heavier creative phases once we have the coverage, without ever re-extracting PDFs.

