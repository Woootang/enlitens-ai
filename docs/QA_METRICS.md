## QA Metrics Logger

- Run `python process_multi_agent_corpus.py <pdf_dir> output.json` (same command you already use).
- After each document finishes, the processor now appends a JSON line to `logs/qa_metrics.jsonl`.
- Fields captured per run:
  - `document_id`, `timestamp`
  - `quality_score`, `confidence_score`
  - `quality_breakdown` (per agent metrics from ValidationAgent)
  - `validation_warnings`
  - `processing_time_seconds`
- Open the log with `jq` or import into a spreadsheet to spot coverage gaps.
- Delete/reset the log by removing the file before a new pilot: `rm logs/qa_metrics.jsonl`

This gives you a running QA ledger whenever you pilot the pipeline on new research PDFs. No extra flags required.

