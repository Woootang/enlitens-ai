# Pilot Run Playbook

**Objective:** Verify the upgraded single-model pipeline end-to-end, capture quality metrics, and note follow-up refinements.

## 1. Pre-flight
- `source venv/bin/activate`
- Ensure `./scripts/start_vllm_qwen3_14b_128k.sh` is running (port 8000).
- Clear prior logs: `rm -f logs/enlitens_complete_processing.log`.
- Confirm dashboard server is running (`flask --app dashboard/server.py run`).

## 2. Launch Pilot Run
- Run a focused document test:  
  `python process_multi_agent_corpus.py --input-dir enlitens_corpus/test_input --output-file enlitens_knowledge_base/test_output.json`
- Optional: full corpus once pilot succeeds.

## 3. Monitor in Real Time
- Dashboard ▸ Progress tab
  - `Last Quality` target ≥ 0.65
  - `Validation warnings` & `Clinician review checklist`
- Dashboard ▸ Alerts tab for fresh warnings.
- `tail -f logs/enlitens_complete_processing.log` for agent-level context.

## 4. Post-run Checklist
- Review `review_checklist` and `validation_warnings` in the dashboard or final JSON.
- Spot-check marketing/blog outputs for:
  - Persona alignment (segments + GA4 queries)
  - St. Louis localization & compliance tag
- Verify `enlitens_knowledge_base/test_output.json` appended the `compliance_message`.

## 5. Backlog Items to Capture
- Any repeated warnings surfaced via dashboard metrics.
- Manual voice edits required despite new prompts.
- Suggestions from clinicians during review loop.

Log findings in [`docs/backlog_refinements.md`](backlog_refinements.md).


## 6. Web Intelligence Snapshot (Daily)
- Run `PYTHONPATH=. scripts/run_web_intel_snapshot.py` (cron-friendly) to refresh local news, policy, resource, and research caches.
- Inspect JSONL drops in `data/` (`local_news.jsonl`, `policy_updates.jsonl`, etc.) for anomalies before main processing.
- Clear caches with `rm -rf cache/http/*` if a domain changes robots permissions.


