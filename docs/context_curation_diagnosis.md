# Context Curation Failure Diagnosis – November 2025

## Summary

The three-document validation run on November 10th repeatedly surfaced `"revise"`
verdicts from the `ContextVerificationAgent`. Despite the retry loop, the
pipeline advanced to the supervisor stage and crashed with
`AttributeError: 'NoneType' object has no attribute 'ainvoke'`.

## Findings

1. **Verifier Verdicts Remain `"revise"`**
   - For documents `2021-43536-001` and `2023-67353-007`, the verifier reported:
     - “Personas and health brief do not directly connect to the source paper's focus…”
     - “The content does not include specific statistical data from the source paper.”
     - “The current content does not fully reflect Liz's voice…”
   - The health brief prompt is still delivering broad city context rather than
     specific, paper-linked mechanisms with Liz-tone statistics, so the verifier
     legitimately rejects the bundle.

2. **Supervisor Still Runs After Verification Failure**
   - `MultiAgentProcessor.process_document` raises a `RuntimeError` when the
     final context status is not `"pass"`, but the exception is caught by the
     surrounding `try/except` and only logged as a warning
     ([process_multi_agent_corpus.py:660-692](../process_multi_agent_corpus.py)).
   - After logging, the function sets `context["curated_context"] = None` and
     continues to execute the supervisor workflow, leading to the
     `NoneType.ainvoke` crash when `SupervisorAgent` expects a populated
     workflow graph.

3. **Verification Loop Exhausts Retries Without Adjusting Criteria**
   - `ContextCuratorAgent` allows up to three passes, but the retry feedback only
     tweaks persona and health prompts; the guardrails in
     `ContextVerificationAgent` still require explicit statistical grounding and
     Liz voice evidence, which the current prompts do not guarantee.

## Impact

- Context curation fails cleanly but the exception handling masks the failure,
  so downstream processing attempts to operate on empty context and crashes.
- Repeated retries waste GPU cycles because the synthesis prompts never inject
  the verifier’s required evidence (stats, voice receipts, paper linkage).

## Next Steps

- Harden `process_multi_agent_corpus.process_document` so a non-`"pass"` status
  aborts the document cleanly.
- Update verifier prompts/criteria to acknowledge adjacent alignments when the
  bridge is spelled out, and instrument the health brief/persona prompts to
  surface Liz-tone statistics explicitly.
- Regenerate the health digest with persona-aware slices so the health brief has
  concrete, citeable numbers for the verifier.


