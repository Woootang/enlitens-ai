# Runtime Status — 30 Oct 2025

## Snapshot
- **Environment**: workstation RTX 3090 (24 GB VRAM), Python 3.10, vLLM 0.6.3, Docling enhanced PDF pipeline
- **Command used**: `./stable_run.sh`
- **Latest run**: 2025-10-30 16:07–16:13 CT

## High-Priority Issues

### 1. Docling enhanced extraction exhausts GPU memory
- **Symptoms**: `torch.OutOfMemoryError` raised repeatedly while Docling RT-DETR vision stack processes PDFs (see `logs/enlitens_complete_processing.log`, lines 20–550).
- **Impact**: Every PDF triggers 3 failed “enhanced extraction” attempts before falling back to plain text extraction. Adds large retry loops and delays.
- **Likely cause**: RT-DETR-v2 model + full-resolution page batches exceed remaining VRAM because vLLM keeps ~14.5 GB resident even when idle.
- **Suggested mitigation**:
  - Reduce Docling batch/page size or force CPU execution (`DOC_LING_DEVICE=cpu` for layout model).
  - Lower vLLM cache footprint (`--gpu-memory-utilization` <= 0.6) or stop secondary model to free VRAM.
  - Explore lighter OCR/layout backends for the first pass.

### 2. LangGraph workflow crashes on shared `stage` channel
- **Error**: `InvalidUpdateError: At key 'stage': Can receive only one value per step. Use an Annotated key to handle multiple values.` (see log lines 706–728, 1151–1169).
- **Impact**: Supervisor aborts each document once an agent (e.g., `science_extraction`) retries and writes to `stage` again.
- **Next action**: Adjust LangGraph node outputs to write to unique keys or wrap the channel in `Annotated[LastValue[str], Addable]`.

### 3. vLLM rejecting structured generation requests
- **Symptom**: Rapid series of `HTTP 400 Bad Request` / `HTTP 500 Internal Server Error` responses after agents request JSON output (see log lines 578–706, 1090–1260).
- **Repro**: Any agent calling `OllamaClient.generate_structured_response` with `model=/home/.../mistral-7b-instruct`.
- **Likely cause**:
  - Guardrails expect JSON-with-grammar; prompts exceed max tokens or fail schema; local model not fine-tuned for structured output.
  - 500s observed during earlier load without errors in server logs after context truncated.
- **Suggested fix**:
  - Reduce prompt size (strip large “Strict Rules” block for extraction agents).
  - Provide `json_schema` via `guided_decoding` or disable structured enforcement temporarily.
  - Validate responses using streaming guard rather than entire JSON schema.

### 4. Monitoring dashboard aesthetics incomplete
- **User feedback**: Needs fully branded UI, interactive panels, accessible theme.
- **Status**: CSS tokens live in `style.css`; monitoring React/Vue components still default.

## Secondary Observations
- vLLM local model loads successfully with `--max-model-len 8192`; GPU cache usage stays under 10% during steady state.
- Supervisor successfully initializes all agents; only fails when science_extraction retries and hits structured-response + stage issues.
- Fallback text extraction succeeds quickly once GPU OOM triggers.

## Next Steps (Proposed)
1. Patch LangGraph supervisor to separate `stage` channel per agent or mark as appendable.
2. Relax structured generation to plain completions until we have validated grammar prompts.
3. Reconfigure Docling to CPU/low-memory mode while vLLM active, or run extractions before LLM spin-up.
4. Address UI design separately (see `docs/ui-redesign.md`).

## References
- Full log: `logs/enlitens_complete_processing.log`
- vLLM foreground run: `~/.venv_vllm/bin/python -m vllm.entrypoints.openai.api_server ...`
