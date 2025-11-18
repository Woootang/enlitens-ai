# KV Cache Compression – Feasibility Notes

## Motivation

- Context curation + supervisor runs peak at ~9k curated tokens per doc, which
  drives GPU VRAM pressure when batching future runs.
- KVzip-style importance scoring would let us keep the vLLM deployment but cut
  cache size by 50–70% without sacrificing accuracy.

## Proposed Approach

1. **Prefill Hook**  
   - Wrap vLLM's `SamplingParams` so we can capture the attention cache after
     the prefill step.  
   - Use the chunk-based scoring routine from the KVzip paper to compute
     head-level importance.

2. **Eviction Strategy**  
   - Start with context-independent scoring (one-off profiling) to avoid run-time
     overhead.  
   - Store head budgets per model under `models/kv_budget/{model}.json`.

3. **Runtime Toggle**  
   - Add `LLM_KVZIP_ENABLED=1` environment flag.  
   - Inject into `VLLMClient.generate_response` so we trim caches before
     decoding.

4. **Instrumentation**  
   - Extend `src/utils/gpu_memory_manager.GPUMemoryManager` to record
     cache size, eviction ratios, attention latency.  
   - Emit stats to dashboard for side-by-side comparison.

## Experiment Plan

1. **Baseline**  
   - Process three documents on vLLM (Qwen 14B) with compression disabled.  
   - Capture GPU memory, latency, verifier pass rate.

2. **Compression Sweep**  
   - Apply 90%, 70%, 50% retention ratios.  
   - Monitor tokens/sec, latency, verifier/warnings to ensure no regression.

3. **Moonshot Comparison**  
   - Repeat on remote Moonshot provider to quantify benefit of local vs. API
     usage (Moonshot removes local KV entirely).

## Known Risks

- The current vLLM build may not expose prefill caches without patching.  
  We might need to fork or plug into the open-source KVzip reference.
- Adjacent long-context prompts (voice guide + personas) may be sensitive to
  aggressive pruning; verifier metrics will reveal if Liz voice drifts.

## Next Steps

- Prototype `kvzip_scoring.py` with PyTorch-based importance scoring.  
- Add optional cache compression hook to `VLLMClient`.  
- Automate benchmarks via `scripts/benchmark_kv_cache.py`.

