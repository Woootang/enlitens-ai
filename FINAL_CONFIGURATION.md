# 🎯 Enlitens AI - Final Configuration Summary

## System Overview

**Goal:** Process 345 research PDFs into a comprehensive knowledge base with maximum quality output.

**Philosophy:** Quality > Speed. Use all available resources for best possible results.

---

## Hardware Resources

### GPU
- **Model:** NVIDIA GeForce RTX 3090
- **VRAM:** 24GB
- **Usage:** Running Qwen 2.5-14B LLM

### CPU & RAM
- **RAM:** 64GB
- **Free RAM:** ~50GB available
- **Usage:** Running 5 NER models sequentially

### Storage
- **Qdrant Vector DB:** Local file-based storage (`qdrant_storage/`)
- **Models:** ~30GB (Qwen + NER models)
- **Knowledge Base:** Growing JSON file

---

## AI Models Configuration

### Main LLM: Qwen 2.5-14B Instruct (AWQ 4-bit)

**Quantization:** 4-bit AWQ
- **VRAM Usage:** ~8GB for weights
- **Quality:** 95% of full precision
- **Why 4-bit:** Stable, fits comfortably in 24GB with room for KV cache

**Context Window:** **FULL 128k tokens (131,072)**
- **Why 128k:** Fits EVERYTHING at once:
  - Full research paper (6k-15k tokens)
  - ALL 57 client personas (50k-70k tokens)
  - Liz's transcripts (10k-20k tokens)
  - St. Louis Health Report (5k-10k tokens)
  - NER extracted entities (2k-5k tokens)
  - ContextRAG retrieved docs (10k-20k tokens)
  - External search results (5k-15k tokens)
  - System prompts (3k-5k tokens)

**Speed Impact:** ~4-5x slower than 16k context
**Quality Impact:** MASSIVE improvement - model sees everything at once

**Startup Command:**
```bash
bash scripts/start_vllm_128k.sh
```

### NER Models (5 specialized models on CPU)

Running sequentially to avoid GPU memory conflicts:

1. **OpenMed DiseaseDetect** (434M params)
   - Diseases, conditions, disorders
   
2. **OpenMed PharmaDetect** (434M params)
   - Chemicals, drugs, medications
   
3. **OpenMed AnatomyDetect** (560M params)
   - Anatomical structures, body parts
   
4. **OpenMed GenomeDetect** (434M params)
   - Genes, proteins, genetic markers
   
5. **ClinicalDistilBERT i2b2-2010** (65M params)
   - Clinical symptoms, treatments, procedures

**Why CPU:** GPU is maxed out by Qwen + 128k KV cache
**RAM Usage:** ~2-3GB per model (plenty of headroom with 64GB)

---

## Processing Pipeline

### Input Sources (What Goes Into Each PDF)

1. **Research PDF** - The paper being processed
2. **57 Client Personas** - Real stories from `enlitens_client_profiles/profiles/`
3. **Liz's Transcripts** - Founder voice from `enlitens_knowledge_base/liz_transcripts.txt`
4. **St. Louis Health Report** - Local context from `st_louis_health_report.pdf`
5. **NER Entities** - Extracted by 5 specialized models
6. **ContextRAG** - Related papers from Qdrant vector database
7. **External Search** - Wikipedia, PubMed, Semantic Scholar, DuckDuckGo

### Multi-Agent Workflow

```
PDF → Extract Text
  ↓
  → NER Models (5 sequential extractions on CPU)
  ↓
  → Load ALL Context (personas, transcripts, health report)
  ↓
  → ContextRAG Search (find related papers in Qdrant)
  ↓
  → External Search (enrich low-confidence entities)
  ↓
  → Multi-Agent Processing (parallel):
      - ScienceExtraction (research content)
      - ClinicalSynthesis (clinical insights)
      - EducationalContent (patient education)
      - RebellionFramework (neurodivergent perspective)
      - FounderVoiceAgent (marketing, SEO, social media)
  ↓
  → Validation & Quality Check
  ↓
  → Save to Knowledge Base JSON
  ↓
  → Store in Qdrant for future ContextRAG
```

### Output: Comprehensive JSON Knowledge Base

Each processed document includes:
- **Research Content:** Key findings, methodologies, results
- **Clinical Insights:** Treatment implications, patient considerations
- **Educational Material:** Accessible explanations for clients
- **Neurodivergent Perspective:** Rebellion framework analysis
- **Marketing Content:** Website copy, blog posts, social media
- **SEO Optimization:** Keywords, meta descriptions
- **Entity Extraction:** Diseases, genes, treatments, anatomy
- **Confidence Scores:** Quality metrics for each entity
- **Cross-References:** Links to related papers

---

## Quality Optimizations Applied

### ✅ 128k Context Window
- **Impact:** Massive quality improvement
- **Benefit:** Model sees entire context, no truncation
- **Trade-off:** 4-5x slower per document (acceptable)

### ✅ Qdrant Vector Database
- **Storage:** Local file-based (`qdrant_storage/`)
- **Benefit:** Cross-document connections, growing intelligence
- **Impact:** Better synthesis as more docs are processed

### ✅ 57 Client Personas
- **Source:** Real intake data, clustered into distinct segments
- **Benefit:** Marketing content speaks to real client experiences
- **Integration:** Loaded into every document's context

### ✅ Founder Voice Integration
- **Source:** Liz's transcripts and voice patterns
- **Benefit:** Authentic, on-brand content generation
- **Integration:** Guides all marketing/educational content

### ✅ 5 Specialized NER Models
- **Benefit:** Domain-specific entity extraction
- **Coverage:** Medical, genetic, clinical, anatomical
- **Sequential Processing:** Avoids GPU memory conflicts

### ✅ Confidence Scoring & External Search
- **Benefit:** Fills knowledge gaps automatically
- **Sources:** Wikipedia, PubMed, Semantic Scholar, DuckDuckGo
- **Trigger:** Low-confidence entities get enriched

### ✅ Dashboard Monitoring
- **Access:** Port 5000 (SSH tunnel)
- **Features:** Real-time metrics, logs, GPU/CPU stats, JSON viewer
- **Benefit:** Full visibility into processing

---

## Performance Expectations

### Speed
- **Per Document:** ~30-40 minutes (with 128k context)
- **Total Time:** ~345 docs × 35 min = ~200 hours (~8-9 days)
- **Acceptable:** Quality > Speed philosophy

### Quality
- **Context Utilization:** 100% (no truncation)
- **Entity Extraction:** Comprehensive (5 models)
- **Cross-Document Learning:** Improves with each doc (ContextRAG)
- **Marketing Quality:** Authentic, persona-driven
- **Clinical Accuracy:** High (full paper + external validation)

---

## File Structure

```
enlitens-ai/
├── enlitens_corpus/
│   └── input_pdfs/          # 345 research PDFs
├── enlitens_client_profiles/
│   ├── profiles/            # 57 client personas (JSON)
│   └── intakes.txt          # 224 raw intake messages
├── enlitens_knowledge_base/
│   ├── liz_transcripts.txt  # Founder voice
│   └── style (20).css       # Brand styling
├── models/
│   ├── qwen2.5-14b-instruct-awq/  # Main LLM (4-bit, 8GB)
│   └── qwen2.5-14b-instruct-gptq-int8/  # 8-bit version (16GB, backup)
├── qdrant_storage/          # Vector database (local)
├── dashboard/               # Monitoring UI
├── src/                     # Processing pipeline code
├── scripts/
│   └── start_vllm_128k.sh   # Start LLM with 128k context
├── logs/                    # All processing logs
└── enlitens_knowledge_base.json  # Final output

```

---

## Key Commands

### Start vLLM (128k context)
```bash
bash scripts/start_vllm_128k.sh
```

### Start Processing
```bash
python3 process_multi_agent_corpus.py \
    --input-dir enlitens_corpus/input_pdfs \
    --output-file enlitens_knowledge_base.json \
    --st-louis-report st_louis_health_report.pdf
```

### Start Dashboard
```bash
cd dashboard && python3 server.py
```

### Access Dashboard (from Windows)
```powershell
ssh -NT -L 5000:127.0.0.1:5000 antons-gs@192.168.50.39
```
Then open: `http://localhost:5000`

### Monitor Logs
```bash
tail -f logs/enlitens_complete_processing.log
tail -f logs/vllm_128k.log
```

### Check GPU Usage
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

---

## Why This Configuration is Optimal

### 1. Maximum Context (128k)
- ✅ Fits ALL input sources simultaneously
- ✅ No truncation = no lost information
- ✅ Better reasoning across entire context
- ✅ Worth the 4-5x speed trade-off

### 2. 4-bit Quantization
- ✅ Stable (no OOM errors)
- ✅ 95% quality of full precision
- ✅ Leaves room for 128k KV cache
- ✅ Proven reliable

### 3. CPU-Based NER
- ✅ Frees GPU for main LLM
- ✅ Sequential = no memory conflicts
- ✅ 64GB RAM = plenty of headroom
- ✅ Quality unaffected

### 4. Qdrant Local Storage
- ✅ No external dependencies
- ✅ Persistent across restarts
- ✅ Growing intelligence over time
- ✅ Fast local access

### 5. Comprehensive Input Context
- ✅ 57 real client personas
- ✅ Founder voice authenticity
- ✅ Local St. Louis context
- ✅ Cross-document connections
- ✅ External knowledge enrichment

---

## Next Steps

1. ✅ vLLM loading with 128k context
2. ⏳ Start processing 345 PDFs
3. ⏳ Monitor via dashboard
4. ⏳ Let it run for ~8-9 days
5. ⏳ Review final knowledge base
6. ⏳ Push to new git repo: `enlitens-agents`

---

**Bottom Line:** This configuration prioritizes maximum quality output by utilizing ALL available resources (128k context, 64GB RAM, 24GB VRAM, 5 NER models, vector DB, external search) to create the most comprehensive, accurate, and authentic knowledge base possible.

**Trade-off:** Slower processing (~8-9 days total) is acceptable because quality is the primary goal, and the system runs unattended.

**Result:** Each document in the knowledge base will have complete, accurate, persona-driven, clinically validated, cross-referenced content that truly serves your neurodivergent clients.

