# PDF Processing Plan - 345 PDFs
**Date:** November 8, 2025  
**PDFs to Process:** 345 research papers

---

## 🤖 AI System Configuration

### **PRIMARY: vLLM (LOCAL) - 100% FREE**

Your system is configured to use **vLLM running locally on your GPU**:

- **Model:** Mistral-7B-Instruct (local in `/home/antons-gs/enlitens-ai/models/`)
- **Server:** `http://localhost:8000/v1` (vLLM API)
- **Cost:** **$0.00** (runs on your RTX 3090)
- **Speed:** ~15-30 seconds per PDF (depends on PDF size)

**✅ NO API COSTS - Everything runs locally on your hardware!**

---

## 📊 Processing Estimates

### Time Estimates
- **Per PDF:** ~20-40 seconds average
- **345 PDFs:** 
  - **Best case:** ~2 hours (20 sec/PDF)
  - **Realistic:** ~3-4 hours (30-40 sec/PDF)
  - **Worst case:** ~5-6 hours (if complex PDFs or GPU throttling)

### GPU Usage
- **Model:** Mistral-7B-Instruct (~7GB VRAM)
- **Your GPU:** RTX 3090 (24GB VRAM)
- **Headroom:** ~17GB free (plenty of room)
- **Temperature:** Should stay under 80°C with your cooling setup

### Cost Breakdown
- **vLLM (local):** $0.00 ✅
- **Electricity:** ~$0.50-1.00 (3-4 hours at full GPU load)
- **External Search (optional):** $0.00 (using free APIs)

**TOTAL COST: ~$0.50-1.00 in electricity**

---

## 🔍 External Search Configuration

For filling knowledge gaps (confidence scoring system), we'll use **FREE** APIs:

### Option 1: Wikipedia API (FREE)
- **Cost:** $0.00
- **Rate Limit:** No strict limit for reasonable use
- **Coverage:** General knowledge, medical terms, neuroscience basics
- **API:** `https://en.wikipedia.org/api/rest_v1/`

### Option 2: PubMed/NCBI API (FREE)
- **Cost:** $0.00
- **Rate Limit:** 3 requests/second (no API key), 10/sec (with free API key)
- **Coverage:** Medical research, neuroscience papers, clinical studies
- **API:** `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`

### Option 3: Semantic Scholar API (FREE)
- **Cost:** $0.00
- **Rate Limit:** 100 requests/5 minutes (no API key), higher with free API key
- **Coverage:** Academic papers, citations, research metadata
- **API:** `https://api.semanticscholar.org/`

### Option 4: DuckDuckGo Instant Answer API (FREE)
- **Cost:** $0.00
- **Rate Limit:** Reasonable use
- **Coverage:** General knowledge, definitions, quick facts
- **API:** `https://api.duckduckgo.com/`

**RECOMMENDED: Use all 4 in cascade (try Wikipedia → PubMed → Semantic Scholar → DuckDuckGo)**

---

## 🎯 Processing Pipeline

### Phase 1: PDF Extraction (vLLM Local)
For each PDF:
1. **Extract text** from PDF
2. **Multi-agent processing:**
   - Science Extraction Agent (research findings, statistics, methods)
   - Marketing/SEO Agent (keywords, copy, target audience)
   - Founder Voice Agent (Liz Wooten's tone and style)
   - Rebellion Framework Agent (proprietary framework application)
3. **Entity extraction** (using local BiomedBERT models)
4. **Generate structured JSON** for knowledge base

**Cost:** $0.00 (all local)  
**Time:** ~30 seconds per PDF

### Phase 2: Confidence Scoring (NEW - We'll Add This)
For each extracted entity/concept:
1. **Count mentions** across all PDFs
2. **Assess context quality** (is it well-explained?)
3. **Calculate confidence score:**
   - High (80-100%): Mentioned 5+ times with good context
   - Medium (50-79%): Mentioned 2-4 times or limited context
   - Low (<50%): Mentioned once or poor context

**Cost:** $0.00 (local processing)  
**Time:** ~5-10 minutes total (after all PDFs processed)

### Phase 3: Knowledge Gap Filling (NEW - We'll Add This)
For low-confidence entities:
1. **Trigger external search** (Wikipedia, PubMed, etc.)
2. **Fetch additional context**
3. **Merge into knowledge base**
4. **Re-calculate confidence score**

**Cost:** $0.00 (free APIs)  
**Time:** ~10-20 minutes (depends on number of low-confidence entities)

### Phase 4: Persona Integration (NEW - We'll Add This)
1. **Load 57 personas** from `enlitens_client_profiles/`
2. **Extract persona insights:**
   - Pain points
   - Therapy goals
   - SEO keywords
   - Local entities (St. Louis)
3. **Merge with knowledge base:**
   - Add "target_audience" section
   - Add "client_pain_points" section
   - Enhance SEO keywords with persona data
   - Add local context (St. Louis neighborhoods, places)

**Cost:** $0.00 (local processing)  
**Time:** ~2-3 minutes

---

## 📦 Output: `enlitens_knowledge_base.json`

### Structure
```json
{
  "metadata": {
    "total_documents": 345,
    "processed_date": "2025-11-08",
    "processing_time_hours": 3.5,
    "total_personas_integrated": 57
  },
  "research_content": [
    {
      "document_id": "paper_001",
      "title": "...",
      "findings": [...],
      "statistics": [...],
      "methodologies": [...],
      "confidence_scores": {
        "ADHD": 0.95,
        "executive_function": 0.87,
        "dopamine": 0.72
      }
    },
    ...
  ],
  "persona_insights": {
    "target_audiences": [...],
    "pain_points": [...],
    "therapy_goals": [...],
    "seo_keywords": [...],
    "local_entities": [...]
  },
  "confidence_index": {
    "high_confidence": ["ADHD", "autism", "neurodivergence", ...],
    "medium_confidence": ["executive_function", "sensory_processing", ...],
    "low_confidence": ["specific_gene_XYZ", ...]
  },
  "external_enrichment": {
    "wikipedia_lookups": 45,
    "pubmed_lookups": 23,
    "semantic_scholar_lookups": 12
  }
}
```

---

## ⚠️ Important Notes

### GPU Considerations
1. **Monitor temperature:** Your setup should handle this fine, but watch for throttling
2. **Sequential processing:** Script processes 1 PDF at a time (memory safe)
3. **Checkpointing:** Saves progress after each PDF (can resume if interrupted)

### vLLM Server Must Be Running
Before starting, ensure vLLM is running:
```bash
# Check if vLLM is running
curl -s http://localhost:8000/v1/models
```

If not running, you'll need to start it. The script will check and warn you.

### Processing Can Be Interrupted
- **Safe to stop:** Press Ctrl+C anytime
- **Resume:** Script will skip already-processed PDFs
- **Checkpoint file:** `enlitens_knowledge_base_TIMESTAMP.json.temp`

---

## 🚀 Execution Plan

### Step 1: Start vLLM Server (if not running)
```bash
# You'll need to start vLLM with Mistral-7B
# (I don't see the exact command in your files, but it should be something like:)
# vllm serve /home/antons-gs/enlitens-ai/models/mistral-7b-instruct --port 8000
```

### Step 2: Run PDF Processing
```bash
cd /home/antons-gs/enlitens-ai
./scripts/start_processing.sh
```

### Step 3: Monitor Progress
- **Live logs:** `tail -f logs/enlitens_complete_processing.log`
- **GPU usage:** `watch -n 1 nvidia-smi`
- **Progress:** Script prints status after each PDF

### Step 4: Review Output
- **Knowledge base:** `enlitens_knowledge_base_YYYYMMDD_HHMMSS.json`
- **Logs:** `logs/enlitens_complete_processing.log`

---

## 🔧 What I'll Add Before Running

### 1. Persona Integration Module
**File:** `src/agents/persona_integration_agent.py`
- Loads 57 personas
- Extracts insights
- Merges with knowledge base

### 2. Confidence Scoring Module
**File:** `src/utils/confidence_scorer.py`
- Counts entity mentions
- Assesses context quality
- Calculates confidence scores

### 3. External Search Module
**File:** `src/retrieval/external_search.py`
- Wikipedia API client
- PubMed API client
- Semantic Scholar API client
- DuckDuckGo API client
- Cascade search strategy

### 4. Update Main Script
**File:** `process_multi_agent_corpus.py`
- Add persona loading
- Add confidence scoring
- Add external search for low-confidence entities
- Merge all data into final knowledge base

---

## 📋 Pre-Flight Checklist

Before running, verify:

- [ ] vLLM server is running (`curl http://localhost:8000/v1/models`)
- [ ] GPU has free VRAM (`nvidia-smi`)
- [ ] 345 PDFs are in `enlitens_corpus/input_pdfs/`
- [ ] 57 personas exist in `enlitens_client_profiles/profiles/`
- [ ] Disk space available (~500MB for output)
- [ ] Cooling is adequate (fans running)

---

## 🎯 Expected Timeline

**Total Time: 3-5 hours**

1. **Setup & Verification:** 5 minutes
2. **PDF Processing (345 PDFs):** 3-4 hours
3. **Confidence Scoring:** 5-10 minutes
4. **External Search (if needed):** 10-20 minutes
5. **Persona Integration:** 2-3 minutes
6. **Final Assembly:** 1-2 minutes

**You can walk away and let it run!** It will checkpoint progress and can be resumed if interrupted.

---

## 💰 Final Cost Summary

| Component | Cost |
|-----------|------|
| vLLM (local) | $0.00 |
| External Search APIs | $0.00 |
| Electricity (3-4 hrs GPU) | ~$0.50-1.00 |
| **TOTAL** | **~$0.50-1.00** |

**✅ NO SURPRISES - Everything is free except electricity!**

---

## ❓ Questions?

1. **What if vLLM isn't running?** - I'll help you start it
2. **What if it's too slow?** - We can optimize (batch processing, smaller model, etc.)
3. **What if GPU overheats?** - Script will pause and resume when cool
4. **What if I need to stop?** - Just Ctrl+C, it will resume from checkpoint

---

**Ready to proceed?** Say "yes" and I'll:
1. Add the 3 new modules (persona integration, confidence scoring, external search)
2. Update the main script
3. Verify vLLM is running
4. Start processing your 345 PDFs!

