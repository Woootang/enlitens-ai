# 🚀 Full Corpus Processing - LIVE NOW

## Status: ✅ RUNNING

**Started:** November 9, 2025 at 7:00 PM  
**Dashboard:** http://localhost:5000  
**Total Documents:** 345 PDFs  
**Estimated Time:** ~174 hours (~7.2 days) at 30 min/doc average

---

## 📊 What's Running

### 1. **Enhanced Dashboard** (Port 5000)
- **URL:** http://localhost:5000
- **Features:**
  - Real-time GPU/CPU metrics with live charts
  - Document processing progress with ETA
  - Quality & confidence tracking
  - Agent status board (see which agents are running)
  - Context curator token usage
  - Live log streaming with filters
  - Recent documents with performance metrics
  - Document performance chart (duration vs quality)
  - Alert monitoring
  - JSON knowledge base viewer & download

### 2. **Multi-Agent Processing Pipeline**
- **Input:** `/home/antons-gs/enlitens-ai/enlitens_corpus/input_pdfs/` (345 PDFs)
- **Output:** `/home/antons-gs/enlitens-ai/enlitens_knowledge_base/enlitens_knowledge_base.json`
- **Logs:** `/home/antons-gs/enlitens-ai/logs/enlitens_complete_processing.log`

---

## 🎯 Active Agents

1. **Context Curator** - Selects top 10 personas, creates health brief, distills voice guide
2. **Science Extraction** - Extracts research content and key findings
3. **Clinical Synthesis** - Creates clinical applications
4. **Educational Content** - Generates educational materials
5. **Rebellion Framework** - Applies Enlitens philosophy
6. **Founder Voice** - Creates marketing content in Liz's voice
7. **Marketing & SEO** - Generates website copy, blog posts, social media
8. **Validation** - Quality assurance and scoring

---

## 📈 Current Progress

**Document 1/345:** Processing `2023-67353-007.pdf`  
**Stage:** Entity extraction (NER models loading on CPU)  
**Status:** ✅ PDF extracted successfully (117,540 characters)

The system is:
- ✅ Using vLLM with Qwen 2.5-14B on GPU (22.83 GB VRAM)
- ✅ Running NER models on CPU (to preserve GPU memory)
- ✅ Falling back gracefully when Docling hits CUDA OOM
- ✅ Tracking quality and confidence scores
- ✅ Saving progress after each document

---

## 🎮 How to Monitor

### Option 1: Dashboard (Recommended)
Open your browser to: **http://localhost:5000**

You'll see:
- Live progress bar
- Current document being processed
- GPU/CPU usage with temperature
- Agent status (which agents are running)
- Recent logs with color coding
- Performance charts

### Option 2: Command Line
```bash
# Watch the main log
tail -f /home/antons-gs/enlitens-ai/logs/enlitens_complete_processing.log

# Check GPU usage
nvidia-smi

# See how many docs are done
grep "processed successfully" /home/antons-gs/enlitens-ai/logs/enlitens_complete_processing.log | wc -l
```

---

## 🛠️ Management Commands

### Check Status
```bash
# Is processing running?
ps aux | grep process_multi_agent_corpus.py

# Is dashboard running?
ps aux | grep dashboard/server.py

# Check latest progress
tail -n 50 /home/antons-gs/enlitens-ai/logs/enlitens_complete_processing.log
```

### Stop Processing (if needed)
```bash
pkill -f process_multi_agent_corpus.py
```

### Restart Dashboard (if needed)
```bash
pkill -f "dashboard/server.py"
cd /home/antons-gs/enlitens-ai
source venv/bin/activate
nohup python dashboard/server.py > logs/dashboard.log 2>&1 &
```

### Resume Processing (if stopped)
```bash
cd /home/antons-gs/enlitens-ai
./scripts/run_full_corpus.sh
```

---

## 📦 Output Structure

Each processed document will have:

```json
{
  "metadata": {
    "document_id": "...",
    "processing_timestamp": "...",
    "processing_time": 1544.97,
    "word_count": 16386
  },
  "extracted_entities": {
    "diseases": [...],
    "treatments": [...],
    "biomarkers": [...],
    "total_entities": 11
  },
  "research_content": {
    "title": "...",
    "key_findings": [...],
    "clinical_implications": [...]
  },
  "clinical_content": {
    "assessment_applications": [...],
    "therapeutic_applications": [...]
  },
  "rebellion_framework": {
    "neuroscience_translation": "...",
    "clinical_rebellion": "..."
  },
  "marketing_content": {
    "about_section": "...",
    "services_section": "..."
  },
  "educational_content": {
    "blog_posts": [...],
    "social_media": [...]
  },
  "content_creation_ideas": {
    "topics": [...],
    "series": [...]
  }
}
```

---

## ⚡ Performance Expectations

Based on test runs:
- **Average time per document:** ~25-30 minutes
- **Quality score:** ~0.83 (target: >0.75)
- **Confidence score:** ~0.75 (target: >0.70)
- **GPU usage:** ~23 GB VRAM (vLLM + inference)
- **CPU usage:** Variable (NER models)

---

## 🎉 What Happens When It's Done

After all 345 documents are processed:

1. **Final JSON file** will be at:
   `/home/antons-gs/enlitens-ai/enlitens_knowledge_base/enlitens_knowledge_base.json`

2. **Download from dashboard:**
   - Go to http://localhost:5000
   - Click "⬇️ Download JSON" button

3. **Vector database** will be populated at:
   `/home/antons-gs/enlitens-ai/qdrant_storage/`
   (Ready for ContextRAG retrieval)

4. **Complete logs** at:
   `/home/antons-gs/enlitens-ai/logs/enlitens_complete_processing.log`

---

## 🚨 Troubleshooting

### If processing stops:
1. Check logs: `tail -n 100 /home/antons-gs/enlitens-ai/logs/enlitens_complete_processing.log`
2. Check if process died: `ps aux | grep process_multi_agent_corpus.py`
3. Restart: `./scripts/run_full_corpus.sh` (it will resume from where it stopped)

### If dashboard is blank:
1. Check if server is running: `ps aux | grep dashboard/server.py`
2. Check dashboard logs: `tail -n 50 /home/antons-gs/enlitens-ai/logs/dashboard.log`
3. Restart dashboard (see commands above)

### If GPU runs out of memory:
- The system will automatically fall back to CPU-based extraction
- NER models already run on CPU
- vLLM is configured with memory management

---

## 📞 Quick Reference

| What | Where |
|------|-------|
| Dashboard | http://localhost:5000 |
| Main Log | `/home/antons-gs/enlitens-ai/logs/enlitens_complete_processing.log` |
| Output JSON | `/home/antons-gs/enlitens-ai/enlitens_knowledge_base/enlitens_knowledge_base.json` |
| Input PDFs | `/home/antons-gs/enlitens-ai/enlitens_corpus/input_pdfs/` |
| Run Script | `./scripts/run_full_corpus.sh` |

---

**Last Updated:** November 9, 2025, 7:00 PM  
**Status:** ✅ Processing document 1/345

