# 🚀 START HERE - Enlitens AI Pipeline Rebuild

**Date**: November 15, 2025  
**Status**: Ready to begin implementation

---

## 📋 What Just Happened?

I've prepared your system for a complete rebuild of the PDF processing pipeline with:

1. ✅ **Updated Tool Stack** - Researched and verified the best tools as of Nov 2025
2. ✅ **Complete Implementation Guide** - 17,000+ word guide with all code
3. ✅ **Dashboard Updates** - Now tracks the new knowledge base location
4. ✅ **Cleanup Scripts** - Ready to clear old data and start fresh

---

## 📚 Three Key Documents

### 1. **PROJECT_MASTER_IMPLEMENTATION_GUIDE.md** (MAIN GUIDE)
- Complete technical documentation
- All code for every module
- Step-by-step instructions
- Troubleshooting guide
- **READ THIS for implementation details**

### 2. **CLEANUP_AND_START_GUIDE.md** (START HERE)
- How to run the cleanup script
- Quick start commands
- Verification steps
- **RUN THIS first before coding**

### 3. **README_START_HERE.md** (THIS FILE)
- Overview and next steps
- Quick reference

---

## 🎯 Your Next Steps (In Order)

### Step 1: Run Cleanup (5 minutes)

```bash
cd /home/antons-gs/enlitens-ai
python3 cleanup_and_prepare.py
```

This will:
- Backup all old data
- Clear logs and old knowledge bases
- Create new folder structure
- Restart dashboard
- Verify system requirements

### Step 2: Verify Dashboard (2 minutes)

Open: **http://localhost:5000**

Should show:
- 0 documents
- Mode: `main_kb_v2` or `none`

### Step 3: Install Ollama & Configure VRAM (30 minutes)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull qwen3:14b-instruct-q6_K

# Configure (see CLEANUP_AND_START_GUIDE.md for full Modelfile)
ollama create qwen3-128k -f models/Qwen3-128k-Modelfile
```

### Step 4: Install PDF Processing Tools (10 minutes)

```bash
source venv/bin/activate
pip install docling pymupdf "camelot-py[cv]" pytesseract pdf2image
```

### Step 5: Implement Modules (2-4 hours)

Copy code from **PROJECT_MASTER_IMPLEMENTATION_GUIDE.md**:

1. `src/utils/llm_client.py`
2. `process_pdfs/ingestion.py`
3. `process_pdfs/extraction.py`
4. `process_pdfs/translation.py`
5. `process_pdfs/main.py`
6. Prompt templates in `process_pdfs/prompts/`

### Step 6: Test Single PDF (5 minutes)

```bash
python3 process_pdfs/main.py "enlitens_corpus/input_pdfs/[FIRST_PDF].pdf"
```

### Step 7: Process All 345 PDFs (Overnight)

```bash
python3 process_pdfs/main.py "enlitens_corpus/input_pdfs/"
```

---

## 🔧 What's New in This Rebuild?

### Tool Stack Updates (November 2025)

| Component | Old | New | Why Changed |
|-----------|-----|-----|-------------|
| **PDF Extraction** | PyMuPDF only | **Docling (primary)** + PyMuPDF (backup) | Docling built for scientific papers, handles tables/equations natively |
| **Context Window** | 6k-12k chars | **Full 128k tokens** | KV Cache Quantization enables full context on 24GB VRAM |
| **Extraction Strategy** | Single pass | **Multi-pass** (scientific → clinical) | Prevents task overload, deeper extraction |
| **Knowledge Base** | Multiple JSON files | **Single JSONL** (`main_kb.jsonl`) | Scalable, robust, one source of truth |
| **Full Text Storage** | Missing | **Included** (`full_text` field) | Enables future re-processing |
| **Architecture** | Monolithic | **Modular** (separate scripts per data type) | Prevents context pollution |

### Key Innovations

1. **KV Cache Quantization** - Compress model memory from 16-bit to 8-bit
   - Enables 128k context on 24GB VRAM
   - No quality loss for extraction tasks

2. **Docling Integration** - IBM's purpose-built PDF tool
   - Better table extraction
   - Native equation/chart handling
   - Structured DocTags output

3. **Multi-Pass Prompting** - Separate extraction and translation
   - Scientific extraction: methods, findings, statistics
   - Clinical translation: interventions, protocols, assessments
   - Each pass gets full model capacity

4. **Separation of Concerns** - Modular codebase
   - `process_pdfs/` - Research paper processing
   - `process_health_reports/` - Future: Health data
   - `process_profiles/` - Future: Client profiles
   - `process_connections/` - Future: Relationship mapping

---

## 📊 Expected Results

### Before (Old Pipeline)
- ❌ Only 6k-12k characters processed
- ❌ Shallow 7-8 word summaries
- ❌ No full text storage
- ❌ Context pollution from mixed data types
- ❌ Poor table/figure handling

### After (New Pipeline)
- ✅ Full 30k-100k character documents processed
- ✅ Detailed 200-500 char summaries per field
- ✅ Complete verbatim text stored
- ✅ Clean separation of data types
- ✅ Native table/figure extraction

### Quality Metrics to Track
- **Text Length**: Should be 20k-80k chars per paper
- **Extraction Depth**: Avg 200+ chars per field
- **Statistics Captured**: N=X, p-values, effect sizes present
- **Processing Time**: 2-5 minutes per paper
- **Success Rate**: >95% of papers processed

---

## 🆘 Quick Troubleshooting

### Cleanup Script Won't Run
```bash
# Run commands manually (see CLEANUP_AND_START_GUIDE.md)
```

### Dashboard Shows Old Data
```bash
# Hard refresh browser: Ctrl+Shift+R
# Or restart dashboard:
pkill -f "dashboard/server.py"
python3 dashboard/server.py --port 5000 &
```

### VRAM Out of Memory
```bash
# Reduce KV cache quantization in Modelfile:
PARAMETER cache_type_k q4_0  # Instead of q8_0
```

### Ollama Not Found
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

---

## 📁 File Locations

### Key Files You'll Create
- `src/utils/llm_client.py` - LLM interface
- `process_pdfs/ingestion.py` - PDF parsing
- `process_pdfs/extraction.py` - Scientific extraction
- `process_pdfs/translation.py` - Clinical translation
- `process_pdfs/main.py` - Orchestrator
- `process_pdfs/prompts/*.txt` - Prompt templates

### Key Files You'll Use
- `data/knowledge_base/main_kb.jsonl` - Output knowledge base
- `logs/processing.log` - Processing logs
- `enlitens_corpus/input_pdfs/` - Input PDFs (345 files)

### Reference Files
- `PROJECT_MASTER_IMPLEMENTATION_GUIDE.md` - Complete guide
- `CLEANUP_AND_START_GUIDE.md` - Cleanup instructions
- `backup_YYYYMMDD_HHMMSS/` - Old data backup

---

## 🎓 Understanding the Architecture

### Data Flow
```
PDF File
  ↓
[Docling/PyMuPDF] → Extract text, tables, figures
  ↓
[LLM Pass 1] → Scientific extraction (methods, findings, stats)
  ↓
[LLM Pass 2] → Clinical translation (interventions, protocols)
  ↓
[Validation] → Quality checks
  ↓
[Enrichment] → Free APIs (Wikipedia, PubMed, CrossRef)
  ↓
[Knowledge Base] → Append to main_kb.jsonl
```

### Why JSONL?
- **Scalable**: Append-only, no file rewrites
- **Robust**: One bad entry doesn't corrupt entire file
- **Portable**: Easy to import into any system
- **Debuggable**: Each line is a complete JSON object

### Why Multi-Pass?
- **Prevents Task Overload**: LLM focuses on one thing at a time
- **Deeper Extraction**: More tokens available per task
- **Better Quality**: Specialized prompts for each task
- **Easier Debugging**: Can test extraction and translation separately

---

## 💡 Pro Tips

1. **Start Small**: Test on 1 PDF before processing all 345
2. **Monitor VRAM**: Watch `nvidia-smi` during first run
3. **Check Logs**: `tail -f logs/processing.log` to see progress
4. **Validate Output**: Check first few entries in `main_kb.jsonl`
5. **Use Resume**: If processing stops, re-run same command (skips completed)

---

## 📞 Getting Help

### If Something Goes Wrong

1. **Check Logs**:
   ```bash
   tail -100 logs/processing.log
   tail -50 /tmp/dashboard_new.log
   ```

2. **Verify Installation**:
   ```bash
   ollama list
   python3 -c "import docling; print('OK')"
   nvidia-smi
   ```

3. **Read Troubleshooting**:
   - See **PROJECT_MASTER_IMPLEMENTATION_GUIDE.md** → Troubleshooting Guide

4. **Check Dashboard**:
   - http://localhost:5000
   - Should show metrics and logs

---

## ✅ Pre-Flight Checklist

Before starting, verify:

- [ ] Cleanup script ran successfully
- [ ] Dashboard shows 0 documents, mode `main_kb_v2`
- [ ] `data/knowledge_base/main_kb.jsonl` exists (empty)
- [ ] `logs/processing.log` exists (empty)
- [ ] Backup folder created with old data
- [ ] GPU detected: `nvidia-smi` works
- [ ] Python 3.9+: `python3 --version`
- [ ] Virtual environment activated: `source venv/bin/activate`

---

## 🚀 Ready to Start?

1. **Run cleanup**: `python3 cleanup_and_prepare.py`
2. **Open guide**: `PROJECT_MASTER_IMPLEMENTATION_GUIDE.md`
3. **Follow Phase 1**: Step-by-step implementation
4. **Test on 1 PDF**: Verify quality
5. **Process all 345**: Let it run overnight

---

**Good luck! You've got this! 🎉**

The complete implementation guide has everything you need. Take it one phase at a time, and you'll have a production-ready pipeline processing research papers with research-grade quality.

