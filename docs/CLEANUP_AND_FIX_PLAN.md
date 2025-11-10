# Cleanup & PDF Processing Fix Plan

## What You Need Working
1. ✅ PDF processing (`process_multi_agent_corpus.py`)
2. ✅ Client profiles (already working in `enlitens_client_profiles/`)
3. ✅ Integrate personas into PDF processing
4. ✅ Add confidence scoring + external search

---

## Safe to Delete (Unused/Duplicate/Old)

### Directories to Remove
- `docs/` - Documentation (can regenerate if needed)
- `golden_dataset/` - Old test data
- `monitoring_ui/` - Monitoring dashboard (not core functionality)
- `pyairports/` - Unrelated package
- `test_input/` - Old test files
- `tests/` - Old test files
- `.worktrees/` - Git worktrees (if exists)
- `__pycache__/` - Python cache (regenerates automatically)

### Files to Remove (Root Directory)
- `*.log` - All log files (falcon_vllm.log, mistral_vllm.log, etc.)
- `monitoring_server*.py` - Monitoring servers (not core)
- `test_monitoring_server.py` - Test file
- `monitor_processing.py` - Monitoring script
- `check_progress.py` - Monitoring script
- `test_dashboard.html` - Test file
- `style.css` - Test file
- `intakes.txt` - Duplicate (you have it in enlitens_knowledge_base/)
- `transcripts.txt` - Duplicate (you have it in enlitens_knowledge_base/)
- `*.sh` except `start_processing.sh` - Old scripts
- `FIXES_SUMMARY.md`, `IMMEDIATE_FIXES.md`, `QUICKSTART_MONITORING.md`, `MONITORING_README.md` - Old docs
- `requirements-monitoring.txt` - Monitoring requirements

### Keep These
✅ `src/` - Core code for PDF processing
✅ `enlitens_client_profiles/` - Your persona system
✅ `enlitens_corpus/` - Your PDFs
✅ `enlitens_knowledge_base/` - Your data (intakes, transcripts)
✅ `process_multi_agent_corpus.py` - Main PDF processing script
✅ `add_new_pdfs.py` - Incremental PDF processing
✅ `start_processing.sh` - Start script
✅ `requirements.txt` - Dependencies
✅ `venv/` - Python environment
✅ `logs/` - Keep directory (will regenerate logs)
✅ `cache/` - Keep directory (caching)
✅ `models/` - Keep if you have model files
✅ All `.pdf`, `.zip`, `.txt`, `.docx` files

---

## What's Actually Needed in `src/`

Based on `process_multi_agent_corpus.py`, you need:

### Required Directories
- `src/agents/` - Multi-agent system
  - `supervisor_agent.py`
  - `extraction_team.py`
- `src/models/` - Data schemas
  - `enlitens_schemas.py`
- `src/extraction/` - PDF extraction
  - `enhanced_pdf_extractor.py`
  - `enhanced_extraction_tools.py`
- `src/utils/` - Utilities
  - `enhanced_logging.py`
  - `terminology.py`
- `src/retrieval/` - Embeddings
  - `embedding_ingestion.py`

### Potentially Unused (Check First)
- `src/cli/` - CLI tools (might not be needed)
- `src/pipeline/` - Pipeline tools (might be redundant)
- `src/synthesis/` - Synthesis tools (check if used)
- `src/validation/` - Validation (check if used)
- `src/knowledge_base/` - Knowledge base tools (check if used)
- `src/schema/` - Schema tools (might be redundant with models/)
- `src/testing/` - Testing tools (not needed for production)
- `src/monitoring/` - Monitoring (not core)

---

## Cleanup Commands

```bash
cd /home/antons-gs/enlitens-ai

# Remove unused directories
rm -rf docs/ golden_dataset/ monitoring_ui/ pyairports/ test_input/ tests/ __pycache__/

# Remove log files
rm -f *.log

# Remove monitoring files
rm -f monitoring_server*.py test_monitoring_server.py monitor_processing.py check_progress.py
rm -f test_dashboard.html style.css requirements-monitoring.txt

# Remove duplicate files
rm -f intakes.txt transcripts.txt

# Remove old scripts (keep start_processing.sh)
rm -f fix_ollama_gpu.sh stable_run.sh start_monitoring.sh

# Remove old docs
rm -f FIXES_SUMMARY.md IMMEDIATE_FIXES.md QUICKSTART_MONITORING.md MONITORING_README.md

# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
```

---

## After Cleanup: Your Project Structure

```
enlitens-ai/
├── src/                          # Core PDF processing code
│   ├── agents/                   # Multi-agent system
│   ├── extraction/               # PDF extraction
│   ├── models/                   # Data schemas
│   ├── retrieval/                # Embeddings
│   └── utils/                    # Utilities
├── enlitens_client_profiles/     # Persona generation system
│   ├── profiles/                 # 57 generated personas
│   ├── clusters/                 # Intake clusters
│   └── *.py                      # Persona scripts
├── enlitens_corpus/              # Your PDFs
│   └── input_pdfs/               # PDFs to process
├── enlitens_knowledge_base/      # Your data
│   ├── intakes.txt               # 224 client intakes
│   └── transcripts.txt           # Session transcripts
├── process_multi_agent_corpus.py # Main PDF processing
├── add_new_pdfs.py               # Incremental processing
├── start_processing.sh           # Start script
├── requirements.txt              # Dependencies
├── venv/                         # Python environment
├── logs/                         # Processing logs
├── cache/                        # Caching
└── README*.md                    # Documentation

TOTAL: ~8 directories, much cleaner!
```

---

## Next Steps

1. ✅ Run cleanup commands
2. ✅ Verify PDF processing still works
3. ✅ Add persona integration
4. ✅ Add confidence scoring + external search
5. ✅ Process your PDFs with personas

