#!/bin/bash
# Organize & Clean - Creates a proper hierarchy and removes unused files
# This will make your project CLEAN and ORGANIZED

set -e

echo "🧹 ENLITENS-AI ORGANIZE & CLEANUP"
echo "===================================="
echo ""
echo "This will:"
echo "  1. Organize loose files into proper folders"
echo "  2. Remove unused/duplicate files"
echo "  3. Create a clean hierarchy"
echo ""
echo "✅ KEEPS:"
echo "  - src/ (PDF processing)"
echo "  - enlitens_client_profiles/ (personas)"
echo "  - enlitens_corpus/ (PDFs)"
echo "  - enlitens_knowledge_base/ (data)"
echo "  - All .pdf, .zip files"
echo ""
echo "❌ REMOVES:"
echo "  - Monitoring files"
echo "  - Test files"
echo "  - Old docs"
echo "  - Duplicate files"
echo "  - Log files"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

cd /home/antons-gs/enlitens-ai

echo ""
echo "📁 Creating organized folder structure..."

# Create organized structure
mkdir -p _archive/old_docs
mkdir -p _archive/old_logs
mkdir -p _archive/old_scripts
mkdir -p docs/reports
mkdir -p scripts

echo "✅ Created: _archive/, docs/, scripts/"

echo ""
echo "📦 Moving documentation files..."
mv -f DEEP_RESEARCH_PROMPT.md docs/ 2>/dev/null || true
mv -f PERSONA_GENERATION_FINAL_REPORT.md docs/reports/ 2>/dev/null || true
mv -f PERSONA_INTEGRATION_PLAN.md docs/ 2>/dev/null || true
mv -f CLEANUP_AND_FIX_PLAN.md docs/ 2>/dev/null || true
mv -f README_KNOWLEDGE_BASE.md docs/ 2>/dev/null || true
mv -f README_MULTI_AGENT_SYSTEM.md docs/ 2>/dev/null || true
echo "✅ Organized: Documentation → docs/"

echo ""
echo "📦 Archiving old documentation..."
mv -f FIXES_SUMMARY.md _archive/old_docs/ 2>/dev/null || true
mv -f IMMEDIATE_FIXES.md _archive/old_docs/ 2>/dev/null || true
mv -f QUICKSTART_MONITORING.md _archive/old_docs/ 2>/dev/null || true
mv -f MONITORING_README.md _archive/old_docs/ 2>/dev/null || true
echo "✅ Archived: Old docs → _archive/old_docs/"

echo ""
echo "📦 Moving scripts..."
mv -f start_processing.sh scripts/ 2>/dev/null || true
mv -f add_new_pdfs.py scripts/ 2>/dev/null || true
echo "✅ Organized: Scripts → scripts/"

echo ""
echo "📦 Archiving old scripts..."
mv -f fix_ollama_gpu.sh _archive/old_scripts/ 2>/dev/null || true
mv -f stable_run.sh _archive/old_scripts/ 2>/dev/null || true
mv -f start_monitoring.sh _archive/old_scripts/ 2>/dev/null || true
echo "✅ Archived: Old scripts → _archive/old_scripts/"

echo ""
echo "📦 Archiving log files..."
mv -f *.log _archive/old_logs/ 2>/dev/null || true
echo "✅ Archived: Logs → _archive/old_logs/"

echo ""
echo "🗑️  Removing monitoring files..."
rm -f monitoring_server*.py test_monitoring_server.py monitor_processing.py check_progress.py 2>/dev/null || true
rm -f test_dashboard.html style.css requirements-monitoring.txt 2>/dev/null || true
rm -f cloudflared 2>/dev/null || true
echo "✅ Removed: Monitoring files"

echo ""
echo "🗑️  Removing duplicate data files..."
rm -f intakes.txt transcripts.txt 2>/dev/null || true
echo "✅ Removed: Duplicate intakes.txt, transcripts.txt (kept in enlitens_knowledge_base/)"

echo ""
echo "🗑️  Removing unused directories..."
rm -rf docs/old_structure 2>/dev/null || true
rm -rf golden_dataset/ monitoring_ui/ pyairports/ test_input/ tests/ 2>/dev/null || true
rm -rf .vscode/ .worktrees/ 2>/dev/null || true
echo "✅ Removed: Unused directories"

echo ""
echo "🗑️  Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo "✅ Cleaned: Python cache"

echo ""
echo "🗑️  Removing unused src/ subdirectories..."
cd src/
rm -rf cli/ monitoring/ pipeline/ schema/ testing/ 2>/dev/null || true
echo "✅ Removed: cli, monitoring, pipeline, schema, testing from src/"

echo ""
echo "🗑️  Cleaning unused files in src/agents/..."
cd agents/
rm -f clinical_synthesis_agent.py context_rag_agent.py educational_content_agent.py 2>/dev/null || true
rm -f enhanced_complete_enlitens_agent.py base_agent.py 2>/dev/null || true
echo "✅ Kept only: extraction_team.py, supervisor_agent.py"

echo ""
echo "🗑️  Cleaning unused files in src/extraction/..."
cd ../extraction/
rm -f enhanced_pdf_extractor_v2.py pdf_extractor.py 2>/dev/null || true
echo "✅ Kept only: enhanced_pdf_extractor.py, enhanced_extraction_tools.py"

echo ""
echo "🗑️  Cleaning unused files in src/retrieval/..."
cd ../retrieval/
rm -f chunker.py hybrid_retriever.py index_maintenance.py 2>/dev/null || true
echo "✅ Kept only: embedding_ingestion.py"

echo ""
echo "🗑️  Cleaning unused files in src/synthesis/..."
cd ../synthesis/
rm -f ai_synthesizer.py enlitens_rebellion_synthesizer.py few_shot_library.py prompts.py 2>/dev/null || true
echo "✅ Kept only: ollama_client.py"

echo ""
echo "🗑️  Cleaning unused files in src/utils/..."
cd ../utils/
rm -f prompt_cache.py retry.py settings.py 2>/dev/null || true
echo "✅ Kept only: enhanced_logging.py, terminology.py"

echo ""
echo "🗑️  Removing src/validation/ (unused)..."
cd ..
rm -rf validation/ 2>/dev/null || true
echo "✅ Removed: validation/"

cd /home/antons-gs/enlitens-ai

echo ""
echo "📝 Creating README.md..."
cat > README.md << 'READMEEOF'
# Enlitens AI - Client Profile & Knowledge Base System

## 📁 Project Structure

```
enlitens-ai/
├── src/                              # Core PDF processing code
│   ├── agents/                       # Multi-agent system
│   │   ├── extraction_team.py        # Extraction agent team
│   │   └── supervisor_agent.py       # Supervisor agent
│   ├── extraction/                   # PDF extraction
│   │   ├── enhanced_pdf_extractor.py
│   │   └── enhanced_extraction_tools.py
│   ├── models/                       # Data schemas
│   │   └── enlitens_schemas.py
│   ├── retrieval/                    # Embeddings & retrieval
│   │   └── embedding_ingestion.py
│   ├── synthesis/                    # LLM synthesis
│   │   └── ollama_client.py
│   ├── utils/                        # Utilities
│   │   ├── enhanced_logging.py
│   │   └── terminology.py
│   └── knowledge_base/               # Knowledge management
│       └── knowledge_manager.py
│
├── enlitens_client_profiles/         # Client persona system
│   ├── profiles/                     # 57 generated personas
│   ├── clusters/                     # Intake clusters
│   └── *.py                          # Persona generation scripts
│
├── enlitens_corpus/                  # Research PDFs
│   └── input_pdfs/                   # PDFs to process
│
├── enlitens_knowledge_base/          # Client data
│   ├── intakes.txt                   # 224 client intakes
│   └── transcripts.txt               # Session transcripts
│
├── scripts/                          # Main scripts
│   ├── start_processing.sh           # Start PDF processing
│   └── add_new_pdfs.py               # Add new PDFs incrementally
│
├── docs/                             # Documentation
│   └── reports/                      # Analysis reports
│
├── process_multi_agent_corpus.py     # Main PDF processing script
├── requirements.txt                  # Python dependencies
├── venv/                             # Python virtual environment
├── logs/                             # Processing logs
├── cache/                            # Caching
└── _archive/                         # Archived old files

```

## 🚀 Quick Start

### Process PDFs to Create Knowledge Base
```bash
cd /home/antons-gs/enlitens-ai
./scripts/start_processing.sh
```

### View Generated Personas
```bash
python -m enlitens_client_profiles.view_persona
```

### Generate New Personas (when you have new intakes)
```bash
python -m enlitens_client_profiles.cluster_intakes 50
python -m enlitens_client_profiles.generate_from_clusters --full
```

## 📊 Current Status

- ✅ **57 Client Personas** generated from 224 real intakes
- ✅ **50 Client Segments** identified via clustering
- ⏳ **Knowledge Base** - Ready to process PDFs
- ⏳ **Confidence Scoring** - To be implemented
- ⏳ **External Search** - To be implemented

## 🎯 Next Steps

1. Process PDFs with persona integration
2. Add confidence scoring system
3. Add external search for knowledge gaps
4. Generate training pairs for fine-tuning

## 📖 Documentation

- [Persona Generation Report](docs/reports/PERSONA_GENERATION_FINAL_REPORT.md)
- [Multi-Agent System](docs/README_MULTI_AGENT_SYSTEM.md)
- [Knowledge Base](docs/README_KNOWLEDGE_BASE.md)

READMEEOF

echo "✅ Created: README.md"

echo ""
echo "✅ ORGANIZATION & CLEANUP COMPLETE!"
echo ""
echo "📊 NEW CLEAN STRUCTURE:"
echo ""
echo "enlitens-ai/"
echo "├── 📁 src/                        # Core code"
echo "├── 📁 enlitens_client_profiles/   # 57 personas"
echo "├── 📁 enlitens_corpus/            # Your PDFs"
echo "├── 📁 enlitens_knowledge_base/    # Data (intakes, transcripts)"
echo "├── 📁 scripts/                    # Main scripts"
echo "├── 📁 docs/                       # Documentation"
echo "├── 📁 _archive/                   # Old files (safe to delete later)"
echo "├── 📄 process_multi_agent_corpus.py"
echo "├── 📄 requirements.txt"
echo "├── 📄 README.md"
echo "└── 📁 venv/"
echo ""
echo "🎯 Next steps:"
echo "   1. Test PDF processing:"
echo "      cd /home/antons-gs/enlitens-ai"
echo "      ./scripts/start_processing.sh"
echo ""
echo "   2. View your personas:"
echo "      python -m enlitens_client_profiles.view_persona"
echo ""
echo "   3. Delete _archive/ folder when ready (contains old files)"

