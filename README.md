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

