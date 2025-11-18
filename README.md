# Enlitens AI - Scientific Knowledge Base & Document Processing System

**A neurodiversity-affirming, trauma-informed AI system for processing scientific literature and building comprehensive knowledge bases.**

---

## 🎯 What is Enlitens?

Enlitens is a production-grade document processing pipeline that:
- Extracts scientific content from PDFs using Docling + Llama 3.1 8B
- Validates and enriches extractions with Gemini 2.5 Pro (1M context)
- Stores knowledge in PostgreSQL, ChromaDB, and Neo4j
- Provides a real-time dashboard for monitoring
- Runs entirely on your own infrastructure (GPU + cloud APIs)

---

## 📁 Project Structure

```
enlitens-ai/
├── src/                              # Core processing pipeline
│   ├── pipeline/                     # Document pipeline orchestration
│   ├── integrations/                 # External APIs (Gemini CLI, Wikipedia, etc.)
│   ├── retrieval/                    # Vector store & external search
│   ├── persistence/                  # PostgreSQL & Neo4j publishers
│   └── utils/                        # Logging, terminology, helpers
│
├── process_pdfs/                     # PDF extraction & enrichment
│   ├── extraction.py                 # Scientific content extraction (Llama)
│   ├── enrichment.py                 # External enrichment (Wikipedia, Crossref, etc.)
│   └── docling_wrapper.py            # PDF parsing (Docling)
│
├── dashboard/                        # Real-time monitoring dashboard
│   ├── server.py                     # Flask API
│   ├── templates/                    # HTML templates
│   └── static/                       # CSS/JS
│
├── scripts/                          # Organized scripts
│   ├── ingestion/                    # Document processing scripts
│   ├── dashboard/                    # Dashboard management
│   ├── model_management/             # vLLM startup scripts
│   ├── backup/                       # Backup scripts
│   └── utilities/                    # Testing & utilities
│
├── ops/                              # Operations & deployment
│   ├── systemd/                      # Systemd service units
│   └── cloudflare/                   # Cloudflare Tunnel config
│
├── docs/                             # Documentation
│   └── hosting_guide.md              # Full deployment guide
│
├── config/                           # Configuration
│   └── local_models.yaml             # Model definitions
│
├── enlitens_corpus/                  # Document corpus
│   ├── input_pdfs/                   # PDFs to process
│   ├── processed/                    # Completed PDFs (organized by date)
│   └── failed/                       # Failed PDFs
│
├── enlitens_knowledge_base/          # Knowledge base storage
│   └── ledger/                       # JSONL ledger (one entry per document)
│
├── data/                             # Database storage
│   ├── vector_store/chroma/          # ChromaDB vector store
│   └── neo4j/                        # Neo4j graph database
│
├── models/                           # LLM model files (Llama 3.1 8B)
├── cache/                            # Docling cache
├── logs/                             # Processing logs
├── backups/                          # Automated backups
└── requirements.txt                  # Python dependencies
```

---

## 🚀 Quick Start

### 1. Prerequisites
- **GPU**: NVIDIA GPU with 24GB+ VRAM (for Llama 3.1 8B via vLLM)
- **RAM**: 32GB+ recommended
- **Disk**: 100GB+ free space
- **OS**: Ubuntu 20.04+ or similar Linux distribution

### 2. Installation
```bash
# Clone the repository
cd /home/antons-gs/enlitens-ai

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Start Processing Documents
```bash
# Start the dashboard (monitor at https://dashboard.enlitens.com)
bash scripts/dashboard/start_dashboard.sh

# Start document ingestion (processes all PDFs in enlitens_corpus/input_pdfs/)
cd /home/antons-gs/enlitens-ai
ENLITENS_ENABLE_POSTGRES=1 \
ENLITENS_ENABLE_NEO4J=1 \
ENLITENS_ENABLE_VECTOR_MIRROR=1 \
DATABASE_URL=postgresql:///enlitens \
ENLITENS_NEO4J_URI=bolt://localhost:7687 \
ENLITENS_NEO4J_USER=neo4j \
ENLITENS_NEO4J_PASSWORD=YourPassword \
./venv/bin/python scripts/ingestion/run_ingest_batch.py --auto-start --auto-stop
```

---

## 📊 System Architecture

### Processing Pipeline
1. **Docling** (CPU) - Extracts text, tables, and metadata from PDFs
2. **Llama 3.1 8B** (GPU) - Extracts scientific content (background, methods, findings, etc.)
3. **External Enrichment** - Fetches Wikipedia, Crossref, Semantic Scholar data
4. **Gemini 2.5 Pro** (Cloud API) - Validates and consolidates all outputs
5. **Storage** - Writes to PostgreSQL, ChromaDB, and Neo4j

### Tech Stack
- **PDF Parsing**: Docling (CPU-based OCR + layout detection)
- **LLM Inference**: vLLM (GPU-accelerated Llama 3.1 8B)
- **Validation**: Gemini CLI (Gemini 2.5 Pro with 1M context)
- **Databases**: PostgreSQL 16 (with pgvector), Neo4j 5.x, ChromaDB
- **Dashboard**: Flask + Jinja2 + responsive CSS
- **Deployment**: Systemd + Cloudflare Tunnel

---

## 🎛️ Configuration

### Environment Variables
All secrets are stored in `/etc/enlitens/enlitens.env`:
```bash
ENLITENS_ENABLE_POSTGRES=1
ENLITENS_ENABLE_NEO4J=1
ENLITENS_ENABLE_VECTOR_MIRROR=1
DATABASE_URL=postgresql:///enlitens
ENLITENS_NEO4J_URI=bolt://localhost:7687
ENLITENS_NEO4J_USER=neo4j
ENLITENS_NEO4J_PASSWORD=YourPassword
```

### Model Configuration
Edit `config/local_models.yaml` to configure LLM models.

---

## 📈 Monitoring

### Dashboard
Access the real-time dashboard at:
- **Local**: http://localhost:5000
- **Remote**: https://dashboard.enlitens.com (via Cloudflare Tunnel)

The dashboard shows:
- Documents processed, pending, and failed
- GPU usage and model status
- Recent processing activity
- Error logs

### Logs
```bash
# View processing logs
tail -f logs/processing.log

# View dashboard logs
tail -f logs/dashboard.log
```

---

## 🔧 Maintenance

### Backups
```bash
# Manual backup
bash scripts/backup/run_backup.sh

# Automated backups are stored in backups/ directory
```

### Clear Cache
```bash
# Clear Docling cache
rm -rf cache/*

# Clear logs
rm -rf logs/*
```

### Restart Services
```bash
# Restart dashboard
bash scripts/dashboard/start_dashboard.sh

# Restart vLLM
bash scripts/model_management/start_local_model.sh llama
```

---

## 📖 Documentation

- **[Hosting Guide](docs/hosting_guide.md)** - Full deployment instructions
- **[Architecture](docs/ARCHITECTURE_V2_SUMMARY.md)** - System architecture overview
- **[Knowledge Base](docs/README_KNOWLEDGE_BASE.md)** - Knowledge base structure

---

## 🛠️ Development

### Adding New PDFs
```bash
# Add PDFs to the input directory
cp your_paper.pdf enlitens_corpus/input_pdfs/

# Run ingestion
python scripts/ingestion/run_ingest_batch.py --auto-start --auto-stop
```

### Testing
```bash
# Run a single document pilot
python scripts/utilities/run_single_document_pilot.py
```

---

## 📝 License

This project is proprietary and confidential.

---

## 🤝 Support

For questions or issues, contact the development team.
