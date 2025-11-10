# 📦 Creating New Git Repo: enlitens-agents

## Overview

Once processing is complete and everything is working perfectly, we'll create a brand new git repository called `enlitens-agents` to house this entire project.

---

## Pre-Push Checklist

### ✅ Things to Include

**Core Code:**
- ✅ `src/` - All processing pipeline code
- ✅ `scripts/` - Startup and utility scripts
- ✅ `dashboard/` - Monitoring dashboard
- ✅ `enlitens_client_profiles/` - Persona generation and profiles
- ✅ `requirements.txt` - Python dependencies
- ✅ `README.md` - Project documentation
- ✅ `FINAL_CONFIGURATION.md` - System configuration
- ✅ `QDRANT_FIXED.md` - Vector DB setup
- ✅ `DASHBOARD_FIXED.md` - Dashboard documentation

**Configuration:**
- ✅ `.gitignore` - Exclude large files and sensitive data
- ✅ `process_multi_agent_corpus.py` - Main processing script

### ❌ Things to Exclude (.gitignore)

**Large Files:**
- ❌ `models/` - AI models (16-30GB each)
- ❌ `enlitens_corpus/input_pdfs/` - 345 PDFs
- ❌ `enlitens_knowledge_base.json` - Output file (will be huge)
- ❌ `qdrant_storage/` - Vector database files
- ❌ `*.safetensors` - Model weight files
- ❌ `*.bin` - Binary model files

**Logs & Temporary Files:**
- ❌ `logs/` - All log files
- ❌ `*.log` - Individual log files
- ❌ `*.temp` - Temporary files
- ❌ `*.progress` - Progress tracking files
- ❌ `__pycache__/` - Python cache
- ❌ `*.pyc` - Compiled Python
- ❌ `.pytest_cache/` - Test cache

**Virtual Environment:**
- ❌ `venv/` - Python virtual environment
- ❌ `.venv/` - Alternative venv name

**Sensitive Data:**
- ❌ `enlitens_knowledge_base/liz_transcripts.txt` - Private transcripts
- ❌ `enlitens_client_profiles/intakes.txt` - Client data (if contains PII)
- ❌ `st_louis_health_report.pdf` - May contain sensitive info
- ❌ `.env` - Environment variables (if any)
- ❌ `*.key` - API keys

**OS & IDE Files:**
- ❌ `.DS_Store` - macOS
- ❌ `Thumbs.db` - Windows
- ❌ `.vscode/` - VS Code settings
- ❌ `.idea/` - PyCharm settings

---

## .gitignore File

Create this file in the project root:

```gitignore
# AI Models (too large for git)
models/
*.safetensors
*.bin
*.gguf
*.ggml

# Input Data
enlitens_corpus/input_pdfs/
*.pdf

# Output Data
enlitens_knowledge_base.json
enlitens_knowledge_base.json.temp
enlitens_knowledge_base.json.progress
*.json.temp
*.json.progress

# Vector Database
qdrant_storage/
*.db

# Logs
logs/
*.log

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.venv/
ENV/
env/
.pytest_cache/
.coverage
htmlcov/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Sensitive Data
enlitens_knowledge_base/liz_transcripts.txt
enlitens_client_profiles/intakes.txt
st_louis_health_report.pdf
.env
*.key
*.pem

# Temporary Files
*.tmp
*.bak
*.cache
```

---

## README.md for New Repo

```markdown
# Enlitens Agents

Multi-agent AI system for processing neurodivergence research papers into a comprehensive, client-focused knowledge base.

## Overview

Enlitens Agents processes research PDFs through a sophisticated multi-agent pipeline to create:
- Clinical insights and treatment implications
- Patient-friendly educational content
- Neurodivergent-affirming perspectives
- Marketing and SEO-optimized content
- Cross-referenced entity extraction

## Key Features

- **128k Context Window** - Processes entire papers with full context
- **57 Client Personas** - Real stories drive authentic content
- **5 Specialized NER Models** - Medical, genetic, clinical entity extraction
- **Vector Database (Qdrant)** - Cross-document connections and RAG
- **External Search Integration** - Auto-enriches knowledge gaps
- **Real-time Dashboard** - Monitor processing, GPU/CPU, logs

## Architecture

- **Main LLM:** Qwen 2.5-14B (4-bit AWQ, 128k context)
- **NER Models:** OpenMed suite + ClinicalDistilBERT (CPU)
- **Vector DB:** Qdrant (local file storage)
- **Pipeline:** Multi-agent workflow with validation

## Requirements

- NVIDIA GPU with 24GB+ VRAM
- 64GB+ RAM
- Python 3.10+
- CUDA 12.1+

## Installation

\`\`\`bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/enlitens-agents.git
cd enlitens-agents

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models (see MODELS.md)
# Add your PDFs to enlitens_corpus/input_pdfs/
# Add client personas to enlitens_client_profiles/profiles/
\`\`\`

## Usage

\`\`\`bash
# Start vLLM with 128k context
bash scripts/start_vllm_128k.sh

# Start processing
python3 process_multi_agent_corpus.py \\
    --input-dir enlitens_corpus/input_pdfs \\
    --output-file enlitens_knowledge_base.json \\
    --st-louis-report st_louis_health_report.pdf

# Start monitoring dashboard
cd dashboard && python3 server.py
\`\`\`

## Documentation

- `FINAL_CONFIGURATION.md` - Complete system configuration
- `QDRANT_FIXED.md` - Vector database setup
- `DASHBOARD_FIXED.md` - Monitoring dashboard guide

## License

[Your License Here]

## Contact

[Your Contact Info]
\`\`\`

---

## Steps to Create New Repo

### 1. Create Repo on GitHub

```bash
# Go to GitHub.com
# Click "New Repository"
# Name: enlitens-agents
# Description: Multi-agent AI system for neurodivergence research processing
# Public or Private: [Your choice]
# Do NOT initialize with README (we have one)
# Click "Create Repository"
```

### 2. Initialize Local Git

```bash
cd /home/antons-gs/enlitens-ai

# Initialize git
git init

# Add .gitignore
# (Create the .gitignore file from above first)

# Add all files
git add .

# First commit
git commit -m "Initial commit: Enlitens multi-agent processing system

- Multi-agent pipeline with 128k context
- 57 client personas integration
- 5 specialized NER models
- Qdrant vector database
- Real-time monitoring dashboard
- Comprehensive documentation"

# Add remote
git remote add origin https://github.com/YOUR_USERNAME/enlitens-agents.git

# Push
git branch -M main
git push -u origin main
```

### 3. Post-Push Setup

**Add Topics on GitHub:**
- `ai`
- `nlp`
- `multi-agent`
- `neurodivergence`
- `research-processing`
- `rag`
- `vector-database`

**Create Releases:**
- Tag: `v1.0.0`
- Title: "Initial Release - 128k Context Multi-Agent System"

**Add Documentation:**
- Wiki pages for detailed setup
- Issues templates
- Contributing guidelines (if open source)

---

## Important Notes

### Model Files
Models are NOT included in git (too large). Users must download separately:

```bash
# Qwen 2.5-14B AWQ
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-AWQ \\
    --local-dir models/qwen2.5-14b-instruct-awq

# NER models download automatically on first run
```

### Client Data
- Personas in `enlitens_client_profiles/profiles/` should be anonymized
- Remove any PII before pushing
- Consider encrypting sensitive data

### Configuration
- Create `.env.example` with required environment variables
- Document all configuration options
- Provide example configs

---

## Timeline

**Do this AFTER:**
1. ✅ vLLM is running with 128k context
2. ✅ Processing pipeline is tested and working
3. ✅ Dashboard is functional
4. ✅ Documentation is complete
5. ✅ All sensitive data is removed/anonymized

**Then:**
1. Create .gitignore
2. Review all files for sensitive data
3. Test on a clean clone
4. Push to new repo

---

**Ready to push when you say so!** 🚀

