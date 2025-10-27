# Enlitens Multi-Agent Knowledge Base System

## 🚀 Overview

This is a sophisticated multi-agent AI system designed to process 344+ neuroscience research papers and generate high-quality, structured content for Enlitens therapy practice. The system uses advanced prompt engineering, multiple specialized AI agents, and comprehensive validation to ensure the highest quality outputs.

## 🏗️ Architecture

### Enhanced Multi-Agent System
- **Supervisor Agent**: Orchestrates the entire processing pipeline with quality-based retry mechanisms
- **Science Extraction Agent**: Extracts and analyzes neuroscience research with deep clinical relevance
- **Clinical Synthesis Agent**: Converts research into practical therapy applications with evidence-based protocols
- **Founder Voice Agent**: Captures Liz Wooten's authentic voice and rebellious brand personality
- **Context RAG Agent**: Provides contextual intelligence using Qdrant vector database with persistent storage
- **Marketing SEO Agent**: Optimizes content for search engines and conversion with St. Louis market intelligence
- **Validation Agent**: Ensures quality, completeness, confidence scoring, and fact checking of all outputs

### Advanced Quality Assurance
- **Confidence Scoring**: Multi-factor confidence assessment for each output
- **Fact Checking**: Validates neuroscience and clinical accuracy against established knowledge
- **Retry Mechanisms**: Automatically retries poor quality outputs with targeted improvements
- **Quality Thresholds**: Ensures minimum quality standards are met (75%+ clinical accuracy)
- **Progress Validation**: Real-time quality monitoring throughout processing

### Key Features
- ✅ GPU memory optimization for 24GB VRAM systems
- ✅ Sequential agent processing to prevent memory issues
- ✅ Comprehensive error handling and recovery
- ✅ St. Louis regional context integration with RAG enhancement
- ✅ Founder voice authenticity preservation with personality analysis
- ✅ Advanced quality validation with confidence scoring and fact checking
- ✅ Robust retry mechanisms for poor quality outputs
- ✅ Progress tracking and comprehensive logging (single file for all 344 documents)
- ✅ BERTopic analysis for discovering client pain points and themes
- ✅ Automatic cleanup of old files and fresh log generation

## 🛠️ Setup Instructions

### 1. Environment Setup
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install NVIDIA drivers (if not already installed)
sudo apt install nvidia-driver-535

# Install Python dependencies
pip install -r requirements.txt

# Set environment variables for optimal performance
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_MAX_QUEUE=1
export OLLAMA_RUNNERS_DIR=/tmp/ollama-runners
export TORCH_USE_CUDA_DSA=1
```

### 2. Ollama Setup
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve

# Pull the required model (in another terminal)
ollama pull qwen3:32b
```

### 3. Verify GPU Setup
```bash
# Check GPU status
nvidia-smi

# Check CUDA version
nvcc --version

# Test PyTorch GPU support
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🚀 Usage

### Option 1: Stable Execution (Recommended)
```bash
# Run with stable execution script (prevents SSH tunnel issues)
./stable_run.sh

# Monitor progress
python3 monitor_processing.py --continuous --interval 30
```

### Option 2: Direct Execution
```bash
# Run the multi-agent processor directly
python3 process_multi_agent_corpus.py \
    --input-dir enlitens_corpus/input_pdfs \
    --output-file enlitens_knowledge_base_final.json \
    --st-louis-report st_louis_health_report.pdf
```

### Option 3: Test First
```bash
# Test the system with a single document
python3 test_multi_agent_system.py
```

## 📊 Monitoring

### Real-time Monitoring
```bash
# Monitor processing in real-time
python3 monitor_processing.py --continuous

# Check system resources
nvidia-smi

# Monitor specific log files
tail -f logs/multi_agent_processing_*.log

# Check Ollama status
curl http://localhost:11434/api/tags
```

### Progress Tracking
- 📁 **Output files**: `enlitens_knowledge_base_*.json`
- 📝 **Log files**: `logs/multi_agent_processing_*.log`
- 💾 **Temp files**: `enlitens_knowledge_base_*.json.temp` (auto-cleaned)
- 🖥️ **System status**: `python3 monitor_processing.py`

## 🔧 Troubleshooting

### SSH Tunnel Issues
If you see thousands of SSH tunnel errors:
```bash
# Kill stuck processes
pkill -f "ssh_tunnel"
pkill -f "process_.*corpus"

# Clear terminal state
stty sane

# Run with stable script
./stable_run.sh
```

### GPU Memory Issues
```bash
# Check GPU memory
nvidia-smi

# Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache()"

# Restart Ollama with memory limits
pkill ollama
ollama serve
```

### Common Issues
1. **Ollama not responding**: Restart with `ollama serve`
2. **GPU out of memory**: Process fewer documents or restart system
3. **JSON parsing errors**: Check log files for specific error messages
4. **Slow processing**: Monitor GPU utilization with `nvidia-smi`

## 📋 Expected Output

### Knowledge Base Structure
The system generates comprehensive JSON files with:
- **Research Content**: Scientific findings and evidence with citations
- **Clinical Applications**: Therapy techniques and protocols with confidence scores
- **Marketing Content**: Headlines, taglines, and value propositions optimized for St. Louis
- **SEO Content**: Keywords and optimization data with local market intelligence
- **Website Copy**: Complete website content sections with conversion elements
- **Blog Content**: Article ideas and educational materials with engagement strategies
- **Social Media**: Post ideas and engagement content with authentic founder voice
- **Educational Content**: Client-friendly explanations with neuroscience backing

### Enhanced Quality Metrics
- **Clinical Accuracy**: 75%+ neuroscience terminology validation and fact checking
- **Founder Voice**: 65%+ authentic Liz Wooten language patterns and rebellious tone
- **Marketing Effectiveness**: 80%+ conversion optimization with St. Louis relevance
- **Completeness**: 90%+ all required sections populated
- **Confidence Scoring**: 70%+ overall confidence threshold with multi-factor assessment
- **Fact Checking**: Validated against established neuroscience and clinical knowledge
- **Retry Success**: Automatic improvement of outputs below quality thresholds

### Quality Assurance Features
- **Multi-Agent Validation**: Each agent output validated by specialized quality checks
- **Retry Mechanisms**: Up to 3 automatic retries for outputs below quality thresholds
- **Confidence Scoring**: Real-time confidence assessment with detailed breakdowns
- **Progress Monitoring**: Quality metrics tracked throughout processing
- **Final Validation**: Comprehensive quality report with improvement suggestions

## 🎯 Enhanced Processing Stages

1. **Science Extraction**: Extract research findings and neuroscience data with evidence validation
2. **Clinical Synthesis**: Convert to practical therapy applications with confidence scoring
3. **Founder Voice Integration**: Apply Liz Wooten's authentic voice with personality analysis
4. **Context Enhancement**: Add St. Louis regional context via RAG and BERTopic analysis
5. **Marketing Optimization**: Optimize for search and conversion with local market intelligence
6. **Quality Validation**: Comprehensive validation with confidence scoring and fact checking
7. **Retry & Improvement**: Automatic retry of poor quality outputs with targeted enhancements
8. **Final Quality Assurance**: Multi-agent validation and quality metric compilation

## 🏙️ St. Louis Context Integration

The system incorporates:
- **Client Challenges**: ADHD, anxiety, trauma, treatment resistance
- **Regional Factors**: Poverty, violence, cultural diversity
- **Healthcare Barriers**: Access issues, stigma, transportation
- **Market Intelligence**: Local search trends and competition

## 🔐 Data Privacy & Ethics

- All processing happens locally on your system
- No data is sent to external services except Ollama (local)
- Client intake data is anonymized and processed locally
- Founder transcripts are used only for voice pattern analysis

## 📞 Support

If you encounter issues:
1. Check the log files in the `logs/` directory
2. Run the test script: `python3 test_multi_agent_system.py`
3. Monitor system resources: `python3 monitor_processing.py`
4. Check GPU status: `nvidia-smi`

## 🚀 Next Steps

After successful processing:
1. Review the generated JSON files for quality
2. Import into your content management system
3. Fine-tune the model with your specific requirements
4. Set up automated weekly processing for new research papers

---

**Built for Enlitens Therapy | Neuroscience-Based Mental Health Care**
**St. Louis, Missouri | Challenging Traditional Therapy Approaches**
