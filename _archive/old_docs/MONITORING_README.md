# Enlitens AI Enhanced Monitoring System

## Overview

The enhanced monitoring system provides comprehensive real-time visibility into the Enlitens AI knowledge base extraction pipeline. It features an intelligent Foreman AI that monitors all agents, tracks quality metrics, and provides insights into processing status.

## Features

### 🤖 Intelligent Foreman AI
- Powered by vLLM (`qwen2.5-32b-instruct-q4_k_m`) with prompt prefix caching
- Automatically falls back to Groq Llama-3.1-8B when the local GPU is offline
- Analyzes errors and provides troubleshooting suggestions
- Monitors agent pipeline and processing status
- Answers questions about quality metrics and system health
- Provides real-time insights and recommendations

### 📊 Real-Time Processing Monitoring
- **Current Document Tracking**: See exactly which document is being processed
- **Progress Tracking**: Live progress percentage and document count
- **Time Metrics**: Track processing time per document and time since last log
- **Agent Status**: Monitor which agents are running, completed, or failed
- **Supervisor Hierarchy**: Visualize supervisor stages and agent pipeline

### ✅ Quality Dashboard
- **Overall Quality Score**: Easy-to-understand quality rating (Excellent/Good/Fair/Poor)
- **Citation Verification**: Track how many citations have been verified
- **Validation Failures**: Monitor quality check failures
- **Empty Fields Detection**: See how many fields are missing data
- **Visual Metrics**: Comprehensive cards showing all quality indicators

### 🔄 Agent Pipeline Visualization
- Real-time agent status (running/completed/failed/idle)
- Supervisor stage tracking
- Agent performance metrics
- Hierarchical view of agent execution

### 📄 JSON Knowledge Base Viewer
- Load and view the latest knowledge base JSON
- Syntax-highlighted display
- Quality validation of extracted data
- Refresh on demand

### 📋 Live Logs
- Real-time log streaming via WebSocket
- Color-coded log levels (Debug/Info/Warning/Error/Critical)
- Filtering by log level and search text
- Auto-scrolling with pause capability
- Export logs to JSON

## Architecture

### Components

1. **Enhanced Monitoring Server** (`monitoring_server_enhanced.py`)
   - FastAPI web server with WebSocket support
   - ProcessingState tracker for comprehensive metrics
   - ForemanAI router that prefers local vLLM and falls back to Groq
   - API endpoints for stats, pipeline, knowledge base, and Foreman health

2. **Enhanced Web UI** (`monitoring_ui/`)
   - `index_enhanced.html`: Main dashboard with multiple views
   - `app_enhanced.js`: Client-side JavaScript for real-time updates
   - `styles.css`: Comprehensive styling with dark theme

3. **Remote Log Forwarding** (`src/utils/enhanced_logging.py`)
   - RemoteLogHandler sends logs to monitoring server
   - Automatic retry with exponential backoff
   - Error suppression to prevent log flooding

## Installation

### Prerequisites

1. **Python 3.8+** with required packages:
   ```bash
   pip install fastapi uvicorn httpx websockets
   ```

2. **vLLM** runtime installed (GPU build recommended):
   ```bash
   pip install "vllm>=0.5.4"
   ```

3. **Quantised Qwen models** downloaded (weights stored locally):
   - `qwen2.5-32b-instruct-q4_k_m`
   - `qwen2.5-3b-instruct-q4_k_m`

4. **Enlitens AI Processing System** configured and ready

### Setup

1. **Start vLLM servers** (main + monitoring models):
   ```bash
   ./stable_run.sh  # starts vLLM daemons before launching processing
   ```
   Main URL: `http://localhost:8000/v1`
   Foreman URL: `http://localhost:8001/v1`

2. **Start Enhanced Monitoring Server**:
   ```bash
   python monitoring_server_enhanced.py --host 0.0.0.0 --port 8765
   ```

   Options:
   - `--host`: Host to bind to (default: 0.0.0.0)
   - `--port`: Port to bind to (default: 8765)
   - `--reload`: Enable auto-reload for development

3. **Configure Remote Logging** in your processing system:
   Set environment variable before running:
   ```bash
   export ENLITENS_MONITOR_URL="http://localhost:8765/api/log"
   ```

   Or pass to `setup_enhanced_logging()`:
   ```python
   logger = setup_enhanced_logging(
       log_file="enlitens_processing.log",
       remote_logging_url="http://localhost:8765/api/log"
   )
   ```

4. **Open Dashboard**:
   Navigate to: `http://localhost:8765`

## Model Switch SOP (Local vLLM ↔ Groq)

1. **Run health checks** before switching:
   ```bash
   python -m src.monitoring.health_checks
   ```
   Ensure the vLLM batching, document latency (<10 minutes), and Foreman responsiveness tests all pass.

2. **Switch to Groq fallback** when the GPU node is unavailable:
   ```bash
   export FOREMAN_LOCAL_URL=""
   export GROQ_API_KEY="<your_groq_key>"
   export GROQ_MODEL="llama-3.1-8b-instant"
   ```
   Restart `monitoring_server_enhanced.py` to pick up the change.

3. **Return to local vLLM** once the GPU is back online:
   ```bash
   export FOREMAN_LOCAL_URL="http://localhost:8001/v1"
   unset GROQ_API_KEY
   ```
   Restart the monitoring server and rerun the health checks.

4. **Document the switch** in the operations log with:
   - Timestamp and reason for the change
   - Health check results before/after
   - Any remediation steps taken

## Usage

### Dashboard Navigation

The sidebar provides access to 6 main views:

1. **📋 Live Logs** - Real-time streaming logs with filtering
2. **🔄 Agent Pipeline** - Supervisor and agent hierarchy visualization
3. **✅ Quality Dashboard** - Comprehensive quality metrics
4. **📄 JSON Viewer** - Knowledge base output viewer
5. **🏗️ Foreman AI** - Chat with intelligent monitoring AI
6. **📈 Statistics** - Detailed processing and performance stats

### Using Foreman AI

The Foreman AI can answer questions about:
- Current processing status
- Recent errors and warnings
- Quality metrics and validation
- Agent performance
- Troubleshooting suggestions

**Example Questions:**
- "What's the current processing status?"
- "Analyze the latest error"
- "How's the quality looking?"
- "Which agents are running?"
- "Why is processing slow?"
- "Are there any validation failures?"

### Quality Scoring

Quality scores are calculated based on:
- ✅ **Bonus**: Citations verified (+1 point each, max +20)
- ❌ **Penalty**: Validation failures (-5 points each)
- ❌ **Penalty**: Errors (-10 points each, max -50)
- ⚠️ **Penalty**: Warnings (-3 points each, max -20)
- ⚠️ **Penalty**: Empty fields (-2 points each, max -20)

**Score Ratings:**
- **90-100%**: 🟢 Excellent
- **75-89%**: 🟢 Good
- **60-74%**: 🟡 Fair
- **0-59%**: 🔴 Poor

### API Endpoints

The monitoring server exposes these REST endpoints:

- `GET /` - Dashboard HTML
- `GET /api/stats` - Current processing statistics
- `GET /api/knowledge-base` - Latest knowledge base JSON
- `GET /api/agent-pipeline` - Agent pipeline hierarchy
- `POST /api/log` - Receive log entries (used by RemoteLogHandler)
- `WebSocket /ws` - Real-time log streaming and Foreman AI

## Configuration

### Environment Variables

- `VLLM_BASE_URL`: Primary inference endpoint (default: `http://localhost:8000/v1`)
- `FOREMAN_LOCAL_URL`: Optional monitoring model endpoint (default: `http://localhost:8001/v1`)
- `GROQ_API_KEY`: Enables Groq fallback when provided
- `GROQ_MODEL`: Groq model identifier (default: `llama-3.1-8b-instant`)
- `ENLITENS_MONITOR_URL`: Monitoring server URL for remote logging

### Customization

**Change Foreman routing**:
Update the environment variables before launching the server:
```bash
export FOREMAN_LOCAL_URL="http://localhost:8001/v1"   # empty string disables local model
export GROQ_API_KEY="..."                             # optional Groq fallback
export GROQ_MODEL="llama-3.1-8b-instant"
```
No code changes are required; restart `monitoring_server_enhanced.py` to apply updates.

**Adjust Update Frequency**:
Edit `app_enhanced.js`:
```javascript
function startStatusPolling() {
    statusUpdateInterval = setInterval(async () => {
        // Poll stats
    }, 2000);  // Change this (milliseconds)
}
```

**Customize Quality Scoring**:
Edit `calculateQualityScore()` in `app_enhanced.js` to adjust weights.

## Troubleshooting

### Monitoring Server Won't Start

**Check port availability**:
```bash
lsof -i :8765  # Check if port is in use
```

**Try different port**:
```bash
python monitoring_server_enhanced.py --port 8766
```

### vLLM Connection Failed

**Verify vLLM is running**:
```bash
curl http://localhost:8000/v1/models | jq '.'
```

**Check Foreman endpoint**:
```bash
curl http://localhost:8001/v1/models | jq '.'
```

**Re-run health checks**:
```bash
python -m src.monitoring.health_checks
```

### No Logs Appearing

**Verify remote logging is configured**:
```python
logger = setup_enhanced_logging(
    log_file="enlitens.log",
    remote_logging_url="http://localhost:8765/api/log"
)
```

**Check network connectivity**:
```bash
curl -X POST http://localhost:8765/api/log \
  -H "Content-Type: application/json" \
  -d '{"type":"log","level":"INFO","message":"test"}'
```

### Foreman AI Not Responding

1. **Verify local monitoring model**:
   ```bash
   curl http://localhost:8001/v1/models | jq '.'
   ```

2. **Check Groq configuration** (if using fallback):
   ```bash
   echo $GROQ_API_KEY
   ```

3. **Use the Foreman health endpoint**:
   ```bash
   curl http://localhost:8765/api/foreman/health | jq '.'
   ```

4. **Run operational health checks**:
   ```bash
   python -m src.monitoring.health_checks
   ```

### Knowledge Base JSON Not Loading

**Verify file exists**:
```bash
ls -lh enlitens_knowledge_base_latest.json
```

**Check file path in server**:
Edit `monitoring_server_enhanced.py`:
```python
KNOWLEDGE_BASE_PATH = Path("enlitens_knowledge_base_latest.json")
```

## Performance

### Resource Usage

- **CPU**: Minimal (< 5% on modern systems)
- **Memory**: ~50-100 MB for server
- **Network**: ~1-10 KB/s for log streaming
- **vLLM main**: ~18GB VRAM for `qwen2.5-32b-instruct-q4_k_m`

### Optimization Tips

1. **Reduce Polling Frequency**: Increase interval in `startStatusPolling()`
2. **Limit Log History**: Adjust `max_history` in ConnectionManager
3. **Use Smaller vLLM Model**: Switch to a quantised 13B/7B checkpoint if GPU memory is constrained
4. **Disable Features**: Comment out unnecessary API calls

## Security Considerations

⚠️ **Warning**: This monitoring system is intended for local development use.

**For Production Deployment**:
1. Add authentication/authorization
2. Use HTTPS/WSS for secure connections
3. Implement rate limiting
4. Restrict CORS origins
5. Validate all inputs
6. Use environment variables for sensitive config

## Advanced Features

### Custom Agent Metrics

Track custom metrics by adding to ProcessingState:
```python
class ProcessingState:
    def __init__(self):
        # ... existing fields
        self.custom_metrics = {}
```

### Export Capabilities

- **Logs**: Click "💾 Export" to download logs as JSON
- **Stats**: Use `/api/stats` endpoint to programmatically access data
- **Knowledge Base**: Use `/api/knowledge-base` for latest extraction

### Integration with External Tools

**Send alerts on errors**:
```python
# In monitoring_server_enhanced.py
def update_from_log(self, log_data: Dict[str, Any]):
    if log_data.get("level") == "ERROR":
        # Send alert via email/Slack/etc
        send_alert(log_data)
```

**Export metrics to time-series database**:
```python
async def broadcast(self, message: Dict[str, Any]):
    await super().broadcast(message)
    # Export to Prometheus/InfluxDB/etc
    export_metric(message)
```

## Support

For issues, questions, or feature requests:
1. Check the troubleshooting section above
2. Review server logs for detailed error messages
3. Check browser console for client-side errors
4. Verify all dependencies are installed correctly

## Version History

### v2.0.0 - Enhanced Monitoring System
- ✅ Intelligent Foreman AI with vLLM + Groq routing
- ✅ Real-time processing status tracking
- ✅ Comprehensive quality dashboard
- ✅ Agent pipeline visualization
- ✅ JSON knowledge base viewer
- ✅ Enhanced UI with multiple views
- ✅ Remote log forwarding
- ✅ WebSocket-based real-time updates

### v1.0.0 - Basic Monitoring
- Basic log viewer
- Simple WebSocket streaming
- Quality metrics tracking

---

**Built for Enlitens AI Knowledge Base Extraction System**
