# Enhanced Monitoring System - Quick Start Guide

## What Was Built

I've created a comprehensive enhanced monitoring system that gives you complete real-time visibility into your Enlitens AI processing pipeline, exactly as you requested. Here's what you now have:

### 🤖 Intelligent Foreman AI
An AI-powered monitoring agent that watches over all your other agents like "HR or Internal Affairs". The Foreman:
- Uses Ollama (qwen3:32b model) for intelligent analysis
- Investigates errors and provides troubleshooting suggestions
- Monitors the supervisor/agent hierarchy in real-time
- Answers questions about processing status, quality, and errors
- Provides actionable insights and recommendations

### 📊 Comprehensive Real-Time Monitoring
As you requested, I've added "as much information and data and stats as possible":
- **Current document** being processed with filename
- **When last log update occurred** (seconds ago)
- **Progress percentage** and document count (X/Y documents)
- **Time spent** on current document
- **Agent status** for every agent (running/completed/failed/idle)
- **Supervisor stages** showing what the supervisor is doing
- **Quality metrics** with citations, validations, and errors
- **Recent errors and warnings** for immediate visibility

### ✅ User-Friendly Quality System
Easy-to-understand quality scoring that shows at a glance:
- **Overall Score**: 0-100% with visual indicators
- **Quality Rating**: Excellent/Good/Fair/Poor (color-coded)
- **Citation Tracking**: How many citations were verified
- **Validation Status**: Any quality check failures
- **Empty Fields**: Missing data detection

### 🔄 Agent Pipeline Visualization
See your "agents doing individual tasks working together answering to supervisor agents":
- Visual hierarchy showing Supervisor → Agents
- Real-time status for each agent
- Current supervisor stage
- Agent performance metrics

### 📄 JSON Quality Viewer
Confirm the quality of your knowledge base extraction:
- Load and view the latest `enlitens_knowledge_base_latest.json`
- Syntax-highlighted JSON display
- Refresh on demand to see latest results

## Quick Start

### 1. Start Ollama (if not running)
```bash
ollama serve
```

Make sure you have the qwen3:32b model:
```bash
ollama pull qwen3:32b
```

### 2. Start the Enhanced Monitoring Server
```bash
cd /home/user/enlitens-ai
python monitoring_server_enhanced.py
```

You should see:
```
================================================================================
🏗️  ENLITENS AI ENHANCED MONITORING SERVER
================================================================================
📊 Dashboard: http://0.0.0.0:8765
🔌 WebSocket: ws://0.0.0.0:8765/ws
📡 Log Endpoint: http://0.0.0.0:8765/api/log
📈 Stats: http://0.0.0.0:8765/api/stats
🤖 Using Ollama at: http://localhost:11434
================================================================================
```

### 3. Open the Dashboard
Open your browser to: **http://localhost:8765**

You'll see the enhanced dashboard with:
- Processing status bar at the top
- Sidebar with 6 navigation views
- Live logs streaming in real-time

### 4. Configure Your Processing System
Before running your knowledge base extraction, set the environment variable:

```bash
export ENLITENS_MONITOR_URL="http://localhost:8765/api/log"
```

This tells the enhanced logging system to forward logs to the monitoring server.

### 5. Run Your Processing
```bash
# Your normal command to run processing, for example:
python main.py
```

As soon as processing starts, you'll see:
- Logs streaming in real-time
- Processing status updating
- Agents appearing in the pipeline
- Quality metrics accumulating

## Testing the Features

### Test 1: View Real-Time Logs
1. Navigate to **📋 Live Logs** (default view)
2. Watch logs stream in as processing happens
3. Try filtering by log level (dropdown)
4. Search for specific text using the search box

### Test 2: Monitor Agent Pipeline
1. Click **🔄 Agent Pipeline** in the sidebar
2. See the Supervisor Agent card
3. Watch as agents appear and their status updates
4. Observe agents transitioning: running → completed

### Test 3: Check Quality Dashboard
1. Click **✅ Quality Dashboard**
2. See overall quality score with rating
3. Review metrics cards:
   - Processing Progress
   - Citations Verified
   - Validation Failures
   - Empty Fields
   - Active Agents
   - Recent Errors
   - Processing Time

### Test 4: View JSON Output
1. Click **📄 JSON Viewer**
2. Click **🔄 Refresh** button
3. View the syntax-highlighted knowledge base JSON
4. Scroll through to verify extraction quality

### Test 5: Chat with Foreman AI
1. Click **🏗️ Foreman AI**
2. Try these questions:
   - "What's the current status?"
   - "Are there any errors?"
   - "How's the quality looking?"
   - "Which agents are running?"
3. Get intelligent AI-powered responses

### Test 6: Review Detailed Statistics
1. Click **📈 Statistics**
2. Review 4 stat boxes:
   - Processing Overview
   - Agent Performance
   - Timing Metrics
   - Quality Breakdown

## Exploring the Dashboard

### Processing Status Bar (Top)
Shows at a glance:
- **Current Document**: Which PDF is being processed
- **Progress**: X% complete (Y/Z documents)
- **Time on Document**: How long on current doc
- **Last Update**: Seconds since last log entry

### Sidebar Quick Stats
Real-time counters:
- **Documents**: X/Y processed
- **Total Logs**: Count of all log entries
- **Errors**: Error count (red)
- **Warnings**: Warning count (orange)
- **Quality Score**: Overall quality %

### Control Buttons
- **🗑️ Clear Logs**: Reset log display
- **⏸️ Pause**: Pause/resume log streaming
- **💾 Export**: Download logs as JSON
- **🔄 Refresh JSON**: Reload knowledge base

## Verifying It's Working

### Check WebSocket Connection
In the top-right corner, you should see:
- Green pulse indicator
- "Connected" status

If disconnected:
- Red pulse indicator
- "Disconnected" status
- Automatic reconnection attempts every 3 seconds

### Check Foreman AI
1. Go to Foreman AI view
2. Type: "test"
3. Send message
4. You should get a response analyzing current stats

If Foreman doesn't respond:
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Check browser console for errors
- The system will fall back to heuristic responses if Ollama is unavailable

### Check Remote Logging
Run a simple test:
```bash
curl -X POST http://localhost:8765/api/log \
  -H "Content-Type: application/json" \
  -d '{
    "type": "log",
    "timestamp": "2024-01-01T12:00:00",
    "level": "INFO",
    "logger": "test",
    "message": "Test log entry from curl"
  }'
```

You should immediately see this log appear in the dashboard.

## What Makes This Special

### Multi-Agent Orchestration
You now have full visibility into your multi-agent system:
- **Supervisor Level**: See what stage the supervisor is on
- **Agent Level**: Monitor each specialized agent
- **Foreman Level**: AI monitoring everything from above

This gives you the "many agents doing individual tasks working together answering to supervisor agents and foremans monitoring it all" that you requested.

### Intelligent Error Analysis
Unlike simple log viewers, the Foreman AI:
- Understands the context of errors
- Can suggest solutions based on current state
- Learns from the processing pipeline
- Provides actionable insights

### Real-Time Everything
Every metric updates automatically:
- Status bar: every 2 seconds
- Logs: instant streaming via WebSocket
- Quality metrics: calculated in real-time
- Agent status: immediate updates

## Troubleshooting

### Port Already in Use
If port 8765 is taken:
```bash
python monitoring_server_enhanced.py --port 8766
```
Then update your URL to `http://localhost:8766`

### No Logs Appearing
1. Verify `ENLITENS_MONITOR_URL` is set:
   ```bash
   echo $ENLITENS_MONITOR_URL
   ```
2. Check processing system is using enhanced logging
3. Verify network connectivity to monitoring server

### Ollama Connection Failed
1. Check Ollama is running:
   ```bash
   ps aux | grep ollama
   ```
2. Test Ollama directly:
   ```bash
   ollama list  # Should show qwen3:32b
   ```
3. If Ollama isn't available, Foreman will still work with fallback responses

### JSON Viewer Shows Error
1. Verify `enlitens_knowledge_base_latest.json` exists:
   ```bash
   ls -lh enlitens_knowledge_base_latest.json
   ```
2. Run processing first to generate the file
3. Check file permissions

## Next Steps

1. **Run a Full Processing Session**: Start processing your PDFs and watch the entire pipeline in real-time
2. **Interact with Foreman**: Ask questions during processing to get insights
3. **Monitor Quality**: Keep an eye on the quality dashboard to catch issues early
4. **Export Results**: Use the JSON viewer to confirm extraction quality
5. **Review Errors**: Use Foreman to analyze any errors that occur

## Files Created

All code is committed and pushed to branch `claude/review-enlightens-snapshots-011CUbybvVxRke7h8pYz4Wo5`:

- `monitoring_server_enhanced.py` - FastAPI server with Foreman AI
- `monitoring_ui/index_enhanced.html` - Enhanced dashboard HTML
- `monitoring_ui/app_enhanced.js` - Client-side JavaScript
- `monitoring_ui/styles.css` - Updated CSS (with new styles)
- `MONITORING_README.md` - Full documentation
- `QUICKSTART_MONITORING.md` - This guide

## Support

For detailed documentation, see `MONITORING_README.md`.

For troubleshooting, check the server logs and browser console for specific error messages.

---

**You now have a production-ready monitoring system with intelligent AI assistance!** 🚀
