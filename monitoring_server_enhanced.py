#!/usr/bin/env python3
"""
Enhanced Real-Time Monitoring Server for Enlitens AI Processing

Features:
- WebSocket log streaming with agent tracking
- Real-time processing metrics with document tracking
- Quality monitoring with detailed breakdown
- Intelligent Foreman AI using Ollama
- JSON output viewer and validator
- Agent pipeline visualization
- Supervisor/Agent hierarchy monitoring
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
import argparse
from collections import defaultdict, deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import httpx

app = FastAPI(title="Enlitens AI Monitor Enhanced", version="2.0.0")

# Configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
KNOWLEDGE_BASE_PATH = Path("enlitens_knowledge_base_latest.json")
STATIC_DIR = Path(__file__).parent / "monitoring_ui"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Processing state tracker
class ProcessingState:
    def __init__(self):
        self.current_document = None
        self.current_document_start = None
        self.last_log_time = None
        self.documents_processed = 0
        self.total_documents = 0
        self.agent_status = {}
        self.agent_pipeline = []
        self.errors = []
        self.warnings = []
        self.quality_metrics = {
            "citation_verified": 0,
            "validation_failures": 0,
            "empty_fields": 0,
            "avg_quality_score": 0.0,
            "precision_at_3": None,
            "recall_at_3": None,
            "faithfulness": None,
            "hallucination_rate": None,
            "layer_failures": [],
            "last_quality_event": None,
            "documents_scored": []
        }
        self.agent_performance = defaultdict(lambda: {
            "executions": 0,
            "successes": 0,
            "failures": 0,
            "avg_time": 0.0,
            "times": deque(maxlen=10)
        })
        self.supervisor_stack = []

    def update_from_log(self, log_data: Dict[str, Any]):
        """Update state from incoming log message."""
        self.last_log_time = datetime.now()

        message = log_data.get("message", "")
        level = log_data.get("level", "INFO")

        message = message.strip()

        # Structured quality metrics payloads
        if message.startswith("QUALITY_METRICS"):
            payload = message[len("QUALITY_METRICS"):].strip()
            try:
                metrics_update = json.loads(payload)
                self.quality_metrics["precision_at_3"] = metrics_update.get("precision_at_3")
                self.quality_metrics["recall_at_3"] = metrics_update.get("recall_at_3")
                self.quality_metrics["faithfulness"] = metrics_update.get("faithfulness")
                self.quality_metrics["hallucination_rate"] = metrics_update.get("hallucination_rate")
                failures = metrics_update.get("layer_failures", []) or []
                self.quality_metrics["layer_failures"] = failures
                if failures:
                    self.quality_metrics["validation_failures"] += len(failures)
                self.quality_metrics["last_quality_event"] = metrics_update.get("evaluated_at")
                self.quality_metrics["documents_scored"].append(metrics_update)
                self.quality_metrics["documents_scored"] = self.quality_metrics["documents_scored"][-20:]
            except json.JSONDecodeError:
                logger.warning("Failed to parse quality metrics payload: %s", payload)

        # Track document processing
        if "Processing file" in message:
            match = re.search(r"Processing file (\d+)/(\d+): (.+)", message)
            if match:
                current, total, filename = match.groups()
                self.current_document = filename
                self.current_document_start = datetime.now()
                self.documents_processed = int(current) - 1
                self.total_documents = int(total)

        # Track document completion
        if "Successfully processed:" in message:
            self.documents_processed += 1

        # Track agent execution
        if "Executing agent:" in message:
            agent_name = message.split("Executing agent:")[-1].strip()
            self.agent_status[agent_name] = "running"
            if agent_name not in self.agent_pipeline:
                self.agent_pipeline.append(agent_name)

        # Track agent completion
        if "Agent" in message and "completed successfully" in message:
            match = re.search(r"Agent (\w+) completed", message)
            if match:
                agent_name = match.group(1)
                self.agent_status[agent_name] = "completed"

        # Track agent failures
        if "Agent" in message and "failed" in message:
            match = re.search(r"Agent (\w+).*failed", message)
            if match:
                agent_name = match.group(1)
                self.agent_status[agent_name] = "failed"
                self.agent_performance[agent_name]["failures"] += 1

        # Track supervisor stages
        if "Stage" in message and "Starting" in message:
            match = re.search(r"Stage \d+: Starting (.+)", message)
            if match:
                stage_name = match.group(1)
                self.supervisor_stack.append(stage_name)

        # Track quality metrics
        if "citation" in message.lower() and "verified" in message.lower():
            self.quality_metrics["citation_verified"] += 1
        if "validation" in message.lower() and "failed" in message.lower():
            self.quality_metrics["validation_failures"] += 1
        if "empty field" in message.lower():
            self.quality_metrics["empty_fields"] += 1

        # Track errors and warnings
        if level == "ERROR":
            self.errors.append({
                "timestamp": log_data.get("timestamp"),
                "message": message,
                "agent": log_data.get("agent_name")
            })
        elif level == "WARNING":
            self.warnings.append({
                "timestamp": log_data.get("timestamp"),
                "message": message
            })

    def get_current_status(self) -> Dict[str, Any]:
        """Get current processing status."""
        time_on_doc = None
        if self.current_document_start:
            time_on_doc = (datetime.now() - self.current_document_start).total_seconds()

        time_since_log = None
        if self.last_log_time:
            time_since_log = (datetime.now() - self.last_log_time).total_seconds()

        return {
            "current_document": self.current_document,
            "time_on_document_seconds": time_on_doc,
            "last_log_seconds_ago": time_since_log,
            "documents_processed": self.documents_processed,
            "total_documents": self.total_documents,
            "progress_percentage": (self.documents_processed / self.total_documents * 100) if self.total_documents > 0 else 0,
            "agent_status": dict(self.agent_status),
            "agent_pipeline": self.agent_pipeline,
            "active_agents": [a for a, s in self.agent_status.items() if s == "running"],
            "supervisor_stack": self.supervisor_stack.copy(),
            "recent_errors": self.errors[-5:],
            "recent_warnings": self.warnings[-5:],
            "quality_metrics": self.quality_metrics,
            "agent_performance": {k: dict(v) for k, v in self.agent_performance.items()}
        }

# Global state
processing_state = ProcessingState()


def format_percentage(value: Optional[float]) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value) * 100:.1f}%"
    except Exception:
        return "--"


# Connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.message_history: List[Dict[str, Any]] = []
        self.max_history = 1000

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

        # Send recent history
        for msg in self.message_history[-100:]:
            try:
                await websocket.send_json(msg)
            except:
                pass

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]

        # Update state from log messages
        if message.get("type") == "log":
            processing_state.update_from_log(message)

        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.add(connection)

        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Intelligent Foreman AI
class ForemanAI:
    """Intelligent AI assistant that uses Ollama to analyze logs and provide insights."""

    def __init__(self, ollama_url: str = OLLAMA_BASE_URL):
        self.ollama_url = ollama_url
        self.client = httpx.AsyncClient(timeout=30.0)

    async def analyze_query(self, query: str, context: Dict[str, Any]) -> str:
        """Use Ollama to analyze the query with current context."""

        # Prepare context summary
        status = processing_state.get_current_status()

        context_prompt = f"""You are the Foreman AI for the Enlitens AI knowledge base extraction system.

Current Processing Status:
- Current Document: {status['current_document'] or 'None'}
- Documents Processed: {status['documents_processed']}/{status['total_documents']}
- Progress: {status['progress_percentage']:.1f}%
- Time on Current Doc: {status['time_on_document_seconds']} seconds
- Active Agents: {', '.join(status['active_agents']) or 'None'}
- Recent Errors: {len(status['recent_errors'])}
- Recent Warnings: {len(status['recent_warnings'])}

Quality Metrics:
- Citations Verified: {status['quality_metrics']['citation_verified']}
- Validation Failures: {status['quality_metrics']['validation_failures']}
- Empty Fields Detected: {status['quality_metrics']['empty_fields']}

Agent Pipeline: {' → '.join(status['agent_pipeline'])}

User Query: {query}

Provide a helpful, concise response analyzing the situation and answering the user's question.
Be direct and actionable. If there are issues, suggest specific solutions."""

        try:
            response = await self.client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "qwen3:32b",
                    "prompt": context_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 500
                    }
                }
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "I'm having trouble analyzing that right now.")
            else:
                return f"Ollama service error: {response.status_code}"

        except Exception as e:
            # Fallback to heuristic response
            return self._fallback_response(query, status)

    def _fallback_response(self, query: str, status: Dict[str, Any]) -> str:
        """Fallback heuristic responses when Ollama is unavailable."""
        query_lower = query.lower()

        if any(word in query_lower for word in ['status', 'progress', 'how']):
            progress = status['progress_percentage']
            current = status['current_document'] or 'initializing'
            active = ', '.join(status['active_agents']) or 'waiting'
            return f"📊 **Processing Status:**\n- Currently on: `{current}`\n- Progress: **{progress:.1f}%** ({status['documents_processed']}/{status['total_documents']})\n- Active agents: {active}\n- Recent errors: {len(status['recent_errors'])}"

        elif any(word in query_lower for word in ['error', 'problem', 'issue', 'wrong']):
            if status['recent_errors']:
                latest = status['recent_errors'][-1]
                return f"⚠️ **Latest Error:**\n```\n{latest['message']}\n```\nAgent: {latest.get('agent', 'Unknown')}\nTime: {latest['timestamp']}"
            else:
                return "✅ No errors detected! System is running smoothly."

        elif any(word in query_lower for word in ['quality', 'hallucination', 'validation']):
            metrics = status['quality_metrics']
            precision = metrics.get('precision_at_3')
            recall = metrics.get('recall_at_3')
            faithfulness = metrics.get('faithfulness')
            hallucinations = metrics.get('hallucination_rate')
            return (
                "🎯 **Quality Metrics:**\n"
                f"- ✅ Citations Verified: {metrics['citation_verified']}\n"
                f"- 📐 Precision@3: {format_percentage(precision)}\n"
                f"- 📊 Recall@3: {format_percentage(recall)}\n"
                f"- 🔒 Faithfulness: {format_percentage(faithfulness)}\n"
                f"- 🧠 Hallucination Rate: {format_percentage(hallucinations)}\n"
                f"- ❌ Validation Failures: {metrics['validation_failures']}\n"
                f"- ⚠️ Empty Fields: {metrics['empty_fields']}\n"
                "\nOverall: "
                f"{'Excellent' if (precision or 0) >= 0.9 and (hallucinations or 0) < 0.1 else 'Needs review'}"
            )

        else:
            return f"I monitor {status['total_documents']} documents with {len(status['agent_pipeline'])} agents in the pipeline. Ask me about status, errors, or quality metrics!"

foreman_ai = ForemanAI()

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the enhanced monitoring dashboard."""
    index_file = STATIC_DIR / "index_enhanced.html"
    if index_file.exists():
        return FileResponse(index_file)
    # Fallback to standard index
    fallback = STATIC_DIR / "index.html"
    if fallback.exists():
        return FileResponse(fallback)
    return HTMLResponse("<h1>Monitoring UI not found</h1>", status_code=404)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            if message.get("type") == "foreman_query":
                query = message.get("query", "")
                response = await foreman_ai.analyze_query(query, {})
                await websocket.send_json({
                    "type": "foreman_response",
                    "query": query,
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/log")
async def receive_log(request: Request):
    """Receive log entries from the processing system."""
    try:
        log_data = await request.json()
        await manager.broadcast(log_data)
        return JSONResponse({"status": "ok"})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.get("/api/stats")
async def get_stats():
    """Get current processing statistics."""
    return JSONResponse({
        "connected_clients": len(manager.active_connections),
        "total_logs": len(manager.message_history),
        "status": "running",
        **processing_state.get_current_status()
    })

@app.get("/api/knowledge-base")
async def get_knowledge_base():
    """Get the latest knowledge base JSON for quality review."""
    if KNOWLEDGE_BASE_PATH.exists():
        try:
            with open(KNOWLEDGE_BASE_PATH, 'r') as f:
                data = json.load(f)
            return JSONResponse(data)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"error": "Knowledge base file not found"}, status_code=404)

@app.get("/api/agent-pipeline")
async def get_agent_pipeline():
    """Get the current agent processing pipeline and hierarchy."""
    status = processing_state.get_current_status()

    # Build hierarchical structure
    pipeline = {
        "supervisor": {
            "name": "SupervisorAgent",
            "status": "running" if status['current_document'] else "idle",
            "current_stage": status['supervisor_stack'][-1] if status['supervisor_stack'] else None,
            "agents": []
        }
    }

    # Add agents to pipeline
    for agent_name in status['agent_pipeline']:
        agent_info = {
            "name": agent_name,
            "status": status['agent_status'].get(agent_name, "idle"),
            "performance": status['agent_performance'].get(agent_name, {})
        }
        pipeline["supervisor"]["agents"].append(agent_info)

    return JSONResponse(pipeline)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enlitens AI Enhanced Monitoring Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print("=" * 80)
    print("🏗️  ENLITENS AI ENHANCED MONITORING SERVER")
    print("=" * 80)
    print(f"📊 Dashboard: http://{args.host}:{args.port}")
    print(f"🔌 WebSocket: ws://{args.host}:{args.port}/ws")
    print(f"📡 Log Endpoint: http://{args.host}:{args.port}/api/log")
    print(f"📈 Stats: http://{args.host}:{args.port}/api/stats")
    print(f"🤖 Using Ollama at: {OLLAMA_BASE_URL}")
    print("=" * 80)

    uvicorn.run(
        "monitoring_server_enhanced:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
