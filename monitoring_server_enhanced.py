#!/usr/bin/env python3
"""
Enhanced Real-Time Monitoring Server for Enlitens AI Processing

Features:
- WebSocket log streaming with agent tracking
- Real-time processing metrics with document tracking
- Quality monitoring with detailed breakdown
- Intelligent Foreman AI using vLLM with Groq fallback
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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx

from src.synthesis.ollama_client import VLLMClient, MONITORING_MODEL
from src.knowledge_base.status_file import STATUS_FILE_NAME, read_processing_status

logger = logging.getLogger(__name__)

app = FastAPI(title="Enlitens AI Monitor Enhanced", version="2.0.0")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
FOREMAN_LOCAL_URL = os.environ.get("FOREMAN_LOCAL_URL", "http://localhost:8001/v1")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
KNOWLEDGE_BASE_PATH = Path("enlitens_knowledge_base_latest.json")
KNOWLEDGE_BASE_STATUS_PATH = KNOWLEDGE_BASE_PATH.parent / STATUS_FILE_NAME
STATIC_DIR = Path(__file__).parent / "monitoring_ui"

# Initialize Groq client if API key is available
GROQ_CLIENT = httpx.AsyncClient(
    base_url="https://api.groq.com/openai/v1",
    headers={"Authorization": f"Bearer {GROQ_API_KEY}"} if GROQ_API_KEY else {},
    timeout=30.0
) if GROQ_API_KEY else None

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
        self.verbose_logging_enabled = False
        self.retry_requests: deque = deque(maxlen=20)

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

        # Ensure JSON-serializable structures (e.g., convert deque to list)
        agent_performance_serializable: Dict[str, Any] = {}
        for agent_name, performance in self.agent_performance.items():
            perf_dict = dict(performance)
            times = perf_dict.get("times")
            if isinstance(times, deque):
                perf_dict["times"] = list(times)
            agent_performance_serializable[agent_name] = perf_dict

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
            "agent_performance": agent_performance_serializable,
            "verbose_logging": self.verbose_logging_enabled,
            "recent_retry_requests": list(self.retry_requests),
        }

    def record_retry_request(self):
        if self.current_document:
            self.retry_requests.append({
                "document": self.current_document,
                "requested_at": datetime.utcnow().isoformat(),
            })

    def toggle_verbose(self) -> bool:
        self.verbose_logging_enabled = not self.verbose_logging_enabled
        return self.verbose_logging_enabled

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
FOREMAN_SYSTEM_PROMPT = (
    "You are the Foreman AI for the Enlitens AI knowledge base extraction system. "
    "Provide concise, operational guidance using the monitoring metrics supplied in "
    "each query. Escalate outages, suggest concrete remediation steps, and cite agent "
    "names when possible."
)


class ForemanModelRouter:
    """Route Foreman prompts between local vLLM and Groq fallback."""

    def __init__(self, local_url, groq_api_key):
        self.local_client = VLLMClient(base_url=local_url, default_model=MONITORING_MODEL) if local_url else None
        self.local_url = local_url
        self.groq_api_key = groq_api_key
        self.groq_client = httpx.AsyncClient(base_url="https://api.groq.com/openai/v1", timeout=30.0) if groq_api_key else None

    async def _ask_local(self, prompt: str):
        if not self.local_client:
            return None
        try:
            if not await self.local_client.check_connection():
                logger.warning("Local Foreman vLLM endpoint unavailable: %s", self.local_url)
                return None
            response = await self.local_client.generate_text(
                prompt,
                model=self.local_client.default_model,
                temperature=0.4,
                num_predict=512,
                system_prompt=FOREMAN_SYSTEM_PROMPT,
                extra_options={"extra_body": {"cache_prompt": True}},
            )
            return response or None
        except Exception as exc:
            logger.warning("Local Foreman generation failed: %s", exc)
            return None

    async def _ask_groq(self, prompt: str):
        if not self.groq_client or not self.groq_api_key:
            return None
        try:
            response = await self.groq_client.post(
                "chat/completions",
                headers={"Authorization": f"Bearer {self.groq_api_key}"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": FOREMAN_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.6,
                    "max_tokens": 600,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as exc:
            logger.error("Groq fallback failed: %s", exc)
            return None

    async def generate(self, prompt: str):
        reply = await self._ask_local(prompt)
        if reply:
            return reply
        return await self._ask_groq(prompt)

    async def availability(self):
        local_ok = False
        if self.local_client:
            try:
                local_ok = await self.local_client.check_connection()
            except Exception:
                local_ok = False
        groq_ok = bool(self.groq_api_key and self.groq_client)
        return {
            "local_url": self.local_url,
            "local_available": local_ok,
            "groq_configured": groq_ok,
        }

    async def close(self):
        if self.local_client:
            await self.local_client.cleanup()
        if self.groq_client:
            await self.groq_client.aclose()


class ForemanAI:
    """Intelligent AI assistant with routing and heuristic fallback."""

    def __init__(self, router=None):
        self.router = router or ForemanModelRouter(FOREMAN_LOCAL_URL, GROQ_API_KEY)

    async def analyze_query(self, query: str, context: Dict[str, Any]) -> str:
        status = processing_state.get_current_status()
        context_prompt = f"""Current Processing Status:
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

Respond as the monitoring foreman."""

        reply = await self.router.generate(context_prompt)
        if reply:
            return reply
        return self._fallback_response(query, status)

    async def health(self):
        return await self.router.availability()

    async def close(self):
        await self.router.close()

    def _fallback_response(self, query: str, status: Dict[str, Any]) -> str:
        query_lower = query.lower()

        if any(word in query_lower for word in ("status", "progress", "how")):
            progress = status['progress_percentage']
            current = status['current_document'] or 'initializing'
            active = ', '.join(status['active_agents']) or 'waiting'
            return (
                "📊 **Processing Status:**\n"
                f"- Currently on: `{current}`\n"
                f"- Progress: **{progress:.1f}%** ({status['documents_processed']}/{status['total_documents']})\n"
                f"- Active agents: {active}\n"
                f"- Recent errors: {len(status['recent_errors'])}"
            )

        if any(word in query_lower for word in ("error", "problem", "issue", "wrong")):
            if status['recent_errors']:
                latest = status['recent_errors'][-1]
                return (
                    "⚠️ **Latest Error:**\n```\n"
                    f"{latest['message']}\n```\n"
                    f"Agent: {latest.get('agent', 'Unknown')}\n"
                    f"Time: {latest['timestamp']}"
                )
            return "✅ No errors detected! System is running smoothly."

        if any(word in query_lower for word in ("quality", "hallucination", "validation")):
            metrics = status['quality_metrics']
            return (
                "🎯 **Quality Metrics:**\n"
                f"- ✅ Citations Verified: {metrics['citation_verified']}\n"
                f"- ❌ Validation Failures: {metrics['validation_failures']}\n"
                f"- ⚠️ Empty Fields: {metrics['empty_fields']}\n"
                f"Overall: {'Good' if metrics['validation_failures'] < 5 else 'Needs attention'}"
            )

        return (
            f"I monitor {status['total_documents']} documents with {len(status['agent_pipeline'])} "
            "agents in the pipeline. Ask me about status, errors, or quality metrics!"
        )

foreman_ai = ForemanAI()


@app.on_event("shutdown")
async def _shutdown_foreman():
    await foreman_ai.close()


@app.get("/api/foreman/health")
async def foreman_health():
    data = await foreman_ai.health()
    return JSONResponse(data)


@app.post("/api/foreman/query")
async def foreman_query(payload: Dict[str, Any]):
    query = payload.get("query", "")
    response = await foreman_ai.analyze_query(query, payload.get("context", {}))
    return JSONResponse({"query": query, "response": response})

# Routes
@app.get("/test", response_class=HTMLResponse)
async def get_test():
    """Serve the minimal test page for debugging."""
    test_file = STATIC_DIR / "test_minimal.html"
    if test_file.exists():
        return FileResponse(test_file)
    # Try minimal.html as fallback
    minimal_file = STATIC_DIR / "minimal.html"
    if minimal_file.exists():
        return FileResponse(minimal_file)
    return HTMLResponse("<h1>Test page not found</h1>", status_code=404)

@app.get("/minimal", response_class=HTMLResponse)
async def get_minimal():
    """Serve the absolute minimal test page."""
    minimal_file = STATIC_DIR / "minimal.html"
    if minimal_file.exists():
        return FileResponse(minimal_file)
    return HTMLResponse("<h1>Minimal page not found</h1>", status_code=404)

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
    status = read_processing_status(KNOWLEDGE_BASE_STATUS_PATH)
    if status and status.status.lower() != "ok":
        return JSONResponse(
            {
                "error": status.reason,
                "status": status.to_dict(),
            },
            status_code=503,
        )
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

@app.post("/api/chat")
async def chat_with_assistant(request: Request):
    """AI chat assistant endpoint using Groq API."""
    try:
        body = await request.json()
        user_message = body.get("message", "")
        context = body.get("context", {})
        
        if not user_message:
            return JSONResponse({"response": "Please provide a message."})
        
        # Build context-aware system prompt
        stats = context.get("stats", {})
        logs = context.get("logs", [])
        
        system_prompt = f"""You are an AI assistant monitoring the Enlitens AI processing pipeline.

Current Status:
- Documents processed: {stats.get('docs', 0)} of {stats.get('total', 0)}
- Errors: {stats.get('errors', 0)}
- Warnings: {stats.get('warnings', 0)}
- Quality score: {stats.get('quality', 0)}%
- Progress: {stats.get('progress', 0)}%

Recent logs show: {len(logs)} recent events.

Answer questions about the processing status, quality metrics, errors, and provide insights.
Be concise and helpful. If you see issues, suggest solutions."""

        # Use Groq API if available, otherwise use local fallback
        if GROQ_CLIENT and GROQ_API_KEY:
            try:
                response = await GROQ_CLIENT.post(
                    "/chat/completions",
                    json={
                        "model": GROQ_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500
                    }
                )
                response.raise_for_status()
                data = response.json()
                ai_response = data["choices"][0]["message"]["content"]
                return JSONResponse({"response": ai_response})
            except Exception as groq_error:
                logger.error(f"Groq API error: {groq_error}")
                # Fall through to local fallback
        
        # Fallback response without API
        fallback_responses = {
            "status": f"Currently processing document {stats.get('docs', 0)} of {stats.get('total', 0)}. {stats.get('errors', 0)} errors detected. Quality score: {stats.get('quality', 0)}%.",
            "error": f"Found {stats.get('errors', 0)} errors. Check the logs for details about what went wrong.",
            "quality": f"Current quality score is {stats.get('quality', 0)}%. {'Excellent!' if stats.get('quality', 0) >= 90 else 'There is room for improvement.' if stats.get('quality', 0) >= 60 else 'Quality needs attention.'}",
            "help": "I can help you with: current status, error analysis, quality metrics, and processing suggestions. What would you like to know?"
        }
        
        message_lower = user_message.lower()
        for key, response in fallback_responses.items():
            if key in message_lower:
                return JSONResponse({"response": response})
        
        return JSONResponse({
            "response": f"I see you're asking about: '{user_message}'. Currently {stats.get('docs', 0)} documents processed with {stats.get('errors', 0)} errors. For detailed AI analysis, please set the GROQ_API_KEY environment variable (free API at https://console.groq.com)."
        })
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return JSONResponse({"response": "Sorry, I encountered an error processing your request."})


@app.post("/api/actions/retry-document")
async def retry_document_action():
    """Operator-triggered request to retry the active document."""
    document = processing_state.current_document
    if not document:
        return JSONResponse({"success": False, "message": "No active document to retry."}, status_code=400)

    processing_state.record_retry_request()
    await manager.broadcast({
        "type": "log",
        "timestamp": datetime.utcnow().isoformat(),
        "level": "INFO",
        "message": f"Operator requested retry for document '{document}'.",
        "agent_name": "operator",
    })
    logger.info("Retry requested for document %s", document)
    return JSONResponse({"success": True, "message": f"Retry request acknowledged for {document}."})


@app.post("/api/actions/toggle-verbose")
async def toggle_verbose_action():
    """Toggle verbose logging flag for downstream processors."""
    new_state = processing_state.toggle_verbose()
    await manager.broadcast({
        "type": "log",
        "timestamp": datetime.utcnow().isoformat(),
        "level": "INFO",
        "message": f"Verbose logging {'enabled' if new_state else 'disabled'} by operator.",
        "agent_name": "operator",
    })
    logger.info("Verbose logging state changed to %s", new_state)
    return JSONResponse({"success": True, "verbose": new_state})

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
    print(f"🤖 Primary vLLM endpoint: {VLLM_BASE_URL}")
    print(f"🛡️ Foreman local model: {FOREMAN_LOCAL_URL or 'disabled'}")
    if GROQ_API_KEY:
        print(f"🌐 Groq fallback model: {GROQ_MODEL}")
    else:
        print("🌐 Groq fallback model: not configured")
    print("=" * 80)

    uvicorn.run(
        "monitoring_server_enhanced:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )
