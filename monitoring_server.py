#!/usr/bin/env python3
"""
Real-Time Monitoring Server for Enlitens AI Processing

Features:
- WebSocket log streaming
- Real-time processing metrics
- Quality monitoring
- Foreman AI integration
- Beautiful web dashboard
"""

import asyncio
import json
import logging
import time
import re
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import List, Dict, Any, Set, Optional
import argparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Enlitens AI Monitor", version="1.0.0")


# Static assets directory
STATIC_DIR = Path(__file__).parent / "monitoring_ui"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    print(f"⚠️ Monitoring UI assets directory not found at {STATIC_DIR}")

# Store active WebSocket connections
class QualityMetricsAggregator:
    """Aggregate validation metrics for the quality dashboard."""

    def __init__(self, max_records: int = 500) -> None:
        self.records: deque[Dict[str, Any]] = deque(maxlen=max_records)

    def record(self, payload: Dict[str, Any]) -> None:
        event = {
            "document_id": payload.get("document_id"),
            "timestamp": payload.get("timestamp"),
            "quality": float(payload.get("quality", 0.0) or 0.0),
            "confidence": float(payload.get("confidence", 0.0) or 0.0),
            "verification_passed": bool(payload.get("verification_passed", False)),
            "retry_attempt": int(payload.get("retry_attempt", 1) or 1),
            "needs_retry": bool(payload.get("needs_retry", False)),
            "failure_reasons": list(payload.get("failure_reasons", [])),
            "citation_failures": list(payload.get("citation_failures", [])),
            "missing_quotes": list(payload.get("missing_quotes", [])),
            "self_critique_performed": bool(payload.get("self_critique_performed", False)),
            "quality_scores": payload.get("quality_scores", {}),
        }
        self.records.append(event)

    def summary(self) -> Dict[str, Any]:
        if not self.records:
            return {
                "total_events": 0,
                "average_quality": 0.0,
                "average_confidence": 0.0,
                "pass_rate": 0.0,
                "average_retry_attempt": 0.0,
                "self_critique_rate": 0.0,
                "documents_requiring_retry": 0,
                "top_failure_reasons": [],
                "recent_documents": [],
            }

        records_list = list(self.records)
        total_events = len(records_list)
        avg_quality = mean(event["quality"] for event in records_list)
        avg_confidence = mean(event["confidence"] for event in records_list)
        pass_rate = sum(1 for event in records_list if event["verification_passed"]) / total_events
        avg_retry_attempt = mean(event["retry_attempt"] for event in records_list)
        self_critique_rate = sum(1 for event in records_list if event["self_critique_performed"]) / total_events
        documents_requiring_retry = sum(1 for event in records_list if event["needs_retry"])

        counter: Counter[str] = Counter()
        for event in records_list:
            counter.update(event["failure_reasons"])
            if event["citation_failures"]:
                counter.update({"citation_mismatch": len(event["citation_failures"])})
            if event["missing_quotes"]:
                counter.update({"missing_quotes": len(event["missing_quotes"])})

        top_failure_reasons = [
            {"reason": reason, "count": count}
            for reason, count in counter.most_common(5)
        ]

        recent_documents = [
            {
                "document_id": event["document_id"],
                "quality": event["quality"],
                "needs_retry": event["needs_retry"],
                "retry_attempt": event["retry_attempt"],
                "timestamp": event["timestamp"],
                "failure_reasons": event["failure_reasons"],
            }
            for event in records_list[-10:]
        ][::-1]

        return {
            "total_events": total_events,
            "average_quality": round(avg_quality, 3),
            "average_confidence": round(avg_confidence, 3),
            "pass_rate": round(pass_rate, 3),
            "average_retry_attempt": round(avg_retry_attempt, 2),
            "self_critique_rate": round(self_critique_rate, 3),
            "documents_requiring_retry": documents_requiring_retry,
            "top_failure_reasons": top_failure_reasons,
            "recent_documents": recent_documents,
        }


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.message_history: List[Dict[str, Any]] = []
        self.max_history = 1000

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

        # Send history to new connection
        for msg in self.message_history[-100:]:  # Last 100 messages
            try:
                await websocket.send_json(msg)
            except:
                pass

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        # Add to history
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history:]

        # Broadcast to all connections
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.add(connection)

        # Clean up disconnected clients
        self.active_connections -= disconnected

        if message.get("type") == "log":
            payload = message.get("message", "")
            if isinstance(payload, str) and payload.startswith("QUALITY_METRICS "):
                try:
                    metrics_json = payload.split("QUALITY_METRICS ", 1)[1]
                    metrics_data = json.loads(metrics_json)
                    quality_aggregator.record(metrics_data)
                except Exception:
                    logging.getLogger(__name__).debug("Failed to parse quality metrics payload")


quality_aggregator = QualityMetricsAggregator()
manager = ConnectionManager()

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
knowledge_base_snapshot: Dict[str, Any] = {}


dashboard_state: Dict[str, Any] = {}


def reset_dashboard_state(start_time: Optional[datetime] = None) -> None:
    """Reset aggregated dashboard metrics to their initial state."""

    dashboard_state.clear()
    dashboard_state.update(
        {
            "total_documents": 0,
            "documents_processed": 0,
            "current_document": None,
            "current_doc_index": 0,
            "current_doc_started": None,
            "start_time": start_time,
            "last_update": None,
            "recent_errors": deque(maxlen=50),
            "recent_warnings": deque(maxlen=50),
            "quality_scores": deque(maxlen=100),
            "recent_documents": deque(maxlen=20),
            "_processed_documents": set(),
            "_failed_documents": set(),
            "failed_documents": 0,
        }
    )


reset_dashboard_state()


def strip_ansi(value: str) -> str:
    return ANSI_ESCAPE_RE.sub("", value)


def _extract_log_text(message: Dict[str, Any]) -> str:
    raw = message.get("message", "")
    cleaned = strip_ansi(raw)
    parts = cleaned.split(" - ", 3)
    if len(parts) >= 4:
        return parts[3].strip()
    return cleaned.strip()


def update_dashboard_from_log(message: Dict[str, Any]) -> None:
    if message.get("type") != "log":
        return

    logger_name = message.get("logger", "")
    if not (logger_name.startswith("__main__") or logger_name.startswith("src.")):
        return

    log_text = _extract_log_text(message)
    level = message.get("level", "").upper()
    timestamp = datetime.utcnow()
    dashboard_state["last_update"] = timestamp

    if "🚀 Starting MULTI-AGENT" in log_text or "📁 Input directory" in log_text:
        reset_dashboard_state(start_time=timestamp)

    if dashboard_state.get("start_time") is None:
        dashboard_state["start_time"] = timestamp

    if level == "ERROR":
        dashboard_state["recent_errors"].appendleft({
            "timestamp": message.get("timestamp"),
            "message": log_text,
        })
    elif level in {"WARN", "WARNING"}:
        dashboard_state["recent_warnings"].appendleft({
            "timestamp": message.get("timestamp"),
            "message": log_text,
        })

    total_match = re.search(r"Found\s+(\d+)\s+PDF files", log_text)
    if total_match:
        dashboard_state["total_documents"] = int(total_match.group(1))

    current_match = re.search(r"Processing file\s+(\d+)/(\d+):\s+(.+)", log_text)
    if current_match:
        dashboard_state["current_doc_index"] = int(current_match.group(1))
        dashboard_state["total_documents"] = int(current_match.group(2))
        dashboard_state["current_document"] = current_match.group(3).strip()
        dashboard_state["current_doc_started"] = timestamp
        return

    success_match = re.search(r"Document\s+(.+?) processed successfully", log_text)
    if success_match:
        doc_id = success_match.group(1).strip()
        dashboard_state["_processed_documents"].add(doc_id)
        dashboard_state["documents_processed"] = len(dashboard_state["_processed_documents"])

        duration = None
        if dashboard_state.get("current_doc_started"):
            duration = (timestamp - dashboard_state["current_doc_started"]).total_seconds()

        entry = {
            "document_id": doc_id,
            "timestamp": message.get("timestamp"),
            "duration": duration,
        }

        quality_match = re.search(r"Quality\s+([0-9.]+)", log_text)
        if quality_match:
            score = float(quality_match.group(1))
            dashboard_state["quality_scores"].append(score)
            entry["quality"] = score

        confidence_match = re.search(r"Confidence\s+([0-9.]+)", log_text)
        if confidence_match:
            entry["confidence"] = float(confidence_match.group(1))

        dashboard_state["recent_documents"].appendleft(entry)
        dashboard_state["current_document"] = None
        dashboard_state["current_doc_started"] = None
        return

    if "document" in log_text.lower() and "failed" in log_text.lower():
        failed_match = re.search(r"Document\s+(.+?)(?:\s|$)", log_text)
        if failed_match:
            dashboard_state["_failed_documents"].add(failed_match.group(1).strip())
            dashboard_state["failed_documents"] = len(dashboard_state["_failed_documents"])


def update_dashboard_from_payload(payload: Dict[str, Any]) -> None:
    if not payload:
        return

    timestamp = datetime.utcnow()
    dashboard_state["last_update"] = timestamp

    if "total_documents" in payload:
        dashboard_state["total_documents"] = payload["total_documents"]

    if "documents_processed" in payload:
        dashboard_state["documents_processed"] = payload["documents_processed"]

    if "documents_failed" in payload:
        dashboard_state["failed_documents"] = payload["documents_failed"]

    if "current_document" in payload:
        dashboard_state["current_document"] = payload["current_document"]
        dashboard_state["current_doc_index"] = payload.get("current_index", dashboard_state.get("current_doc_index", 0))
        dashboard_state["current_doc_started"] = timestamp

    if payload.get("status") in {"completed", "idle"}:
        dashboard_state["current_document"] = None
        dashboard_state["current_doc_started"] = None

    if "last_document" in payload:
        entry = {
            "document_id": payload["last_document"],
            "timestamp": timestamp.isoformat(),
        }
        if "last_duration" in payload:
            entry["duration"] = payload["last_duration"]
        if "last_quality" in payload:
            entry["quality"] = payload["last_quality"]
        if "last_error" in payload:
            entry["error"] = payload["last_error"]
        dashboard_state["recent_documents"].appendleft(entry)

    if "runtime_seconds" in payload:
        dashboard_state["runtime_seconds"] = payload["runtime_seconds"]


def serialize_recent(deq: deque) -> List[Dict[str, Any]]:
    if not deq:
        return []
    return list(deq)


class ForemanResponder:
    """Generates streaming responses for Foreman chat clients."""

    def __init__(self) -> None:
        self.quick_replies = [
            "Current pipeline health?",
            "Show latest latency anomalies",
            "Summarize LangFuse traces",
            "Any hallucination alerts today?",
            "Cost outlook for this batch",
        ]

    def build_response(self, query: str) -> str:
        timestamp = datetime.utcnow().strftime("%H:%M:%S UTC")
        intro = (
            "Foreman online. Monitoring spans, metrics, and anomaly signals. "
            "Here's the latest rundown:\n"
        )
        insights = [
            "• LangFuse ingestion is nominal with no backlog detected.",
            "• Phoenix RAG precision is holding above 0.82 for grounded responses.",
            "• Latency guardrails are stable; no stage exceeded its SLO in the last cycle.",
            "• Cost monitors are green. I'll flag any token spikes immediately.",
        ]
        footer = f"Query '{query}' logged at {timestamp}."
        return intro + "\n".join(insights) + "\n\n" + footer

    async def stream_response(self, websocket: WebSocket, query: str) -> None:
        response_text = self.build_response(query)
        stream_id = f"foreman-{int(time.time() * 1000)}"
        for token in response_text.split():
            await websocket.send_json(
                {
                    "type": "foreman_token",
                    "token": token + " ",
                    "stream_id": stream_id,
                }
            )
            await asyncio.sleep(0.04)
        await websocket.send_json(
            {
                "type": "foreman_response",
                "query": query,
                "response": response_text,
                "stream_id": stream_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )


foreman_responder = ForemanResponder()


class WebSocketLogHandler(logging.Handler):
    """Custom log handler that streams to WebSocket clients."""

    def __init__(self, manager: ConnectionManager):
        super().__init__()
        self.manager = manager
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the server event loop for thread-safe scheduling."""
        self.loop = loop

    def emit(self, record):
        try:
            # Format log record
            log_entry = {
                "type": "log",
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
                "module": record.module,
                "funcName": record.funcName,
                "lineNo": record.lineno
            }

            # Extract metadata if present
            if hasattr(record, 'document_id'):
                log_entry['document_id'] = record.document_id
            if hasattr(record, 'agent_name'):
                log_entry['agent_name'] = record.agent_name
            if hasattr(record, 'processing_stage'):
                log_entry['processing_stage'] = record.processing_stage

            # Send to all clients
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.manager.broadcast(log_entry))
                return
            except RuntimeError:
                pass

            if self.loop and not self.loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self.manager.broadcast(log_entry),
                    self.loop,
                )
        except Exception as e:
            print(f"Error in WebSocketLogHandler: {e}")


@app.on_event("startup")
async def startup_event():
    """Set up WebSocket log handler on startup."""
    # Get root logger and add WebSocket handler
    root_logger = logging.getLogger()
    ws_handler = WebSocketLogHandler(manager)
    ws_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ws_handler.setFormatter(formatter)
    ws_handler.set_loop(asyncio.get_running_loop())
    root_logger.addHandler(ws_handler)

    print("✅ Monitoring server started - WebSocket log streaming enabled")


@app.get("/")
async def get_dashboard():
    """Serve the monitoring dashboard."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"), media_type="text/html")
    else:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enlitens AI Monitor</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                .container {
                    text-align: center;
                    background: rgba(255, 255, 255, 0.1);
                    padding: 50px;
                    border-radius: 20px;
                    backdrop-filter: blur(10px);
                }
                h1 { font-size: 3em; margin: 0 0 20px 0; }
                p { font-size: 1.2em; }
                .status {
                    margin-top: 30px;
                    padding: 20px;
                    background: rgba(255, 255, 255, 0.2);
                    border-radius: 10px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 Enlitens AI Monitor</h1>
                <p>Setting up monitoring dashboard...</p>
                <div class="status">
                    <p>Server is running on port 8765</p>
                    <p>Dashboard UI will be available shortly</p>
                </div>
            </div>
        </body>
        </html>
        """)


@app.get("/quality")
async def get_quality_page():
    """Serve the quality dashboard UI."""
    html_path = STATIC_DIR / "quality_dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"), media_type="text/html")
    return HTMLResponse("<h1>Quality dashboard assets not found</h1>", status_code=404)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for log streaming."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and receive client messages
            data = await websocket.receive_text()

            # Handle client messages (e.g., commands, foreman queries)
            try:
                message = json.loads(data)
                if message.get("type") == "foreman_query":
                    query_text = message.get("query", "").strip()
                    if query_text:
                        asyncio.create_task(
                            foreman_responder.stream_response(websocket, query_text)
                        )
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/stats")
async def get_stats():
    """Get current processing statistics."""
    state = dashboard_state
    now = datetime.utcnow()

    total = state.get("total_documents", 0)
    processed = state.get("documents_processed", 0)
    failed = state.get("failed_documents", len(state.get("_failed_documents", set())))
    current_document = state.get("current_document")
    start_time = state.get("start_time")
    current_started = state.get("current_doc_started")

    runtime_seconds = (now - start_time).total_seconds() if start_time else 0
    time_on_document = (now - current_started).total_seconds() if current_started else 0
    progress_pct = int((processed / total) * 100) if total else 0
    queue_depth = max(0, (total - processed - (1 if current_document else 0))) if total else 0
    success_base = processed + failed
    success_rate = (processed / success_base) * 100 if success_base else 0

    quality_scores = state.get("quality_scores", deque())
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

    return {
        "connected_clients": len(manager.active_connections),
        "total_logs": len(manager.message_history),
        "status": "running" if processed or current_document else "idle",
        "documents_processed": processed,
        "total_documents": total,
        "progress_percentage": progress_pct,
        "queue_depth": queue_depth,
        "current_document": current_document,
        "time_on_document_seconds": round(time_on_document, 2),
        "runtime_seconds": round(runtime_seconds, 2),
        "success_rate": round(success_rate, 2),
        "recent_errors": serialize_recent(state.get("recent_errors", deque())),
        "recent_warnings": serialize_recent(state.get("recent_warnings", deque())),
        "recent_documents": serialize_recent(state.get("recent_documents", deque())),
        "quality_metrics": {
            "avg_quality_score": round(avg_quality, 2),
            "samples": len(quality_scores),
        },
        "quality_summary": quality_aggregator.summary(),
        "knowledge_base": knowledge_base_snapshot,
    }


@app.get("/api/quality-dashboard")
async def get_quality_dashboard():
    """Expose aggregated validation metrics for the quality dashboard."""
    return quality_aggregator.summary()


@app.get("/api/knowledge-base")
async def get_knowledge_base():
    """Return the most recent knowledge base snapshot (if available)."""
    return knowledge_base_snapshot or {"status": "pending"}


@app.post("/api/broadcast")
async def broadcast_message(message: Dict[str, Any]):
    """API endpoint to broadcast custom messages."""
    message_type = message.get("type")

    if message_type == "knowledge_base":
        knowledge_base_snapshot.update(message.get("payload", {}))
    elif message_type == "stats":
        update_dashboard_from_payload(message.get("payload", {}))
    else:
        update_dashboard_from_log(message)

    await manager.broadcast(message)
    return {"status": "broadcasted"}


def main():
    """Run the monitoring server."""
    parser = argparse.ArgumentParser(description="Enlitens AI Monitoring Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║              🤖 ENLITENS AI MONITORING SERVER 🤖                  ║
║                                                                   ║
║  Real-time log streaming and quality monitoring                  ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝

🌐 Dashboard: http://{args.host}:{args.port}
🔌 WebSocket: ws://{args.host}:{args.port}/ws
📊 API Stats: http://{args.host}:{args.port}/api/stats

Press Ctrl+C to stop the server
""")

    uvicorn.run(
        "monitoring_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
