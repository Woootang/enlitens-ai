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
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Set
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

manager = ConnectionManager()


class WebSocketLogHandler(logging.Handler):
    """Custom log handler that streams to WebSocket clients."""

    def __init__(self, manager: ConnectionManager):
        super().__init__()
        self.manager = manager
        self.loop = None

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
            if self.loop is None:
                self.loop = asyncio.get_event_loop()

            asyncio.run_coroutine_threadsafe(
                self.manager.broadcast(log_entry),
                self.loop
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
                    # Handle Foreman AI query (to be implemented)
                    await websocket.send_json({
                        "type": "foreman_response",
                        "query": message.get("query"),
                        "response": "Foreman AI is analyzing your question..."
                    })
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/stats")
async def get_stats():
    """Get current processing statistics."""
    return {
        "connected_clients": len(manager.active_connections),
        "total_logs": len(manager.message_history),
        "server_uptime": "N/A",
        "status": "running"
    }


@app.post("/api/broadcast")
async def broadcast_message(message: Dict[str, Any]):
    """API endpoint to broadcast custom messages."""
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
