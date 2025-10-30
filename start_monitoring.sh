#!/bin/bash
# Start Enlitens AI Monitoring Server

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                                                                   ║"
echo "║              🤖 ENLITENS AI MONITORING SERVER 🤖                  ║"
echo "║                                                                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if requirements are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}📦 Installing monitoring requirements...${NC}"
    pip install -r requirements-monitoring.txt
fi

# Start the monitoring server
echo -e "${GREEN}🚀 Starting monitoring server...${NC}"
echo ""
echo -e "${BLUE}Dashboard will be available at:${NC}"
echo -e "   ${GREEN}http://localhost:8765${NC}"
echo ""
echo -e "${BLUE}WebSocket endpoint:${NC}"
echo -e "   ${GREEN}ws://localhost:8765/ws${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

python3 monitoring_server.py --host 0.0.0.0 --port 8765
