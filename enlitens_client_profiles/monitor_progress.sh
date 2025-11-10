#!/bin/bash
# Monitor persona generation progress

PROFILES_DIR="/home/antons-gs/enlitens-ai/enlitens_client_profiles/profiles"
LOG_FILE="/home/antons-gs/enlitens-ai/enlitens_client_profiles/logs/cluster_generation_full.log"

while true; do
    clear
    echo "================================"
    echo "PERSONA GENERATION PROGRESS"
    echo "================================"
    echo ""
    
    # Count generated personas
    TOTAL=$(ls -1 $PROFILES_DIR/persona_cluster_*.json 2>/dev/null | wc -l)
    echo "Generated: $TOTAL / 100 personas"
    echo ""
    
    # Show percentage
    PCT=$((TOTAL * 100 / 100))
    echo "Progress: $PCT%"
    echo ""
    
    # Show last 5 generated
    echo "Last 5 generated:"
    tail -20 $LOG_FILE | grep "SUCCESS:" | tail -5
    echo ""
    
    # Show any failures
    FAILURES=$(grep "FAILED\|ERROR:" $LOG_FILE | wc -l)
    if [ $FAILURES -gt 0 ]; then
        echo "⚠️  Failures detected: $FAILURES"
        echo ""
    fi
    
    # Check if complete
    if [ $TOTAL -ge 100 ]; then
        echo "✅ COMPLETE! All 100 personas generated!"
        break
    fi
    
    echo "Refreshing in 30 seconds... (Ctrl+C to stop monitoring)"
    sleep 30
done

