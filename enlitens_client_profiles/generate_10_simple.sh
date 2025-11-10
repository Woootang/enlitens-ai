#!/bin/bash
# Generate 10 personas one at a time using the working script

cd /home/antons-gs/enlitens-ai
source venv/bin/activate

echo "Generating 10 unique personas..."
echo "================================"
echo ""

for i in {1..10}; do
    echo "Generating persona #$i..."
    python -m enlitens_client_profiles.generate_real_stories_gemini
    
    if [ $? -eq 0 ]; then
        echo "✅ Persona #$i generated successfully"
    else
        echo "❌ Persona #$i failed"
        exit 1
    fi
    
    echo ""
    sleep 2  # Brief pause between generations
done

echo ""
echo "✅ All 10 personas generated!"
echo ""
echo "Generated personas:"
ls -1 enlitens_client_profiles/profiles/persona_real_story_*.json | nl

