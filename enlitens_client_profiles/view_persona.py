"""
Quick persona viewer - displays a formatted view of any persona.
Usage: python -m enlitens_client_profiles.view_persona [persona_number]
"""
import json
import sys
from pathlib import Path


def view_persona(persona_file: Path):
    """Display a formatted view of a persona."""
    with open(persona_file) as f:
        p = json.load(f)
    
    print("="*80)
    print(f"PERSONA: {p['meta']['persona_name']}")
    print(f"ID: {p['meta']['profile_id']}")
    print("="*80)
    
    # Demographics
    demo = p['identity_demographics']
    print(f"\n📋 DEMOGRAPHICS")
    print(f"   Age: {demo.get('age_range', 'N/A')}")
    print(f"   Gender: {demo.get('gender', 'N/A')} ({demo.get('pronouns', 'N/A')})")
    print(f"   Situation: {demo.get('current_life_situation', 'N/A')}")
    print(f"   Occupation: {demo.get('occupation', 'N/A')}")
    print(f"   Locality: {demo.get('locality', 'N/A')}")
    
    # Neurodivergence
    neuro = p['neurodivergence_mental_health']
    print(f"\n🧠 NEURODIVERGENCE")
    print(f"   Identities: {', '.join(neuro.get('identities', []))}")
    print(f"   Diagnosis: {neuro.get('diagnosis_journey', 'N/A')[:150]}...")
    
    # Developmental Story
    dev = p['developmental_story']
    print(f"\n👶 DEVELOPMENTAL STORY")
    print(f"   Childhood: {dev.get('childhood_environment', 'N/A')[:200]}...")
    print(f"   Adversities: {', '.join(dev.get('formative_adversities', [])[:3])}")
    
    # Executive Function & Sensory
    ef = p['executive_function_sensory']
    print(f"\n🍽️  FOOD SENSORY PROFILE")
    food = ef.get('food_sensory_details', 'N/A')
    print(f"   {food[:300]}...")
    
    # Current Life Context
    life = p['current_life_context']
    print(f"\n🏠 CURRENT LIFE")
    print(f"   Where: {life.get('where_they_live', 'N/A')[:150]}...")
    print(f"   Work/School: {life.get('work_school_situation', 'N/A')[:150]}...")
    print(f"   Safe Spaces: {', '.join(life.get('safe_spaces', [])[:5])}")
    
    # Goals & Barriers
    goals = p['goals_barriers']
    print(f"\n🎯 THERAPY GOALS")
    print(f"   Why Now: {goals.get('why_therapy_now', 'N/A')[:200]}...")
    print(f"   Wants to Change: {', '.join(goals.get('what_they_want_to_change', [])[:3])}")
    
    # Narrative Voice
    voice = p['narrative_voice']
    print(f"\n💬 VOICE")
    print(f"   Struggle: \"{voice.get('quote_struggle', 'N/A')}\"")
    print(f"   Hope: \"{voice.get('quote_hope', 'N/A')}\"")
    
    # Marketing/SEO
    seo = p['marketing_seo']
    print(f"\n📍 LOCAL ENTITIES (SEO)")
    print(f"   {', '.join(seo.get('local_entities', [])[:10])}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    profiles_dir = Path("/home/antons-gs/enlitens-ai/enlitens_client_profiles/profiles")
    personas = sorted(profiles_dir.glob("persona_cluster_*.json"))
    
    if len(sys.argv) > 1:
        # View specific persona by number
        try:
            num = int(sys.argv[1])
            if 0 < num <= len(personas):
                view_persona(personas[num - 1])
            else:
                print(f"❌ Persona #{num} not found. Available: 1-{len(personas)}")
        except ValueError:
            print(f"❌ Invalid persona number: {sys.argv[1]}")
    else:
        # List all personas
        print(f"✅ Found {len(personas)} personas\n")
        print("="*80)
        print("PERSONA LIST")
        print("="*80)
        
        for i, p_file in enumerate(personas, 1):
            with open(p_file) as f:
                p = json.load(f)
            
            name = p['meta']['persona_name']
            age = p['identity_demographics'].get('age_range', 'N/A')
            situation = p['identity_demographics'].get('current_life_situation', 'N/A')[:50]
            
            print(f"{i:3d}. {name}")
            print(f"      Age: {age} | {situation}")
        
        print("\n" + "="*80)
        print(f"Usage: python -m enlitens_client_profiles.view_persona [1-{len(personas)}]")

