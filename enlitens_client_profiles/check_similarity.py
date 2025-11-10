"""
Check similarity between all generated personas.
Run this AFTER generating personas to verify uniqueness.
"""
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from enlitens_client_profiles.schema_v2_real_stories import ClientProfileV2RealStories


def extract_comparable_text(persona_dict: dict) -> str:
    """Extract text from persona dict for similarity comparison."""
    parts = []
    
    # Identity & Demographics
    id_demo = persona_dict.get("identity_demographics", {})
    parts.append(f"Age: {id_demo.get('age_range')}")
    parts.append(f"Gender: {id_demo.get('gender')}")
    parts.append(f"Situation: {id_demo.get('current_life_situation')}")
    parts.append(f"Occupation: {id_demo.get('occupation')}")
    parts.append(f"Education: {id_demo.get('education')}")
    parts.append(f"Locality: {id_demo.get('locality')}")
    parts.append(f"Cultural identities: {' '.join(id_demo.get('cultural_faith_identities', []))}")
    
    # Developmental Story
    dev = persona_dict.get("developmental_story", {})
    parts.append(f"Childhood: {dev.get('childhood_environment')}")
    parts.append(f"Adolescence: {dev.get('adolescence')}")
    parts.append(f"Early adulthood: {dev.get('early_adulthood')}")
    parts.append(f"Family: {dev.get('family_structure')}")
    parts.append(f"Adversities: {' '.join(dev.get('formative_adversities', []))}")
    parts.append(f"Education journey: {dev.get('educational_journey')}")
    parts.append(f"Pivotal moments: {' '.join(dev.get('pivotal_moments', []))}")
    parts.append(f"Intergenerational: {dev.get('intergenerational_patterns')}")
    
    # Neurodivergence & Mental Health
    neuro = persona_dict.get("neurodivergence_mental_health", {})
    parts.append(f"Identities: {' '.join(neuro.get('identities', []))}")
    parts.append(f"Diagnosis: {neuro.get('diagnosis_journey')}")
    parts.append(f"How it shows up: {neuro.get('how_it_shows_up')}")
    parts.append(f"Nervous system: {neuro.get('nervous_system_pattern')}")
    parts.append(f"Strengths: {' '.join(neuro.get('strengths_superpowers', []))}")
    parts.append(f"Coping: {' '.join(neuro.get('current_coping_strategies', []))}")
    
    # Executive Function & Sensory
    ef = persona_dict.get("executive_function_sensory", {})
    parts.append(f"EF strengths: {' '.join(ef.get('ef_strengths', []))}")
    parts.append(f"EF friction: {' '.join(ef.get('ef_friction_points', []))}")
    parts.append(f"EF workarounds: {' '.join(ef.get('ef_workarounds', []))}")
    parts.append(f"Sensory: {ef.get('sensory_profile')}")
    parts.append(f"Food: {ef.get('food_sensory_details')}")
    
    # Current Life Context
    context = persona_dict.get("current_life_context", {})
    parts.append(f"Where: {context.get('where_they_live')}")
    parts.append(f"Work/school: {context.get('work_school_situation')}")
    parts.append(f"Daily life: {context.get('commute_daily_rhythms')}")
    parts.append(f"Stressors: {' '.join(context.get('local_stressors', []))}")
    parts.append(f"Safe spaces: {' '.join(context.get('safe_spaces', []))}")
    parts.append(f"Support: {context.get('support_system')}")
    
    # Goals & Barriers
    goals = persona_dict.get("goals_barriers", {})
    parts.append(f"Why now: {goals.get('why_therapy_now')}")
    parts.append(f"Want to change: {' '.join(goals.get('what_they_want_to_change', []))}")
    parts.append(f"Barriers: {' '.join(goals.get('whats_in_the_way', []))}")
    
    # Narrative Voice
    narrative = persona_dict.get("narrative_voice", {})
    parts.append(f"Struggle quote: {narrative.get('quote_struggle')}")
    parts.append(f"Hope quote: {narrative.get('quote_hope')}")
    parts.append(f"Additional quotes: {' '.join(narrative.get('quotes_additional', []))}")
    parts.append(f"Clinical narrative: {narrative.get('liz_clinical_narrative')}")
    parts.append(f"Therapy prefs: {narrative.get('therapy_preferences')}")
    
    return "\n".join(str(p) for p in parts if p and p != "None")


def main():
    project_root = Path("/home/antons-gs/enlitens-ai")
    profiles_dir = project_root / "enlitens_client_profiles" / "profiles"
    
    # Load all personas
    persona_files = sorted(profiles_dir.glob("persona_real_story_*.json"))
    
    if not persona_files:
        print("❌ No personas found!")
        return
    
    print(f"Found {len(persona_files)} personas")
    print("\nLoading personas...")
    
    personas = []
    for f in persona_files:
        with open(f) as fp:
            persona_dict = json.load(fp)
            personas.append({
                "file": f.name,
                "name": persona_dict.get("meta", {}).get("persona_name", "Unknown"),
                "data": persona_dict
            })
    
    print(f"✅ Loaded {len(personas)} personas\n")
    
    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    print("✅ Model loaded\n")
    
    # Calculate embeddings
    print("Calculating embeddings...")
    embeddings = []
    for p in personas:
        text = extract_comparable_text(p["data"])
        embedding = model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        embeddings.append(embedding)
    print("✅ Embeddings calculated\n")
    
    # Calculate similarity matrix
    print("="*80)
    print("SIMILARITY ANALYSIS")
    print("="*80 + "\n")
    
    threshold = 0.41
    violations = []
    
    for i in range(len(personas)):
        for j in range(i + 1, len(personas)):
            similarity = float(np.dot(embeddings[i], embeddings[j]))
            
            if similarity > threshold:
                violations.append({
                    "persona1": personas[i]["name"],
                    "persona2": personas[j]["name"],
                    "similarity": similarity
                })
    
    # Print results
    if violations:
        print(f"⚠️  FOUND {len(violations)} SIMILARITY VIOLATIONS (>{threshold*100:.0f}%):\n")
        for v in sorted(violations, key=lambda x: x["similarity"], reverse=True):
            print(f"  {v['similarity']*100:.1f}% - {v['persona1']} ↔ {v['persona2']}")
    else:
        print(f"✅ ALL PERSONAS ARE UNIQUE (all similarities <{threshold*100:.0f}%)\n")
    
    # Print full similarity matrix
    print("\n" + "="*80)
    print("FULL SIMILARITY MATRIX")
    print("="*80 + "\n")
    
    # Print header
    print(f"{'':30}", end="")
    for i, p in enumerate(personas, 1):
        print(f"{i:>6}", end="")
    print()
    
    # Print matrix
    for i, p1 in enumerate(personas, 1):
        print(f"{i:2}. {p1['name'][:26]:26}", end="")
        for j, p2 in enumerate(personas):
            if i-1 == j:
                print(f"{'---':>6}", end="")
            elif i-1 < j:
                similarity = float(np.dot(embeddings[i-1], embeddings[j]))
                print(f"{similarity*100:5.1f}%", end="")
            else:
                print(f"{'':>6}", end="")
        print()
    
    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80 + "\n")
    
    all_similarities = []
    for i in range(len(personas)):
        for j in range(i + 1, len(personas)):
            similarity = float(np.dot(embeddings[i], embeddings[j]))
            all_similarities.append(similarity)
    
    if all_similarities:
        print(f"Total comparisons: {len(all_similarities)}")
        print(f"Average similarity: {np.mean(all_similarities)*100:.1f}%")
        print(f"Max similarity: {np.max(all_similarities)*100:.1f}%")
        print(f"Min similarity: {np.min(all_similarities)*100:.1f}%")
        print(f"Std deviation: {np.std(all_similarities)*100:.1f}%")
        print(f"\nViolations (>{threshold*100:.0f}%): {len(violations)}")
        print(f"Unique pairs: {len(all_similarities) - len(violations)}")
    
    print("\n" + "="*80)
    print("PERSONA LIST")
    print("="*80 + "\n")
    
    for i, p in enumerate(personas, 1):
        demo = p["data"].get("identity_demographics", {})
        print(f"{i:2}. {p['name']}")
        print(f"    {demo.get('age_range')} | {demo.get('current_life_situation')}")
        print(f"    {demo.get('locality')}")
        print()


if __name__ == "__main__":
    main()

