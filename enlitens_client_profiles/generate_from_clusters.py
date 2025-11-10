"""
Generate personas from intake clusters.
Each persona represents a real client segment with guaranteed diversity.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from enlitens_client_profiles.gemini_client import GeminiClient
from enlitens_client_profiles.schema_v2_real_stories import ClientProfileV2RealStories


def build_cluster_persona_prompt(cluster_id: int, cluster_data: Dict, cluster_number: int, total_clusters: int) -> str:
    """
    Build a prompt to generate a persona from a specific cluster.
    Uses the ACTUAL intake text from that cluster.
    """
    
    # Get representative samples from this cluster
    samples = cluster_data["representative_samples"][:5]  # Top 5 most representative
    all_texts = cluster_data["all_texts"]
    
    # Build intake examples from this cluster
    intake_examples = "\n\n".join([f"INTAKE {i+1}:\n{s['text']}" for i, s in enumerate(samples)])
    
    prompt = f"""You are generating a REAL, DEEPLY HUMAN client persona for Enlitens, a neurodivergent-affirming therapy practice in St. Louis, MO.

**THIS PERSONA REPRESENTS CLIENT SEGMENT #{cluster_number} of {total_clusters}**

This segment contains {len(all_texts)} real client intakes with similar characteristics. You MUST create a persona that represents this SPECIFIC client segment.

---

**REAL INTAKE MESSAGES FROM THIS SEGMENT:**

{intake_examples}

---

**CRITICAL INSTRUCTIONS:**

1. **USE THE ACTUAL LANGUAGE** from these intakes. These are REAL people - use their words, their struggles, their voice.

2. **REPRESENT THIS SEGMENT:** Create a persona that captures the common themes, struggles, and characteristics across these intakes.

3. **DEVELOPMENTAL STORY IS MANDATORY:** Create a rich childhood-to-present story (0-25 years):
   - Childhood environment (poverty? stability? chaos? who raised them?)
   - Family structure (parents, siblings, losses, divorces, who was actually there)
   - Formative adversities (abuse, neglect, instability, loss, trauma, moves, poverty)
   - Educational journey (school changes, disruptions, struggles, achievements)
   - Pivotal moments that shaped who they are
   - Intergenerational patterns (was mom also neurodivergent? generational trauma?)

4. **FOOD SENSORY MUST BE SPECIFIC:** NOT "pizza and nuggets". Be DETAILED:
   - Specific textures (e.g. "avoids anything slimy or with visible fat")
   - Specific preparations (e.g. "only eats chicken if it's grilled, room temperature")
   - Specific brands or foods (e.g. "safe foods: plain Cheerios, specific brand of crackers")
   - How food impacts functioning (e.g. "binge eating when stressed", "skips meals when overwhelmed")

5. **NO TOURIST TRAPS:** Do NOT mention Pappy's, Imo's, Ted Drewes unless genuinely relevant.

6. **REAL HUMAN DEPTH:** Create a person with:
   - Actual adversity and trauma
   - Complex family dynamics
   - Real developmental challenges
   - Genuine human struggle and resilience

7. **INFER DETAILS:** The intakes are brief. You MUST infer and expand:
   - If they mention "late-diagnosed ADHD" → create their full diagnosis journey
   - If they mention "mom of 2" → create their parenting story, family dynamics
   - If they mention "burnout" → create what led to it, the years of struggle
   - If they mention a specific struggle → trace it back to childhood

---

**YOUR TASK:**

Generate ONE complete, deeply human persona as a SINGLE JSON OBJECT (NOT an array) matching this EXACT structure:

{{
  "meta": {{
    "profile_id": "cluster-{cluster_id:03d}-persona",
    "persona_name": "Descriptive Name based on the intakes (e.g. 'Burned-Out ADHD Professional', 'Struggling Single Mom')",
    "persona_tagline": "Short tagline",
    "created_at": "{datetime.utcnow().isoformat()}",
    "source_documents": ["cluster_{cluster_id}", "intake_segment"],
    "llm_model": "gemini-cluster-based",
    "version": "2.0",
    "attribute_tags": ["relevant", "tags", "from", "intakes"]
  }},
  "identity_demographics": {{
    "age_range": "infer from intakes",
    "gender": "infer from intakes",
    "pronouns": "infer from intakes",
    "orientation": "infer from intakes if mentioned",
    "ethnicity": "infer from intakes if mentioned",
    "current_life_situation": "infer from intakes (e.g. 'single mom of 2', 'grad student')",
    "occupation": "infer from intakes",
    "education": "infer from intakes",
    "locality": "infer St. Louis area from intakes if mentioned, otherwise choose appropriate area",
    "cultural_faith_identities": ["infer from intakes"]
  }},
  "developmental_story": {{
    "childhood_environment": "CREATE based on current struggles - what childhood led to this?",
    "adolescence": "CREATE - what happened in teen years?",
    "early_adulthood": "CREATE - what happened 19-25?",
    "family_structure": "CREATE - who raised them, siblings, dynamics",
    "formative_adversities": ["CREATE list of specific adversities"],
    "educational_journey": "CREATE - schools, disruptions, achievements",
    "pivotal_moments": ["CREATE list of 3-5 pivotal moments"],
    "intergenerational_patterns": "CREATE - family patterns"
  }},
  "neurodivergence_mental_health": {{
    "identities": ["from intakes"],
    "diagnosis_journey": "CREATE full story",
    "how_it_shows_up": "from intakes + expand",
    "nervous_system_pattern": "CREATE",
    "strengths_superpowers": ["CREATE"],
    "current_coping_strategies": ["CREATE based on intakes"]
  }},
  "executive_function_sensory": {{
    "ef_strengths": ["CREATE"],
    "ef_friction_points": ["from intakes + expand"],
    "ef_workarounds": ["CREATE"],
    "sensory_profile": "CREATE",
    "food_sensory_details": "CREATE DETAILED food profile - be SPECIFIC"
  }},
  "current_life_context": {{
    "where_they_live": "CREATE based on intakes",
    "work_school_situation": "from intakes + expand",
    "commute_daily_rhythms": "CREATE",
    "local_stressors": ["CREATE"],
    "safe_spaces": ["CREATE"],
    "support_system": "CREATE based on intakes"
  }},
  "goals_barriers": {{
    "why_therapy_now": "from intakes - what brought them here NOW",
    "what_they_want_to_change": ["from intakes + expand"],
    "whats_in_the_way": ["CREATE - internal + external barriers"]
  }},
  "narrative_voice": {{
    "quote_struggle": "USE ACTUAL WORDS from intakes if possible",
    "quote_hope": "USE ACTUAL WORDS from intakes if possible",
    "quotes_additional": ["USE ACTUAL WORDS from intakes"],
    "liz_clinical_narrative": "150-250 word narrative in Liz Wooten's compassionate tone",
    "therapy_preferences": "infer from intakes"
  }},
  "marketing_seo": {{
    "website_copy_snippets": ["2-3 snippets that would resonate with this segment"],
    "primary_keywords": ["keywords for this segment"],
    "local_entities": ["St. Louis areas relevant to this segment"],
    "content_angles": ["2-3 content angles for this segment"]
  }}
}}

**CRITICAL:**
- Return a SINGLE JSON OBJECT, not an array
- Use REAL language from the intakes
- CREATE the developmental story (don't leave it blank)
- Be SPECIFIC about food/sensory
- INFER and EXPAND from the brief intakes
- Make this a REAL person with depth

Generate the complete JSON object now.
"""
    
    return prompt


def generate_persona_from_cluster(
    cluster_id: int,
    cluster_data: Dict,
    cluster_number: int,
    total_clusters: int,
    gemini_client: GeminiClient
) -> ClientProfileV2RealStories:
    """Generate one persona from a cluster."""
    
    prompt = build_cluster_persona_prompt(cluster_id, cluster_data, cluster_number, total_clusters)
    
    result = gemini_client.generate_structured(
        prompt=prompt,
        response_model=ClientProfileV2RealStories,
        temperature=0.8,
    )
    
    return result


def main(test_mode: bool = True, num_test: int = 3):
    """
    Generate personas from clusters.
    
    Args:
        test_mode: If True, only generate num_test personas for quality check
        num_test: Number of personas to generate in test mode
    """
    
    project_root = Path("/home/antons-gs/enlitens-ai")
    
    # Find the most recent clusters file
    clusters_dir = project_root / "enlitens_client_profiles" / "clusters"
    cluster_files = sorted(clusters_dir.glob("clusters_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not cluster_files:
        raise FileNotFoundError(f"No cluster files found in {clusters_dir}")
    
    clusters_file = cluster_files[0]
    print(f"✅ Using cluster file: {clusters_file.name}", flush=True)
    
    output_dir = project_root / "enlitens_client_profiles" / "profiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load clusters
    print("Loading cluster data...", flush=True)
    with open(clusters_file) as f:
        cluster_data = json.load(f)
    
    total_clusters = cluster_data["n_clusters"]
    clusters = cluster_data["clusters"]
    
    print(f"✅ Loaded {total_clusters} clusters\n", flush=True)
    
    # Sort clusters by size (largest first)
    sorted_clusters = sorted(
        [(int(k), v) for k, v in clusters.items()],
        key=lambda x: x[1]["size"],
        reverse=True
    )
    
    if test_mode:
        print(f"🧪 TEST MODE: Generating {num_test} personas from largest clusters\n", flush=True)
        sorted_clusters = sorted_clusters[:num_test]
    else:
        print(f"🚀 FULL MODE: Generating {total_clusters} personas\n", flush=True)
    
    # Setup Gemini
    import os
    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyChf4y7bqezULJtipsNXOJvaK3MVW0XvXI")
    print(f"Initializing Gemini client...", flush=True)
    gemini_client = GeminiClient(api_key=api_key)
    print(f"✅ Gemini client ready\n", flush=True)
    
    # Generate personas
    print("="*80)
    print("GENERATING PERSONAS FROM CLUSTERS")
    print("="*80 + "\n")
    
    generated = []
    failed = []
    
    for i, (cluster_id, cluster_info) in enumerate(sorted_clusters, 1):
        cluster_size = cluster_info["size"]
        print(f"\n{i}/{len(sorted_clusters)}. Cluster #{cluster_id} ({cluster_size} intakes)")
        print("-"*80)
        
        try:
            persona = generate_persona_from_cluster(
                cluster_id=cluster_id,
                cluster_data=cluster_info,
                cluster_number=i,
                total_clusters=len(sorted_clusters),
                gemini_client=gemini_client
            )
            
            if persona is None:
                print(f"  ❌ FAILED: Gemini returned None")
                failed.append(cluster_id)
                continue
            
            # Save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"persona_cluster_{cluster_id:03d}_{timestamp}.json"
            output_file = output_dir / filename
            
            if hasattr(persona, 'model_dump'):
                data = persona.model_dump()
            else:
                data = persona
            
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"  ✅ SUCCESS: {persona.meta.persona_name}")
            print(f"     Age: {persona.identity_demographics.age_range}")
            print(f"     Situation: {persona.identity_demographics.current_life_situation}")
            print(f"     Saved: {filename}")
            
            generated.append({
                "cluster_id": cluster_id,
                "cluster_size": cluster_size,
                "persona_name": persona.meta.persona_name,
                "file": filename
            })
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed.append(cluster_id)
    
    # Summary
    print("\n" + "="*80)
    print("GENERATION SUMMARY")
    print("="*80 + "\n")
    
    print(f"✅ Successfully generated: {len(generated)}")
    print(f"❌ Failed: {len(failed)}")
    
    if generated:
        print("\nGenerated Personas:")
        for p in generated:
            print(f"  • Cluster #{p['cluster_id']} ({p['cluster_size']} intakes): {p['persona_name']}")
    
    if failed:
        print(f"\nFailed Clusters: {failed}")
    
    if test_mode:
        print("\n" + "="*80)
        print("TEST MODE COMPLETE")
        print("="*80)
        print("\nReview the generated personas above.")
        print("If quality is good, run with test_mode=False to generate all 100.\n")
    
    return generated, failed


if __name__ == "__main__":
    import sys
    
    # Check if full mode requested
    test_mode = "--full" not in sys.argv
    
    generated, failed = main(test_mode=test_mode, num_test=3)

