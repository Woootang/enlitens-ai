"""
Generate 10 UNIQUE client personas with 41% similarity threshold.
Tests deduplication and provides feedback to Gemini on rejections.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

from enlitens_client_profiles.config import ProfilePipelineConfig
from enlitens_client_profiles.data_ingestion import load_ingestion_bundle
from enlitens_client_profiles.gemini_client import GeminiClient
from enlitens_client_profiles.schema_v2_real_stories import ClientProfileV2RealStories
from sentence_transformers import SentenceTransformer
import numpy as np


def build_persona_prompt(
    bundle,
    persona_number: int,
    existing_personas: List[ClientProfileV2RealStories] = None,
    rejection_feedback: str = None
) -> str:
    """
    Build a prompt for generating a unique persona.
    Includes feedback if this is a retry after rejection.
    """
    
    # Sample intakes
    intake_samples = [t.raw_text[:500] for t in bundle.transcripts[:10] if hasattr(t, 'raw_text')]
    intake_text = "\n\n".join(intake_samples)
    
    # Analytics summary
    analytics_summary = ""
    if bundle.analytics:
        analytics_summary = f"""
ANALYTICS CONTEXT:
- Top search queries: {', '.join(bundle.analytics.top_queries[:5]) if hasattr(bundle.analytics, 'top_queries') else 'N/A'}
- Top landing pages: {', '.join(bundle.analytics.top_landing_pages[:3]) if hasattr(bundle.analytics, 'top_landing_pages') else 'N/A'}
"""
    
    # Build existing personas summary for uniqueness
    existing_summary = ""
    if existing_personas:
        existing_summary = "\n**EXISTING PERSONAS (you MUST be different from these):**\n\n"
        for i, p in enumerate(existing_personas, 1):
            existing_summary += f"{i}. {p.meta.persona_name}\n"
            existing_summary += f"   - Age: {p.identity_demographics.age_range}\n"
            existing_summary += f"   - Situation: {p.identity_demographics.current_life_situation}\n"
            existing_summary += f"   - Locality: {p.identity_demographics.locality}\n"
            existing_summary += f"   - Key adversities: {', '.join(p.developmental_story.formative_adversities[:3])}\n"
            existing_summary += f"   - Identities: {', '.join(p.neurodivergence_mental_health.identities)}\n\n"
    
    # Rejection feedback
    rejection_section = ""
    if rejection_feedback:
        rejection_section = f"""
⚠️ **REJECTION FEEDBACK - PREVIOUS ATTEMPT WAS TOO SIMILAR:**

{rejection_feedback}

**YOU MUST MAKE THIS PERSONA MORE UNIQUE. Change:**
- Age range and life stage
- Family structure and dynamics
- Specific adversities and trauma
- Occupation and education path
- Locality within St. Louis
- Specific neurodivergent presentation
- Cultural/faith identities

DO NOT create another version of the same person. Create someone COMPLETELY DIFFERENT.

---
"""
    
    prompt = f"""You are generating REAL, DEEPLY HUMAN client persona #{persona_number} for Enlitens, a neurodivergent-affirming therapy practice in St. Louis, MO.

{rejection_section}

{existing_summary}

**CRITICAL INSTRUCTIONS:**

1. **USE THE INTAKE LANGUAGE**: Below are REAL intake messages from actual people. Use their ACTUAL WORDS and patterns. Don't sanitize or genericize.

2. **DEVELOPMENTAL STORY IS MANDATORY**: Every adult has a childhood story (0-25 years). You MUST create:
   - Childhood environment (poverty? stability? chaos? who raised them?)
   - Family structure (parents, siblings, losses, divorces, who was actually there)
   - Formative adversities (abuse, neglect, instability, loss, trauma, moves, poverty)
   - Educational journey (school changes, disruptions, struggles, achievements)
   - Pivotal moments that shaped who they are
   - Intergenerational patterns (was mom also neurodivergent? generational trauma?)

3. **FOOD SENSORY MUST BE SPECIFIC**: NOT "pizza and nuggets". Be DETAILED:
   - Specific textures (e.g. "avoids anything slimy or with visible fat")
   - Specific preparations (e.g. "only eats chicken if it's grilled, room temperature, cut into small pieces")
   - Specific brands or foods (e.g. "safe foods: plain Cheerios, specific brand of crackers, grilled chicken breast")
   - How food impacts functioning (e.g. "binge eating when stressed", "can't eat in social situations")

4. **NO TOURIST TRAPS**: Do NOT mention Pappy's, Imo's, Ted Drewes, or any "famous" St. Louis places unless it's genuinely relevant. If you mention a place, it should reveal something about the PERSON.

5. **REAL HUMAN DEPTH**: These are REAL people with REAL struggles. Create personas with:
   - Actual adversity and trauma
   - Complex family dynamics
   - Real developmental challenges
   - Genuine human struggle and resilience

6. **REMOVE REPETITION**: Don't repeat the same information across multiple fields. Each field should add NEW information.

7. **BE UNIQUE**: This persona MUST be significantly different from any existing personas listed above. Vary:
   - Age, gender, orientation, ethnicity
   - Life situation (single, married, divorced, parent, childless, student, worker)
   - Locality (North County, South City, West County, Clayton, University City, etc.)
   - Adversities (don't repeat the same trauma stories)
   - Neurodivergent presentation (ADHD-I vs ADHD-C vs AuDHD vs late-diagnosed autism, etc.)

---

**REAL INTAKE EXAMPLES (use this language and these patterns):**

{intake_text}

---

{analytics_summary}

---

**YOUR TASK:**

Generate ONE complete, deeply human persona as a SINGLE JSON OBJECT (NOT an array) matching this EXACT structure:

{{
  "meta": {{
    "profile_id": "persona-real-{persona_number:03d}",
    "persona_name": "Descriptive Name (e.g. 'Late-Diagnosed Ray', 'Burned-Out College Student')",
    "persona_tagline": "Short tagline",
    "created_at": "{datetime.utcnow().isoformat()}",
    "source_documents": ["intake_samples", "analytics"],
    "llm_model": "gemini-2.5-real-stories",
    "version": "2.0",
    "attribute_tags": ["adhd", "autistic", "late_diagnosed", "student", etc.]
  }},
  "identity_demographics": {{ ... }},
  "developmental_story": {{ ... }},
  "neurodivergence_mental_health": {{ ... }},
  "executive_function_sensory": {{ ... }},
  "current_life_context": {{ ... }},
  "goals_barriers": {{ ... }},
  "narrative_voice": {{ ... }},
  "marketing_seo": {{ ... }}
}}

**CRITICAL:**
- Return a SINGLE JSON OBJECT, not an array
- Use REAL intake language
- Create DEEP developmental stories
- Be SPECIFIC about food/sensory
- NO tourist traps
- REAL human struggle and depth
- Each field adds NEW information
- BE UNIQUE from existing personas

Generate the complete JSON object now (NOT wrapped in an array).
"""
    
    return prompt


def extract_comparable_text(persona: ClientProfileV2RealStories) -> str:
    """
    Extract text from persona for similarity comparison.
    Excludes meta and marketing_seo as those are derivative.
    """
    parts = []
    
    # Identity & Demographics
    parts.append(f"Age: {persona.identity_demographics.age_range}")
    parts.append(f"Gender: {persona.identity_demographics.gender}")
    parts.append(f"Situation: {persona.identity_demographics.current_life_situation}")
    parts.append(f"Occupation: {persona.identity_demographics.occupation}")
    parts.append(f"Education: {persona.identity_demographics.education}")
    parts.append(f"Locality: {persona.identity_demographics.locality}")
    parts.append(f"Cultural identities: {' '.join(persona.identity_demographics.cultural_faith_identities)}")
    
    # Developmental Story
    parts.append(f"Childhood: {persona.developmental_story.childhood_environment}")
    parts.append(f"Adolescence: {persona.developmental_story.adolescence}")
    parts.append(f"Early adulthood: {persona.developmental_story.early_adulthood}")
    parts.append(f"Family: {persona.developmental_story.family_structure}")
    parts.append(f"Adversities: {' '.join(persona.developmental_story.formative_adversities)}")
    parts.append(f"Education journey: {persona.developmental_story.educational_journey}")
    parts.append(f"Pivotal moments: {' '.join(persona.developmental_story.pivotal_moments)}")
    parts.append(f"Intergenerational: {persona.developmental_story.intergenerational_patterns}")
    
    # Neurodivergence & Mental Health
    parts.append(f"Identities: {' '.join(persona.neurodivergence_mental_health.identities)}")
    parts.append(f"Diagnosis: {persona.neurodivergence_mental_health.diagnosis_journey}")
    parts.append(f"How it shows up: {persona.neurodivergence_mental_health.how_it_shows_up}")
    parts.append(f"Nervous system: {persona.neurodivergence_mental_health.nervous_system_pattern}")
    parts.append(f"Strengths: {' '.join(persona.neurodivergence_mental_health.strengths_superpowers)}")
    parts.append(f"Coping: {' '.join(persona.neurodivergence_mental_health.current_coping_strategies)}")
    
    # Executive Function & Sensory
    parts.append(f"EF strengths: {' '.join(persona.executive_function_sensory.ef_strengths)}")
    parts.append(f"EF friction: {' '.join(persona.executive_function_sensory.ef_friction_points)}")
    parts.append(f"EF workarounds: {' '.join(persona.executive_function_sensory.ef_workarounds)}")
    parts.append(f"Sensory: {persona.executive_function_sensory.sensory_profile}")
    parts.append(f"Food: {persona.executive_function_sensory.food_sensory_details}")
    
    # Current Life Context
    parts.append(f"Where: {persona.current_life_context.where_they_live}")
    parts.append(f"Work/school: {persona.current_life_context.work_school_situation}")
    parts.append(f"Daily life: {persona.current_life_context.commute_daily_rhythms}")
    parts.append(f"Stressors: {' '.join(persona.current_life_context.local_stressors)}")
    parts.append(f"Safe spaces: {' '.join(persona.current_life_context.safe_spaces)}")
    parts.append(f"Support: {persona.current_life_context.support_system}")
    
    # Goals & Barriers
    parts.append(f"Why now: {persona.goals_barriers.why_therapy_now}")
    parts.append(f"Want to change: {' '.join(persona.goals_barriers.what_they_want_to_change)}")
    parts.append(f"Barriers: {' '.join(persona.goals_barriers.whats_in_the_way)}")
    
    # Narrative Voice
    parts.append(f"Struggle quote: {persona.narrative_voice.quote_struggle}")
    parts.append(f"Hope quote: {persona.narrative_voice.quote_hope}")
    parts.append(f"Additional quotes: {' '.join(persona.narrative_voice.quotes_additional)}")
    parts.append(f"Clinical narrative: {persona.narrative_voice.liz_clinical_narrative}")
    parts.append(f"Therapy prefs: {persona.narrative_voice.therapy_preferences}")
    
    return "\n".join(str(p) for p in parts if p)


def calculate_similarity(
    new_persona: ClientProfileV2RealStories,
    existing_personas: List[ClientProfileV2RealStories],
    embedding_model: SentenceTransformer
) -> Tuple[float, ClientProfileV2RealStories]:
    """
    Calculate similarity between new persona and all existing personas.
    Returns (max_similarity, most_similar_persona).
    """
    if not existing_personas:
        return 0.0, None
    
    new_text = extract_comparable_text(new_persona)
    new_embedding = embedding_model.encode([new_text], normalize_embeddings=True, show_progress_bar=False)[0]
    
    max_similarity = 0.0
    most_similar = None
    
    for existing in existing_personas:
        existing_text = extract_comparable_text(existing)
        existing_embedding = embedding_model.encode([existing_text], normalize_embeddings=True, show_progress_bar=False)[0]
        
        # Cosine similarity (embeddings are already normalized)
        similarity = float(np.dot(new_embedding, existing_embedding))
        
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar = existing
    
    return max_similarity, most_similar


def main():
    # Setup
    project_root = Path("/home/antons-gs/enlitens-ai")
    config = ProfilePipelineConfig(project_root=project_root)
    output_dir = project_root / "enlitens_client_profiles" / "profiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading ingestion bundle...")
    bundle = load_ingestion_bundle(config)
    print(f"Loaded {len(bundle.transcripts)} intake messages")
    
    # Setup clients
    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyChf4y7bqezULJtipsNXOJvaK3MVW0XvXI")
    gemini_client = GeminiClient(api_key=api_key)
    
    print("Loading embedding model for similarity detection...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    print("✅ Model loaded\n")
    
    # Track results
    accepted_personas: List[ClientProfileV2RealStories] = []
    metrics = {
        "total_attempts": 0,
        "rejections": 0,
        "threshold_adjustments": 0,
        "similarity_scores": []
    }
    
    print("\n" + "="*80)
    print("GENERATING 10 UNIQUE PERSONAS WITH 41% SIMILARITY THRESHOLD")
    print("="*80 + "\n")
    
    # Generate 10 personas
    for i in range(1, 11):
        print(f"\n{'='*80}")
        print(f"GENERATING PERSONA #{i}")
        print(f"{'='*80}\n")
        
        threshold = 0.41  # Start with 41% threshold
        max_retries = 5
        retry_count = 0
        rejection_feedback = None
        
        while retry_count < max_retries:
            metrics["total_attempts"] += 1
            
            # Build prompt
            prompt = build_persona_prompt(
                bundle=bundle,
                persona_number=i,
                existing_personas=accepted_personas,
                rejection_feedback=rejection_feedback
            )
            
            # Generate
            print(f"Attempt {retry_count + 1}/{max_retries} (threshold: {threshold*100:.0f}%)")
            
            try:
                result = gemini_client.generate_structured(
                    prompt=prompt,
                    response_model=ClientProfileV2RealStories,
                    temperature=0.9,  # Higher temp for more diversity
                )
                
                if result is None:
                    print(f"  ❌ ERROR: Gemini returned None")
                    retry_count += 1
                    continue
                
                # Check similarity
                if accepted_personas:
                    similarity, most_similar = calculate_similarity(
                        result, accepted_personas, embedding_model
                    )
                    
                    print(f"  Similarity to existing: {similarity*100:.1f}%")
                    
                    if similarity > threshold:
                        # REJECTED - too similar
                        metrics["rejections"] += 1
                        retry_count += 1
                        
                        print(f"  ❌ REJECTED - Too similar to '{most_similar.meta.persona_name}'")
                        
                        # Build feedback for next attempt
                        rejection_feedback = f"""
Your previous attempt was {similarity*100:.1f}% similar to "{most_similar.meta.persona_name}".

That persona has:
- Age: {most_similar.identity_demographics.age_range}
- Situation: {most_similar.identity_demographics.current_life_situation}
- Locality: {most_similar.identity_demographics.locality}
- Adversities: {', '.join(most_similar.developmental_story.formative_adversities[:3])}
- Identities: {', '.join(most_similar.neurodivergence_mental_health.identities)}

You MUST create someone COMPLETELY DIFFERENT. Change the age, life situation, locality, adversities, and neurodivergent presentation.
"""
                        
                        # If we've tried 5 times, raise threshold to make it easier
                        if retry_count >= max_retries:
                            threshold = 0.50
                            max_retries += 1  # Give one more chance with easier threshold
                            metrics["threshold_adjustments"] += 1
                            print(f"  ⚠️  Raising threshold to {threshold*100:.0f}% for next attempt")
                        
                        continue
                    else:
                        # ACCEPTED
                        print(f"  ✅ ACCEPTED - Unique enough ({similarity*100:.1f}% < {threshold*100:.0f}%)")
                        metrics["similarity_scores"].append(similarity)
                else:
                    # First persona - always accept
                    print(f"  ✅ ACCEPTED - First persona")
                    metrics["similarity_scores"].append(0.0)
                
                # Save and add to accepted list
                accepted_personas.append(result)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = output_dir / f"persona_{i:03d}_{timestamp}.json"
                
                # Handle both Pydantic model and dict
                if hasattr(result, 'model_dump'):
                    data = result.model_dump()
                else:
                    data = result
                
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                
                print(f"\n  Saved: {result.meta.persona_name}")
                print(f"  File: {output_file.name}")
                
                break  # Success - move to next persona
                
            except Exception as e:
                print(f"  ❌ ERROR: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"\n  ⛔ FAILED after {max_retries} attempts. Stopping.")
                    print(f"\n  Need to reassess approach. Generated {len(accepted_personas)} personas so far.")
                    
                    # Print metrics and exit
                    print("\n" + "="*80)
                    print("METRICS")
                    print("="*80)
                    print(f"Total attempts: {metrics['total_attempts']}")
                    print(f"Rejections: {metrics['rejections']}")
                    print(f"Threshold adjustments: {metrics['threshold_adjustments']}")
                    print(f"Accepted personas: {len(accepted_personas)}")
                    if metrics["similarity_scores"]:
                        avg_sim = sum(metrics["similarity_scores"]) / len(metrics["similarity_scores"])
                        print(f"Average similarity: {avg_sim*100:.1f}%")
                    
                    return
    
    # Success - generated all 10
    print("\n" + "="*80)
    print("✅ SUCCESS - GENERATED 10 UNIQUE PERSONAS")
    print("="*80 + "\n")
    
    print("METRICS:")
    print(f"  Total attempts: {metrics['total_attempts']}")
    print(f"  Rejections: {metrics['rejections']}")
    print(f"  Threshold adjustments: {metrics['threshold_adjustments']}")
    print(f"  Acceptance rate: {len(accepted_personas) / metrics['total_attempts'] * 100:.1f}%")
    
    if metrics["similarity_scores"]:
        avg_sim = sum(metrics["similarity_scores"]) / len(metrics["similarity_scores"])
        max_sim = max(metrics["similarity_scores"])
        print(f"  Average similarity: {avg_sim*100:.1f}%")
        print(f"  Max similarity: {max_sim*100:.1f}%")
    
    print("\nGENERATED PERSONAS:")
    for i, p in enumerate(accepted_personas, 1):
        print(f"  {i}. {p.meta.persona_name}")
        print(f"     {p.identity_demographics.age_range} | {p.identity_demographics.current_life_situation}")
        print(f"     {p.identity_demographics.locality}")


if __name__ == "__main__":
    main()

