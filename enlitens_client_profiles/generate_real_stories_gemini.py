"""
Generate REAL client personas with depth using Gemini 2.5 Pro.
Focus on developmental stories, real human struggle, and intake language.
"""
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from enlitens_client_profiles.config import ProfilePipelineConfig
from enlitens_client_profiles.data_ingestion import load_ingestion_bundle
from enlitens_client_profiles.gemini_client import GeminiClient
from enlitens_client_profiles.schema_v2_real_stories import ClientProfileV2RealStories


def build_real_story_prompt(bundle, num_personas: int = 1) -> str:
    """
    Build a prompt that demands REAL human stories with developmental depth.
    Uses the actual language from intakes.txt.
    """
    
    # Sample intakes to show REAL language
    intake_samples = [t.raw_text[:500] for t in bundle.transcripts[:10] if hasattr(t, 'raw_text')]
    intake_text = "\n\n".join(intake_samples)
    
    # Sample analytics
    analytics_summary = ""
    if bundle.analytics:
        analytics_summary = f"""
ANALYTICS CONTEXT:
- Top search queries: {', '.join(bundle.analytics.top_queries[:5]) if hasattr(bundle.analytics, 'top_queries') else 'N/A'}
- Top landing pages: {', '.join(bundle.analytics.top_landing_pages[:3]) if hasattr(bundle.analytics, 'top_landing_pages') else 'N/A'}
"""
    
    prompt = f"""You are generating REAL, DEEPLY HUMAN client personas for Enlitens, a neurodivergent-affirming therapy practice in St. Louis, MO.

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
    "profile_id": "persona-real-001",
    "persona_name": "Descriptive Name (e.g. 'Late-Diagnosed Allison', 'Overwhelmed Mom of Four')",
    "persona_tagline": "Short tagline",
    "created_at": "2025-11-08T...",
    "source_documents": ["intake_samples", "analytics"],
    "llm_model": "gemini-2.5-real-stories",
    "version": "2.0",
    "attribute_tags": ["adhd", "late_diagnosed", "parent", etc.]
  }},
  "identity_demographics": {{
    "age_range": "late 30s",
    "gender": "Female",
    "pronouns": "She/Her",
    "orientation": null,
    "ethnicity": null,
    "current_life_situation": "Single mom of 4, youngest is 3 months",
    "occupation": "Stay-at-home parent",
    "education": "Some college",
    "locality": "North County",
    "cultural_faith_identities": ["Christian", "Working class"]
  }},
  "developmental_story": {{
    "childhood_environment": "Grew up in a chaotic household with an alcoholic father and depressed mother. Moved frequently due to evictions. Often went without basic necessities.",
    "adolescence": "Became parentified at 13, taking care of younger siblings. Dropped out of high school at 16 to work and support family. Got GED at 19.",
    "early_adulthood": "Had first child at 20. Struggled with undiagnosed ADHD and depression. Series of unstable relationships.",
    "family_structure": "Raised by mother after father left when she was 8. Three younger siblings. Mother struggled with depression and was emotionally unavailable. No extended family support.",
    "formative_adversities": [
      "Poverty and housing instability",
      "Parentification and loss of childhood",
      "Father's abandonment",
      "Mother's depression and emotional neglect",
      "Educational disruption"
    ],
    "educational_journey": "Attended 5 different elementary schools due to moves. Struggled academically but teachers never identified ADHD. Dropped out at 16. Got GED at 19. Started community college twice but couldn't finish due to childcare and money.",
    "pivotal_moments": [
      "Father leaving when she was 8 - learned she couldn't rely on anyone",
      "Becoming primary caregiver for siblings at 13 - lost her childhood",
      "Having her first child at 20 - wanted to break the cycle but didn't have tools",
      "Recent ADHD diagnosis at 38 - finally understanding why life has been so hard"
    ],
    "intergenerational_patterns": "Mother likely also had undiagnosed ADHD and depression. Generational pattern of poverty, instability, and untreated mental health issues."
  }},
  "neurodivergence_mental_health": {{
    "identities": ["ADHD (recently diagnosed)", "Depression", "Anxiety"],
    "diagnosis_journey": "Diagnosed with ADHD at 38 after her oldest child was diagnosed. Finally understood why she's struggled her whole life. Feels angry about all the years of not knowing.",
    "how_it_shows_up": "Can't keep up with daily tasks. House is always chaotic. Forgets appointments. Overwhelmed by kids' big feelings. Shuts down when stressed. Struggles to regulate emotions.",
    "nervous_system_pattern": "Constantly in fight-or-flight. Easily overwhelmed by noise and chaos. Shuts down when overstimulated. Difficulty transitioning between tasks.",
    "strengths_superpowers": [
      "Incredibly resilient",
      "Deeply empathetic",
      "Creative problem-solver when not overwhelmed",
      "Fiercely protective of her kids"
    ],
    "current_coping_strategies": [
      "Scrolling on phone to dissociate",
      "Emotional eating",
      "Withdrawing when overwhelmed",
      "Asking older kids for help (sometimes too much)"
    ]
  }},
  "executive_function_sensory": {{
    "ef_strengths": [
      "Can hyperfocus in crisis situations",
      "Creative solutions to problems"
    ],
    "ef_friction_points": [
      "Task initiation - can't start things",
      "Time blindness - always running late",
      "Working memory - forgets what she was doing",
      "Planning and organization - house is chaotic"
    ],
    "ef_workarounds": [
      "Phone alarms for everything",
      "Older kids help with routines",
      "Keeps essentials in one place (when she remembers)"
    ],
    "sensory_profile": "Overwhelmed by noise and chaos (hard with 4 kids). Needs quiet to think. Sensitive to textures in clothing. Seeks deep pressure (weighted blanket).",
    "food_sensory_details": "Binge eats when stressed - usually carbs and sweets (cookies, chips, ice cream straight from container). Skips meals when overwhelmed. Avoids cooking because it's too many steps. Safe foods for kids: chicken nuggets (frozen, specific brand), mac and cheese (box only), PB&J. She eats whatever is fastest - often cold leftovers standing at the fridge. Texture aversions: slimy foods, anything with bones. Struggles with 'family meals' because kids are picky and she's too exhausted to fight about food."
  }},
  "current_life_context": {{
    "where_they_live": "Rents a small house in North County. Neighborhood is affordable but not great schools. Wants to move but can't afford it. House is cluttered and chaotic.",
    "work_school_situation": "Stay-at-home parent. Wants to work but childcare costs more than she'd make. Feels trapped. Homeschooling oldest because school wasn't working.",
    "commute_daily_rhythms": "No commute. Days are chaos - trying to manage 4 kids' needs, appointments, meals. No routine. Every day feels like survival mode.",
    "local_stressors": [
      "Isolation - no family support nearby",
      "Financial stress - barely making it",
      "Neighborhood safety concerns",
      "Lack of childcare options",
      "Feeling judged by other moms"
    ],
    "safe_spaces": [
      "Her bedroom when kids are asleep",
      "Local park early morning before it's crowded",
      "Online support groups"
    ],
    "support_system": "Minimal. No family nearby. Few friends because she's too overwhelmed to maintain relationships. Oldest daughter (14) helps a lot - maybe too much. Partner works long hours and doesn't understand ADHD. Feels very alone."
  }},
  "goals_barriers": {{
    "why_therapy_now": "Reached a breaking point. Can't keep up with daily life. Scared she's failing her kids. Recent ADHD diagnosis made her realize she needs help. Wants to break the cycle and be a better mom.",
    "what_they_want_to_change": [
      "Learn to regulate emotions",
      "Develop coping skills for overwhelm",
      "Figure out how to manage daily tasks",
      "Stop yelling at kids",
      "Understand herself and her ADHD",
      "Heal from childhood trauma"
    ],
    "whats_in_the_way": [
      "Cost of therapy",
      "Finding childcare for appointments",
      "Shame and guilt about struggling",
      "Perfectionism - feels like she should be able to do this",
      "Exhaustion - barely has energy for basics",
      "Partner doesn't understand why she needs therapy"
    ]
  }},
  "narrative_voice": {{
    "quote_struggle": "I'm struggling with regulating myself, keeping up with daily tasks, managing to stay calm when my kids are having big feelings, and also figuring out what I want to do with my life and learning about who I am.",
    "quote_hope": "I know I need to do better for my kids. I don't want them to go through what I went through. I just need help figuring out how.",
    "quotes_additional": [
      "I feel like I'm drowning and no one sees it.",
      "My kids deserve better than what I'm giving them right now.",
      "I'm so tired of feeling like a failure."
    ],
    "liz_clinical_narrative": "This is a woman who has survived incredible adversity and is now fighting to break generational cycles. She's carrying the weight of untreated ADHD, childhood trauma, and the overwhelming demands of parenting four children with minimal support. Her recent diagnosis is both validating and painful - understanding why life has been so hard, while grieving all the years of struggle without knowing. She's exhausted, overwhelmed, and feeling like she's failing, but she's here seeking help because she's determined to do better for her children. Her resilience is remarkable. With the right support, she can develop the tools to manage her ADHD, process her trauma, and build the life she wants for herself and her family. She deserves compassion, practical strategies, and someone who sees her strength even when she can't.",
    "therapy_preferences": "Needs practical, concrete strategies. No abstract talk therapy. Wants someone who gets ADHD and trauma. Needs flexibility with scheduling and maybe telehealth because childcare is hard. Prefers direct communication - no therapist-speak."
  }},
  "marketing_seo": {{
    "website_copy_snippets": [
      "Overwhelmed mom struggling to keep up? You're not failing - your brain just works differently.",
      "Late ADHD diagnosis? We can help you understand yourself and develop strategies that actually work.",
      "Breaking generational cycles is hard. You don't have to do it alone."
    ],
    "primary_keywords": [
      "ADHD therapy for moms",
      "Overwhelmed parent support",
      "Late ADHD diagnosis",
      "Trauma therapy St. Louis"
    ],
    "local_entities": [
      "North County",
      "St. Louis",
      "Community resources"
    ],
    "content_angles": [
      "How ADHD shows up differently in moms",
      "Breaking generational trauma cycles",
      "Practical strategies for overwhelmed parents"
    ]
  }}
}}

**CRITICAL:**
- Return a SINGLE JSON OBJECT, not an array
- Use REAL intake language
- Create DEEP developmental stories
- Be SPECIFIC about food/sensory
- NO tourist traps
- REAL human struggle and depth
- Each field adds NEW information

Generate the complete JSON object now (NOT wrapped in an array).
"""
    
    return prompt


def main():
    # Setup
    project_root = Path("/home/antons-gs/enlitens-ai")
    config = ProfilePipelineConfig(project_root=project_root)
    
    # Load data
    print("Loading ingestion bundle...")
    bundle = load_ingestion_bundle(config)
    
    print(f"Loaded {len(bundle.transcripts)} intake messages")
    print(f"Loaded {len(bundle.knowledge_assets)} knowledge assets")
    
    # Setup Gemini client
    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyChf4y7bqezULJtipsNXOJvaK3MVW0XvXI")
    gemini_client = GeminiClient(api_key=api_key)
    
    # Build prompt
    print("\nBuilding real story prompt...")
    prompt = build_real_story_prompt(bundle, num_personas=1)
    
    print(f"\nPrompt length: {len(prompt)} characters")
    print("\nGenerating persona with Gemini 2.5 Pro...")
    print("This will take 30-60 seconds...\n")
    
    # Generate
    try:
        result = gemini_client.generate_structured(
            prompt=prompt,
            response_model=ClientProfileV2RealStories,
            temperature=0.8,
        )
        
        # Save raw output
        output_dir = project_root / "enlitens_client_profiles" / "profiles"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_file = output_dir / f"persona_real_story_{timestamp}.json"
        
        with open(raw_file, "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        
        print(f"✅ SUCCESS! Saved to: {raw_file}")
        print(f"\nGenerated persona: {result.meta.persona_name}")
        print(f"Attribute tags: {', '.join(result.meta.attribute_tags)}")
        
        # Print key sections to verify depth
        print("\n" + "="*80)
        print("DEVELOPMENTAL STORY SAMPLE:")
        print("="*80)
        print(f"Childhood: {result.developmental_story.childhood_environment[:200]}...")
        print(f"\nFormative adversities: {result.developmental_story.formative_adversities[:3]}")
        
        print("\n" + "="*80)
        print("FOOD SENSORY SAMPLE:")
        print("="*80)
        print(result.executive_function_sensory.food_sensory_details[:300] + "...")
        
        print("\n" + "="*80)
        print("NARRATIVE VOICE:")
        print("="*80)
        print(f"Struggle: {result.narrative_voice.quote_struggle}")
        print(f"Hope: {result.narrative_voice.quote_hope}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

