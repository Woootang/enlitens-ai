"""
Generate ONE complete persona using Gemini 2.5 Pro.
Focus: ALL fields filled, no hallucinations, use REAL data only.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import google.generativeai as genai

from enlitens_client_profiles.config import ProfilePipelineConfig
from enlitens_client_profiles.data_ingestion import load_ingestion_bundle
from enlitens_client_profiles.schema_simplified import ClientProfileDocumentSimplified

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_strict_prompt(bundle) -> str:
    """Build prompt that forces complete output and prevents hallucination."""
    
    # Get REAL data
    intake_samples = [str(i) for i in bundle.intakes[:10]]
    transcript_samples = [t.raw_text[:300] for t in bundle.transcripts[:10]]
    
    # Get REAL localities
    localities = set()
    for intake in bundle.intakes:
        if hasattr(intake, 'locality') and intake.locality:
            localities.add(intake.locality)
    real_localities = list(localities)[:15]
    
    # Real St. Louis places/institutions
    stl_colleges = ["Washington University", "Saint Louis University", "University of Missouri-St. Louis (UMSL)", 
                    "Webster University", "Fontbonne University", "Maryville University", "Lindenwood University"]
    
    stl_neighborhoods = ["Central West End", "The Hill", "Soulard", "Tower Grove", "Dogtown", "The Grove", 
                        "Cherokee Street", "South Grand", "Delmar Loop", "Lafayette Square"]
    
    stl_landmarks = ["Forest Park", "The Arch", "City Museum", "Missouri Botanical Garden (Shaw's Garden)", 
                    "The Magic House", "Grant's Farm", "Laumeier Sculpture Park", "Cahokia Mounds"]
    
    stl_restaurants = ["Pappy's Smokehouse", "Ted Drewes", "Imo's Pizza", "Sugarfire", "The Fountain on Locust",
                      "Kaldi's Coffee", "Blueprint Coffee", "Meshuggah Cafe", "Cafe Osage", "Seedz Cafe"]
    
    prompt = f"""You are generating ONE complete client persona for a therapy practice.

**CRITICAL RULES:**
1. Fill EVERY SINGLE FIELD with detailed content - no empty fields allowed
2. ONLY use information from the data provided below - DO NOT make up schools, places, or details
3. If a locality is mentioned, it must be from this list: {real_localities}
4. Mark items as [direct] if from the data, [inferred] if you're inferring
5. Generate 3-5+ items for every list field
6. Generate 2-3 sentences for every text field

**REAL CLIENT DATA TO USE:**

Intake samples (use these exact phrases where relevant):
{chr(10).join(f'{i+1}. {s[:200]}' for i, s in enumerate(intake_samples[:5]))}

Transcript samples (use these themes):
{chr(10).join(f'{i+1}. {s[:200]}' for i, s in enumerate(transcript_samples[:3]))}

Valid St. Louis localities (ONLY use these): {', '.join(real_localities)}

**REAL ST. LOUIS PLACES (use these for hyper-local details):**
- Colleges/Universities: {', '.join(stl_colleges)}
- Neighborhoods: {', '.join(stl_neighborhoods)}
- Landmarks/Parks: {', '.join(stl_landmarks)}
- Restaurants/Cafes: {', '.join(stl_restaurants)}

**HYPER-LOCAL REQUIREMENTS:**
- Choose ONE specific municipality from the localities list as their home base
- Mention 2-3 specific places they actually go (restaurants, parks, cafes, gyms)
- If student: specify which college from the list above
- If working: specify neighborhood where they work
- Commute: mention specific highways (I-64, I-44, I-70) or neighborhoods they pass through
- Safe spaces: use actual St. Louis places from the lists above

**GENERATE THIS EXACT STRUCTURE - ALL FIELDS REQUIRED:**

{{
  "meta": {{
    "profile_id": "persona-complete-001",
    "persona_name": "Creative memorable name",
    "persona_tagline": "One sentence descriptor",
    "created_at": "{datetime.utcnow().isoformat()}Z",
    "source_documents": ["intake_001", "transcript_042", "analytics_data"],
    "llm_model": "gemini-2.5-complete",
    "version": "1.0",
    "attribute_tags": ["tag1", "tag2", "tag3", "tag4", "tag5"]
  }},
  "demographics": {{
    "age_range": "specific range like 'late 20s' or '15-17'",
    "gender": "gender identity",
    "pronouns": "preferred pronouns",
    "orientation": "sexual orientation or null",
    "ethnicity": "ethnic/racial identity or null",
    "family_status": "detailed family structure",
    "occupation": "current occupation or school",
    "education": "education level",
    "locality": "MUST be from the valid localities list above"
  }},
  "neurodivergence_clinical": {{
    "identities": ["[direct] identity1", "[inferred] identity2", "identity3"],
    "diagnosis_notes": "2-3 sentences about diagnosis journey",
    "language_preferences": ["preference1", "preference2", "preference3"],
    "presenting_issues": ["issue1", "issue2", "issue3", "issue4", "issue5"],
    "nervous_system_pattern": "2-3 sentences explaining nervous system tendencies",
    "mood_patterns": ["pattern1", "pattern2", "pattern3"],
    "trauma_history": "1-2 sentences or 'No significant trauma reported'",
    "strengths": ["strength1", "strength2", "strength3", "strength4", "strength5"],
    "coping_skills": ["skill1", "skill2", "skill3", "skill4", "skill5"]
  }},
  "executive_sensory": {{
    "executive_function_strengths": ["strength1", "strength2", "strength3"],
    "executive_function_friction_points": ["friction1", "friction2", "friction3"],
    "executive_function_workarounds": ["workaround1", "workaround2", "workaround3"],
    "sensory_sensitivities": ["MUST include food textures/tastes they avoid", "sound sensitivity", "light sensitivity", "touch/fabric sensitivity", "smell sensitivity"],
    "sensory_seeking_behaviors": ["behavior1", "behavior2", "behavior3"],
    "sensory_regulation_methods": ["method1", "method2", "method3"]
  }},
  "food_sensory_profile": {{
    "safe_foods": ["List 5-7 specific foods they regularly eat and feel comfortable with"],
    "texture_aversions": ["Specific textures they avoid (mushy, slimy, crunchy, etc.)"],
    "taste_preferences": ["Sweet, salty, bland, spicy preferences"],
    "food_related_challenges": ["Describe eating challenges - limited variety, social eating anxiety, etc."],
    "favorite_stl_restaurants": ["List 2-3 restaurants from the STL list above where they feel comfortable eating"]
  }},
  "goals_barriers": {{
    "therapy_goals": ["goal1", "goal2", "goal3", "goal4"],
    "life_goals": ["goal1", "goal2", "goal3"],
    "motivations": ["motivation1", "motivation2", "motivation3"],
    "why_now": "2-3 sentences about urgency/catalyst",
    "internal_barriers": ["barrier1", "barrier2", "barrier3"],
    "systemic_barriers": ["barrier1", "barrier2", "barrier3"],
    "access_constraints": ["constraint1", "constraint2", "constraint3"]
  }},
  "local_cultural_context": {{
    "cultural_identities": ["identity1", "identity2", "identity3"],
    "community_roles": ["role1", "role2"],
    "cultural_notes": "2-3 sentences about cultural/faith influences or null",
    "home_environment": "2-3 sentences describing home IN THEIR SPECIFIC MUNICIPALITY (not just 'St. Louis')",
    "work_environment": "2-3 sentences describing work/school - MUST mention specific college from list or specific neighborhood where they work",
    "commute": "MUST mention specific route: which highways (I-64/I-44/I-70), which neighborhoods they pass through, how long",
    "local_stressors": ["MUST include St. Louis-specific stressors like traffic on specific highways, specific neighborhood issues, etc."],
    "safe_spaces": ["MUST use actual places from the STL lists above - specific parks, cafes, restaurants, landmarks"],
    "places_they_frequent": ["List 3-5 SPECIFIC St. Louis places they go regularly - use names from the lists above"],
    "supportive_allies": ["ally1", "ally2", "ally3"],
    "caregiving_roles": ["role1", "role2"] or [],
    "support_gaps": ["gap1", "gap2", "gap3"]
  }},
  "narrative_voice": {{
    "quotes_struggle": "Direct quote expressing struggle",
    "quotes_hope": "Direct quote expressing hope",
    "quotes_additional": ["quote1", "quote2", "quote3"],
    "liz_voice_narrative": "200-250 word narrative in warm, empathetic, strengths-based tone about this person's journey",
    "narrative_highlights": ["highlight1", "highlight2", "highlight3", "highlight4", "highlight5"],
    "therapy_preferred_styles": ["style1", "style2", "style3"],
    "therapy_disliked_approaches": ["approach1", "approach2", "approach3"],
    "therapy_past_experiences": "2-3 sentences about past therapy or null"
  }},
  "marketing_seo": {{
    "website_about": ["About copy 1 (1-2 sentences)", "About copy 2", "About copy 3"],
    "landing_page_intro": ["Intro snippet 1", "Intro snippet 2", "Intro snippet 3"],
    "email_nurture": ["Email snippet 1", "Email snippet 2", "Email snippet 3"],
    "social_snippets": ["Social post 1", "Social post 2", "Social post 3"],
    "primary_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "long_tail_keywords": ["long tail 1", "long tail 2", "long tail 3", "long tail 4", "long tail 5"],
    "local_entities": ["entity1", "entity2", "entity3", "entity4", "entity5"],
    "content_angles": ["angle1", "angle2", "angle3", "angle4", "angle5"],
    "recommended_offers": ["offer1", "offer2", "offer3"],
    "referral_needs": ["need1", "need2", "need3"]
  }}
}}

Generate the complete JSON now. EVERY field must be filled. Use ONLY the real data provided above.
"""
    
    return prompt


def main():
    logger.info("=" * 60)
    logger.info("Gemini 2.5 Pro - ONE Complete Persona Generation")
    logger.info("=" * 60)
    
    # Setup
    project_root = Path("/home/antons-gs/enlitens-ai")
    config = ProfilePipelineConfig(project_root=project_root)
    output_dir = project_root / "enlitens_client_profiles" / "profiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Gemini API key
    api_key = "AIzaSyChf4y7bqezULJtipsNXOJvaK3MVW0XvXI"
    genai.configure(api_key=api_key)
    
    # Load data
    logger.info("Loading real client data...")
    bundle = load_ingestion_bundle(config)
    logger.info(f"Loaded: {len(bundle.intakes)} intakes, {len(bundle.transcripts)} transcripts")
    
    # Build prompt
    logger.info("Building strict anti-hallucination prompt...")
    prompt = build_strict_prompt(bundle)
    logger.info(f"Prompt length: {len(prompt)} characters")
    
    # Generate with Gemini
    logger.info("\nCalling Gemini 2.5 Pro...")
    logger.info("Estimated cost: ~$0.05-$0.10")
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=8000,
                response_mime_type="application/json"
            )
        )
        
        result_text = response.text
        logger.info(f"\n✓ Gemini returned {len(result_text)} characters")
        
        # Parse and validate
        profile_data = json.loads(result_text)
        
        # Check completeness
        required_sections = ['meta', 'demographics', 'neurodivergence_clinical', 'executive_sensory', 
                           'goals_barriers', 'local_cultural_context', 'narrative_voice', 'marketing_seo']
        
        missing = [s for s in required_sections if s not in profile_data]
        if missing:
            logger.error(f"Missing sections: {missing}")
        else:
            logger.info("✓ All 8 sections present")
        
        # Count fields
        total_fields = sum(len(v) if isinstance(v, dict) else 1 for v in profile_data.values())
        logger.info(f"✓ Generated {total_fields} total fields")
        
        # Try Pydantic validation
        try:
            validated = ClientProfileDocumentSimplified.model_validate(profile_data)
            logger.info("✓ Pydantic validation PASSED!")
            
            # Save
            filename = f"persona_gemini_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(validated.model_dump(), f, indent=2, default=str)
            
            logger.info(f"\n{'=' * 60}")
            logger.info(f"✅ SUCCESS! Complete persona saved to:")
            logger.info(f"   {filepath}")
            logger.info(f"{'=' * 60}")
            
        except Exception as e:
            logger.warning(f"Pydantic validation failed: {e}")
            logger.info("Saving raw output for inspection...")
            
            filename = f"persona_gemini_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(profile_data, f, indent=2, default=str)
            
            logger.info(f"Saved to: {filepath}")
            logger.info("Check the file to see what fields are missing or incorrect")
        
    except Exception as e:
        logger.error(f"❌ Gemini call failed: {e}")
        raise


if __name__ == "__main__":
    main()

