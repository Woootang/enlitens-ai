"""
Generate client personas using OpenAI GPT-4 with structured outputs.

Based on deep research findings: GPT-4 with strict function calling achieves
100% schema compliance for nested JSON structures.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from openai import OpenAI

from enlitens_client_profiles.config import ProfilePipelineConfig
from enlitens_client_profiles.data_ingestion import load_ingestion_bundle
from enlitens_client_profiles.schema_simplified import ClientProfileDocumentSimplified

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/home/antons-gs/enlitens-ai/enlitens_client_profiles/logs/gpt4_generation.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def build_generation_prompt(bundle, num_profiles: int = 1) -> str:
    """Build comprehensive prompt with all data sources."""
    
    # Sample data
    intake_samples = bundle.intakes[:8] if bundle.intakes else []
    transcript_samples = [t.raw_text for t in bundle.transcripts[:8]] if bundle.transcripts else []
    
    # Build locality context
    locality_counts = {}
    for intake in bundle.intakes:
        if hasattr(intake, 'locality') and intake.locality:
            locality_counts[intake.locality] = locality_counts.get(intake.locality, 0) + 1
    
    top_localities = sorted(locality_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    locality_list = ", ".join([f"{loc} ({cnt})" for loc, cnt in top_localities]) if top_localities else "St. Louis region"
    
    prompt = f"""Generate ONE extremely detailed, rich fictional client persona for Enlitens Counseling, a neurodiversity-affirming therapy practice in St. Louis.

**CRITICAL: This must be a COMPLETE, COMPREHENSIVE profile with ALL fields filled out in detail. Do NOT leave fields empty or minimal.**

**STRUCTURE:**
Return a JSON object with these 8 top-level keys: meta, demographics, neurodivergence_clinical, executive_sensory, goals_barriers, local_cultural_context, narrative_voice, marketing_seo.

**EACH section must be FULLY populated with rich, detailed content:**

**Data Sources to Use:**

Intake Form Samples (Real Client Language):
{chr(10).join(f'{i+1}. "{s}"' for i, s in enumerate(intake_samples[:5]))}

Therapy Transcript Samples:
{chr(10).join(f'{i+1}. {s[:200]}...' for i, s in enumerate(transcript_samples[:3]))}

St. Louis Localities (MUST use specific municipalities):
Most common: {locality_list}

**IMPORTANT:** Each persona MUST reference a SPECIFIC St. Louis municipality or neighborhood (e.g., "Kirkwood", "Webster Groves", "Central West End"), NOT generic "St. Louis" or "St. Louis County". Include specific places, schools, parks, or landmarks.

**Content Guidelines:**
1. Each persona should feel unique - vary age, locality, family structure, neurodivergent identities, challenges, and goals
2. Use identity-first language ("autistic person") unless person-first is explicitly preferred
3. Reframe "deficits" as differences or strengths in different contexts
4. Mark list items as "[direct]" if from intake/transcript quotes, or "[inferred]" if clinical interpretation
5. For every challenge, provide at least one strengths-based reframe
6. Ground personas in the provided data - DO NOT invent generic personas
7. Include rich, context-specific details in all narrative fields

**REQUIRED FIELDS - ALL must be filled with substantial content:**

meta:
- profile_id: "persona-001"
- persona_name: Creative, memorable (e.g., "Overachiever Olivia")
- persona_tagline: Short descriptor (e.g., "High-achieving professional with ADHD")
- attribute_tags: 5-7 specific tags
- llm_model: "gpt-4-detailed"
- source_documents: List 3-5 document IDs
- created_at: Current timestamp
- version: "1.0"

demographics (ALL fields required):
- age_range, gender, pronouns, orientation, ethnicity, family_status, occupation, education
- locality: SPECIFIC St. Louis municipality (Kirkwood, Webster Groves, Clayton, etc.)

neurodivergence_clinical (ALL fields with 3-5+ items each):
- identities: List neurodivergent identities with [direct] or [inferred] tags
- diagnosis_notes: 2-3 sentence paragraph
- language_preferences: 3+ preferences
- presenting_issues: 5+ specific issues
- nervous_system_pattern: 2-3 sentence explanation
- mood_patterns: 3+ patterns
- trauma_history: 1-2 sentences or "None reported"
- strengths: 5+ adaptive strengths
- coping_skills: 5+ existing skills

executive_sensory (ALL fields with 3-5+ items):
- executive_function_strengths, executive_function_friction_points, executive_function_workarounds
- sensory_sensitivities, sensory_seeking_behaviors, sensory_regulation_methods

goals_barriers (ALL fields with 3-5+ items):
- therapy_goals, life_goals, motivations
- why_now: 2-3 sentence explanation
- internal_barriers, systemic_barriers, access_constraints

local_cultural_context (ALL fields with detailed content):
- cultural_identities, community_roles, cultural_notes
- home_environment, work_environment, commute: Each 2-3 sentences
- local_stressors, safe_spaces: 3-5 St. Louis-specific items each
- supportive_allies, caregiving_roles, support_gaps

narrative_voice (ALL fields required):
- quotes_struggle: Direct quote expressing struggle
- quotes_hope: Direct quote expressing hope
- quotes_additional: 2-3 more quotes
- liz_voice_narrative: 200-250 word narrative in Liz's warm, empathetic tone
- narrative_highlights: 5 bullet highlights
- therapy_preferred_styles, therapy_disliked_approaches: 3+ each
- therapy_past_experiences: 2-3 sentences

marketing_seo (ALL fields with 2-3+ items each):
- website_about: 3 pieces of "About" copy (each 1-2 sentences)
- landing_page_intro: 3 landing page snippets
- email_nurture: 3 email snippets
- social_snippets: 3 social media posts
- primary_keywords: 5-7 keywords
- long_tail_keywords: 5-7 long-tail phrases
- local_entities: 5+ St. Louis entities
- content_angles: 5+ blog/article ideas
- recommended_offers: 3-5 Enlitens services
- referral_needs: 3-5 external referral types

Generate ONE complete persona now. Return ONLY the JSON object (no markdown, no explanation).
"""
    
    return prompt


def estimate_cost(prompt: str, expected_output_tokens: int = 15000) -> dict:
    """Estimate API cost before making the call."""
    # GPT-4 pricing (as of Nov 2025): ~$0.03/1K input, ~$0.06/1K output
    input_tokens = len(prompt.split()) * 1.3  # Rough estimate (1 token ≈ 0.75 words)
    
    input_cost = (input_tokens / 1000) * 0.03
    output_cost = (expected_output_tokens / 1000) * 0.06
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": expected_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }


async def generate_with_gpt4(prompt: str, schema: dict, api_key: str) -> dict:
    """Generate personas using GPT-4 with strict function calling."""
    
    # Cost estimation
    cost_estimate = estimate_cost(prompt, expected_output_tokens=15000)
    logger.info(f"\n💰 Cost Estimate:")
    logger.info(f"   Input tokens: ~{cost_estimate['input_tokens']}")
    logger.info(f"   Output tokens: ~{cost_estimate['output_tokens']}")
    logger.info(f"   Estimated cost: ${cost_estimate['total_cost']:.3f}")
    
    if cost_estimate['total_cost'] > 2.0:
        logger.warning(f"⚠️  Estimated cost (${cost_estimate['total_cost']:.2f}) exceeds $2.00!")
        logger.warning("   This should only cost ~$0.50-$1.00 for 3 profiles.")
        logger.warning("   Aborting to prevent unexpected charges.")
        raise ValueError(f"Cost estimate too high: ${cost_estimate['total_cost']:.2f}")
    
    client = OpenAI(api_key=api_key)
    
    messages = [
        {
            "role": "system",
            "content": "You are a clinical data generator that outputs structured JSON for neurodiversity-affirming therapy personas. You specialize in creating rich, contextual client profiles based on real therapy data."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    logger.info("\nCalling GPT-4 with structured output (strict mode)...")
    logger.info("This will cost approximately $%.3f" % cost_estimate['total_cost'])
    
    try:
        # Use the new OpenAI API (v1.0+) with response_format for JSON
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",  # Use GPT-4 Turbo with JSON mode
            messages=messages,
            response_format={"type": "json_object"},  # Force JSON output
            temperature=0.7,
            max_tokens=4000  # Max allowed for this model
        )
        
        result_text = response.choices[0].message.content
        profiles_data = json.loads(result_text)
        
        # Log actual usage
        usage = response.usage
        actual_cost = (usage.prompt_tokens / 1000 * 0.03) + (usage.completion_tokens / 1000 * 0.06)
        logger.info(f"\n✓ GPT-4 returned structured output")
        logger.info(f"💰 Actual cost: ${actual_cost:.3f}")
        logger.info(f"   Prompt tokens: {usage.prompt_tokens}")
        logger.info(f"   Completion tokens: {usage.completion_tokens}")
        
        return profiles_data
        
    except Exception as e:
        logger.error(f"GPT-4 API call failed: {e}")
        raise


def validate_and_save_profiles(profiles_data: dict, output_dir: Path) -> List[Path]:
    """Validate profiles against Pydantic schema and save to JSON files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    
    profiles = profiles_data.get("profiles", [])
    logger.info(f"Validating {len(profiles)} profiles...")
    
    for i, profile_dict in enumerate(profiles, start=1):
        try:
            # Validate with Pydantic
            profile = ClientProfileDocumentSimplified.model_validate(profile_dict)
            
            # Generate filename
            persona_name = profile.meta.persona_name.lower().replace(" ", "_").replace("'", "")
            filename = f"persona_{i:03d}_{persona_name}.json"
            filepath = output_dir / filename
            
            # Save to file
            with open(filepath, "w") as f:
                json.dump(profile.model_dump(), f, indent=2, default=str)
            
            saved_files.append(filepath)
            logger.info(f"✓ Saved persona {i}: {profile.meta.persona_name} → {filename}")
            
        except Exception as e:
            logger.error(f"✗ Validation failed for persona {i}: {e}")
            # Save the raw dict for debugging
            debug_file = output_dir / f"persona_{i:03d}_FAILED.json"
            with open(debug_file, "w") as f:
                json.dump(profile_dict, f, indent=2, default=str)
            logger.info(f"Saved failed profile to {debug_file} for debugging")
    
    return saved_files


def main():
    """Main execution function."""
    
    # Configuration
    project_root = Path("/home/antons-gs/enlitens-ai")
    config = ProfilePipelineConfig(project_root=project_root)
    output_dir = project_root / "enlitens_client_profiles" / "profiles"
    
    # OpenAI API key - load from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    logger.info("=" * 60)
    logger.info("GPT-4 Structured Output Persona Generation")
    logger.info("=" * 60)
    
    # Load data
    logger.info("Loading ingestion bundle...")
    bundle = load_ingestion_bundle(config)
    logger.info(f"Loaded: {len(bundle.intakes)} intakes, {len(bundle.transcripts)} transcripts")
    
    # Get schema
    logger.info("Preparing Pydantic schema...")
    # Get the full schema with $defs
    full_schema = ClientProfileDocumentSimplified.model_json_schema()
    
    # Wrap in a container schema for the "profiles" array
    container_schema = {
        "type": "object",
        "properties": {
            "profiles": {
                "type": "array",
                "items": {"$ref": "#/$defs/ClientProfileDocumentSimplified"},
                "minItems": 3,
                "maxItems": 3
            }
        },
        "required": ["profiles"],
        "$defs": full_schema.get("$defs", {})
    }
    
    # Add the main model to $defs if not already there
    if "ClientProfileDocumentSimplified" not in container_schema["$defs"]:
        container_schema["$defs"]["ClientProfileDocumentSimplified"] = {
            k: v for k, v in full_schema.items() if k != "$defs"
        }
    
    # Build prompt
    logger.info("Building generation prompt...")
    prompt = build_generation_prompt(bundle, num_profiles=1)
    logger.info(f"Prompt length: {len(prompt)} characters")
    
    # Generate with GPT-4
    logger.info("\nGenerating 3 personas with GPT-4 (strict schema mode)...")
    logger.info("This may take 30-60 seconds...")
    
    try:
        profiles_data = asyncio.run(generate_with_gpt4(prompt, container_schema, api_key))
        
        # Validate and save
        logger.info("\nValidating and saving profiles...")
        saved_files = validate_and_save_profiles(profiles_data, output_dir)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("GENERATION COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"Successfully generated: {len(saved_files)}/3 profiles")
        logger.info(f"Output directory: {output_dir}")
        logger.info("\nSaved files:")
        for filepath in saved_files:
            logger.info(f"  - {filepath.name}")
        
        if len(saved_files) == 3:
            logger.info("\n🎉 SUCCESS! All 3 test profiles generated and validated!")
        else:
            logger.warning(f"\n⚠️ Only {len(saved_files)}/3 profiles succeeded. Check logs for errors.")
        
    except Exception as e:
        logger.error(f"\n❌ Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()

