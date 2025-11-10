"""Generate client profiles using Qwen2.5-14B (local vLLM)."""

import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path

from enlitens_client_profiles.config import ProfilePipelineConfig
from enlitens_client_profiles.data_ingestion import load_ingestion_bundle
from enlitens_client_profiles.profile_builder_gemini import GeminiProfileBuilder
from enlitens_client_profiles.schema_simplified import ClientProfileDocumentSimplified
from src.synthesis.ollama_client import OllamaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Generate 3 test client profiles using Qwen2.5-14B."""
    logger.info("Loading ingestion bundle (intakes, transcripts, health data, KB, analytics)...")

    # Create config (paths auto-set from project_root)
    config = ProfilePipelineConfig(
        project_root=Path("/home/antons-gs/enlitens-ai"),
        ga_property_id="430717984",
        gsc_site_url="https://enlitens.com/",
        analytics_lookback_days=120,
    )

    # Load data
    bundle = load_ingestion_bundle(config)

    logger.info(
        f"Loaded: {len(bundle.intake_sentence_pool)} intake sentences, "
        f"{len(bundle.transcripts)} transcripts, "
        f"{len(bundle.locality_counts)} localities, "
        f"{len(bundle.site_documents)} site pages"
    )

    # Prepare data for builder
    transcript_pool = [f"{t.speaker or 'Client'}: {t.raw_text}" for t in bundle.transcripts[:100]]
    knowledge_dict = {asset.name or f"asset_{i}": asset.content for i, asset in enumerate(bundle.knowledge_assets[:10])}
    health_dict = {"health_report": bundle.health_report_markdown[:3000]}
    analytics_dict = asdict(bundle.analytics) if bundle.analytics else {}

    # Initialize Qwen client (local vLLM)
    llm_client = OllamaClient(
        base_url="http://localhost:8010/v1",
        default_model="/home/antons-gs/enlitens-ai/models/qwen2.5-14b-instruct-awq",
    )
    logger.info("Initialized Qwen2.5-14B client (local vLLM)")

    # Use the same prompt builder as Gemini (it's model-agnostic)
    builder = GeminiProfileBuilder()

    # Output directory
    output_dir = Path("/home/antons-gs/enlitens-ai/enlitens_client_profiles/profiles")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate 3 personas
    for i in range(1, 4):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Generating Persona {i}/3...")
        logger.info(f"{'=' * 60}")

        # Build prompt
        import random

        intake_samples = random.sample(bundle.intake_sentence_pool, min(4, len(bundle.intake_sentence_pool)))
        transcript_samples = random.sample(transcript_pool, min(4, len(transcript_pool)))
        founder_samples = random.sample(bundle.founder_voice_snippets, min(3, len(bundle.founder_voice_snippets)))

        # Build prompt components
        locality_context = builder._build_locality_context(bundle.locality_counts)
        analytics_context = builder._build_analytics_context(analytics_dict)
        site_context = builder._build_site_context(bundle.site_documents, max_chars=2000)
        brand_context = builder._build_brand_context(bundle.brand_mentions, max_chars=1000)
        kb_context = builder._build_kb_context(knowledge_dict, max_chars=3000)
        health_context = builder._build_health_context(health_dict, max_chars=1500)

        system_instruction = builder._build_system_instruction()
        user_prompt = builder._build_user_prompt(
            intake_samples=intake_samples,
            transcript_samples=transcript_samples,
            founder_samples=founder_samples,
            locality_context=locality_context,
            analytics_context=analytics_context,
            site_context=site_context,
            brand_context=brand_context,
            kb_context=kb_context,
            health_context=health_context,
        )

        full_prompt = f"{system_instruction}\n\n{user_prompt}"

        logger.info(f"Generating persona with Qwen2.5-14B (prompt: {len(full_prompt)} chars)")

        # Generate with structured response (uses incremental fallback if needed)
        profile = asyncio.run(
            llm_client.generate_structured_response(
                prompt=full_prompt, response_model=ClientProfileDocumentSimplified, temperature=0.8, max_retries=3
            )
        )

        if profile:
            # Save to file
            output_file = output_dir / f"persona_{i}_{profile.meta.profile_id}.json"
            with output_file.open("w") as f:
                json.dump(profile.model_dump(), f, indent=2, default=str)

            logger.info(f"✓ Saved persona {i}: {profile.meta.persona_name} → {output_file}")
            logger.info(f"  Locality: {profile.demographics.locality}")
            logger.info(f"  Neuro identities: {', '.join(profile.neurodivergence_clinical.neuro_identities[:3])}")
            logger.info(f"  Age: {profile.demographics.age_range}")
        else:
            logger.error(f"✗ Failed to generate persona {i}")

    logger.info("\n" + "=" * 60)
    logger.info("Persona generation complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

