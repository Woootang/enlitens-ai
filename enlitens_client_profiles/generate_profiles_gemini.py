"""Generate client profiles using Gemini 2.5 Pro."""

import json
import logging
from dataclasses import asdict
from pathlib import Path

from enlitens_client_profiles.config import ProfilePipelineConfig
from enlitens_client_profiles.data_ingestion import load_ingestion_bundle
from enlitens_client_profiles.profile_builder_gemini import GeminiProfileBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Generate 3 test client profiles using Gemini."""
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
    transcript_pool = [f"{t.speaker or 'Client'}: {t.raw_text}" for t in bundle.transcripts[:100]]  # Convert to strings
    knowledge_dict = {asset.name or f"asset_{i}": asset.content for i, asset in enumerate(bundle.knowledge_assets[:10])}
    health_dict = {"health_report": bundle.health_report_markdown[:3000]}
    analytics_dict = asdict(bundle.analytics) if bundle.analytics else {}

    # Initialize Gemini builder
    builder = GeminiProfileBuilder()

    # Output directory
    output_dir = Path("/home/antons-gs/enlitens-ai/enlitens_client_profiles/profiles")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate 3 personas
    for i in range(1, 4):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Generating Persona {i}/3...")
        logger.info(f"{'=' * 60}")

        profile = builder.generate_profile(
            intake_pool=bundle.intake_sentence_pool,
            transcript_pool=transcript_pool,
            knowledge_assets=knowledge_dict,
            health_insights=health_dict,
            locality_counts=bundle.locality_counts,
            analytics_summary=analytics_dict,
            site_documents=bundle.site_documents,
            brand_mentions=bundle.brand_mentions,
            founder_voice_snippets=bundle.founder_voice_snippets,
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

