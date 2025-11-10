#!/usr/bin/env python3
"""CLI entrypoint for generating enriched Enlitens client profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ProfilePipelineConfig
from .profile_pipeline import PipelineResult, run_profile_pipeline
from .telemetry import ClientProfileTelemetry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate enriched Enlitens client personas.")
    parser.add_argument("--count", type=int, default=5, help="Number of new profiles to generate")
    parser.add_argument("--model", type=str, help="Override LLM model for generation")
    parser.add_argument(
        "--monitor-url",
        type=str,
        help="Optional monitoring endpoint for dashboard updates (defaults to ENLITENS_MONITOR_URL env)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable cache reuse and force regeneration")
    parser.add_argument("--allow-duplicates", action="store_true", help="Do not halt on similarity >= 0.41")
    parser.add_argument("--config-dump", action="store_true", help="Print resolved configuration before running")
    parser.add_argument("--google-credentials", type=Path, help="Path to Google service account JSON for GA/GSC pulls")
    parser.add_argument("--ga-property", type=str, help="Google Analytics 4 property id (numbers only)")
    parser.add_argument("--gsc-site", type=str, help="Google Search Console property URL (e.g. https://example.com/)")
    parser.add_argument("--analytics-lookback", type=int, help="Days of analytics history to consider (default 90)")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = ProfilePipelineConfig()
    if args.model:
        config.llm_model = args.model
    if args.monitor_url:
        config.monitor_url = args.monitor_url
    if args.google_credentials:
        config.google_credentials_path = args.google_credentials
    if args.ga_property:
        config.ga_property_id = args.ga_property
    if args.gsc_site:
        config.gsc_site_url = args.gsc_site
    if args.analytics_lookback:
        config.analytics_lookback_days = max(7, args.analytics_lookback)
    if args.no_cache:
        config.reuse_existing = False

    if args.config_dump:
        from dataclasses import asdict

        serialisable = {}
        for key, value in asdict(config).items():
            if isinstance(value, Path):
                serialisable[key] = str(value)
            else:
                serialisable[key] = value
        print(json.dumps(serialisable, indent=2))

    telemetry = ClientProfileTelemetry(config)
    result: PipelineResult = run_profile_pipeline(
        config,
        desired_profiles=args.count,
        telemetry=telemetry,
        allow_duplicates=args.allow_duplicates,
    )
    print(f"Generated {len(result.generated)} profiles → {result.output_dir}")
    print(f"Manifest updated → {result.manifest_path}")


if __name__ == "__main__":
    main()

