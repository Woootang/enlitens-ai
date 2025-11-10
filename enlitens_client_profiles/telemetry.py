"""Telemetry hooks for monitoring client profile pipeline progress."""

from __future__ import annotations

import json
import time
from typing import Any, Dict

import httpx

from .config import ProfilePipelineConfig
from .schema import ClientProfileDocument


class ClientProfileTelemetry:
    """Send lightweight events to the monitoring dashboard and local logs."""

    def __init__(self, config: ProfilePipelineConfig) -> None:
        self.config = config
        self.monitor_url = config.resolve_monitor_url()

    def log_event(self, event: str, payload: Dict[str, Any]) -> None:
        enriched = {
            "event": event,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **payload,
        }
        log_path = self.config.log_dir / "profile_pipeline.log"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched) + "\n")

        if self.monitor_url:
            try:
                with httpx.Client(timeout=5.0) as client:
                    client.post(self.monitor_url, json=enriched)
            except Exception:
                # Fallback to local log only
                pass

    def log_profile_created(self, document: ClientProfileDocument) -> None:
        self.log_event(
            "profile_created",
            {
                "profile_id": document.meta.profile_id,
                "persona_name": document.meta.persona_name,
                "locality": document.demographics.locality,
                "age_range": document.demographics.age_range,
                "identities": document.neurodivergence_profile.identities,
                "attribute_tags": document.meta.attribute_tags,
            },
        )

