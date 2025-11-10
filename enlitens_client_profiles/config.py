"""Configuration helpers for the client profile generation pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _default_base_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_project_root() -> Path:
    return _default_base_dir().parent


@dataclass(slots=True)
class ProfilePipelineConfig:
    """Runtime configuration for client profile generation."""

    # Core paths
    project_root: Path = field(default_factory=_default_project_root)
    data_root: Path = field(init=False)
    intakes_path: Path = field(init=False)
    transcripts_path: Path = field(init=False)
    health_report_path: Path = field(init=False)
    knowledge_base_dir: Path = field(init=False)
    enlitens_site_root: str = "https://www.enlitens.com/"

    # Output management
    output_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    site_cache_dir: Path = field(init=False)
    log_dir: Path = field(init=False)
    monitor_url: Optional[str] = None

    # Generation controls
    llm_model: Optional[str] = None
    llm_temperature: float = 0.3
    llm_top_p: float = 0.92
    batch_size: int = 5
    max_profiles: int = 500
    profile_depth_tokens: int = 4096
    reuse_existing: bool = True
    enforce_citation_tags: bool = True
    crawl_site: bool = True
    site_cache_limit: int = 200

    # Analytics + research
    google_credentials_path: Optional[Path] = None
    ga_property_id: Optional[str] = None
    gsc_site_url: Optional[str] = None
    analytics_lookback_days: int = 90

    # Brand intelligence + knowledge graph
    brand_snapshot_path: Path = field(init=False)
    brave_api_key: Optional[str] = None
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None
    neo4j_database: Optional[str] = None

    # Geographic amplification
    stl_counties: List[str] = field(default_factory=lambda: [
        "City of St. Louis",
        "St. Louis County",
        "St. Charles County",
        "Jefferson County",
        "Franklin County",
        "Madison County (IL)",
        "St. Clair County (IL)",
        "Monroe County (IL)",
        "Washington County",
    ])

    def __post_init__(self) -> None:
        kb_dir = self.project_root / "enlitens_knowledge_base"
        self.data_root = kb_dir
        self.intakes_path = kb_dir / "intakes.txt"
        self.transcripts_path = kb_dir / "transcripts.txt"
        self.health_report_path = kb_dir / "st_louis_health_report.pdf"
        self.knowledge_base_dir = kb_dir

        base_output = self.project_root / "enlitens_client_profiles"
        self.output_dir = base_output / "profiles"
        self.cache_dir = base_output / "cache"
        self.site_cache_dir = base_output / "site_cache"
        self.log_dir = base_output / "logs"
        self.brand_snapshot_path = self.cache_dir / "brand_snapshot.json"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.site_cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if not self.google_credentials_path:
            env_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if env_path:
                candidate = Path(env_path).expanduser()
                if candidate.exists():
                    self.google_credentials_path = candidate
        if not self.google_credentials_path:
            default_credentials = self.project_root / "enlitens_client_profiles" / "credentials" / "service_account.json"
            if default_credentials.exists():
                self.google_credentials_path = default_credentials

        if not self.ga_property_id:
            self.ga_property_id = os.environ.get("GA4_PROPERTY_ID")
        if not self.gsc_site_url:
            self.gsc_site_url = os.environ.get("GSC_SITE_URL")
        if not self.brave_api_key:
            self.brave_api_key = os.environ.get("BRAVE_API_KEY")
        if not self.neo4j_uri:
            self.neo4j_uri = os.environ.get("NEO4J_URI")
        if not self.neo4j_user:
            self.neo4j_user = os.environ.get("NEO4J_USER")
        if not self.neo4j_password:
            self.neo4j_password = os.environ.get("NEO4J_PASSWORD")
        if not self.neo4j_database:
            self.neo4j_database = os.environ.get("NEO4J_DATABASE")

    @property
    def cache_manifest_path(self) -> Path:
        return self.cache_dir / "profile_generation_manifest.json"

    @property
    def stats_path(self) -> Path:
        return self.output_dir / "profiles.stats.json"

    def resolve_monitor_url(self) -> Optional[str]:
        if self.monitor_url:
            return self.monitor_url
        return None

