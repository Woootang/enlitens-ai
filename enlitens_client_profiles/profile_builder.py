"""High-level orchestrator that converts raw materials into ClientProfileDocuments."""

from __future__ import annotations

import hashlib
import json
import random
import time
from typing import Dict, Iterable, List, Optional, Sequence

from .config import ProfilePipelineConfig
from .data_ingestion import IngestionBundle
from .deep_research import ResearchCache
from .foundation_builder import PersonaFoundation
from .llm import ProfileLLMClient
from .prompt_builder import SYSTEM_PROMPT, build_profile_prompt
from .schema import ClientProfileDocument
from .stl_geography import build_geographic_context


class ClientProfileBuilder:
    """Create rich client profiles using curated prompt templates and LLM."""

    def __init__(self, config: ProfilePipelineConfig) -> None:
        self.config = config
        self.llm = ProfileLLMClient(
            model=config.llm_model,
            temperature=config.llm_temperature,
            top_p=config.llm_top_p,
            max_tokens=config.profile_depth_tokens,
        )

    def _sample_intakes(self, sentences: Sequence[str], *, k: int = 4) -> List[str]:
        pool = [s for s in sentences if len(s.strip()) > 60]
        if not pool:
            return list(sentences)[:k]
        sample_size = min(k, len(pool))
        return random.sample(pool, k=sample_size)

    def _sample_transcripts(self, snippets: Sequence[str], *, k: int = 4) -> List[str]:
        pool = [s for s in snippets if len(s.strip()) > 40]
        if not pool:
            return list(snippets)[:k]
        sample_size = min(k, len(pool))
        return random.sample(pool, k=sample_size)

    def _site_map_context(self, site_documents: Sequence) -> str:
        sections: List[str] = []
        for doc in site_documents[:3]:
            title = getattr(doc, "title", "") or doc.url
            summary = getattr(doc, "summary", "")
            sections.append(f"## {title}\n{summary[:700]}")
        return "\n\n".join(sections)

    def _knowledge_asset_subset(self, bundle: IngestionBundle) -> Dict[str, str]:
        subset: Dict[str, str] = {}
        for asset in bundle.knowledge_assets:
            if asset.name.lower().startswith("enlitens_philosophy"):
                subset[asset.name] = asset.content
            elif "intake" in asset.name.lower() or "framework" in asset.name.lower():
                subset[asset.name] = asset.content
            if len(subset) >= 3:
                break
        if not subset and bundle.knowledge_assets:
            for asset in bundle.knowledge_assets[:2]:
                subset[asset.name] = asset.content
        return subset

    def _hash_prompt(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _write_cache(self, cache_key: str, payload: Dict[str, any]) -> None:
        cache_file = self.config.cache_dir / f"{cache_key}.json"
        cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_cache(self, cache_key: str) -> Dict[str, any]:
        cache_file = self.config.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))
        return {}

    def _ensure_profile_id(self, payload: Dict[str, any]) -> str:
        meta = payload.setdefault("meta", {})
        profile_id = meta.get("profile_id")
        if not profile_id:
            profile_id = f"persona-{int(time.time()*1000)}"
            meta["profile_id"] = profile_id
        return profile_id

    def _foundation_summary(self, foundation: Optional[PersonaFoundation]) -> str:
        if not foundation:
            return ""
        lines = ["### Demographic hypotheses"]
        for key, value in foundation.demographics.items():
            lines.append(f"- {key}: {value}")
        if foundation.locality_hypotheses:
            lines.append("\n### Locality hypotheses")
            lines.extend(f"- {item}" for item in foundation.locality_hypotheses)
        if foundation.family_clues:
            lines.append("\n### Family system clues")
            lines.extend(f"- {clue}" for clue in foundation.family_clues[:5])
        if foundation.occupation_clues:
            lines.append("\n### Occupation clues")
            lines.extend(f"- {clue}" for clue in foundation.occupation_clues[:5])
        if foundation.gaps:
            lines.append("\n### Outstanding research gaps")
            lines.extend(f"- {gap}" for gap in foundation.gaps)
        return "\n".join(lines)

    def _research_summary(self, research: Optional[ResearchCache]) -> str:
        if not research:
            return ""
        return research.narrative_block()

    def generate_profile(
        self,
        bundle: IngestionBundle,
        *,
        foundation: Optional[PersonaFoundation] = None,
        research: Optional[ResearchCache] = None,
    ) -> ClientProfileDocument:
        intake_samples = self._sample_intakes(bundle.intake_sentence_pool)
        transcript_samples = self._sample_transcripts(bundle.founder_voice_snippets)
        if not transcript_samples:
            # Fallback to transcript records if founder snippets unavailable
            transcript_samples = self._sample_transcripts(
                [snippet.raw_text for snippet in bundle.transcripts if snippet.raw_text]
            )

        intake_samples = [sample[:600] for sample in intake_samples]
        transcript_samples = [sample[:500] for sample in transcript_samples]

        site_context = self._site_map_context(bundle.site_documents)[:2200]
        brand_site_context = (bundle.brand_site_block or site_context)[:2000]
        brand_mentions_context = (bundle.brand_mentions_block or "")[:900]

        foundation_summary = self._foundation_summary(foundation)
        research_summary = self._research_summary(research)
        prompt = build_profile_prompt(
            intake_samples=intake_samples,
            transcript_samples=transcript_samples,
            health_insights=bundle.health_report_markdown,
            knowledge_assets=self._knowledge_asset_subset(bundle),
            geo_reference=build_geographic_context(),
            site_map_context=site_context,
            brand_site_context=brand_site_context,
            brand_mentions_context=brand_mentions_context,
            locality_counts=bundle.locality_counts,
            analytics_summary=bundle.analytics_summary_block(),
            analytics_lookback_days=bundle.analytics.lookback_days if bundle.analytics else None,
            foundation_summary=foundation_summary,
            research_summary=research_summary,
        )

        cache_key = self._hash_prompt(prompt)
        if self.config.reuse_existing:
            cached = self._load_cache(cache_key)
            if cached:
                document = ClientProfileDocument.model_validate(cached)
                if not document.meta.attribute_tags:
                    document.meta.attribute_tags = document.attribute_set()
                return document

        fallback_context = build_profile_prompt(
            intake_samples=intake_samples,
            transcript_samples=transcript_samples,
            health_insights=bundle.health_report_markdown,
            knowledge_assets=self._knowledge_asset_subset(bundle),
            geo_reference=build_geographic_context(),
            site_map_context=site_context,
            brand_site_context=brand_site_context,
            brand_mentions_context=brand_mentions_context,
            locality_counts=bundle.locality_counts,
            analytics_summary=bundle.analytics_summary_block(),
            analytics_lookback_days=bundle.analytics.lookback_days if bundle.analytics else None,
            foundation_summary=foundation_summary,
            research_summary=research_summary,
        )

        profile_model = self.llm.generate_structured(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            response_model=ClientProfileDocument,
            fallback_prompt=fallback_context,
        )

        profile_payload = profile_model.model_dump()
        profile_id = self._ensure_profile_id(profile_payload)
        profile_payload.setdefault("meta", {}).setdefault("llm_model", self.config.llm_model)
        profile_payload.setdefault("meta", {}).setdefault("persona_tagline", None)
        profile_payload["meta"]["source_documents"] = [
            self.config.intakes_path.name,
            self.config.transcripts_path.name,
            self.config.health_report_path.name,
        ]

        document = ClientProfileDocument.model_validate(profile_payload)
        if not document.meta.attribute_tags:
            document.meta.attribute_tags = document.attribute_set()
        payload = document.model_dump()
        if not document.analytics.similarity_fingerprint:
            document.analytics.similarity_fingerprint = self._hash_prompt(json.dumps(payload, sort_keys=True))
            payload = document.model_dump()
        self._write_cache(cache_key, payload)
        return document

    def generate_profiles(self, bundle: IngestionBundle, count: int) -> Iterable[ClientProfileDocument]:
        for _ in range(count):
            yield self.generate_profile(bundle)

