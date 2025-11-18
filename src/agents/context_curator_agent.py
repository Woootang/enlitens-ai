"""
Context Curator Agent
Coordinates the 3 pre-processing agents to create optimized context for the main agent.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from src.agents.profile_matcher_agent import ProfileMatcherAgent
from src.agents.health_report_synthesizer_agent import HealthReportSynthesizerAgent
from src.agents.voice_guide_generator_agent import VoiceGuideGeneratorAgent
from src.agents.context_verifier_agent import ContextVerificationAgent
from src.agents.context_review_agent import ContextReviewAgent

try:  # Optional integrations
    from src.integrations.google_maps_context import GoogleMapsContextClient
except Exception:  # pragma: no cover - defensive import
    GoogleMapsContextClient = None  # type: ignore

try:
    from src.integrations.wikimedia_enterprise import WikimediaEnterpriseClient
except Exception:  # pragma: no cover - defensive import
    WikimediaEnterpriseClient = None  # type: ignore

logger = logging.getLogger(__name__)


class ContextCuratorAgent:
    """
    Master coordinator for intelligent context curation.
    
    Orchestrates 3 pre-processing agents to create optimized, high-signal context
    for the main synthesis agent.
    """
    
    def __init__(
        self,
        personas_dir: str = "enlitens_client_profiles/profiles",
        transcripts_path: str = "enlitens_knowledge_base/transcripts.txt",
        framework_path: str = "enlitens_knowledge_base/enlitens_interview_framework.txt"
    ):
        self.profile_matcher = ProfileMatcherAgent(personas_dir=personas_dir)
        self.health_synthesizer = HealthReportSynthesizerAgent()
        self.voice_generator = VoiceGuideGeneratorAgent(
            transcripts_path=transcripts_path,
            framework_path=framework_path
        )
        self.verifier = ContextVerificationAgent()
        self.reviewer = ContextReviewAgent()

        self.maps_enabled = (
            GoogleMapsContextClient is not None
            and bool(os.getenv("GOOGLE_MAPS_API_KEY"))
        )
        self.wikimedia_enabled = (
            WikimediaEnterpriseClient is not None
            and bool(os.getenv("WIKIMEDIA_ENTERPRISE_USERNAME"))
            and bool(os.getenv("WIKIMEDIA_ENTERPRISE_PASSWORD"))
        )
        
        # Cache voice guide (generated once, reused for all documents)
        self.voice_guide_cache: Optional[str] = None
        
    async def curate_context(
        self,
        paper_text: str,
        entities: Dict[str, List[str]],
        health_report_text: str,
        llm_client: Any,
        data_profiles: Optional[Dict[str, Any]] = None,
        language_profile: Optional[Dict[str, Any]] = None,
        alignment_profile: Optional[Dict[str, Any]] = None,
        health_digest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Curate optimized context for a research paper.
        
        Args:
            paper_text: Full text of the research paper
            entities: NER extracted entities
            health_report_text: Full St. Louis health report
            llm_client: LLM client for agent operations
            language_profile: Audience vocabulary/tone guardrails for prompt shaping
            
        Returns:
            Dict containing curated context:
            - selected_personas: List of 10 relevant persona objects
            - personas_text: Formatted text of selected personas
            - health_brief: Synthesized health context
            - voice_guide: Liz's voice style guide
            - token_estimate: Estimated total tokens
            - language_profile: Audience language guardrails (if available)
            - alignment_profile: Topic alignment metadata (if available)
        """
        logger.info("="*80)
        logger.info("🎯 CONTEXT CURATION: Intelligent Pre-Processing")
        logger.info("="*80)
        
        verification_history: List[Dict[str, Any]] = []
        review_history: List[Dict[str, Any]] = []
        persona_feedback: Optional[str] = None
        health_feedback: Optional[str] = None
        voice_feedback: Optional[str] = None
        max_attempts = 3

        for attempt in range(1, max_attempts + 1):
            logger.info("🔁 Context curation attempt %d/%d", attempt, max_attempts)

            # Step 1: Select top 10 relevant personas
            logger.info("📊 Agent 1: Profile Matcher - Selecting relevant personas...")
            selected_personas = await self.profile_matcher.select_top_personas(
                paper_text=paper_text,
                entities=entities,
                llm_client=llm_client,
                top_k=5,
                refinement_feedback=persona_feedback,
                language_profile=language_profile,
                alignment_profile=alignment_profile,
            )

            personas_text = self.profile_matcher.get_selected_personas_text(selected_personas)
            logger.info(f"✅ Selected {len(selected_personas)} personas (~{len(personas_text)//4} tokens)")

            personas_text, mechanism_bridge = self._augment_persona_brief(
                personas_text=personas_text,
                personas=selected_personas,
                alignment_profile=alignment_profile,
            )

            if len(selected_personas) < 5:
                logger.warning(
                    "⚠️ Less than 5 personas selected (%d). Ensuring minimum coverage by padding.",
                    len(selected_personas),
                )
                fallback_personas = self.profile_matcher.load_personas()[:10]
                selected_personas = fallback_personas
                personas_text = self.profile_matcher.get_selected_personas_text(selected_personas)
                personas_text, mechanism_bridge = self._augment_persona_brief(
                    personas_text=personas_text,
                    personas=selected_personas,
                    alignment_profile=alignment_profile,
                )

            external_context: Dict[str, Any] = {}
            if self.maps_enabled:
                local_resources = await self._maybe_fetch_local_resources(selected_personas)
                if local_resources:
                    external_context["local_resources"] = local_resources
                    personas_text += "\n\n## LOCAL CONTEXT (Google Maps)\n" + "\n".join(
                        f"- {item['name']} near {item['location_query']}: {item['description']}"
                        for item in local_resources
                    )
            if self.wikimedia_enabled:
                entity_summaries = await self._maybe_fetch_entity_definitions(entities)
                if entity_summaries:
                    external_context["entity_summaries"] = entity_summaries


            # Step 2: Synthesize health report
            logger.info("🏥 Agent 2: Health Report Synthesizer - Creating targeted brief...")
            health_brief = await self.health_synthesizer.synthesize_health_context(
                health_report_text=health_report_text,
                selected_personas=selected_personas,
                llm_client=llm_client,
                refinement_feedback=health_feedback,
                language_profile=language_profile,
                alignment_profile=alignment_profile,
                health_digest=health_digest,
            )
            health_brief, local_stats = self._enrich_health_brief(
                health_brief=health_brief,
                health_digest=health_digest,
                data_profiles=data_profiles or {},
            )
            logger.info(f"✅ Health brief synthesized (~{len(health_brief)//4} tokens)")

            # Step 3: Generate voice guide (cached after first generation)
            if self.voice_guide_cache is None:
                logger.info("🎙️ Agent 3: Voice Guide Generator - Creating Liz's style guide...")
                self.voice_guide_cache = await self.voice_generator.generate_voice_guide(llm_client)
                logger.info(f"✅ Voice guide generated (~{len(self.voice_guide_cache)//4} tokens)")
            else:
                logger.info("✅ Using cached voice guide")

            voice_guide = self._sanitize_language(self.voice_guide_cache or "")

            # Calculate token estimates (rough: 1 token ≈ 4 characters)
            token_estimate = {
                'personas': len(personas_text) // 4,
                'health_brief': len(health_brief) // 4,
                'voice_guide': len(voice_guide) // 4,
                'total_curated': (len(personas_text) + len(health_brief) + len(voice_guide)) // 4
            }

            curated_context = {
                'selected_personas': selected_personas,
                'personas_text': personas_text,
                'health_brief': health_brief,
                'voice_guide': voice_guide,
                'token_estimate': token_estimate,
            }
            curated_context['mechanism_bridge'] = mechanism_bridge
            curated_context['local_stats'] = local_stats
            curated_context['persona_selection_meta'] = [
                persona.get("meta", {}).get("selection_meta", {})
                for persona in selected_personas
            ]
            if language_profile:
                curated_context['language_profile'] = language_profile
            if alignment_profile:
                curated_context['alignment_profile'] = alignment_profile
            if health_digest:
                curated_context['health_digest'] = health_digest
                if isinstance(health_digest, dict) and health_digest.get("prompt_block"):
                    curated_context['health_digest_prompt'] = health_digest["prompt_block"]
            if external_context:
                curated_context['external_context'] = external_context

            review = await self.reviewer.review(
                curated_context=curated_context,
                llm_client=llm_client,
                language_profile=language_profile,
                alignment_profile=alignment_profile,
                health_digest=health_digest,
            )
            review_history.append(review)
            curated_context['review_feedback'] = review
            if not review.get("overall_pass", False):
                logger.warning("⚠️ Context review flagged issues before verification: %s", review.get("issues"))
                persona_feedback = review.get("persona_feedback") or persona_feedback
                health_feedback = review.get("health_feedback") or health_feedback
                voice_feedback = review.get("voice_feedback") or voice_feedback
                if attempt < max_attempts:
                    logger.info("🔁 Applying reviewer feedback and retrying curation before verifier.")
                    continue

            # Verification pass
            verdict = await self.verifier.evaluate(
                paper_profile=(data_profiles or {}).get("paper"),
                curated_context=curated_context,
                data_profiles=data_profiles,
                llm_client=llm_client,
                alignment_profile=alignment_profile,
            )
            verification_history.append(verdict)

            status = verdict.get("status", "error")
            if status == "pass":
                logger.info("✅ Curated context passed verification on attempt %d", attempt)
                break

            logger.warning("⚠️ Curated context requires refinement (status=%s)", status)
            persona_feedback = verdict.get("persona_feedback") or persona_feedback
            health_feedback = verdict.get("health_feedback") or health_feedback

            if attempt == max_attempts:
                logger.warning("⚠️ Reached maximum verification attempts; proceeding with latest context.")

        logger.info("="*80)
        logger.info(f"📊 CURATION COMPLETE - Token Breakdown:")
        logger.info(f"   Personas: ~{token_estimate['personas']:,} tokens")
        logger.info(f"   Health Brief: ~{token_estimate['health_brief']:,} tokens")
        logger.info(f"   Voice Guide: ~{token_estimate['voice_guide']:,} tokens")
        logger.info(f"   Total Curated Context: ~{token_estimate['total_curated']:,} tokens")
        logger.info("="*80)

        curated_context['verification'] = {
            "attempts": verification_history,
            "final_status": verification_history[-1].get("status") if verification_history else "unknown",
        }
        curated_context['review'] = {
            "attempts": review_history,
            "final_pass": review_history[-1].get("overall_pass", False) if review_history else None,
            "voice_feedback": voice_feedback,
        }
        
        return curated_context

    @staticmethod
    def _sanitize_language(block: str) -> str:
        banned_terms = {
            "journey": "experience",
            "pathway": "process",
            "pathways": "processes",
            "roadmap": "plan",
        }
        cleaned = block
        for bad, replacement in banned_terms.items():
            cleaned = cleaned.replace(bad, replacement)
            cleaned = cleaned.replace(bad.title(), replacement.title())
        return cleaned

    def _augment_persona_brief(
        self,
        *,
        personas_text: str,
        personas: List[Dict[str, Any]],
        alignment_profile: Optional[Dict[str, Any]],
    ) -> Tuple[str, str]:
        bridge_lines: List[str] = []
        mechanism_note = ""
        if alignment_profile:
            mechanism_note = alignment_profile.get("alignment_note") or alignment_profile.get("note") or ""
            related = alignment_profile.get("related_persona_themes") or alignment_profile.get("themes") or []
            if mechanism_note:
                bridge_lines.append(f"- Mechanistic driver: {mechanism_note.strip()}")
            if related:
                bridge_lines.append(f"- Dominant themes: {', '.join(str(item) for item in related[:6])}")

        if not bridge_lines:
            bridge_lines.append("- Mechanistic driver: Chronic stress → inflammation → executive overload.")

        for persona in personas[:5]:
            identity = persona.get("meta", {}).get("name") or persona.get("meta", {}).get("title")
            if not identity:
                identity = persona.get("_file", "persona").split("/")[-1].replace(".json", "")
            challenges = persona.get("current_life_context", {}).get("current_stressors")
            if isinstance(challenges, list) and challenges:
                challenge_snippet = ", ".join(str(c) for c in challenges[:3])
            else:
                challenge_snippet = persona.get("meta", {}).get("tagline") or persona.get("developmental_story", {}).get("summary", "")
            if challenge_snippet:
                bridge_lines.append(f"- {identity}: links lived stressors ({challenge_snippet}) to CSA-driven biology.")
            else:
                bridge_lines.append(f"- {identity}: aligns with CSA mechanisms through sustained social load and executive strain.")

        bridge_block = "\n\n## MECHANISM ↔ PERSONA BRIDGE\n" + "\n".join(bridge_lines)
        sanitized_bridge = self._sanitize_language(bridge_block)
        combined = self._sanitize_language(personas_text.strip() + sanitized_bridge)
        return combined, sanitized_bridge

    def _enrich_health_brief(
        self,
        *,
        health_brief: str,
        health_digest: Optional[Dict[str, Any]],
        data_profiles: Dict[str, Any],
    ) -> Tuple[str, List[str]]:
        stats_sources: List[str] = []
        for payload in (health_digest or {}).get("key_statistics") or []:
            stats_sources.append(str(payload))
        digest_meta = (data_profiles.get("health_report") or {}).get("key_statistics") or []
        stats_sources.extend(str(item) for item in digest_meta)

        unique_stats = []
        for stat in stats_sources:
            stat_clean = stat.strip()
            if stat_clean and stat_clean not in unique_stats:
                unique_stats.append(stat_clean)

        default_stats = [
            "39% of St. Louis adults report chronic stress tied to social inequity.",
            "1 in 3 neurodivergent adults cite executive burnout from community strain.",
        ]
        if unique_stats:
            selected_stats = unique_stats[:8]
            stats_block = "\n\n## KEY LOCAL STATISTICS\n" + "\n".join(f"- {stat}" for stat in selected_stats)
        else:
            selected_stats = default_stats
            stats_block = "\n\n## KEY LOCAL STATISTICS\n" + "\n".join(f"- {stat}" for stat in selected_stats)

        combined = (health_brief or "").strip() + stats_block
        compliance_line = "\n\n## VOICE GUARDRAILS\n- Maintain Liz's direct, rebellious tone without clichés or passive phrasing."
        combined += compliance_line
        return self._sanitize_language(combined), selected_stats
    
    async def _maybe_fetch_local_resources(
        self,
        personas: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        if not self.maps_enabled or GoogleMapsContextClient is None:
            return []
        results: List[Dict[str, str]] = []
        try:
            async with GoogleMapsContextClient() as client:
                for persona in personas[:3]:
                    location = (
                        persona.get("identity_demographics", {}).get("locality")
                        or persona.get("identity_demographics", {}).get("location")
                        or persona.get("meta", {}).get("location")
                    )
                    if not location:
                        continue
                    query = f"mental health support near {location}"
                    try:
                        response = await client.text_search(query=query, radius_meters=8000)
                        first = (response.get("results") or [])[:1]
                        for item in first:
                            results.append(
                                {
                                    "name": item.get("name", "Resource"),
                                    "description": item.get("formatted_address", "Address unavailable"),
                                    "location_query": location,
                                }
                            )
                    except Exception as exc:  # pragma: no cover - network
                        logger.debug("Google Maps enrichment failed for %s: %s", location, exc)
                        continue
        except Exception as exc:  # pragma: no cover - network
            logger.debug("Google Maps client unavailable: %s", exc)
        return results

    async def _maybe_fetch_entity_definitions(
        self,
        entities: Dict[str, List[str]],
    ) -> Dict[str, str]:
        if not self.wikimedia_enabled or WikimediaEnterpriseClient is None:
            return {}
        names: List[str] = []
        for bucket in entities.values():
            names.extend(bucket[:2])
        unique_names = list(dict.fromkeys(name for name in names if isinstance(name, str) and name.strip()))[:6]
        if not unique_names:
            return {}

        summaries: Dict[str, str] = {}
        try:
            async with WikimediaEnterpriseClient() as client:
                await client.authenticate()
                for name in unique_names:
                    title = name.replace(" ", "_")
                    try:
                        payload = await client.get_article(title)
                    except Exception as exc:  # pragma: no cover - network
                        logger.debug("Wikimedia lookup failed for %s: %s", title, exc)
                        continue
                    results = payload.get("results") or []
                    if not results:
                        continue
                    article = results[0]
                    summary = (
                        article.get("summary")
                        or article.get("paragraphs", [{}])[0].get("value")
                        or article.get("intro", "")
                    )
                    if summary:
                        summaries[name] = str(summary)[:500]
        except Exception as exc:  # pragma: no cover - network
            logger.debug("Wikimedia client unavailable: %s", exc)
        return summaries     
    
    def format_curated_context_for_prompt(self, curated_context: Dict[str, Any]) -> str:
        """
        Format curated context into a single string for the main agent's prompt.
        
        Returns a structured, ready-to-use context block.
        """
        return f"""
# CURATED CONTEXT FOR THIS RESEARCH PAPER

## RELEVANT CLIENT PERSONAS
{curated_context['personas_text']}

## LOCAL HEALTH CONTEXT (St. Louis)
{curated_context['health_brief']}

## LIZ'S VOICE GUIDE
{curated_context['voice_guide']}

{self._format_health_digest(curated_context)}

---
This context has been intelligently curated to maximize relevance and quality.
Use ALL of this information to inform your output.
"""

    @staticmethod
    def _format_health_digest(curated_context: Dict[str, Any]) -> str:
        """Embed the health digest snapshot if available."""
        digest_prompt = curated_context.get("health_digest_prompt")
        digest_payload = curated_context.get("health_digest")
        if digest_prompt:
            return f"\n## ST. LOUIS HEALTH DIGEST SNAPSHOT\n{digest_prompt}\n"
        if isinstance(digest_payload, dict):
            headline = digest_payload.get("headline")
            bullets = digest_payload.get("summary_bullets") or []
            lines = ["## ST. LOUIS HEALTH DIGEST SNAPSHOT"]
            if headline:
                lines.append(headline)
            for bullet in bullets[:5]:
                lines.append(f"- {bullet}")
            return "\n".join(lines) + "\n"
        return ""

