"""
Educational Content Agent - Extracts and generates educational materials.
"""

import logging
from typing import Dict, Any, Optional, List
from textwrap import dedent

from .base_agent import BaseAgent
from src.models.enlitens_schemas import EducationalContent
from src.synthesis.few_shot_library import FEW_SHOT_LIBRARY
from src.synthesis.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class EducationalContentAgent(BaseAgent):
    """Agent specialized in generating educational content for clients."""

    GENERAL_PERSONA_FALLBACK = (
        "Neurodivergent adults and caregivers in St. Louis who crave neuroscience explanations, "
        "trauma-informed validation, and practical strategies they can try immediately."
    )

    REGIONAL_FALLBACK = (
        "St. Louis metro (2.8M): high trauma exposure, ADHD/executive-function struggles, "
        "racial and economic disparities, transportation and access barriers, strong desire for "
        "science-backed, culturally aware care."
    )

    def __init__(self):
        super().__init__(
            name="EducationalContent",
            role="Client Education Material Generation",
        )
        self.ollama_client = None

    async def initialize(self) -> bool:
        """Initialize the educational content agent."""
        try:
            self.ollama_client = OllamaClient(default_model=self.model)
            if not await self.ollama_client.check_connection():
                raise RuntimeError(
                    f"vLLM server is not reachable at {self.ollama_client.base_url}. Please run stable_run.sh or start the vLLM server."
                )
            self.is_initialized = True
            logger.info(f"✅ {self.name} agent initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            return False

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate educational content from research."""
        try:
            document_text = context.get("document_text", "")[:8000]
            research_content = context.get("science_data", {}).get("research_content", {})
            clinical_content = context.get("clinical_content", {})
            curated_context = context.get("curated_context") or {}

            persona_brief = self._build_persona_brief(
                context.get("client_insights") or {},
                curated_context.get("personas_text")
            )
            regional_brief = self._build_regional_brief(
                context.get("regional_context") or context.get("st_louis_context") or {},
                curated_context.get("health_brief"),
                context.get("regional_digest_chunks"),
                context.get("regional_prompt_block"),
            )
            language_guardrails = self._build_language_guardrails(
                curated_context.get("language_profile") or context.get("language_profile") or {},
                context.get("language_watchouts") or {},
            )
            mechanism_bridge = ""
            if curated_context.get("mechanism_bridge"):
                mechanism_bridge = self._truncate(curated_context["mechanism_bridge"].strip(), max_chars=500)
            mechanism_block = (
                f"MECHANISM ↔ PERSONA BRIDGE:\n{mechanism_bridge}\n"
                if mechanism_bridge else
                "MECHANISM ↔ PERSONA BRIDGE:\nSpell out how CSA/stress biology maps to persona-level burnout, sensory overload, and inflammation.\n"
            )
            stats_list = curated_context.get("local_stats") or []
            stats_block = ""
            if stats_list:
                stats_lines = "\n".join(f"- {stat}" for stat in stats_list[:6])
                stats_block = f"LOCAL STATS ANCHORS:\n{stats_lines}\n"
            research_payload = self._safe_json_dump(research_content)
            clinical_payload = self._safe_json_dump(clinical_content)
            output_example = self._educational_schema_example()

            few_shot_block = FEW_SHOT_LIBRARY.render_for_prompt(
                task="educational_content",
                query=document_text,
                k=2,
            )

            exemplars = (
                "FEW-SHOT EXEMPLARS (match tone, show client-level clarity):\n"
                f"{few_shot_block}\n\n" if few_shot_block else ""
            )

            prompt = f"""
You are an educational content specialist creating client-friendly educational materials.
Your goal is to translate complex neuroscience research into accessible educational content.

STRICT RULES:
✓ Prioritize the provided research findings for factual claims and cite them precisely
✓ Use analogies and examples to make concepts accessible and memorable
✓ Clearly mark hypothetical examples as "[HYPOTHETICAL EXAMPLE]"
✓ If you must add widely accepted neuroscience or trauma-informed insight that is not in the sources,
  prefix the entry with "[GENERAL INSIGHT]" and ensure it aligns with the research themes
✓ If the document offers limited specifics, create useful [GENERAL INSIGHT] entries instead of refusing the task
✗ DO NOT fabricate research findings, statistics, or citations
✗ DO NOT include client testimonials or practice statistics
✗ DO NOT mention internal AI behavior or apologize

{exemplars}
AUDIENCE INSIGHTS:
{persona_brief}

REGIONAL CONTEXT:
{regional_brief}

LANGUAGE GUARDRAILS:
{language_guardrails}

{mechanism_block}
{stats_block}

AVAILABLE RESEARCH DATA (use for citations):
{research_payload}

AVAILABLE CLINICAL INSIGHTS:
{clinical_payload}

DOCUMENT TEXT:
{document_text}

OUTPUT SCHEMA EXAMPLE:
{output_example}

Create comprehensive educational content for ALL sections below (5-10 items per section):

1. EXPLANATIONS: Clear, accessible explanations of neuroscience concepts from the research. Break down complex ideas into understandable language. Explain HOW things work in the brain.

2. EXAMPLES: Concrete, relatable examples that illustrate neuroscience concepts. Use everyday situations clients would recognize. Show concepts in action.

3. ANALOGIES: Creative analogies and metaphors that make neuroscience accessible. Compare brain processes to familiar things (e.g., "Your amygdala is like a smoke alarm").

4. DEFINITIONS: Simple definitions of technical terms, brain regions, processes, and neuroscience jargon. Make terminology accessible without dumbing down.

5. PROCESSES: Step-by-step explanations of how brain processes work. Explain sequences like "How anxiety develops in the brain" or "How memory consolidation happens."

6. COMPARISONS: Compare and contrast different concepts, brain regions, or processes. Show how things are similar or different. Help clients understand relationships.

7. VISUAL AIDS: Descriptions of diagrams, charts, or visual aids that would help explain concepts. Describe what should be illustrated and why it would help learning.

8. LEARNING OBJECTIVES: Clear learning objectives for each topic. What should clients understand after learning this? What should they be able to do with this knowledge?

EXTRACTION GUIDELINES:
- Make everything client-accessible (8th grade reading level)
- Use concrete examples from daily life
- Start each entry with a mini prediction-error hook (e.g., "You were told X, but the brain actually does Y")
- Explain "why this matters" for each concept
- Avoid jargon or explain it when necessary
- Each item should be substantial (2-4 sentences) and tied to persona/local needs
- Focus on practical understanding, not academic detail
- Each section needs 5-10 items minimum. If source coverage is thin, synthesize high-value [GENERAL INSIGHT] entries rather than leaving a list empty.
- Include citations like "…" ONLY when referencing source findings.
- Do not add citations to [GENERAL INSIGHT] items.

Return as JSON with these EXACT field names:
{{"explanations": [list], "examples": [list], "analogies": [list], "definitions": [list], "processes": [list], "comparisons": [list], "visual_aids": [list], "learning_objectives": [list]}}
"""

            cache_kwargs = self._cache_kwargs(context)
            result = await self.ollama_client.generate_structured_response(
                prompt=prompt,
                response_model=EducationalContent,
                temperature=0.5,
                max_retries=3,
                enforce_grammar=True,
                **cache_kwargs,
            )

            if result:
                payload = result.model_dump()
                if self._content_has_substance(payload):
                    return {
                        "educational_content": payload,
                        "generation_quality": "high",
                    }

            fallback = self._build_fallback_content(
                context=context,
                research=research_content,
                clinical=clinical_content,
            )
            logger.warning("EducationalContentAgent: falling back to synthesized educational content")
            return {
                "educational_content": fallback.model_dump(),
                "generation_quality": "fallback",
            }

        except Exception as e:
            logger.error(f"Educational content generation failed: {e}")
            fallback = self._build_fallback_content(
                context=context,
                research=context.get("science_data", {}).get("research_content", {}),
                clinical=context.get("clinical_content", {}),
            )
            return {
                "educational_content": fallback.model_dump(),
                "generation_quality": "fallback",
            }

    def _content_has_substance(self, payload: Dict[str, Any]) -> bool:
        if not payload:
            return False
        for field in (
            "explanations",
            "examples",
            "analogies",
            "definitions",
            "processes",
            "comparisons",
            "visual_aids",
            "learning_objectives",
        ):
            items = payload.get(field) or []
            if any(isinstance(item, str) and item.strip() for item in items):
                return True
        return False

    def _build_fallback_content(
        self,
        *,
        context: Dict[str, Any],
        research: Dict[str, Any],
        clinical: Dict[str, Any],
    ) -> EducationalContent:
        topic = research.get("primary_theme") or clinical.get("primary_focus") or "brain adaptation"
        doc_id = context.get("document_id", "this document")
        region = "St. Louis" if context.get("st_louis_context") else "your community"

        return EducationalContent(
            explanations=[
                f"[GENERAL INSIGHT] {topic.title()} shows how the brain protects you when life keeps hitting hard. Even without granular study data, we can translate the pattern for clients.",
                "[GENERAL INSIGHT] Your nervous system learns safety or danger from repetition—when stress never lets up, it stays on high alert."
            ],
            examples=[
                "[GENERAL INSIGHT] Picture a St. Louis nurse finishing a 12-hour shift then caregiving at home. Her brain keeps pumping stress chemistry, so even small chores feel overwhelming."
            ],
            analogies=[
                "[GENERAL INSIGHT] Chronic cortisol is like idling a car all night—the engine overheats and stalls the next day."
            ],
            definitions=[
                "[GENERAL INSIGHT] Neuroplasticity: the brain's ability to rewire toward safety when we rehearse calmer experiences on purpose."
            ],
            processes=[
                "[GENERAL INSIGHT] Stress loop: trigger → body alarms → brain writes a danger story → behavior follows. Mapping that loop lets clients interrupt it."
            ],
            comparisons=[
                "[GENERAL INSIGHT] Burnout isn't laziness—it is a brain stuck in survival mode with zero recharge built in."
            ],
            visual_aids=[
                f"[GENERAL INSIGHT] Visual: a traffic light showing green (regulated), yellow (mobilized), red (shutdown) with {region} stressors at each level."
            ],
            learning_objectives=[
                f"[GENERAL INSIGHT] After engaging this content, clients from {region} can name their state, spot triggers faster, and choose one sensory reset that fits their daily life."
            ],
        )


    def _build_persona_brief(
        self,
        persona: Dict[str, Any],
        curated_personas_text: Optional[str],
    ) -> str:
        if curated_personas_text:
            return dedent(
                self._truncate(
                    curated_personas_text.strip(),
                    max_chars=800,
                )
            )

        if not persona:
            return self.GENERAL_PERSONA_FALLBACK

        lines = []
        summary = persona.get("persona_summary") or {}
        top_audience_pain = summary.get("top_pain_points") or persona.get("audience_pain_points") or []
        if top_audience_pain:
            lines.append("Community pain points: " + ", ".join(top_audience_pain[:5]))
        top_keywords = summary.get("top_keywords") or persona.get("audience_keywords") or []
        if top_keywords:
            lines.append("High-intent keywords: " + ", ".join(top_keywords[:5]))
        segments = summary.get("representative_segments") or persona.get("persona_segments") or []
        if segments:
            segment = segments[0]
            seg_name = segment.get("name") or "Representative persona"
            seg_pains = ", ".join(segment.get("core_pain_points", [])[:2])
            lines.append(f"{seg_name}: {seg_pains or 'neurodivergent, trauma-aware client'}")
        pain_points = persona.get("pain_points") or persona.get("challenges") or []
        if pain_points:
            lines.append(
                "Top pain points: " + ", ".join(pain_points[:5])
            )
        priorities = persona.get("priorities") or []
        if priorities:
            lines.append(
                "Immediate goals: " + ", ".join(priorities[:5])
            )
        themes = persona.get("key_themes") or []
        if themes:
            lines.append("Recurring themes: " + ", ".join(themes[:4]))

        enhanced = persona.get("enhanced_analysis") or {}
        summary = enhanced.get("narrative_summary") or enhanced.get("summary")
        if summary:
            lines.append(self._truncate(summary, 300))
        persona_prompt_block = persona.get("persona_prompt_block")
        if persona_prompt_block:
            lines.append("Audience language snapshot:\n" + self._truncate(persona_prompt_block, 400))

        if not lines:
            return self.GENERAL_PERSONA_FALLBACK
        return "\n".join(lines)

    def _build_regional_brief(
        self,
        regional: Dict[str, Any],
        health_brief: Optional[str],
        digest_chunks: Optional[Any] = None,
        prompt_block: Optional[str] = None,
    ) -> str:
        if health_brief:
            return self._truncate(health_brief.strip(), max_chars=600)

        demographics = regional or {}
        lines = []
        population = demographics.get("population")
        if population:
            lines.append(f"Population: {population}")
        mh_challenges = demographics.get("mental_health_challenges") or []
        if mh_challenges:
            lines.append("Mental health pressures: " + ", ".join(mh_challenges[:5]))
        socioeconomic = demographics.get("socioeconomic_factors") or []
        if socioeconomic:
            lines.append("Social determinants: " + ", ".join(socioeconomic[:4]))
        summary_bullets = demographics.get("summary_bullets") or []
        if summary_bullets:
            lines.append("Regional summary bullets:\n- " + "\n- ".join(summary_bullets[:4]))
        flashpoints = demographics.get("cultural_flashpoints") or []
        if flashpoints:
            labels = [flash.get("label") for flash in flashpoints if isinstance(flash, dict)]
            if labels:
                lines.append("Cultural flashpoints to name: " + ", ".join(label for label in labels[:4] if label))
        stats = demographics.get("key_statistics") or []
        if stats:
            lines.append("Key stats:\n- " + "\n- ".join(stats[:4]))
        if digest_chunks:
            chunk_preview = "\n\n".join(str(chunk) for chunk in digest_chunks[:2] if chunk)
            if chunk_preview:
                lines.append("Health digest snapshot:\n" + self._truncate(chunk_preview, 500))
        if prompt_block:
            lines.append("Digest prompt block:\n" + self._truncate(prompt_block, 400))

        if not lines:
            return self.REGIONAL_FALLBACK
        return "\n".join(lines)

    def _build_language_guardrails(
        self,
        language_profile: Dict[str, Any],
        watchouts: Dict[str, Any],
    ) -> str:
        lines: List[str] = []
        prompt_block = language_profile.get("prompt_block")
        if prompt_block:
            lines.append(self._truncate(prompt_block, 400))
        words_to_use = watchouts.get("words_to_use") if isinstance(watchouts, dict) else None
        if words_to_use:
            lines.append("Words that resonate: " + ", ".join(words_to_use[:10]))
        words_to_avoid = watchouts.get("words_to_avoid") if isinstance(watchouts, dict) else None
        if words_to_avoid:
            lines.append("Words to avoid unless cited directly: " + ", ".join(words_to_avoid[:8]))
        banned_terms = language_profile.get("banned_terms") if isinstance(language_profile, dict) else None
        if banned_terms:
            lines.append("Hard no terms: " + ", ".join(banned_terms[:8]))
        if lines:
            return "\n".join(lines)
        return "Use Liz's direct tone—plain speech, no therapy clichés, no mindfulness/manifestation fluff."

    def _educational_schema_example(self) -> str:
        return dedent("""\
        {{
          "explanations": [
            "Your amygdala works like a smoke alarm that keeps ringing when chronic stress teaches it the world is unsafe."
          ],
          "examples": [
            "[GENERAL INSIGHT] Think of a teacher stuck in noisy hallways all day—by evening her brain is so overstimulated that even a loved one's simple question feels like an attack."
          ],
          "analogies": [
            "Chronic cortisol is like leaving a car idling all night—the engine (your nervous system) overheats and stalls the next day."
          ],
          "definitions": [
            "Neuroception: your brainstem's split-second scan for safety or danger before you are even aware of it."
          ],
          "processes": [
            "How stress becomes inflammation: trigger → cortisol surge → immune activation → inflamed brain cells → brain fog."
          ],
          "comparisons": [
            "Freeze vs. shutdown: freeze keeps you alert but immobile; dorsal shutdown makes you numb and disconnected."
          ],
          "visual_aids": [
            "Diagram idea: show the vagus nerve as a safety ladder—top rung = social engagement, middle = fight/flight, bottom = collapse."
          ],
          "learning_objectives": [
            "Explain how chronic stress reshapes the brain's alarm systems and identify two practices that help recalibrate them."
          ]
        }}""")

    def _truncate(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _safe_json_dump(self, payload: Any, max_chars: int = 1200) -> str:
        try:
            import json

            serialized = json.dumps(payload, ensure_ascii=False, indent=2)
            return self._truncate(serialized, max_chars=max_chars)
        except Exception:
            return self._truncate(str(payload), max_chars=max_chars)

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate the generated educational content."""
        educational_content = output.get("educational_content", {})

        has_content = any([
            educational_content.get("explanations"),
            educational_content.get("examples"),
            educational_content.get("analogies")
        ])

        return has_content

    async def cleanup(self):
        """Clean up resources."""
        logger.info(f"Cleaning up {self.name} agent")
