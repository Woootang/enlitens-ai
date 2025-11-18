"""
Health Report Synthesizer Agent
Creates targeted briefs from the St. Louis Health Report based on selected personas.
"""

import logging
import re
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthReportSynthesizerAgent:
    """Synthesizes relevant health report data for selected client personas."""
    
    def __init__(self):
        self.health_report_cache: str = ""
        
    async def synthesize_health_context(
        self,
        health_report_text: str,
        selected_personas: List[Dict[str, Any]],
        llm_client: Any,
        refinement_feedback: str | None = None,
        language_profile: Optional[Dict[str, Any]] = None,
        alignment_profile: Optional[Dict[str, Any]] = None,
        health_digest: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a targeted health brief connecting personas to local health data.
        
        Args:
            health_report_text: Full St. Louis health report text
            selected_personas: The 10 selected client personas
            llm_client: LLM client for synthesis
            language_profile: Optional vocabulary/tone guardrails from analytics
            alignment_profile: Optional topic bridge clarifying indirect relevance
            
        Returns:
            Synthesized health brief (500-1000 words)
        """
        if not health_report_text or not selected_personas:
            logger.warning("Missing health report or personas for synthesis")
            return "No local health context available."
        
        # Extract persona demographics
        demographics = self._extract_demographics(selected_personas)
        
        # Chain of Thought: Explain synthesis strategy
        logger.info(f"🧠 THINKING: Analyzing {demographics['count']} personas. Age range: {demographics['age_range']}, Diagnoses: {', '.join(demographics['diagnoses'][:3])}. From the {len(health_report_text)} char health report, I'll extract ONLY the data relevant to these specific conditions and demographics in St. Louis.")
        
        # Create synthesis prompt
        prompt = self._create_synthesis_prompt(
            health_report_text,
            demographics,
            refinement_feedback=refinement_feedback,
            language_profile=language_profile,
            alignment_profile=alignment_profile,
            health_digest=health_digest,
        )
        
        try:
            logger.info(f"🧠 THINKING: Sending synthesis request to LLM. Looking for: local prevalence statistics, environmental factors (air quality, urban stress), healthcare access barriers, and community resources specific to their age/condition profile.")
            
            response = await llm_client.generate_text(
                prompt=prompt,
                temperature=0.4,
                num_predict=1500
            )
            
            # Chain of Thought: Summarize what was extracted
            key_terms = ['prevalence', 'statistics', 'access', 'environmental', 'socioeconomic']
            found_topics = [term for term in key_terms if term.lower() in response.lower()]
            logger.info(f"🧠 THINKING: Synthesis complete. Generated {len(response)} chars covering: {', '.join(found_topics)}. This brief connects the cohort's specific needs to local health realities.")
            
            # Validate that statistics are present
            stats_indicators = [r'\d+%', r'\d+\s*percent', r'\d+\s*in\s*\d+', r'\d+\.\d+', r'\$\d+']
            stats_found = sum(1 for pattern in stats_indicators if re.search(pattern, response, re.IGNORECASE))
            if stats_found < 2:
                logger.warning(f"⚠️ Health brief contains only {stats_found} statistical indicators; quality may be low")
            else:
                logger.info(f"✅ Health brief contains {stats_found} statistical indicators")
            
            logger.info(f"✅ Synthesized health context ({len(response)} chars)")
            return response
            
        except Exception as e:
            logger.error(f"Health report synthesis failed: {e}")
            return "Health context synthesis unavailable."
    
    def _extract_demographics(self, personas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract demographic summary from selected personas."""
        age_ranges: List[str] = []
        age_values: List[int] = []
        gender_counter: Counter[str] = Counter()
        location_counter: Counter[str] = Counter()
        diagnosis_counter: Counter[str] = Counter()
        life_stage_counter: Counter[str] = Counter()
        
        for persona in personas:
            identity = persona.get('identity_demographics', {})
            current_life = persona.get('current_life_context', {})
            neuro = persona.get('neurodivergence_mental_health', {})
            meta = persona.get('meta', {})

            age_range = identity.get('age_range') or meta.get('age_range')
            if isinstance(age_range, str) and age_range.strip():
                age_ranges.append(age_range)
                age_values.extend([int(val) for val in re.findall(r"\d+", age_range)])

            gender = identity.get('gender') or meta.get('gender')
            if isinstance(gender, str) and gender.strip():
                gender_counter[gender.strip()] += 1

            location = identity.get('locality') or identity.get('location') or meta.get('location')
            if isinstance(location, str) and location.strip():
                location_counter[location.strip()] += 1

            for key in ("formal_diagnoses", "self_identified_traits", "identities", "diagnoses", "primary_diagnoses"):
                values = neuro.get(key) if key != "primary_diagnoses" else meta.get(key)
                if isinstance(values, list):
                    for item in values:
                        if isinstance(item, str) and item.strip():
                            diagnosis_counter[item.strip()] += 1

            life_stage = current_life.get('life_stage') or identity.get('current_life_situation') or meta.get('life_stage')
            if isinstance(life_stage, str) and life_stage.strip():
                life_stage_counter[life_stage.strip()] += 1

        age_summary = "Unknown"
        if age_values:
            age_summary = f"{min(age_values)}-{max(age_values)}"
        elif age_ranges:
            age_summary = ", ".join(sorted(set(age_ranges)))
        
        return {
            'age_range': age_summary,
            'age_ranges': sorted(set(age_ranges)),
            'genders': list(gender_counter.keys()),
            'locations': list(location_counter.keys()),
            'diagnoses': list(diagnosis_counter.keys()),
            'life_stages': list(life_stage_counter.keys()),
            'diagnosis_counts': diagnosis_counter.most_common(10),
            'life_stage_counts': life_stage_counter.most_common(10),
            'location_counts': location_counter.most_common(10),
            'count': len(personas)
        }
    
    def _create_synthesis_prompt(
        self,
        health_report_text: str,
        demographics: Dict[str, Any],
        refinement_feedback: str | None = None,
        language_profile: Optional[Dict[str, Any]] = None,
        alignment_profile: Optional[Dict[str, Any]] = None,
        health_digest: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create prompt for health report synthesis."""
        
        # Truncate health report if too long (keep first 8000 chars)
        report_text = health_report_text[:8000] + "..." if len(health_report_text) > 8000 else health_report_text
        
        refinement_block = ""
        if refinement_feedback:
            refinement_block = f"""

ADDITIONAL QUALITY REVIEW FEEDBACK:
{refinement_feedback.strip()}

Explicitly incorporate these missing statistics, demographics, or context details in the brief above."""

        diagnosis_counts = demographics.get("diagnosis_counts", [])
        diagnoses_line = ", ".join(f"{label} ({count})" for label, count in diagnosis_counts[:8]) if diagnosis_counts else ', '.join(demographics.get('diagnoses', []))

        life_stage_counts = demographics.get("life_stage_counts", [])
        life_stage_line = ", ".join(f"{label} ({count})" for label, count in life_stage_counts[:6]) if life_stage_counts else ', '.join(demographics.get('life_stages', []))

        location_counts = demographics.get("location_counts", [])
        location_line = ", ".join(f"{label} ({count})" for label, count in location_counts[:6]) if location_counts else ', '.join(demographics.get('locations', []))

        language_block = ""
        if language_profile:
            snippet = language_profile.get("prompt_block")
            if snippet:
                language_block = f"""

AUDIENCE LANGUAGE SNAPSHOT (use this tone, avoid the banned terms):
{snippet}
"""

        alignment_block = ""
        if alignment_profile:
            note = alignment_profile.get("alignment_note")
            confidence = alignment_profile.get("alignment_confidence", "adjacent")
            themes = alignment_profile.get("related_persona_themes") or []
            if note:
                themes_line = f"Highlight lived themes like: {', '.join(themes[:6])}." if themes else ""
                alignment_block = f"""

RESEARCH ALIGNMENT NOTE (confidence={confidence}):
{note}
{themes_line}

Spell out the bridge between the research mechanism and the neurodivergent cohort so downstream agents don't miss the relevance.
"""

        mechanism_bridge = ""
        if alignment_profile and alignment_profile.get("alignment_note"):
            mechanism_bridge = alignment_profile["alignment_note"]

        digest_block = ""
        if health_digest:
            headline = health_digest.get("headline")
            summary_bullets = health_digest.get("summary_bullets") or []
            flashpoints = health_digest.get("cultural_flashpoints") or []
            digest_lines = []
            if headline:
                digest_lines.append(f"Headline: {headline}")
            if summary_bullets:
                digest_lines.extend(f"- {bullet}" for bullet in summary_bullets[:4])
            if isinstance(flashpoints, list) and flashpoints:
                labels = []
                for flash in flashpoints[:4]:
                    if isinstance(flash, dict):
                        label = flash.get("label")
                        if label:
                            labels.append(label)
                if labels:
                    digest_lines.append(f"Cultural flashpoints to address: {', '.join(labels)}")
            prompt_block = health_digest.get("prompt_block")
            if prompt_block:
                digest_lines.append("Prompt block guidance:")
                digest_lines.append(prompt_block[:800])
            digest_block = "\n".join(digest_lines)

        health_digest_block = ""
        if digest_block:
            health_digest_block = f"""

ENLITENS HEALTH DIGEST SNAPSHOT (use this to stay in Liz's lane):
{digest_block}
"""
        
        prompt = f"""You are a public health analyst for Enlitens, a neurodivergent-affirming practice in St. Louis.

Your mission: Extract and synthesize ONLY the information from the St. Louis Health Report that is directly relevant to our current client cohort.

CLIENT COHORT DEMOGRAPHICS ({demographics['count']} clients):
- Age Range: {demographics['age_range']}
- Genders: {', '.join(demographics.get('genders', []))}
- Locations: {location_line or 'Unknown'}
- Primary Diagnoses: {diagnoses_line or 'Unknown'}
- Life Stages: {life_stage_line or 'Unknown'}
{language_block}
{alignment_block}

ST. LOUIS HEALTH REPORT:
{report_text}

YOUR TASK:
Create a targeted brief (500-1000 words) that connects this cohort to local health realities.

INCLUDE:
1. **Mechanism Bridge**: Explain in Liz's voice how the paper's mechanism ties to the lived experiences/themes above. Reference: "{mechanism_bridge or 'Use alignment note to connect the dots.'}"
2. **Demographics & Prevalence**: Local statistics for their age groups and conditions (cite specific numbers/percentages and mention whether it comes from the paper or the St. Louis digest).
3. **Environmental Factors**: St. Louis-specific factors affecting their health (air quality, noise, urban stress, etc.) with at least one concrete metric (e.g., pollution index, violent crime rate).
4. **Healthcare Access**: Barriers and resources specific to their needs in St. Louis. Include wait times, coverage rates, or provider availability numbers when available.
5. **Community Context**: Relevant local programs, support systems, third places, or gaps in care that match this cohort.
6. **Socioeconomic Factors**: How local economic conditions impact their access to care (income levels, unemployment rates, school funding numbers, etc.).
7. **Liz Voice Receipts**: Sprinkle short first-person riffs (e.g., “I’m not sugarcoating this…”) that echo the voice guide and call out banned terms if they appear in the source.

EXCLUDE:
- Data not relevant to this cohort (e.g., elderly care if cohort is young adults)
- Generic national statistics (focus on St. Louis)
- Conditions not represented in this cohort

FORMAT:
Deliver the brief using the following structure:
1. **Headline** – 1 sentence in Liz's voice (max 20 words).
2. **Mechanism Bridge** – 2-3 sentences tying the research mechanism to the cohort (mention the alignment themes).
3. **Key Numbers That Matter** – Bullet list, minimum 4 bullets, each with an explicit number + the stem “That means…” to translate impact.
4. **How It Lands For Our Personas** – 2 short paragraphs (or bullet clusters) referencing the persona diagnoses/life stages and weaving in local context.
5. **What We Need To Say Out Loud** – Bullet list of 3 Liz-voiced statements (no jargon, no mindfulness language).
6. **Local Solutions / Gaps** – Split into “What’s working” vs “Where we’re still exposed,” with at least one resource name or gap per side.

Keep sentences plain-spoken, direct, and grounded in the data above. Use first-person plural (“we”, “our clients”) or second-person where natural. Call out if the source material lacks a crucial stat so downstream agents know there’s a gap.

This brief will be used to ground all marketing and educational content in local reality.
{refinement_block}
{health_digest_block}
"""
        return prompt

