"""
Profile Matcher Agent
Selects the top 10 most relevant client personas for a given research paper.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.llm_response_cleaner import (
    extract_json_object,
    extract_persona_filenames,
    strip_reasoning_artifacts,
)

logger = logging.getLogger(__name__)


class ProfileMatcherAgent:
    """Intelligent agent that selects the most relevant client personas for a research paper."""
    
    def __init__(self, personas_dir: str = "enlitens_client_profiles/profiles"):
        self.personas_dir = Path(personas_dir)
        self.personas_cache: Optional[List[Dict[str, Any]]] = None
        
    def load_personas(self) -> List[Dict[str, Any]]:
        """Load all client personas with metadata."""
        if self.personas_cache is not None:
            return self.personas_cache
            
        personas = []
        if not self.personas_dir.exists():
            logger.warning(f"Personas directory not found: {self.personas_dir}")
            return personas
            
        for persona_file in self.personas_dir.glob("*.json"):
            try:
                with open(persona_file, 'r') as f:
                    persona = json.load(f)
                    # Add file reference
                    persona['_file'] = str(persona_file)
                    personas.append(persona)
            except Exception as e:
                logger.warning(f"Failed to load persona {persona_file}: {e}")
                
        logger.info(f"✅ Loaded {len(personas)} client personas")
        self.personas_cache = personas
        return personas
    
    def extract_persona_metadata(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metadata from a persona for matching."""
        meta = persona.get('meta', {})
        identity = persona.get('identity_demographics', {})
        dev_story = persona.get('developmental_story', {})
        neuro_mh = persona.get('neurodivergence_mental_health', {})
        exec_func = persona.get('executive_function_sensory', {})
        current_life = persona.get('current_life_context', {})
        goals_barriers = persona.get('goals_barriers', {})
        
        # Extract text fields safely
        def safe_str(val):
            if isinstance(val, str):
                return val
            elif isinstance(val, dict):
                return str(val)
            elif isinstance(val, list):
                return ', '.join([str(v) for v in val])
            else:
                return str(val) if val else ''
        
        # Extract diagnoses safely
        diagnoses = []
        if isinstance(neuro_mh.get('formal_diagnoses'), list):
            diagnoses.extend([str(d) for d in neuro_mh.get('formal_diagnoses', [])])
        if isinstance(neuro_mh.get('self_identified_traits'), list):
            diagnoses.extend([str(d) for d in neuro_mh.get('self_identified_traits', [])])
        if isinstance(neuro_mh.get('identities'), list):
            diagnoses.extend([str(d) for d in neuro_mh.get('identities', [])])
        if isinstance(meta.get('primary_diagnoses'), list):
            diagnoses.extend([str(d) for d in meta.get('primary_diagnoses', [])])
        if isinstance(neuro_mh.get('diagnoses'), list):
            diagnoses.extend([str(d) for d in neuro_mh.get('diagnoses', [])])
        
        life_stage = current_life.get('life_stage') or identity.get('current_life_situation') or meta.get('life_stage')

        key_challenges: List[str] = []
        if isinstance(current_life.get('current_stressors'), list):
            key_challenges.extend([str(item) for item in current_life.get('current_stressors', [])])
        elif current_life.get('current_stressors'):
            key_challenges.append(safe_str(current_life.get('current_stressors')))

        if isinstance(goals_barriers.get('what_they_want_to_change'), list):
            key_challenges.extend([str(item) for item in goals_barriers.get('what_they_want_to_change', [])])

        if isinstance(dev_story.get('formative_adversities'), list):
            key_challenges.extend([str(item) for item in dev_story.get('formative_adversities', [])])

        return {
            'id': persona.get('_file', 'unknown'),
            'age': identity.get('age_range', 'Unknown'),
            'gender': identity.get('gender', 'Unknown'),
            'location': identity.get('locality', 'Unknown'),
            'diagnoses': diagnoses,
            'key_challenges': key_challenges or [
                safe_str(dev_story.get('early_childhood_experience', '')),
                safe_str(exec_func.get('executive_summary', '')),
            ],
            'sensory_profile': exec_func.get('sensory_sensitivities', []) if isinstance(exec_func.get('sensory_sensitivities'), list) else [],
            'life_stage': life_stage or 'Unknown',
            'full_text': json.dumps(persona)  # For semantic search
        }
    
    async def select_top_personas(
        self,
        paper_text: str,
        entities: Dict[str, List[str]],
        llm_client: Any,
        top_k: int = 10,
        refinement_feedback: Optional[str] = None,
        language_profile: Optional[Dict[str, Any]] = None,
        alignment_profile: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Select the top K most relevant personas for this research paper.
        """
        personas = self.load_personas()
        if not personas:
            raise RuntimeError("No personas available for matching.")

        personas_metadata = [self.extract_persona_metadata(p) for p in personas]
        attempt_policies = [
            {"max_personas": 30, "temperature": 0.40, "num_predict": 2000, "use_json_mode": False},
            {"max_personas": 25, "temperature": 0.35, "num_predict": 1800, "use_json_mode": False},
            {"max_personas": 20, "temperature": 0.30, "num_predict": 1600, "use_json_mode": False},
        ]

        retry_hint = refinement_feedback or ""
        failure_reasons: List[str] = []

        for attempt_index, policy in enumerate(attempt_policies, start=1):
            logger.info(
                "🎯 Starting persona matching with LLM (attempt %d/%d)...",
                attempt_index,
                len(attempt_policies),
            )
            logger.info(
                "🧠 THINKING: Evaluating %d personas with %d diseases, %d chemicals, %d symptoms.",
                len(personas),
                len(entities.get("diseases", [])),
                len(entities.get("chemicals", [])),
                len(entities.get("symptoms", [])),
            )

            prompt = self._create_matching_prompt(
                paper_text,
                entities,
                personas_metadata,
                top_k,
                refinement_feedback=refinement_feedback,
                language_profile=language_profile,
                alignment_profile=alignment_profile,
                max_personas=policy["max_personas"],
                retry_hint=retry_hint,
            )

            try:
                if policy.get("use_json_mode", False):
                    generation = await llm_client.generate_response(
                        prompt=prompt,
                        temperature=policy["temperature"],
                        num_predict=policy["num_predict"],
                        response_format="json_object",
                    )
                else:
                    generation = await llm_client.generate_response(
                        prompt=prompt,
                        temperature=policy["temperature"],
                        num_predict=policy["num_predict"],
                    )
            except Exception as exc:
                logger.warning(
                    "⚠️ Persona matching attempt %d failed to reach the model: %s",
                    attempt_index,
                    exc,
                )
                failure_reasons.append(f"transport error: {exc}")
                retry_hint = (
                    "Previous attempt failed due to a transport error. Respond with pure JSON as specified."
                )
                continue

            response_text = generation.get("response", "")
            logger.info("✅ LLM generated response of length: %d", len(response_text))
            logger.info("🔍 LLM response snapshot:\n%s", response_text[:1000])

            selected_ids, parse_issue = self._parse_selection_response(response_text)
            if selected_ids:
                logger.info("🔍 Parsed persona IDs: %s", selected_ids)
                matched_personas: List[Dict[str, Any]] = []
                known_filenames = {Path(p.get("_file", "")).name: p for p in personas if p.get("_file")}
                for persona_id in selected_ids:
                    persona = known_filenames.get(persona_id)
                    if persona:
                        matched_personas.append(persona)
                        persona.setdefault("meta", {}).setdefault("selection_meta", {})["llm_selected"] = True
                        logger.info("✅ Matched persona: %s", persona_id)

                if matched_personas:
                    if len(matched_personas) < top_k:
                        logger.warning(
                            "⚠️ Persona matcher returned only %d personas (requested %d). Padding with next-best matches.",
                            len(matched_personas),
                            top_k,
                        )
                        scored_candidates = self._score_personas_for_fallback(
                            personas_metadata=personas_metadata,
                            selected_ids=set(selected_ids),
                            entities=entities,
                            paper_text=paper_text,
                        )
                        for persona_id, score in scored_candidates:
                            persona = known_filenames.get(persona_id)
                            if not persona:
                                continue
                            persona.setdefault("meta", {}).setdefault("selection_meta", {})["fallback_score"] = score
                            matched_personas.append(persona)
                            logger.info("🔄 Padding with persona: %s (score=%.3f)", persona_id, score)
                            if len(matched_personas) >= top_k:
                                break
                    return matched_personas[:top_k]

                failure_reasons.append(
                    "LLM returned persona IDs that are not present in the catalog."
                )
                retry_hint = (
                    "Previous attempt referenced unknown persona IDs. Use the filenames provided in the prompt exactly."
                )
                continue

            reason = parse_issue or "no persona IDs parsed"
            logger.warning(
                "⚠️ Persona matching attempt %d yielded no IDs: %s", attempt_index, reason
            )
            failure_reasons.append(reason)
            retry_hint = (
                f"Previous attempt failed because {reason}. "
                "Return a JSON object containing a 'selected_persona_ids' array of persona filenames and optional 'justifications'. "
                "Do not output error objects or natural language."
            )

        failure_summary = "; ".join(failure_reasons) if failure_reasons else "unknown reasons"
        raise RuntimeError(f"Persona matcher failed after retries: {failure_summary}")

    def _score_personas_for_fallback(
        self,
        *,
        personas_metadata: List[Dict[str, Any]],
        selected_ids: set,
        entities: Dict[str, List[str]],
        paper_text: str,
    ) -> List[Tuple[str, float]]:
        entity_terms: List[str] = []
        for bucket in entities.values():
            for item in bucket:
                if not isinstance(item, str):
                    continue
                token = item.strip().lower()
                if token:
                    entity_terms.append(token)

        document_terms = set()
        for token in paper_text.lower().split():
            cleaned = token.strip().strip(",.()[]{}\"'")
            if len(cleaned) > 4:
                document_terms.add(cleaned)

        scored: List[Tuple[str, float]] = []
        for meta in personas_metadata:
            persona_id = Path(meta.get("id", "unknown")).name
            if persona_id in selected_ids:
                continue
            full_text = (meta.get("full_text") or "").lower()
            score = 0.0
            for term in entity_terms:
                if term and term in full_text:
                    score += 2.0
            for keyword in (meta.get("diagnoses") or []):
                if isinstance(keyword, str) and keyword.lower() in document_terms:
                    score += 1.5
            for challenge in (meta.get("key_challenges") or []):
                if isinstance(challenge, str):
                    hits = sum(1 for term in entity_terms if term and term in challenge.lower())
                    score += 0.75 * hits
            if not score:
                score = 0.5  # minimal baseline so we can still order deterministically
            scored.append((persona_id, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored
    
    def _create_matching_prompt(
        self,
        paper_text: str,
        entities: Dict[str, List[str]],
        personas_metadata: List[Dict[str, Any]],
        top_k: int,
        refinement_feedback: Optional[str] = None,
        language_profile: Optional[Dict[str, Any]] = None,
        alignment_profile: Optional[Dict[str, Any]] = None,
        *,
        max_personas: Optional[int] = None,
        retry_hint: Optional[str] = None,
    ) -> str:
        """Create the prompt for LLM-based persona matching."""
        
        # Extract paper summary (first 2000 chars)
        paper_summary = paper_text[:2000] + "..." if len(paper_text) > 2000 else paper_text
        
        # Format entities (safely convert to strings)
        entities_str = "\n".join([
            f"- {category.title()}: {', '.join([str(item) for item in items[:10]])}"
            for category, items in entities.items() if items
        ])
        
        # Format personas (condensed)
        persona_subset = personas_metadata if max_personas is None else personas_metadata[:max_personas]

        personas_str = ""
        for i, p in enumerate(persona_subset, 1):
            # Safely format diagnoses
            diagnoses_list = [str(d) for d in p['diagnoses']] if p['diagnoses'] else []
            diagnoses = ', '.join(diagnoses_list) if diagnoses_list else 'None listed'
            
            # Safely format challenges
            challenges_list = [str(c)[:50] for c in p['key_challenges'] if c]
            challenges = ', '.join(challenges_list)[:200] + '...' if challenges_list else 'Not provided'
            
            personas_str += f"""
{i}. ID: {p['id']}
   Age: {p['age']}, Gender: {p['gender']}, Location: {p['location']}
   Diagnoses: {diagnoses}
   Life Stage: {p['life_stage']}
   Key Challenges: {challenges}
"""
        
        refinement_block = ""
        if refinement_feedback:
            refinement_block = f"""

ADDITIONAL GUIDANCE FROM QUALITY REVIEW:
{refinement_feedback.strip()}

Adjust your selection to address the above gaps explicitly."""

        language_block = ""
        if language_profile:
            snippet = language_profile.get("prompt_block")
            if snippet:
                language_block = f"""

AUDIENCE LANGUAGE SNAPSHOT (align selections with these lived expressions):
{snippet}
"""

        alignment_block = ""
        if alignment_profile:
            note = alignment_profile.get("alignment_note")
            confidence = alignment_profile.get("alignment_confidence", "adjacent")
            themes = alignment_profile.get("related_persona_themes") or []
            if note:
                themes_line = f"Related persona themes to emphasise: {', '.join(themes[:6])}." if themes else ""
                alignment_block = f"""

RESEARCH ALIGNMENT NOTE (confidence={confidence}):
{note}
{themes_line}

If diagnoses are not explicit in the paper, pick personas whose lived experience demonstrates these mechanisms in real life. Clearly justify each match using the bridge above.
"""

        retry_block = ""
        if retry_hint:
            retry_block = f"""

IMPORTANT:
{retry_hint.strip()}
"""

        prompt = f"""You are an expert clinical intelligence analyst for Enlitens, a neurodivergent-affirming practice.

Your mission: Select EXACTLY {top_k} client personas whose lived experiences are MOST relevant to this research paper.

CRITICAL: You MUST return exactly {top_k} persona IDs. If fewer than {top_k} are highly relevant, include adjacent matches to reach the required count.

RESEARCH PAPER SUMMARY:
{paper_summary}

EXTRACTED ENTITIES:
{entities_str}

AVAILABLE CLIENT PERSONAS ({len(personas_metadata)} total):
{personas_str}

SELECTION CRITERIA (in priority order):
1. **Diagnosis Overlap**: Does the persona's diagnosis match the paper's focus?
2. **Life Stage Relevance**: Does the paper's context match their current life stage?
3. **Challenge Alignment**: Do their specific challenges relate to the paper's findings?
4. **Sensory/Executive Themes**: If the paper discusses sensory or executive function, prioritize those with related profiles.
5. **Diversity**: Select a diverse set (age, gender, life stage) to represent the full client base.
{refinement_block}
{language_block}
{alignment_block}
{retry_block}

OUTPUT FORMAT (STRICT JSON ONLY):

EXAMPLE 1 (for a paper about inflammation and stress):
{{
  "selected_persona_ids": [
    "persona_cluster_008_20251108_143624.json",
    "persona_cluster_005_20251108_143428.json",
    "persona_cluster_016_20251108_143157.json",
    "persona_cluster_023_20251108_143217.json",
    "persona_cluster_034_20251108_143459.json"
  ],
  "justifications": {{
    "persona_cluster_008_20251108_143624.json": "ADHD and trauma history align with chronic stress mechanisms.",
    "persona_cluster_005_20251108_143428.json": "Sensory processing challenges link to inflammation pathways."
  }}
}}

YOUR OUTPUT (copy this structure exactly, replace with your selected IDs):
{{
  "selected_persona_ids": [
    "REPLACE_WITH_ACTUAL_FILENAME_1.json",
    "REPLACE_WITH_ACTUAL_FILENAME_2.json",
    "REPLACE_WITH_ACTUAL_FILENAME_3.json",
    "REPLACE_WITH_ACTUAL_FILENAME_4.json",
    "REPLACE_WITH_ACTUAL_FILENAME_5.json"
  ],
  "justifications": {{
    "REPLACE_WITH_ACTUAL_FILENAME_1.json": "Brief explanation here."
  }}
}}

CRITICAL RULES:
- Return ONLY the JSON object above, nothing else
- Do NOT add any text before or after the JSON
- Do NOT return an `error` field
- Use the EXACT filenames from the persona list above
- Include exactly {top_k} persona IDs

Be strategic. These personas will shape ALL marketing and educational content for this paper.
"""
        return prompt
    
    def _parse_selection_response(self, response: str) -> Tuple[List[str], Optional[str]]:
        """Parse LLM response to extract selected persona IDs and optional error notes."""
        if not response:
            return [], "empty response"

        error_message: Optional[str] = None
        cleaned = strip_reasoning_artifacts(response or "")
        json_blob = extract_json_object(response)
        payload: Any = None
        if json_blob:
            try:
                payload = json.loads(json_blob)
            except json.JSONDecodeError as exc:
                error_message = error_message or f"invalid JSON ({exc})"
                payload = None

        if payload is None:
            try:
                payload = json.loads(cleaned)
            except json.JSONDecodeError:
                payload = None

        selected_ids: List[str] = []

        if isinstance(payload, dict):
            if payload.get("error"):
                error_message = str(payload.get("error"))
            candidate_lists = (
                payload.get("selected_persona_ids"),
                payload.get("personas"),
                payload.get("selected_personas"),
                payload.get("selections"),
                payload.get("ids"),
            )
            for candidate in candidate_lists:
                if isinstance(candidate, list):
                    provisional: List[str] = []
                    for item in candidate:
                        if isinstance(item, str):
                            provisional.append(Path(item).name)
                        elif isinstance(item, dict):
                            identifier = item.get("id") or item.get("persona_id")
                            if isinstance(identifier, str):
                                provisional.append(Path(identifier).name)
                    provisional = [pid for pid in provisional if pid.startswith(("persona_", "profile_"))]
                    if provisional:
                        selected_ids = provisional
                        break
            if not selected_ids and isinstance(payload.get("justifications"), dict):
                selected_ids = [
                    Path(str(key)).name
                    for key in payload["justifications"].keys()
                    if str(key).startswith(("persona_", "profile_"))
                ]
        elif isinstance(payload, list):
            selected_ids = [
                Path(str(item)).name
                for item in payload
                if isinstance(item, str) and str(item).startswith(("persona_", "profile_"))
            ]

        if selected_ids:
            return selected_ids, error_message

        regex_matches = extract_persona_filenames(response or "")
        if regex_matches:
            fallback_ids: List[str] = []
            for match in regex_matches:
                candidate_name = Path(match).name
                if candidate_name not in fallback_ids:
                    fallback_ids.append(candidate_name)
            if fallback_ids:
                return fallback_ids, error_message

        if error_message:
            return [], error_message

        lowered = cleaned.strip().lower()
        if lowered.startswith("error"):
            return [], cleaned.strip()

        return [], "unable to locate persona filenames in response"
    
    def get_selected_personas_text(self, personas: List[Dict[str, Any]]) -> str:
        """
        Convert selected personas to formatted text for main agent context.
        
        Returns a concise but complete representation of each persona.
        """
        if not personas:
            return "No personas selected."
        
        output = f"# SELECTED CLIENT PERSONAS ({len(personas)} profiles)\n\n"
        
        for i, persona in enumerate(personas, 1):
            meta = persona.get('meta', {})
            identity = persona.get('identity_demographics', {})
            dev_story = persona.get('developmental_story', {})
            neuro_mh = persona.get('neurodivergence_mental_health', {})
            exec_func = persona.get('executive_function_sensory', {})
            current_life = persona.get('current_life_context', {})
            goals_barriers = persona.get('goals_barriers', {})
            narrative = persona.get('narrative_voice', {})
            
            # Extract diagnoses
            diagnoses: List[str] = []
            for key in ("formal_diagnoses", "self_identified_traits", "identities", "diagnoses"):
                value = neuro_mh.get(key)
                if isinstance(value, list):
                    diagnoses.extend([str(item) for item in value])
            if isinstance(meta.get('primary_diagnoses'), list):
                diagnoses.extend([str(d) for d in meta.get('primary_diagnoses', [])])

            diagnoses_str = ', '.join(diagnoses) if diagnoses else 'None listed'
            
            # Safe string extraction
            def safe_str(val, max_len=300):
                if isinstance(val, str):
                    return val[:max_len] + ('...' if len(val) > max_len else '')
                elif isinstance(val, dict):
                    return str(val)[:max_len] + '...'
                elif isinstance(val, list):
                    return ', '.join([str(v) for v in val])[:max_len] + '...'
                else:
                    return 'Not provided'

            def bulletize(items: List[Any], max_items: int = 4) -> str:
                cleaned = [str(item).strip() for item in items if str(item).strip()]
                cleaned = cleaned[:max_items]
                return "\n".join([f"- {item[:200]}" + ("..." if len(item) > 200 else "") for item in cleaned]) if cleaned else "None provided"

            developmental_highlights = bulletize(
                [
                    dev_story.get('childhood_environment'),
                    dev_story.get('adolescence'),
                    dev_story.get('early_adulthood'),
                ],
                max_items=3,
            )

            exec_challenges = bulletize(exec_func.get('ef_friction_points', []) if isinstance(exec_func.get('ef_friction_points'), list) else [exec_func.get('executive_summary')])
            sensory_details = safe_str(exec_func.get('sensory_profile', exec_func.get('sensory_sensitivities', 'Not provided')), 300)
            current_stressors = bulletize(current_life.get('local_stressors', [])) if isinstance(current_life.get('local_stressors'), list) else safe_str(current_life.get('current_stressors', 'Not provided'), 300)
            goals_change = bulletize(goals_barriers.get('what_they_want_to_change', [])) if isinstance(goals_barriers.get('what_they_want_to_change'), list) else "None provided"
            support_system = bulletize(current_life.get('support_system', [])) if isinstance(current_life.get('support_system'), list) else safe_str(current_life.get('support_system', 'Not provided'), 200)
            quote_candidates: List[Any] = [narrative.get('quote_struggle'), narrative.get('quote_hope')]
            additional_quotes = narrative.get('quotes_additional')
            if isinstance(additional_quotes, list):
                quote_candidates.extend(additional_quotes)
            quotes = bulletize(quote_candidates, max_items=3)

            life_stage = current_life.get('life_stage') or identity.get('current_life_situation') or meta.get('life_stage', 'Not provided')
            
            output += f"""## Persona {i}: {identity.get('age_range', 'Unknown')} {identity.get('gender', 'Unknown')} from {identity.get('locality', 'Unknown')}

**Diagnoses:** {diagnoses_str}

**Developmental Highlights:**
{developmental_highlights}

**Current Challenges / Stressors:**
{current_stressors}

**Executive Function Friction Points:**
{exec_challenges}

**Sensory Profile:**
{sensory_details}

**Goals For Change:**
{goals_change}

**Support System:**
{support_system}

**Client Voice Samples:**
{quotes}

**Life Stage:** {life_stage}

---

"""
        
        return output

