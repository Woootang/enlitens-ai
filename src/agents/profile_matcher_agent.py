"""
Profile Matcher Agent
Selects the top 10 most relevant client personas for a given research paper.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        
        return {
            'id': persona.get('_file', 'unknown'),
            'age': identity.get('age_range', 'Unknown'),
            'gender': identity.get('gender', 'Unknown'),
            'location': identity.get('locality', 'Unknown'),
            'diagnoses': diagnoses,
            'key_challenges': [
                safe_str(dev_story.get('early_childhood_experience', '')),
                safe_str(dev_story.get('school_experience', '')),
                safe_str(exec_func.get('executive_summary', '')),
                safe_str(current_life.get('current_stressors', ''))
            ],
            'sensory_profile': exec_func.get('sensory_sensitivities', []) if isinstance(exec_func.get('sensory_sensitivities'), list) else [],
            'life_stage': current_life.get('life_stage', 'Unknown'),
            'full_text': json.dumps(persona)  # For semantic search
        }
    
    async def select_top_personas(
        self,
        paper_text: str,
        entities: Dict[str, List[str]],
        llm_client: Any,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Select the top K most relevant personas for this research paper.
        
        Args:
            paper_text: Full text of the research paper
            entities: NER extracted entities (diseases, genes, treatments, etc.)
            llm_client: LLM client for intelligent matching
            top_k: Number of personas to select (default 10)
            
        Returns:
            List of top K most relevant personas (full objects)
        """
        personas = self.load_personas()
        
        if not personas:
            logger.warning("No personas available for matching")
            return []
        
        # Extract metadata for all personas
        personas_metadata = [self.extract_persona_metadata(p) for p in personas]
        
        # Create matching prompt
        prompt = self._create_matching_prompt(paper_text, entities, personas_metadata, top_k)
        
        # Use LLM to intelligently select personas
        try:
            logger.info(f"🎯 Starting persona matching with LLM...")
            
            # Chain of Thought: Explain the reasoning
            logger.info(f"🧠 THINKING: I have {len(personas)} personas to analyze. Looking at the paper's entities: {len(entities.get('diseases', []))} diseases, {len(entities.get('chemicals', []))} chemicals, {len(entities.get('symptoms', []))} symptoms. I'll match these against each persona's diagnoses, medications, and life challenges to find the top {top_k} most relevant.")
            
            response = await llm_client.generate_text(
                prompt=prompt,
                temperature=0.3,  # Lower temp for more consistent selection
                num_predict=2000
            )
            logger.info(f"✅ LLM generated response of length: {len(response)}")
            
            # Debug: Log the raw LLM response
            logger.info(f"🔍 LLM response for persona matching:\n{response[:1000]}")
            
            # Parse LLM response to get selected persona IDs
            selected_ids = self._parse_selection_response(response)
            logger.info(f"🔍 Parsed persona IDs: {selected_ids}")
            logger.info(f"🔍 Number of IDs parsed: {len(selected_ids)}")
            
            # Chain of Thought: Explain selection reasoning
            logger.info(f"🧠 THINKING: The LLM identified {len(selected_ids)} personas. Now matching these filenames against my loaded persona database to retrieve full profiles.")
            
            # Return full persona objects for selected IDs
            selected_personas = []
            for persona in personas:
                persona_file = persona.get('_file', '')
                # Extract just the filename for matching
                persona_filename = persona_file.split('/')[-1]
                logger.info(f"🔍 Checking persona file: {persona_filename} (from {persona_file})")
                if persona_filename in selected_ids:
                    selected_personas.append(persona)
                    logger.info(f"✅ Matched persona: {persona_filename}")
                    
            logger.info(f"✅ Selected {len(selected_personas)} relevant personas")
            
            # Chain of Thought: Summarize selection
            if len(selected_personas) > 0:
                diagnoses_found = set()
                for p in selected_personas[:3]:  # Sample first 3
                    neuro = p.get('neurodivergence_mental_health', {})
                    dx_list = neuro.get('diagnoses', [])
                    # Handle both strings and dicts in diagnoses
                    for dx in dx_list:
                        if isinstance(dx, str):
                            diagnoses_found.add(dx)
                        elif isinstance(dx, dict):
                            diagnoses_found.add(dx.get('condition', 'Unknown'))
                
                # Get disease list as strings
                disease_list = entities.get('diseases', [])
                disease_names = []
                for d in disease_list[:3]:
                    if isinstance(d, str):
                        disease_names.append(d)
                    elif isinstance(d, dict):
                        disease_names.append(d.get('text', 'Unknown'))
                
                logger.info(f"🧠 THINKING: Successfully matched {len(selected_personas)} personas. They include conditions like: {', '.join(list(diagnoses_found)[:5])}. These personas should align well with the paper's focus on {', '.join(disease_names)}.")
            
            if len(selected_personas) == 0:
                logger.error(f"❌ CRITICAL: NO PERSONAS SELECTED! This should not happen!")
                logger.error(f"Available persona files: {[p.get('_file') for p in personas[:5]]}")
                logger.error(f"Selected IDs from LLM: {selected_ids}")
                # Return first 10 as fallback
                logger.warning("🔥 FALLING BACK to first 10 personas")
                return personas[:top_k]
                
            return selected_personas[:top_k]
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Profile matching failed with exception: {e}")
            logger.exception(e)
            # Fallback: return first 10 personas
            logger.warning("🔥 FALLING BACK to first 10 personas due to exception")
            return personas[:top_k]
    
    def _create_matching_prompt(
        self,
        paper_text: str,
        entities: Dict[str, List[str]],
        personas_metadata: List[Dict[str, Any]],
        top_k: int
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
        personas_str = ""
        for i, p in enumerate(personas_metadata, 1):
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
        
        prompt = f"""You are an expert clinical intelligence analyst for Enlitens, a neurodivergent-affirming practice.

Your mission: Select the {top_k} client personas whose lived experiences are MOST relevant to this research paper.

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

OUTPUT FORMAT:
List the top {top_k} persona IDs (just the filenames) with brief justification:

1. [ID] - [One sentence why this persona is relevant]
2. [ID] - [One sentence why this persona is relevant]
...

Be strategic. These personas will shape ALL marketing and educational content for this paper.
"""
        return prompt
    
    def _parse_selection_response(self, response: str) -> List[str]:
        """Parse LLM response to extract selected persona IDs."""
        selected_ids = []
        
        # Look for patterns like "1. profile_001.json" or "profile_001.json -"
        lines = response.split('\n')
        for line in lines:
            # Try to find .json filenames
            if '.json' in line:
                # Extract the filename
                parts = line.split('.json')[0].split()
                if parts:
                    filename = parts[-1] + '.json'
                    # Clean up any leading numbers or punctuation
                    filename = filename.split('/')[-1]  # Get just the filename
                    if filename.startswith('profile_') or filename.startswith('persona_'):
                        selected_ids.append(filename)
        
        return selected_ids
    
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
            
            # Extract diagnoses
            diagnoses = []
            if isinstance(neuro_mh.get('formal_diagnoses'), list):
                diagnoses.extend([str(d) for d in neuro_mh.get('formal_diagnoses', [])])
            if isinstance(neuro_mh.get('self_identified_traits'), list):
                diagnoses.extend([str(d) for d in neuro_mh.get('self_identified_traits', [])])
            
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
            
            output += f"""## Persona {i}: {identity.get('age_range', 'Unknown')} {identity.get('gender', 'Unknown')} from {identity.get('locality', 'Unknown')}

**Diagnoses:** {diagnoses_str}

**Developmental Story:**
{safe_str(dev_story.get('early_childhood_experience', 'Not provided'))}

**Current Challenges:**
{safe_str(current_life.get('current_stressors', 'Not provided'))}

**Executive Function:**
{safe_str(exec_func.get('executive_summary', 'Not provided'), 200)}

**Sensory Profile:**
{safe_str(exec_func.get('sensory_sensitivities', 'Not provided'), 200)}

**Life Stage:** {current_life.get('life_stage', 'Not provided')}

---

"""
        
        return output

