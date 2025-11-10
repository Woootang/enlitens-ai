"""
Health Report Synthesizer Agent
Creates targeted briefs from the St. Louis Health Report based on selected personas.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class HealthReportSynthesizerAgent:
    """Synthesizes relevant health report data for selected client personas."""
    
    def __init__(self):
        self.health_report_cache: str = ""
        
    async def synthesize_health_context(
        self,
        health_report_text: str,
        selected_personas: List[Dict[str, Any]],
        llm_client: Any
    ) -> str:
        """
        Create a targeted health brief connecting personas to local health data.
        
        Args:
            health_report_text: Full St. Louis health report text
            selected_personas: The 10 selected client personas
            llm_client: LLM client for synthesis
            
        Returns:
            Synthesized health brief (500-1000 words)
        """
        if not health_report_text or not selected_personas:
            logger.warning("Missing health report or personas for synthesis")
            return "No local health context available."
        
        # Extract persona demographics
        demographics = self._extract_demographics(selected_personas)
        
        # Create synthesis prompt
        prompt = self._create_synthesis_prompt(health_report_text, demographics)
        
        try:
            response = await llm_client.generate_text(
                prompt=prompt,
                temperature=0.4,
                num_predict=1500
            )
            
            logger.info(f"✅ Synthesized health context ({len(response)} chars)")
            return response
            
        except Exception as e:
            logger.error(f"Health report synthesis failed: {e}")
            return "Health context synthesis unavailable."
    
    def _extract_demographics(self, personas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract demographic summary from selected personas."""
        ages = []
        genders = []
        locations = []
        diagnoses = []
        life_stages = []
        
        for persona in personas:
            meta = persona.get('meta', {})
            current_life = persona.get('current_life_context', {})
            
            if meta.get('age'):
                ages.append(meta['age'])
            if meta.get('gender'):
                genders.append(meta['gender'])
            if meta.get('location'):
                locations.append(meta['location'])
            if meta.get('primary_diagnoses'):
                diagnoses.extend(meta['primary_diagnoses'])
            if current_life.get('life_stage'):
                life_stages.append(current_life['life_stage'])
        
        return {
            'age_range': f"{min(ages) if ages else 'Unknown'}-{max(ages) if ages else 'Unknown'}",
            'ages': ages,
            'genders': list(set(genders)),
            'locations': list(set(locations)),
            'diagnoses': list(set(diagnoses)),
            'life_stages': list(set(life_stages)),
            'count': len(personas)
        }
    
    def _create_synthesis_prompt(
        self,
        health_report_text: str,
        demographics: Dict[str, Any]
    ) -> str:
        """Create prompt for health report synthesis."""
        
        # Truncate health report if too long (keep first 8000 chars)
        report_text = health_report_text[:8000] + "..." if len(health_report_text) > 8000 else health_report_text
        
        prompt = f"""You are a public health analyst for Enlitens, a neurodivergent-affirming practice in St. Louis.

Your mission: Extract and synthesize ONLY the information from the St. Louis Health Report that is directly relevant to our current client cohort.

CLIENT COHORT DEMOGRAPHICS ({demographics['count']} clients):
- Age Range: {demographics['age_range']}
- Genders: {', '.join(demographics['genders'])}
- Locations: {', '.join(demographics['locations'])}
- Primary Diagnoses: {', '.join(demographics['diagnoses'])}
- Life Stages: {', '.join(demographics['life_stages'])}

ST. LOUIS HEALTH REPORT:
{report_text}

YOUR TASK:
Create a targeted brief (500-1000 words) that connects this cohort to local health realities.

INCLUDE:
1. **Demographics & Prevalence**: Local statistics for their age groups and conditions
2. **Environmental Factors**: St. Louis-specific factors affecting their health (air quality, noise, urban stress, etc.)
3. **Healthcare Access**: Barriers and resources specific to their needs in St. Louis
4. **Community Context**: Relevant local programs, support systems, or gaps in care
5. **Socioeconomic Factors**: How local economic conditions impact their access to care

EXCLUDE:
- Data not relevant to this cohort (e.g., elderly care if cohort is young adults)
- Generic national statistics (focus on St. Louis)
- Conditions not represented in this cohort

FORMAT:
Write in clear, direct language. Use bullet points for data. Connect each point back to how it impacts THIS specific cohort.

This brief will be used to ground all marketing and educational content in local reality.
"""
        return prompt

