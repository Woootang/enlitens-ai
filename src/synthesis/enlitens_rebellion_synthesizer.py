"""
Enlitens Rebellion Synthesizer - The Voice of the Rebellion

This synthesizer translates research into the voice of the rebellion:
- Neurobiological truth over pathology
- Strengths-based insights
- System critique and validation
- Aha moments for clients
- Content for each of the 5 Enlitens modules
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class EnlitensRebellionSynthesizer:
    """
    Synthesizer that translates research into the voice of the rebellion.
    
    This synthesizer:
    - Speaks in the voice of the rebellion
    - Focuses on neurobiological truth over pathology
    - Extracts strengths-based insights
    - Creates aha moments for clients
    - Organizes content for the 5 Enlitens modules
    """
    
    def __init__(self):
        self.rebellion_voice = {
            'tone': 'direct, authentic, rebellious',
            'language': 'profanity when appropriate, clinical when needed',
            'focus': 'neurobiological truth, strengths, system critique',
            'purpose': 'empowerment, validation, insight'
        }
    
    async def synthesize_rebellion_content(self, extraction_result: Dict[str, Any], 
                                        rebellion_content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize research into rebellion-focused content.
        
        Args:
            extraction_result: Raw extraction result
            rebellion_content: Rebellion-focused content
            
        Returns:
            Synthesized rebellion content
        """
        try:
            logger.info("Enlitens Rebellion Synthesizer: Starting synthesis")
            
            # Get basic info
            title = extraction_result.get('source_metadata', {}).get('title', 'Unknown Title')
            abstract = extraction_result.get('source_metadata', {}).get('abstract', '')
            
            # Synthesize for each module
            synthesis_result = {
                'enlitens_takeaway': await self._create_enlitens_takeaway(title, abstract, rebellion_content),
                'module_1_narrative': await self._synthesize_module_1(rebellion_content.get('module_1_narrative', {})),
                'module_2_sensory': await self._synthesize_module_2(rebellion_content.get('module_2_sensory', {})),
                'module_3_executive': await self._synthesize_module_3(rebellion_content.get('module_3_executive', {})),
                'module_4_social': await self._synthesize_module_4(rebellion_content.get('module_4_social', {})),
                'module_5_strengths': await self._synthesize_module_5(rebellion_content.get('module_5_strengths', {})),
                'rebellion_themes': await self._synthesize_rebellion_themes(rebellion_content.get('rebellion_themes', {})),
                'aha_moments': await self._synthesize_aha_moments(rebellion_content.get('aha_moments', [])),
                'clinical_applications': await self._synthesize_clinical_applications(rebellion_content),
                'system_critique': await self._synthesize_system_critique(rebellion_content)
            }
            
            logger.info("Enlitens Rebellion Synthesizer: Completed synthesis")
            return synthesis_result
            
        except Exception as e:
            logger.error(f"Enlitens Rebellion Synthesizer: Synthesis failed: {e}")
            return {}
    
    async def _create_enlitens_takeaway(self, title: str, abstract: str, rebellion_content: Dict[str, Any]) -> str:
        """Create the main Enlitens takeaway"""
        takeaway = f"""
## ENLITENS REBELLION TAKEAWAY

**Research:** {title}

**The Fucking Truth:** {abstract[:200]}...

**Why This Matters for the Rebellion:**
This research provides neurobiological evidence that supports our core belief: people aren't broken, systems are. The findings validate the lived experience of our clients and give us scientific ammunition against the pathologizing bullshit of the current system.

**The Aha Moment for Clients:**
This research can create that powerful "holy shit, it wasn't my fault" moment that transforms shame into science. It validates their experience and gives them the neurobiological truth they need to rewrite their story.

**Clinical Power:**
This isn't just interesting research - it's a weapon in the fight against professional gaslighting. It gives us the scientific evidence to validate our clients' experiences and challenge the system that has failed them.
"""
        return takeaway.strip()
    
    async def _synthesize_module_1(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize content for Module 1: Narrative & Systemic Deconstruction"""
        synthesis = {
            'narrative_insights': [],
            'systemic_critiques': [],
            'validation_evidence': [],
            'clinical_applications': []
        }
        
        # Process narrative insights
        for insight in content.get('narrative_insights', []):
            synthesis['narrative_insights'].append({
                'insight': insight,
                'rebellion_angle': 'This validates the client\'s story and challenges the pathologizing narrative',
                'clinical_use': 'Use this to validate the client\'s experience and reframe their narrative'
            })
        
        # Process systemic critiques
        for critique in content.get('systemic_critiques', []):
            synthesis['systemic_critiques'].append({
                'critique': critique,
                'rebellion_angle': 'This exposes the systemic failures that created the client\'s struggles',
                'clinical_use': 'Use this to shift blame from the client to the system that failed them'
            })
        
        return synthesis
    
    async def _synthesize_module_2(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize content for Module 2: Sensory & Autonomic Profiling"""
        synthesis = {
            'sensory_insights': [],
            'polyvagal_evidence': [],
            'regulation_strategies': [],
            'clinical_applications': []
        }
        
        # Process sensory insights
        for insight in content.get('sensory_insights', []):
            synthesis['sensory_insights'].append({
                'insight': insight,
                'rebellion_angle': 'This validates the client\'s sensory experience as neurobiological reality, not "overreacting"',
                'clinical_use': 'Use this to validate their sensory experience and create a sensory map'
            })
        
        # Process polyvagal evidence
        for evidence in content.get('polyvagal_evidence', []):
            synthesis['polyvagal_evidence'].append({
                'evidence': evidence,
                'rebellion_angle': 'This provides scientific proof that their nervous system is working exactly as designed',
                'clinical_use': 'Use this to explain their autonomic responses and build regulation strategies'
            })
        
        return synthesis
    
    async def _synthesize_module_3(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize content for Module 3: Executive Function & Cognitive Dynamics"""
        synthesis = {
            'executive_insights': [],
            'dopamine_evidence': [],
            'cognitive_load': [],
            'clinical_applications': []
        }
        
        # Process executive insights
        for insight in content.get('executive_insights', []):
            synthesis['executive_insights'].append({
                'insight': insight,
                'rebellion_angle': 'This validates their cognitive experience as neurobiological reality, not "laziness"',
                'clinical_use': 'Use this to reframe their struggles as fuel system issues, not character flaws'
            })
        
        # Process dopamine evidence
        for evidence in content.get('dopamine_evidence', []):
            synthesis['dopamine_evidence'].append({
                'evidence': evidence,
                'rebellion_angle': 'This proves their brain is a high-performance engine that needs specific fuel',
                'clinical_use': 'Use this to explain their motivation patterns and build interest-based strategies'
            })
        
        return synthesis
    
    async def _synthesize_module_4(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize content for Module 4: Social Processing & Communication"""
        synthesis = {
            'social_insights': [],
            'double_empathy_evidence': [],
            'masking_cost': [],
            'clinical_applications': []
        }
        
        # Process social insights
        for insight in content.get('social_insights', []):
            synthesis['social_insights'].append({
                'insight': insight,
                'rebellion_angle': 'This validates their social experience as neurobiological reality, not "social awkwardness"',
                'clinical_use': 'Use this to reframe their social struggles as communication differences, not deficits'
            })
        
        # Process masking cost
        for cost in content.get('masking_cost', []):
            synthesis['masking_cost'].append({
                'cost': cost,
                'rebellion_angle': 'This validates their exhaustion as the predictable result of cognitive overload',
                'clinical_use': 'Use this to validate their fatigue and build energy management strategies'
            })
        
        return synthesis
    
    async def _synthesize_module_5(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize content for Module 5: Strengths Synthesis & Strategic Planning"""
        synthesis = {
            'strengths_evidence': [],
            'resilience_factors': [],
            'strategic_insights': [],
            'clinical_applications': []
        }
        
        # Process strengths evidence
        for strength in content.get('strengths_evidence', []):
            synthesis['strengths_evidence'].append({
                'strength': strength,
                'rebellion_angle': 'This validates their capabilities and challenges the deficit model',
                'clinical_use': 'Use this to build their strengths-based identity and strategic plan'
            })
        
        # Process resilience factors
        for factor in content.get('resilience_factors', []):
            synthesis['resilience_factors'].append({
                'factor': factor,
                'rebellion_angle': 'This validates their survival skills and adaptive strategies',
                'clinical_use': 'Use this to honor their resilience and build on their existing strengths'
            })
        
        return synthesis
    
    async def _synthesize_rebellion_themes(self, themes: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize rebellion themes"""
        synthesis = {
            'neurobiological_truth': [],
            'strengths_based': [],
            'system_critique': [],
            'validation_evidence': []
        }
        
        # Process neurobiological truth
        for truth in themes.get('neurobiological_truth', []):
            synthesis['neurobiological_truth'].append({
                'truth': truth,
                'rebellion_angle': 'This provides scientific evidence for their neurobiological reality',
                'clinical_use': 'Use this to validate their experience with scientific proof'
            })
        
        # Process strengths-based content
        for strength in themes.get('strengths_based', []):
            synthesis['strengths_based'].append({
                'strength': strength,
                'rebellion_angle': 'This challenges the deficit model and validates their capabilities',
                'clinical_use': 'Use this to build their strengths-based identity'
            })
        
        return synthesis
    
    async def _synthesize_aha_moments(self, aha_moments: List[str]) -> List[Dict[str, Any]]:
        """Synthesize aha moments"""
        synthesis = []
        
        for moment in aha_moments:
            synthesis.append({
                'moment': moment,
                'rebellion_angle': 'This can create a powerful "holy shit" moment for the client',
                'clinical_use': 'Use this to facilitate insight and paradigm shifts'
            })
        
        return synthesis
    
    async def _synthesize_clinical_applications(self, rebellion_content: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize clinical applications"""
        applications = {
            'interventions': [],
            'assessments': [],
            'outcomes': [],
            'protocols': []
        }
        
        # Extract intervention ideas
        for module_content in rebellion_content.values():
            if isinstance(module_content, dict):
                for key, value in module_content.items():
                    if isinstance(value, list) and len(value) > 0:
                        applications['interventions'].append({
                            'type': key,
                            'content': value[0] if value else '',
                            'clinical_use': f'Use this for {key} interventions'
                        })
        
        return applications
    
    async def _synthesize_system_critique(self, rebellion_content: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize system critique"""
        critique = {
            'systemic_failures': [],
            'pathologizing_evidence': [],
            'validation_evidence': [],
            'rebellion_ammunition': []
        }
        
        # Extract system critique content
        for module_content in rebellion_content.values():
            if isinstance(module_content, dict):
                for key, value in module_content.items():
                    if isinstance(value, list) and len(value) > 0:
                        critique['systemic_failures'].append({
                            'failure': key,
                            'evidence': value[0] if value else '',
                            'rebellion_angle': 'This exposes how the system has failed our clients'
                        })
        
        return critique
