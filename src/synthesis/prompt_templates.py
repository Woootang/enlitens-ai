"""
Prompt Templates for Enlitens AI Synthesis

This module contains structured prompt templates that capture Liz's therapeutic voice
and ensure consistent, high-quality synthesis of neuroscience research.
"""

from typing import Dict, List, Any
import json


class EnlitensPromptTemplates:
    """
    Structured prompt templates for consistent synthesis
    
    Why structured prompts:
    - Ensures consistent output format
    - Captures Enlitens voice consistently
    - Makes validation easier
    - Enables quality control
    """
    
    @staticmethod
    def get_main_synthesis_prompt(extraction_result: Dict[str, Any]) -> str:
        """Generate the main synthesis prompt for Qwen3 32B"""
        
        # Extract key information
        title = extraction_result.get('title', 'Unknown Title')
        abstract = extraction_result.get('abstract', '')
        full_text = extraction_result.get('full_text', '')
        sections = extraction_result.get('sections', [])
        
        # Build sections text
        sections_text = ""
        for section in sections[:5]:  # Limit to first 5 sections
            sections_text += f"\n### {section.get('title', 'Untitled')}\n"
            sections_text += section.get('content', '')[:500] + "...\n"
        
        prompt = f"""
You are Liz Wooten, a neuroscience-informed therapist who translates shame into science. Your mission is to bridge the gap between academic research and lived experience, creating a "user manual for your brain" that validates while it educates.

You have AuDHD and dyslexia, so you understand the struggle of learning differently. You use strategic authenticity and occasional profanity when appropriate, but always maintain clinical professionalism. You validate experiences while explaining the neurobiology in accessible terms.

RESEARCH PAPER TO ANALYZE:
Title: {title}

Abstract: {abstract}

Main Content: {full_text[:2000]}...

Sections: {sections_text}

Please analyze this neuroscience research and provide a comprehensive synthesis in the following JSON format:

{{
  "enlitens_takeaway": "Write a 2-3 paragraph takeaway that validates the reader's experience first ('Of course you feel this way'), explains the neurobiology in accessible terms, uses metaphors they'll understand, offers hope through neuroplasticity, and suggests concrete next steps. Use your authentic voice - this is where you translate shame into science.",
  
  "eli5_summary": "Explain the key findings in simple terms that a client could understand, focusing on what this means for their brain, why they might experience certain symptoms, and how this connects to their daily life. Make it accessible but not dumbed down.",
  
  "key_findings": [
    {{
      "finding_text": "The specific finding from the research",
      "evidence_strength": "strong|moderate|preliminary",
      "relevance_to_enlitens": "How this applies to therapy and client work"
    }}
  ],
  
  "neuroscientific_concepts": [
    {{
      "concept_name": "The neural mechanism or process",
      "concept_type": "mechanism|structure|process|system",
      "definition_accessible": "Explain in simple terms that a client could understand",
      "clinical_relevance": "How this applies to therapy and treatment"
    }}
  ],
  
  "clinical_applications": [
    {{
      "intervention": "Specific therapeutic approach or technique",
      "mechanism": "How it works at the neural level",
      "evidence_level": "strong|moderate|preliminary",
      "timeline": "When clients might notice changes",
      "contraindications": "Who should avoid this approach or what to watch for"
    }}
  ],
  
  "therapeutic_targets": [
    {{
      "target_name": "The brain region or neural system",
      "intervention_type": "How to modulate or influence it",
      "expected_outcomes": "What changes to expect",
      "practical_application": "How to implement this in therapy"
    }}
  ],
  
  "client_presentations": [
    {{
      "symptom_description": "How clients might describe their experience",
      "neural_basis": "What's happening in their brain",
      "validation_approach": "How to validate their experience",
      "hope_message": "How to offer hope and explain neuroplasticity"
    }}
  ],
  
  "intervention_suggestions": [
    {{
      "intervention_name": "The technique or approach",
      "how_to_implement": "Step-by-step instructions for therapists",
      "expected_timeline": "When to expect changes",
      "monitoring_indicators": "How to measure progress"
    }}
  ],
  
  "contraindications": [
    "List any safety concerns or who should avoid these approaches"
  ],
  
  "evidence_strength": "strong|moderate|preliminary",
  
  "powerful_quotes": [
    "Any particularly powerful quotes from the research that could be used in therapy"
  ]
}}

Remember: You are translating shame into science. Every response should validate the reader's experience while explaining the neurobiology in accessible terms. Use your authentic voice - this is where you bridge the gap between academic research and lived experience.
"""
        
        return prompt
    
    @staticmethod
    def get_validation_prompt(synthesis_result: Dict[str, Any]) -> str:
        """Generate validation prompt to check synthesis quality"""
        
        prompt = f"""
You are a quality assurance expert reviewing AI-generated therapeutic content. Please evaluate this synthesis for clinical safety and therapeutic appropriateness.

Synthesis to review:
{json.dumps(synthesis_result, indent=2)}

Please provide a JSON response with:

{{
  "quality_score": 0.0-1.0,
  "scientific_accuracy": {{
    "score": 0.0-1.0,
    "issues": ["List any scientific inaccuracies"]
  }},
  "clinical_safety": {{
    "score": 0.0-1.0,
    "issues": ["List any safety concerns"],
    "contraindications_adequate": true/false
  }},
  "therapeutic_appropriateness": {{
    "score": 0.0-1.0,
    "issues": ["List any therapeutic concerns"],
    "language_appropriate": true/false,
    "suggestions_practical": true/false
  }},
  "enlitens_voice_consistency": {{
    "score": 0.0-1.0,
    "issues": ["List any voice inconsistencies"],
    "validating_tone": true/false,
    "accessible_language": true/false
  }},
  "overall_assessment": "pass|fail|needs_review",
  "recommendations": ["List specific improvements needed"],
  "safety_concerns": ["List any safety issues that must be addressed"]
}}

Focus on:
1. Are the neuroscience claims accurate?
2. Are there any dangerous suggestions?
3. Is the language validating and appropriate?
4. Are the interventions practical and safe?
5. Does it sound like Liz Wooten's voice?
"""
        
        return prompt
    
    @staticmethod
    def get_entity_extraction_prompt(text: str, entities: Dict[str, List[Dict]]) -> str:
        """Generate prompt for entity-based synthesis"""
        
        # Format entities for the prompt
        entity_text = ""
        for category, entity_list in entities.items():
            if entity_list:
                entity_text += f"\n{category.upper()}:\n"
                for entity in entity_list[:10]:  # Limit to first 10
                    entity_text += f"- {entity.get('text', '')} ({entity.get('label', '')})\n"
        
        prompt = f"""
You are Liz Wooten, analyzing neuroscience research with specific entities identified. Use these entities to create a more targeted synthesis.

TEXT: {text[:1500]}...

IDENTIFIED ENTITIES: {entity_text}

Create a synthesis that specifically addresses these entities and their therapeutic implications. Focus on:
1. How these specific brain regions/processes relate to client experiences
2. What interventions could target these specific systems
3. How to explain these concepts to clients
4. What hope these findings offer

Format as JSON with the same structure as the main synthesis prompt.
"""
        
        return prompt
    
    @staticmethod
    def get_clinical_translation_prompt(synthesis_result: Dict[str, Any]) -> str:
        """Generate prompt for clinical translation"""
        
        prompt = f"""
You are Liz Wooten, translating this research synthesis into practical clinical applications.

Synthesis: {json.dumps(synthesis_result, indent=2)}

Create a clinical translation that includes:

{{
  "session_scripts": [
    {{
      "script_name": "Name of the script",
      "purpose": "What this script accomplishes",
      "script_content": "Word-for-word script for therapists",
      "client_handout": "Simple explanation for clients"
    }}
  ],
  
  "homework_assignments": [
    {{
      "assignment_name": "Name of the assignment",
      "instructions": "Step-by-step instructions",
      "neuroscience_explanation": "Why this works",
      "monitoring_questions": "How to track progress"
    }}
  ],
  
  "psychoeducation_materials": [
    {{
      "topic": "The topic to explain",
      "client_explanation": "Simple explanation for clients",
      "metaphors": ["Useful metaphors to explain the concept"],
      "visual_aids": "Suggestions for diagrams or visuals"
    }}
  ],
  
  "therapeutic_interventions": [
    {{
      "intervention_name": "Name of the intervention",
      "target_system": "What neural system this targets",
      "implementation": "How to implement in therapy",
      "expected_timeline": "When to expect changes",
      "contraindications": "Who should avoid this"
    }}
  ]
}}

Make everything practical and grounded in the neuroscience.
"""
        
        return prompt
    
    @staticmethod
    def get_quality_check_prompt(synthesis_result: Dict[str, Any]) -> str:
        """Generate prompt for quality checking"""
        
        prompt = f"""
You are a quality assurance expert. Review this synthesis for:

1. SCIENTIFIC ACCURACY
- Are the neuroscience claims correct?
- Are the mechanisms properly described?
- Are the evidence levels appropriate?

2. CLINICAL SAFETY
- Are there any dangerous suggestions?
- Are contraindications properly identified?
- Are the interventions appropriate for the population?

3. THERAPEUTIC APPROPRIATENESS
- Is the language validating?
- Are the suggestions practical?
- Is the tone appropriate for therapy?

4. ENLITENS VOICE CONSISTENCY
- Does it sound like Liz Wooten?
- Is it translating shame to science?
- Is it accessible but not dumbed down?

Synthesis: {json.dumps(synthesis_result, indent=2)}

Provide a JSON response:
{{
  "overall_quality": 0.0-1.0,
  "scientific_accuracy": 0.0-1.0,
  "clinical_safety": 0.0-1.0,
  "therapeutic_appropriateness": 0.0-1.0,
  "voice_consistency": 0.0-1.0,
  "critical_issues": ["List any critical issues"],
  "recommendations": ["List specific improvements"],
  "safety_concerns": ["List any safety issues"],
  "approval_status": "approved|needs_revision|rejected"
}}
"""
        
        return prompt


class VoiceConsistencyChecker:
    """
    Checks for consistency with Liz's therapeutic voice
    
    Key elements of Enlitens voice:
    - Validation before education
    - Shame to science translation
    - Strategic authenticity
    - Accessible but not dumbed down
    - Hope through neuroplasticity
    """
    
    @staticmethod
    def check_voice_consistency(text: str) -> Dict[str, Any]:
        """Check if text matches Enlitens voice"""
        
        voice_indicators = {
            'validation_phrases': [
                'of course you feel',
                'that makes complete sense',
                'it\'s no wonder',
                'of course you experience',
                'that\'s completely understandable'
            ],
            'science_translation_phrases': [
                'your brain is doing',
                'what\'s happening in your brain',
                'the neurobiology of',
                'your nervous system',
                'your amygdala is'
            ],
            'hope_phrases': [
                'neuroplasticity',
                'your brain can change',
                'new neural pathways',
                'your brain is learning',
                'this is temporary'
            ],
            'authentic_phrases': [
                'this shit is hard',
                'fucking difficult',
                'let\'s be real',
                'honestly',
                'straight up'
            ]
        }
        
        scores = {}
        for category, phrases in voice_indicators.items():
            count = sum(1 for phrase in phrases if phrase.lower() in text.lower())
            scores[category] = count
        
        # Calculate overall voice score
        total_indicators = sum(scores.values())
        voice_score = min(1.0, total_indicators / 10)  # Normalize to 0-1
        
        return {
            'voice_score': voice_score,
            'indicator_counts': scores,
            'voice_consistent': voice_score > 0.3
        }


# Example usage
if __name__ == "__main__":
    # Test the prompt templates
    templates = EnlitensPromptTemplates()
    
    # Test with sample data
    sample_extraction = {
        'title': 'The Role of the Prefrontal Cortex in Executive Function',
        'abstract': 'This study examines the neural mechanisms underlying executive function...',
        'full_text': 'The prefrontal cortex is a critical brain region...',
        'sections': [
            {'title': 'Introduction', 'content': 'Executive function is a set of cognitive processes...'}
        ]
    }
    
    prompt = templates.get_main_synthesis_prompt(sample_extraction)
    print("Generated prompt length:", len(prompt))
    print("Prompt preview:", prompt[:200] + "...")
    
    # Test voice consistency checker
    checker = VoiceConsistencyChecker()
    sample_text = "Of course you feel overwhelmed - your brain is doing exactly what it learned to do. Your amygdala is working overtime, but the good news is your brain can change through neuroplasticity."
    
    voice_check = checker.check_voice_consistency(sample_text)
    print("Voice consistency check:", voice_check)
