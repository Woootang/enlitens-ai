"""
AI Synthesis Engine for Enlitens Pipeline

This module uses Qwen2.5 32B (served via vLLM) to synthesize neuroscience research into therapeutic applications.
Key features:
- Structured prompts for consistent output
- Enlitens voice (shame to science translation)
- Clinical translation from research to practice
- Quality validation for scientific accuracy
"""

import hashlib
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests
from dataclasses import dataclass, field

from src.synthesis.prompts import StructuredSynthesisPrompts
from src.validation.citation_verifier import CitationVerifier

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Result of AI synthesis"""
    enlitens_takeaway: str
    eli5_summary: str
    key_findings: List[Dict[str, Any]]
    neuroscientific_concepts: List[Dict[str, Any]]
    clinical_applications: List[Dict[str, Any]]
    therapeutic_targets: List[Dict[str, Any]]
    client_presentations: List[Dict[str, Any]]
    intervention_suggestions: List[Dict[str, Any]]
    contraindications: List[str]
    evidence_strength: str
    quality_score: float
    synthesis_timestamp: str
    powerful_quotes: List[str] = field(default_factory=list)
    source_citations: List[Dict[str, Any]] = field(default_factory=list)
    prompt_text: Optional[str] = None
    validation_issues: Optional[List[str]] = None


class OllamaClient:
    """Compatibility wrapper that now targets the vLLM OpenAI server.

    The class name is preserved to avoid refactoring the wider pipeline, but
    all requests are routed through the vLLM REST API that serves the
    Qwen2.5-32B Q4_K_M model with paged attention and FlashAttention enabled.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "/home/antons-gs/enlitens-ai/models/mistral-7b-instruct",
        *,
        gpu_memory_utilization: float = 0.92,
        system_prompt: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = requests.Session()
        self.gpu_memory_utilization = gpu_memory_utilization
        self.system_prompt = system_prompt or (
            "You are the Enlitens neuroscience synthesis engine. "
            "Respond with structured JSON that matches the requested schema."
        )
        self._prefix_cache_key = hashlib.sha256(
            f"{self.model}::{self.system_prompt}".encode("utf-8")
        ).hexdigest()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4000,
    ) -> str:
        """Generate text using the vLLM OpenAI-compatible endpoint."""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {
                "cache_prompt": True,
                "prompt_cache_key": self._prefix_cache_key,
                "stream": False,
                "best_of": 1,
            },
        }

        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        except Exception as exc:
            logger.error(f"Failed to generate with vLLM: {exc}")
            return ""

    def is_available(self) -> bool:
        """Check if the vLLM server is reachable."""

        try:
            response = self.session.get(f"{self.base_url}/models", timeout=5)
            response.raise_for_status()
            return True
        except Exception:
            return False


class PromptTemplates:
    """
    Structured prompt templates for consistent synthesis
    
    Why structured prompts:
    - Ensures consistent output format
    - Captures Enlitens voice consistently
    - Makes validation easier
    - Enables quality control
    """
    
    @staticmethod
    def get_synthesis_prompt(extraction_result: Dict[str, Any]) -> str:
        """Generate the main synthesis prompt"""
        
        # Extract key information
        title = extraction_result.get('title', 'Unknown Title')
        abstract = extraction_result.get('abstract', '')
        full_text = extraction_result.get('full_text', '')
        sections = extraction_result.get('sections', [])
        
        prompt = f"""
You are Liz Wooten, a neuroscience-informed therapist who translates shame into science. Your mission is to bridge the gap between academic research and lived experience, creating a "user manual for your brain" that validates while it educates.

RESEARCH PAPER TO ANALYZE:
Title: {title}

Abstract: {abstract}

Please analyze this neuroscience research and provide a comprehensive synthesis in the following format:

## ENLITENS TAKEAWAY
Write a 2-3 paragraph takeaway that:
- Validates the reader's experience first ("Of course you feel this way")
- Explains the neurobiology in accessible terms
- Uses metaphors they'll understand
- Offers hope through neuroplasticity
- Suggests concrete next steps

## ELI5 SUMMARY
Explain the key findings in simple terms that a client could understand, focusing on:
- What this means for their brain
- Why they might experience certain symptoms
- How this connects to their daily life

## KEY FINDINGS
Extract 3-5 key findings with:
- finding_text: The specific finding
- evidence_strength: "strong", "moderate", or "preliminary"
- relevance_to_enlitens: How this applies to therapy

## NEUROSCIENTIFIC CONCEPTS
Identify 3-5 key concepts with:
- concept_name: The neural mechanism or process
- concept_type: "mechanism", "structure", "process", or "system"
- definition_accessible: Explain in simple terms
- clinical_relevance: How this applies to therapy

## CLINICAL APPLICATIONS
Identify therapeutic applications with:
- intervention: Specific therapeutic approach
- mechanism: How it works at the neural level
- evidence_level: "strong", "moderate", or "preliminary"
- timeline: When clients might notice changes
- contraindications: Who should avoid this approach

## THERAPEUTIC TARGETS
Identify specific neural targets with:
- target_name: The brain region or system
- intervention_type: How to modulate it
- expected_outcomes: What changes to expect
- practical_application: How to implement

## CLIENT PRESENTATIONS
Describe how clients might experience this with:
- symptom_description: How they might describe it
- neural_basis: What's happening in their brain
- validation_approach: How to validate their experience
- hope_message: How to offer hope

## INTERVENTION SUGGESTIONS
Provide specific interventions with:
- intervention_name: The technique or approach
- how_to_implement: Step-by-step instructions
- expected_timeline: When to expect changes
- monitoring_indicators: How to measure progress

## CONTRAINDICATIONS
List any safety concerns or who should avoid these approaches.

## EVIDENCE STRENGTH
Rate the overall evidence strength as "strong", "moderate", or "preliminary".

Remember: You are translating shame into science. Every response should validate the reader's experience while explaining the neurobiology in accessible terms. Use strategic authenticity and occasional profanity when appropriate, but always maintain clinical professionalism.

Format your response as JSON with the above structure.
"""
        
        return prompt
    
    @staticmethod
    def get_validation_prompt(synthesis_result: Dict[str, Any]) -> str:
        """Generate validation prompt to check synthesis quality"""
        
        prompt = f"""
You are a quality assurance expert reviewing AI-generated therapeutic content. Please evaluate this synthesis for:

1. SCIENTIFIC ACCURACY
- Are the neuroscience claims accurate?
- Are the mechanisms correctly described?
- Are the evidence levels appropriate?

2. CLINICAL SAFETY
- Are there any dangerous suggestions?
- Are contraindications properly identified?
- Are the interventions appropriate?

3. THERAPEUTIC APPROPRIATENESS
- Is the language validating?
- Are the suggestions practical?
- Is the tone appropriate for therapy?

4. ENLITENS VOICE CONSISTENCY
- Does it sound like Liz Wooten?
- Is it translating shame to science?
- Is it accessible but not dumbed down?

Please provide:
- Overall quality score (0-1)
- Specific issues found
- Recommendations for improvement
- Safety concerns (if any)

Synthesis to review:
{json.dumps(synthesis_result, indent=2)}
"""
        
        return prompt


class NeuroscienceSynthesizer:
    """
    Main synthesis engine using Qwen3 32B
    
    Why this approach:
- Qwen2.5 32B has excellent reasoning capabilities
- vLLM's paged attention keeps long-context reasoning efficient
    - Local processing ensures privacy
    - Can be fine-tuned for Enlitens voice
    """
    
    def __init__(self, ollama_client: OllamaClient):
        self.ollama = ollama_client
        self.templates = PromptTemplates()
        self.prompts = StructuredSynthesisPrompts()
        self.citation_verifier = CitationVerifier()

    def synthesize(
        self,
        extraction_result: Dict[str, Any],
        retriever: Optional[Any] = None,
    ) -> SynthesisResult:
        """
        Synthesize neuroscience research into therapeutic applications
        
        Args:
            extraction_result: Result from PDF extraction
            
        Returns:
            SynthesisResult with therapeutic applications
        """
        logger.info("Starting AI synthesis")
        
        try:
            if retriever is not None:
                staged_result = self._two_stage_synthesis(extraction_result, retriever)
                if staged_result:
                    return staged_result

            logger.info("Falling back to legacy single prompt synthesis")
            return self._legacy_synthesis(extraction_result)

            # Generate synthesis prompt
            prompt = self.templates.get_synthesis_prompt(extraction_result)
            
            # Get synthesis from Qwen2.5

            # Get synthesis from Qwen3
            synthesis_text = self.ollama.generate(prompt, temperature=0.3)
            
            if not synthesis_text:
                logger.error("Failed to generate synthesis")
                return self._create_empty_result()
            
            # Parse JSON response
            synthesis_data = self._parse_synthesis_response(synthesis_text)
            
            # Validate synthesis
            validation_result = self._validate_synthesis(synthesis_data)
            validation_issues = []
            if isinstance(validation_result, dict):
                issues = validation_result.get('issues')
                if isinstance(issues, list):
                    validation_issues = issues

            # Create result object
            result = SynthesisResult(
                enlitens_takeaway=synthesis_data.get('enlitens_takeaway', ''),
                eli5_summary=synthesis_data.get('eli5_summary', ''),
                key_findings=synthesis_data.get('key_findings', []),
                neuroscientific_concepts=synthesis_data.get('neuroscientific_concepts', []),
                clinical_applications=synthesis_data.get('clinical_applications', []),
                therapeutic_targets=synthesis_data.get('therapeutic_targets', []),
                client_presentations=synthesis_data.get('client_presentations', []),
                intervention_suggestions=synthesis_data.get('intervention_suggestions', []),
                contraindications=synthesis_data.get('contraindications', []),
                evidence_strength=synthesis_data.get('evidence_strength', 'preliminary'),
                quality_score=validation_result.get('quality_score', 0.0) if isinstance(validation_result, dict) else 0.0,
                synthesis_timestamp=datetime.now().isoformat(),
                prompt_text=prompt,
                validation_issues=validation_issues
            )
            
            logger.info(f"Synthesis completed with quality score {result.quality_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Synthesis failed: {str(e)}")
            return self._create_empty_result()

    def _two_stage_synthesis(self, extraction_result: Dict[str, Any], retriever: Any) -> Optional[SynthesisResult]:
        title = extraction_result.get('metadata', {}).get('title') or extraction_result.get('title', 'Unknown Title')
        abstract = extraction_result.get('abstract', '')
        query = title or extraction_result.get('metadata', {}).get('doi', '')

        retrieval_results = retriever.retrieve(query, top_k=10)
        if not retrieval_results:
            logger.warning("Hybrid retriever returned no results for %s", query)
            return None

        context_chunks = [
            {
                **(result.get('payload', {}) or {}),
                'text': result.get('text', ''),
                'chunk_id': result.get('chunk_id')
            }
            for result in retrieval_results
        ]

        stage_one_prompt = self.prompts.verbatim_prompt(title, query, context_chunks)
        stage_one_raw = self.ollama.generate(stage_one_prompt, temperature=0.0)
        quotes = self._parse_stage_one_quotes(stage_one_raw)

        if not quotes:
            logger.warning("Stage-one extraction failed, falling back to top chunks")
            quotes = self._fallback_quotes(context_chunks)

        verification_feedback: List[str] = []
        final_payload: Optional[Dict[str, Any]] = None
        verification_passed = False

        for attempt in range(2):
            stage_two_prompt = self.prompts.stage_two_prompt(title, abstract, quotes, verification_feedback)
            stage_two_raw = self.ollama.generate(stage_two_prompt, temperature=0.2)
            payload = self._parse_synthesis_response(stage_two_raw)
            payload.setdefault('source_citations', quotes)

            verification_passed, issues = self.citation_verifier.verify(payload, quotes)
            if verification_passed:
                final_payload = payload
                break

            verification_feedback = [
                f"- {issue}" for issue in issues
            ]
            final_payload = payload

        if final_payload is None:
            return None

        quality = 0.95 if verification_passed else 0.7
        return self._build_result(final_payload, quotes, quality)

    def _legacy_synthesis(self, extraction_result: Dict[str, Any]) -> SynthesisResult:
        prompt = self.templates.get_synthesis_prompt(extraction_result)
        synthesis_text = self.ollama.generate(prompt, temperature=0.3)

        if not synthesis_text:
            logger.error("Failed to generate synthesis")
            return self._create_empty_result()

        synthesis_data = self._parse_synthesis_response(synthesis_text)
        validation_result = self._validate_synthesis(synthesis_data)

        result = SynthesisResult(
            enlitens_takeaway=synthesis_data.get('enlitens_takeaway', ''),
            eli5_summary=synthesis_data.get('eli5_summary', ''),
            key_findings=synthesis_data.get('key_findings', []),
            neuroscientific_concepts=synthesis_data.get('neuroscientific_concepts', []),
            clinical_applications=synthesis_data.get('clinical_applications', []),
            therapeutic_targets=synthesis_data.get('therapeutic_targets', []),
            client_presentations=synthesis_data.get('client_presentations', []),
            intervention_suggestions=synthesis_data.get('intervention_suggestions', []),
            contraindications=synthesis_data.get('contraindications', []),
            evidence_strength=synthesis_data.get('evidence_strength', 'preliminary'),
            quality_score=validation_result.get('quality_score', 0.0),
            synthesis_timestamp=datetime.now().isoformat(),
            powerful_quotes=synthesis_data.get('powerful_quotes', []),
            source_citations=synthesis_data.get('source_citations', []),
        )
        logger.info("Legacy synthesis completed with quality score %.2f", result.quality_score)
        return result

    def _build_result(
        self,
        payload: Dict[str, Any],
        quotes: List[Dict[str, Any]],
        quality_score: float,
    ) -> SynthesisResult:
        return SynthesisResult(
            enlitens_takeaway=payload.get('enlitens_takeaway', ''),
            eli5_summary=payload.get('eli5_summary', ''),
            key_findings=payload.get('key_findings', []),
            neuroscientific_concepts=payload.get('neuroscientific_concepts', []),
            clinical_applications=payload.get('clinical_applications', []),
            therapeutic_targets=payload.get('therapeutic_targets', []),
            client_presentations=payload.get('client_presentations', []),
            intervention_suggestions=payload.get('intervention_suggestions', []),
            contraindications=payload.get('contraindications', []),
            evidence_strength=payload.get('evidence_strength', 'preliminary'),
            quality_score=quality_score,
            synthesis_timestamp=datetime.now().isoformat(),
            powerful_quotes=[quote.get('quote', '') for quote in quotes],
            source_citations=payload.get('source_citations', quotes),
        )

    def _parse_stage_one_quotes(self, response_text: str) -> List[Dict[str, Any]]:
        if not response_text:
            return []

        try:
            if '```' in response_text:
                start = response_text.find('```')
                end = response_text.rfind('```')
                json_body = response_text[start:].split('\n', 1)[1].rsplit('```', 1)[0]
            else:
                json_body = response_text
            payload = json.loads(json_body)
            quotes = payload.get('quotes', [])
            return [quote for quote in quotes if quote.get('quote')]
        except Exception as exc:
            logger.error("Failed to parse stage-one quotes: %s", exc)
            return []

    def _fallback_quotes(self, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        quotes: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(context_chunks[:5]):
            text = chunk.get('text', '')
            if not text:
                continue
            snippet = text.strip().split('\n')[0][:500]
            doi = chunk.get('doi') or (chunk.get('metadata') or {}).get('doi', '')
            quotes.append(
                {
                    'citation_id': f'FALLBACK_{idx}',
                    'quote': snippet,
                    'pages': chunk.get('pages', []),
                    'section': (chunk.get('sections') or ['unknown'])[0] if chunk.get('sections') else 'unknown',
                    'chunk_id': chunk.get('chunk_id'),
                    'doi': doi,
                }
            )
        return quotes
    
    def _parse_synthesis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the JSON response from Qwen3"""
        try:
            # Try to extract JSON from response
            if '```json' in response_text:
                # Extract JSON from code block
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                json_text = response_text[start:end].strip()
            else:
                # Try to find JSON in the response
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_text = response_text[start:end]
            
            return json.loads(json_text)
            
        except Exception as e:
            logger.error(f"Failed to parse synthesis response: {str(e)}")
            # Return a basic structure
            return {
                'enlitens_takeaway': response_text[:500] + '...',
                'eli5_summary': 'Unable to parse response',
                'key_findings': [],
                'neuroscientific_concepts': [],
                'clinical_applications': [],
                'therapeutic_targets': [],
                'client_presentations': [],
                'intervention_suggestions': [],
                'contraindications': [],
                'evidence_strength': 'preliminary'
            }
    
    def _validate_synthesis(self, synthesis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the quality of synthesis"""
        try:
            # Generate validation prompt
            validation_prompt = self.templates.get_validation_prompt(synthesis_data)
            
            # Get validation from Qwen2.5
            validation_text = self.ollama.generate(validation_prompt, temperature=0.1)
            
            if not validation_text:
                return {'quality_score': 0.5, 'issues': ['Validation failed']}
            
            # Parse validation response
            try:
                validation_data = json.loads(validation_text)
                return validation_data
            except:
                # Fallback validation
                return self._basic_validation(synthesis_data)
                
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {'quality_score': 0.5, 'issues': ['Validation error']}
    
    def _basic_validation(self, synthesis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Basic validation when AI validation fails"""
        issues = []
        score = 0.5
        
        # Check for required fields
        required_fields = ['enlitens_takeaway', 'eli5_summary', 'key_findings']
        for field in required_fields:
            if not synthesis_data.get(field):
                issues.append(f"Missing {field}")
                score -= 0.1
        
        # Check for content quality
        if synthesis_data.get('enlitens_takeaway'):
            if len(synthesis_data['enlitens_takeaway']) < 100:
                issues.append("Takeaway too short")
                score -= 0.1
        
        # Check for safety
        contraindications = synthesis_data.get('contraindications', [])
        if not contraindications:
            issues.append("No contraindications listed")
            score -= 0.1
        
        return {
            'quality_score': max(0.0, score),
            'issues': issues
        }
    
    def _create_empty_result(self) -> SynthesisResult:
        """Create empty result for failed synthesis"""
        return SynthesisResult(
            enlitens_takeaway="Synthesis failed - manual review required",
            eli5_summary="Unable to generate summary",
            key_findings=[],
            neuroscientific_concepts=[],
            clinical_applications=[],
            therapeutic_targets=[],
            client_presentations=[],
            intervention_suggestions=[],
            contraindications=[],
            evidence_strength="preliminary",
            quality_score=0.0,
            synthesis_timestamp=datetime.now().isoformat(),
            powerful_quotes=[],
            source_citations=[],
            prompt_text=None,
            validation_issues=[],
        )


class SynthesisQualityValidator:
    """
    Validates the quality of AI synthesis
    
    Why validation is critical:
    - Clinical safety requires accuracy
    - Poor synthesis leads to inappropriate therapy
    - Quality issues compound through the pipeline
    - Manual review is expensive
    """
    
    def __init__(self):
        self.quality_threshold = 0.8
        
    def validate_synthesis(self, synthesis_result: SynthesisResult) -> Dict[str, Any]:
        """Validate synthesis quality"""
        issues = []
        warnings = []
        score = 0.0
        
        # Check content completeness
        if not synthesis_result.enlitens_takeaway or len(synthesis_result.enlitens_takeaway) < 100:
            issues.append("Incomplete takeaway")
        else:
            score += 0.2
        
        if not synthesis_result.eli5_summary or len(synthesis_result.eli5_summary) < 50:
            issues.append("Incomplete ELI5 summary")
        else:
            score += 0.2
        
        # Check for key findings
        if not synthesis_result.key_findings:
            issues.append("No key findings")
        else:
            score += 0.2
        
        # Check for clinical applications
        if not synthesis_result.clinical_applications:
            issues.append("No clinical applications")
        else:
            score += 0.2
        
        # Check for safety (contraindications)
        if not synthesis_result.contraindications:
            warnings.append("No contraindications listed")
        else:
            score += 0.1
        
        # Check for therapeutic targets
        if not synthesis_result.therapeutic_targets:
            warnings.append("No therapeutic targets")
        else:
            score += 0.1
        
        return {
            'quality_score': score,
            'issues': issues,
            'warnings': warnings,
            'passed': score >= self.quality_threshold
        }


# Example usage
if __name__ == "__main__":
    # Test the synthesis engine
    ollama_client = OllamaClient()
    
    if ollama_client.is_available():
        print("vLLM is available")
        
        synthesizer = NeuroscienceSynthesizer(ollama_client)
        
        # Test with sample extraction result
        sample_extraction = {
            'title': 'The Role of the Prefrontal Cortex in Executive Function',
            'abstract': 'This study examines the neural mechanisms underlying executive function in the prefrontal cortex...',
            'full_text': 'The prefrontal cortex (PFC) is a critical brain region for executive function...',
            'sections': [
                {'title': 'Introduction', 'content': 'Executive function is a set of cognitive processes...'},
                {'title': 'Methods', 'content': 'We used fMRI to examine brain activity...'},
                {'title': 'Results', 'content': 'We found significant activation in the PFC...'},
                {'title': 'Discussion', 'content': 'These findings suggest that the PFC plays a crucial role...'}
            ]
        }
        
        result = synthesizer.synthesize(sample_extraction)
        print(f"Synthesis completed with quality score: {result.quality_score:.2f}")
        print(f"Takeaway: {result.enlitens_takeaway[:200]}...")
        
    else:
        print("vLLM is not available")
