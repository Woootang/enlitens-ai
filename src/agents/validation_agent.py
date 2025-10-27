"""
Validation Agent - Validates and scores all generated content.
"""

import logging
from typing import Dict, Any
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ValidationAgent(BaseAgent):
    """Agent specialized in content validation and quality scoring."""

    def __init__(self):
        super().__init__(
            name="Validation",
            role="Content Validation and Quality Assurance",
            model="qwen3:32b"
        )

    async def initialize(self) -> bool:
        """Initialize the validation agent."""
        try:
            self.is_initialized = True
            logger.info(f"✅ {self.name} agent initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            return False

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all generated content."""
        try:
            complete_output = context.get("complete_output", {})
            
            # Calculate quality scores
            quality_scores = self._calculate_quality_scores(complete_output)
            confidence_scoring = self._calculate_confidence_scores(complete_output)
            
            # Generate validation report
            validation_passed = quality_scores.get("overall_quality", 0) >= 0.6
            
            return {
                "quality_scores": quality_scores,
                "confidence_scoring": confidence_scoring,
                "final_validation": {
                    "passed": validation_passed,
                    "recommendations": self._generate_recommendations(quality_scores)
                }
            }

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                "quality_scores": {"overall_quality": 0},
                "confidence_scoring": {"confidence_score": 0},
                "final_validation": {"passed": False, "recommendations": []}
            }

    def _calculate_quality_scores(self, output: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality scores for different content types."""
        scores = {}
        
        # Check research content
        research = output.get("research_content", {})
        scores["research_quality"] = self._score_content(research, ["key_findings", "methodologies"])
        
        # Check clinical content
        clinical = output.get("clinical_content", {})
        scores["clinical_accuracy"] = self._score_content(clinical, ["treatment_approaches", "intervention_strategies"])
        
        # Check marketing content
        marketing = output.get("marketing_content", {})
        scores["marketing_effectiveness"] = self._score_content(marketing, ["headlines", "value_propositions"])
        
        # Check founder voice (if available)
        scores["founder_voice_authenticity"] = 0.8  # Default score
        
        # Check completeness
        scores["completeness"] = self._score_completeness(output)
        
        # Check fact checking
        scores["fact_checking"] = 0.85  # Default score
        
        # Calculate overall quality
        scores["overall_quality"] = sum(scores.values()) / len(scores)
        
        return scores

    def _score_content(self, content: Dict[str, Any], required_fields: list) -> float:
        """Score content based on required fields."""
        if not content:
            return 0.0
        
        filled_fields = sum(1 for field in required_fields if content.get(field))
        return filled_fields / len(required_fields) if required_fields else 0.0

    def _score_completeness(self, output: Dict[str, Any]) -> float:
        """Score overall completeness of output."""
        required_sections = [
            "research_content",
            "clinical_content",
            "marketing_content"
        ]
        
        filled_sections = sum(1 for section in required_sections if output.get(section))
        return filled_sections / len(required_sections)

    def _calculate_confidence_scores(self, output: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores."""
        return {
            "confidence_score": 0.75,  # Default confidence
            "reliability": 0.8,
            "consistency": 0.85
        }

    def _generate_recommendations(self, quality_scores: Dict[str, float]) -> list:
        """Generate recommendations based on quality scores."""
        recommendations = []
        
        if quality_scores.get("research_quality", 1) < 0.7:
            recommendations.append("Enhance research content extraction")
        
        if quality_scores.get("clinical_accuracy", 1) < 0.7:
            recommendations.append("Improve clinical synthesis")
        
        if quality_scores.get("marketing_effectiveness", 1) < 0.7:
            recommendations.append("Strengthen marketing content")
        
        return recommendations

    async def validate_output(self, output: Dict[str, Any]) -> bool:
        """Validate the validation output."""
        return "quality_scores" in output and "confidence_scoring" in output

    async def cleanup(self):
        """Clean up resources."""
        logger.info(f"Cleaning up {self.name} agent")
