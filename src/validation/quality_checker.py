"""
Comprehensive Quality Validation System

This module provides multi-level quality validation for the Enlitens pipeline:
1. Extraction validation
2. Entity accuracy validation
3. Relationship verification
4. Clinical appropriateness validation
5. Training data quality validation
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of quality validation"""
    passed: bool
    score: float
    issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    critical_issues: List[str]


class QualityValidator:
    """
    Comprehensive quality validation system
    
    Why validation is critical:
    - Clinical safety requires accuracy
    - Poor quality compounds through the pipeline
    - Manual review is expensive
    - Quality issues lead to inappropriate therapy
    """
    
    def __init__(self):
        self.quality_thresholds = {
            'extraction_completeness': 0.95,
            'entity_precision': 0.90,
            'entity_recall': 0.85,
            'relationship_accuracy': 0.85,
            'clinical_safety': 1.00,  # No unsafe content allowed
            'synthesis_quality': 0.90,
            'voice_consistency': 0.85
        }
    
    def validate_extraction_quality(self, extraction_result: Dict[str, Any]) -> ValidationResult:
        """Validate PDF extraction quality"""
        issues = []
        warnings = []
        recommendations = []
        critical_issues = []
        
        score = 0.0
        checks = 0
        
        # Check title extraction
        title = extraction_result.get('title', '')
        if title and title != 'Unknown Title':
            score += 0.2
        else:
            issues.append("Title extraction failed or missing")
        checks += 1
        
        # Check abstract extraction
        abstract = extraction_result.get('abstract', '')
        if abstract and len(abstract.strip()) > 50:
            score += 0.2
        else:
            issues.append("Abstract extraction failed or too short")
        checks += 1
        
        # Check main content
        full_text = extraction_result.get('full_text', '')
        if full_text and len(full_text) > 1000:
            score += 0.2
        else:
            issues.append("Main content extraction failed or too short")
        checks += 1
        
        # Check sections
        sections = extraction_result.get('sections', [])
        if len(sections) >= 3:
            score += 0.2
        else:
            issues.append(f"Too few sections extracted: {len(sections)}")
        checks += 1
        
        # Check tables
        tables = extraction_result.get('tables', [])
        if tables:
            score += 0.1
        else:
            warnings.append("No tables extracted")
        checks += 1
        
        # Check references
        references = extraction_result.get('references', [])
        if len(references) >= 5:
            score += 0.1
        else:
            warnings.append(f"Few references extracted: {len(references)}")
        checks += 1
        
        # Calculate final score
        final_score = score / checks if checks > 0 else 0.0
        
        # Generate recommendations
        if final_score < 0.8:
            recommendations.append("Consider using different extraction parameters")
        if not tables:
            recommendations.append("Check if document contains tables that weren't extracted")
        if len(references) < 5:
            recommendations.append("Verify reference extraction is working properly")
        
        return ValidationResult(
            passed=final_score >= self.quality_thresholds['extraction_completeness'],
            score=final_score,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            critical_issues=critical_issues
        )
    
    def validate_entity_extraction(self, entities: Dict[str, List[Dict]]) -> ValidationResult:
        """Validate entity extraction quality"""
        issues = []
        warnings = []
        recommendations = []
        critical_issues = []
        
        score = 0.0
        total_entities = 0
        valid_entities = 0
        
        # Check each entity category
        for category, entity_list in entities.items():
            if not entity_list:
                continue
            
            total_entities += len(entity_list)
            
            # Validate entities in this category
            for entity in entity_list:
                if self._is_valid_entity(entity):
                    valid_entities += 1
                else:
                    issues.append(f"Invalid entity in {category}: {entity.get('text', '')}")
        
        # Calculate precision
        if total_entities > 0:
            precision = valid_entities / total_entities
            score = precision
        else:
            issues.append("No entities extracted")
            score = 0.0
        
        # Check for critical issues
        if score < 0.5:
            critical_issues.append("Entity extraction quality is critically low")
        
        # Generate recommendations
        if score < 0.8:
            recommendations.append("Review entity extraction parameters")
        if total_entities < 10:
            recommendations.append("Consider adjusting entity extraction sensitivity")
        
        return ValidationResult(
            passed=score >= self.quality_thresholds['entity_precision'],
            score=score,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            critical_issues=critical_issues
        )
    
    def _is_valid_entity(self, entity: Dict[str, Any]) -> bool:
        """Check if an entity is valid"""
        # Check required fields
        if not entity.get('text') or not entity.get('label'):
            return False
        
        # Check confidence score
        confidence = entity.get('confidence', 0.0)
        if confidence < 0.5:
            return False
        
        # Check text length
        text = entity.get('text', '')
        if len(text) < 2 or len(text) > 100:
            return False
        
        # Check for common extraction artifacts
        if text.lower() in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']:
            return False
        
        return True
    
    def validate_synthesis_quality(self, synthesis_result: Any) -> ValidationResult:
        """Validate AI synthesis quality"""
        issues = []
        warnings = []
        recommendations = []
        critical_issues = []
        
        score = 0.0
        checks = 0
        
        # Check takeaway
        takeaway = getattr(synthesis_result, 'enlitens_takeaway', '')
        if takeaway and len(takeaway) > 200:
            score += 0.2
        else:
            issues.append("Enlitens takeaway missing or too short")
        checks += 1
        
        # Check ELI5 summary
        eli5 = getattr(synthesis_result, 'eli5_summary', '')
        if eli5 and len(eli5) > 100:
            score += 0.2
        else:
            issues.append("ELI5 summary missing or too short")
        checks += 1
        
        # Check key findings
        key_findings = getattr(synthesis_result, 'key_findings', [])
        if len(key_findings) >= 3:
            score += 0.2
        else:
            issues.append(f"Too few key findings: {len(key_findings)}")
        checks += 1
        
        # Check clinical applications
        clinical_apps = getattr(synthesis_result, 'clinical_applications', [])
        if len(clinical_apps) >= 2:
            score += 0.2
        else:
            issues.append(f"Too few clinical applications: {len(clinical_apps)}")
        checks += 1
        
        # Check contraindications
        contraindications = getattr(synthesis_result, 'contraindications', [])
        if contraindications:
            score += 0.2
        else:
            warnings.append("No contraindications listed")
        checks += 1
        
        # Check for safety issues
        if self._check_safety_issues(synthesis_result):
            critical_issues.append("Safety issues detected in synthesis")
            score -= 0.5  # Penalty for safety issues
        
        # Calculate final score
        final_score = max(0.0, score / checks if checks > 0 else 0.0)
        
        # Generate recommendations
        if final_score < 0.8:
            recommendations.append("Review synthesis prompts and parameters")
        if not contraindications:
            recommendations.append("Ensure contraindications are always included")
        
        return ValidationResult(
            passed=final_score >= self.quality_thresholds['synthesis_quality'],
            score=final_score,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            critical_issues=critical_issues
        )
    
    def _check_safety_issues(self, synthesis_result: Any) -> bool:
        """Check for safety issues in synthesis"""
        # Check for dangerous suggestions
        dangerous_phrases = [
            'ignore medical advice',
            'stop taking medication',
            'self-diagnose',
            'dangerous',
            'harmful',
            'unsafe'
        ]
        
        text_to_check = [
            getattr(synthesis_result, 'enlitens_takeaway', ''),
            getattr(synthesis_result, 'eli5_summary', ''),
            str(getattr(synthesis_result, 'clinical_applications', [])),
            str(getattr(synthesis_result, 'intervention_suggestions', []))
        ]
        
        for text in text_to_check:
            for phrase in dangerous_phrases:
                if phrase.lower() in text.lower():
                    return True
        
        return False
    
    def validate_voice_consistency(self, synthesis_result: Any) -> ValidationResult:
        """Validate Enlitens voice consistency"""
        issues = []
        warnings = []
        recommendations = []
        critical_issues = []
        
        # Check for voice indicators
        voice_indicators = {
            'validation_phrases': [
                'of course you feel',
                'that makes complete sense',
                'it\'s no wonder',
                'of course you experience'
            ],
            'science_translation_phrases': [
                'your brain is doing',
                'what\'s happening in your brain',
                'the neurobiology of',
                'your nervous system'
            ],
            'hope_phrases': [
                'neuroplasticity',
                'your brain can change',
                'new neural pathways'
            ]
        }
        
        text_to_check = getattr(synthesis_result, 'enlitens_takeaway', '')
        
        score = 0.0
        total_indicators = 0
        found_indicators = 0
        
        for category, phrases in voice_indicators.items():
            for phrase in phrases:
                total_indicators += 1
                if phrase.lower() in text_to_check.lower():
                    found_indicators += 1
        
        if total_indicators > 0:
            score = found_indicators / total_indicators
        
        # Check for inappropriate language
        if self._check_inappropriate_language(text_to_check):
            issues.append("Inappropriate language detected")
        
        # Check for clinical appropriateness
        if not self._check_clinical_appropriateness(text_to_check):
            issues.append("Language not clinically appropriate")
        
        # Generate recommendations
        if score < 0.3:
            recommendations.append("Review voice consistency in synthesis prompts")
        if not self._check_validation_present(text_to_check):
            recommendations.append("Ensure validation language is present")
        
        return ValidationResult(
            passed=score >= self.quality_thresholds['voice_consistency'],
            score=score,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            critical_issues=critical_issues
        )
    
    def _check_inappropriate_language(self, text: str) -> bool:
        """Check for inappropriate language"""
        inappropriate_phrases = [
            'stupid',
            'dumb',
            'idiot',
            'crazy',
            'insane'
        ]
        
        for phrase in inappropriate_phrases:
            if phrase.lower() in text.lower():
                return True
        
        return False
    
    def _check_clinical_appropriateness(self, text: str) -> bool:
        """Check if language is clinically appropriate"""
        # Check for professional tone
        unprofessional_phrases = [
            'lol',
            'haha',
            'omg',
            'wtf'
        ]
        
        for phrase in unprofessional_phrases:
            if phrase.lower() in text.lower():
                return False
        
        return True
    
    def _check_validation_present(self, text: str) -> bool:
        """Check if validation language is present"""
        validation_phrases = [
            'of course',
            'that makes sense',
            'it\'s understandable',
            'that\'s normal'
        ]
        
        for phrase in validation_phrases:
            if phrase.lower() in text.lower():
                return True
        
        return False
    
    def validate_complete_document(self, document: Any) -> ValidationResult:
        """Validate complete document quality"""
        issues = []
        warnings = []
        recommendations = []
        critical_issues = []
        
        # Validate extraction
        extraction_result = {
            'title': getattr(document.source_metadata, 'title', ''),
            'abstract': getattr(document.archival_content, 'abstract_markdown', ''),
            'full_text': getattr(document.archival_content, 'full_document_text_markdown', ''),
            'sections': getattr(document.archival_content, 'sections', []),
            'tables': getattr(document.archival_content, 'tables', []),
            'references': getattr(document.archival_content, 'references', [])
        
        extraction_validation = self.validate_extraction_quality(extraction_result)
        if not extraction_validation.passed:
            issues.extend(extraction_validation.issues)
            critical_issues.extend(extraction_validation.critical_issues)
        
        # Validate entity extraction
        if hasattr(document, 'entity_extraction') and document.entity_extraction:
            entity_validation = self.validate_entity_extraction({
                'biomedical': getattr(document.entity_extraction, 'biomedical', []),
                'scientific': getattr(document.entity_extraction, 'scientific', []),
                'neuroscience_specific': getattr(document.entity_extraction, 'neuroscience_specific', [])
            })
            if not entity_validation.passed:
                issues.extend(entity_validation.issues)
                critical_issues.extend(entity_validation.critical_issues)
        
        # Validate synthesis
        if hasattr(document, 'ai_synthesis') and document.ai_synthesis:
            synthesis_validation = self.validate_synthesis_quality(document.ai_synthesis)
            if not synthesis_validation.passed:
                issues.extend(synthesis_validation.issues)
                critical_issues.extend(synthesis_validation.critical_issues)
            
            # Validate voice consistency
            voice_validation = self.validate_voice_consistency(document.ai_synthesis)
            if not voice_validation.passed:
                issues.extend(voice_validation.issues)
                warnings.extend(voice_validation.warnings)
        
        # Calculate overall score
        overall_score = document.get_quality_score()
        
        # Generate recommendations
        if overall_score < 0.8:
            recommendations.append("Overall document quality is below threshold")
        if critical_issues:
            recommendations.append("Address critical issues before proceeding")
        
        return ValidationResult(
            passed=overall_score >= 0.8 and not critical_issues,
            score=overall_score,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            critical_issues=critical_issues
        )
    
    def generate_quality_report(self, document: Any) -> str:
        """Generate a comprehensive quality report"""
        validation = self.validate_complete_document(document)
        
        report = f"""
Quality Validation Report
========================

Overall Score: {validation.score:.2f} / 1.0
Status: {'PASSED' if validation.passed else 'FAILED'}

Issues Found ({len(validation.issues)}):
"""
        
        for issue in validation.issues:
            report += f"- {issue}\n"
        
        if validation.warnings:
            report += f"\nWarnings ({len(validation.warnings)}):\n"
            for warning in validation.warnings:
                report += f"- {warning}\n"
        
        if validation.critical_issues:
            report += f"\nCritical Issues ({len(validation.critical_issues)}):\n"
            for issue in validation.critical_issues:
                report += f"- {issue}\n"
        
        if validation.recommendations:
            report += f"\nRecommendations ({len(validation.recommendations)}):\n"
            for rec in validation.recommendations:
                report += f"- {rec}\n"
        
        return report


# Example usage
if __name__ == "__main__":
    # Test the quality validator
    validator = QualityValidator()
    
    # Test extraction validation
    sample_extraction = {
        'title': 'Sample Research Paper',
        'abstract': 'This is a sample abstract that should be long enough to pass validation.',
        'full_text': 'This is the full text content of the paper. ' * 100,
        'sections': [
            {'title': 'Introduction', 'content': 'Introduction content'},
            {'title': 'Methods', 'content': 'Methods content'},
            {'title': 'Results', 'content': 'Results content'}
        ],
        'tables': [{'content': 'Table content'}],
        'references': ['Ref 1', 'Ref 2', 'Ref 3', 'Ref 4', 'Ref 5']
    }
    
    extraction_result = validator.validate_extraction_quality(sample_extraction)
    print("Extraction validation:", extraction_result.passed, extraction_result.score)
    
    # Test entity validation
    sample_entities = {
        'biomedical': [
            {'text': 'dopamine', 'label': 'NEUROTRANSMITTER', 'confidence': 0.9},
            {'text': 'prefrontal cortex', 'label': 'BRAIN_REGION', 'confidence': 0.8}
        ]
    }
    
    entity_result = validator.validate_entity_extraction(sample_entities)
    print("Entity validation:", entity_result.passed, entity_result.score)
