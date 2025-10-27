"""
Extraction Team - Specialized Entity Recognition

This team uses specialized biomedical models to extract entities from research papers:
- BiomedBERT for general biomedical entities
- NeuroBERT for neuroscience-specific entities  
- PsychBERT for psychology entities
- GatorTron for clinical entities
"""

import logging
from typing import Dict, List, Any, Optional
import asyncio
from transformers import pipeline

logger = logging.getLogger(__name__)


class ExtractionTeam:
    """
    Extraction team specialized in entity recognition using biomedical models.
    
    Models:
    - BiomedBERT: General biomedical entities
    - NeuroBERT: Neuroscience entities
    - PsychBERT: Psychology entities
    - GatorTron: Clinical entities
    """
    
    def __init__(self):
        self.models = {}
        self.entity_types = {
            'biomedical': ['DISEASE', 'CHEMICAL', 'GENE', 'PROTEIN', 'CELL_LINE'],
            'neuroscience': ['BRAIN_REGION', 'NEURAL_PATHWAY', 'NEUROTRANSMITTER', 'COGNITIVE_FUNCTION'],
            'psychology': ['PSYCHOLOGICAL_CONCEPT', 'BEHAVIOR', 'EMOTION', 'MENTAL_STATE'],
            'clinical': ['SYMPTOM', 'TREATMENT', 'DIAGNOSIS', 'INTERVENTION', 'OUTCOME']
        }
        
    async def extract_entities(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities from the document using specialized models.
        
        Args:
            extraction_result: Raw extraction result from PDF
            
        Returns:
            Dictionary of extracted entities by type
        """
        try:
            logger.info("Extraction Team: Starting entity extraction")
            
            # Get text content
            text_content = self._get_text_content(extraction_result)
            if not text_content:
                logger.warning("Extraction Team: No text content found")
                return {}
            
            # Extract entities using different models
            entities = {}
            
            # Biomedical entities
            biomedical_entities = await self._extract_biomedical_entities(text_content)
            entities['biomedical'] = biomedical_entities
            
            # Neuroscience entities
            neuro_entities = await self._extract_neuroscience_entities(text_content)
            entities['neuroscience'] = neuro_entities
            
            # Psychology entities
            psych_entities = await self._extract_psychology_entities(text_content)
            entities['psychology'] = psych_entities
            
            # Clinical entities
            clinical_entities = await self._extract_clinical_entities(text_content)
            entities['clinical'] = clinical_entities
            
            # Statistical entities
            statistical_entities = await self._extract_statistical_entities(text_content)
            entities['statistical'] = statistical_entities
            
            logger.info(f"Extraction Team: Extracted {sum(len(v) for v in entities.values())} entities")
            return entities
            
        except Exception as e:
            logger.error(f"Extraction Team: Entity extraction failed: {e}")
            return {}
    
    def _get_text_content(self, extraction_result: Dict[str, Any]) -> str:
        """Extract text content from extraction result"""
        if 'archival_content' in extraction_result:
            return extraction_result['archival_content'].get('full_document_text_markdown', '')
        return ''
    
    async def _extract_biomedical_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract biomedical entities using BiomedBERT"""
        try:
            # Load BiomedBERT model
            if 'biomedbert' not in self.models:
                self.models['biomedbert'] = pipeline(
                    "ner",
                    model="dmis-lab/biobert-base-cased-v1.1",
                    aggregation_strategy="simple"
                )
            
            # Extract entities
            entities = self.models['biomedbert'](text)
            
            # Filter for biomedical entity types
            biomedical_entities = []
            for entity in entities:
                if entity['entity_group'] in self.entity_types['biomedical']:
                    biomedical_entities.append({
                        'text': entity['word'],
                        'label': entity['entity_group'],
                        'confidence': entity['score'],
                        'start': entity['start'],
                        'end': entity['end']
                    })
            
            return biomedical_entities
            
        except Exception as e:
            logger.error(f"Biomedical entity extraction failed: {e}")
            return []
    
    async def _extract_neuroscience_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract neuroscience entities using NeuroBERT"""
        try:
            # Load NeuroBERT model
            if 'neurobert' not in self.models:
                self.models['neurobert'] = pipeline(
                    "ner",
                    model="allenai/scibert_scivocab_uncased",
                    aggregation_strategy="simple"
                )
            
            # Extract entities
            entities = self.models['neurobert'](text)
            
            # Filter for neuroscience entity types
            neuro_entities = []
            for entity in entities:
                if entity['entity_group'] in self.entity_types['neuroscience']:
                    neuro_entities.append({
                        'text': entity['word'],
                        'label': entity['entity_group'],
                        'confidence': entity['score'],
                        'start': entity['start'],
                        'end': entity['end']
                    })
            
            return neuro_entities
            
        except Exception as e:
            logger.error(f"Neuroscience entity extraction failed: {e}")
            return []
    
    async def _extract_psychology_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract psychology entities using PsychBERT"""
        try:
            # Load PsychBERT model
            if 'psychbert' not in self.models:
                self.models['psychbert'] = pipeline(
                    "ner",
                    model="mental/mental-bert-base-uncased",
                    aggregation_strategy="simple"
                )
            
            # Extract entities
            entities = self.models['psychbert'](text)
            
            # Filter for psychology entity types
            psych_entities = []
            for entity in entities:
                if entity['entity_group'] in self.entity_types['psychology']:
                    psych_entities.append({
                        'text': entity['word'],
                        'label': entity['entity_group'],
                        'confidence': entity['score'],
                        'start': entity['start'],
                        'end': entity['end']
                    })
            
            return psych_entities
            
        except Exception as e:
            logger.error(f"Psychology entity extraction failed: {e}")
            return []
    
    async def _extract_clinical_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract clinical entities using GatorTron"""
        try:
            # Load GatorTron model
            if 'gator' not in self.models:
                self.models['gator'] = pipeline(
                    "ner",
                    model="emilyalsentzer/Bio_ClinicalBERT",
                    aggregation_strategy="simple"
                )
            
            # Extract entities
            entities = self.models['gator'](text)
            
            # Filter for clinical entity types
            clinical_entities = []
            for entity in entities:
                if entity['entity_group'] in self.entity_types['clinical']:
                    clinical_entities.append({
                        'text': entity['word'],
                        'label': entity['entity_group'],
                        'confidence': entity['score'],
                        'start': entity['start'],
                        'end': entity['end']
                    })
            
            return clinical_entities
            
        except Exception as e:
            logger.error(f"Clinical entity extraction failed: {e}")
            return []
    
    async def _extract_statistical_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract statistical entities using regex patterns"""
        import re
        
        statistical_entities = []
        
        # Statistical patterns
        patterns = {
            'p_value': r'p\s*[<>=]\s*[\d.]+',
            'correlation': r'r\s*=\s*[\d.-]+',
            'beta': r'β\s*=\s*[\d.-]+',
            'odds_ratio': r'OR\s*=\s*[\d.]+',
            'confidence_interval': r'\d+%\s*CI\s*:\s*[\d.-]+',
            'sample_size': r'n\s*=\s*\d+',
            'mean': r'mean\s*=\s*[\d.]+',
            'standard_deviation': r'SD\s*=\s*[\d.]+'
        }
        
        for pattern_name, pattern in patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                statistical_entities.append({
                    'text': match.group(),
                    'label': pattern_name,
                    'confidence': 1.0,
                    'start': match.start(),
                    'end': match.end()
                })
        
        return statistical_entities
