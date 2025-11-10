"""
Extraction Team - Specialized Entity Recognition

This team uses specialized biomedical models to extract entities from research papers:
- BiomedBERT for general biomedical entities
- NeuroBERT for neuroscience-specific entities  
- PsychBERT for psychology entities
- GatorTron for clinical entities
"""

import logging
import os
from typing import Dict, List, Any, Optional, Callable
import asyncio

import torch
from transformers import pipeline

from src.utils.gpu_memory_manager import get_gpu_manager

logger = logging.getLogger(__name__)


class LazyPipelineLoader:
    """Lazy loader that caches Hugging Face pipelines with device awareness."""

    def __init__(
        self,
        pipeline_factory: Callable[..., Any] = pipeline,
        force_cpu: Optional[bool] = None,
    ) -> None:
        self._pipeline_factory = pipeline_factory
        self._cache: Dict[str, Any] = {}
        if force_cpu is None:
            force_cpu = os.getenv("EXTRACTION_FORCE_CPU", "false").lower() in {"1", "true", "yes"}
        self._force_cpu = force_cpu

    def get(
        self,
        cache_key: str,
        task: str,
        model_name: str,
        **pipeline_kwargs: Any,
    ) -> Any:
        """Return a cached pipeline, creating it if necessary."""

        if cache_key in self._cache:
            return self._cache[cache_key]

        device = pipeline_kwargs.pop("device", None)
        if device is None:
            if self._force_cpu:
                device = -1
            else:
                device = 0 if torch.cuda.is_available() else -1

        logger.info(
            "ExtractionTeam: loading pipeline %s on %s",
            model_name,
            "CPU" if device == -1 else f"device {device}",
        )

        try:
            self._cache[cache_key] = self._pipeline_factory(
                task,
                model=model_name,
                device=device,
                **pipeline_kwargs,
            )
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {str(e)[:200]}")
            raise
        return self._cache[cache_key]


class ExtractionTeam:
    """
    Extraction team specialized in entity recognition using biomedical models.
    
    Models:
    - BiomedBERT: General biomedical entities
    - NeuroBERT: Neuroscience entities
    - PsychBERT: Psychology entities
    - GatorTron: Clinical entities
    """
    
    def __init__(
        self,
        pipeline_factory: Callable[..., Any] = pipeline,
        force_cpu: Optional[bool] = None,
    ):
        # FORCE CPU for NER models since vLLM uses all GPU memory
        self.pipeline_loader = LazyPipelineLoader(
            pipeline_factory=pipeline_factory,
            force_cpu=True,  # Always use CPU for NER
        )
        self.models = {}
        self.gpu_manager = get_gpu_manager()
        self.entity_types = {
            'biomedical': ['DISEASE', 'CHEMICAL', 'GENE', 'PROTEIN', 'CELL_LINE'],
            'neuroscience': ['BRAIN_REGION', 'NEURAL_PATHWAY', 'NEUROTRANSMITTER', 'COGNITIVE_FUNCTION'],
            'psychology': ['PSYCHOLOGICAL_CONCEPT', 'BEHAVIOR', 'EMOTION', 'MENTAL_STATE'],
            'clinical': ['SYMPTOM', 'TREATMENT', 'DIAGNOSIS', 'INTERVENTION', 'OUTCOME']
        }
        logger.info("⚠️  NER models will run on CPU (vLLM occupies GPU)")
        
    async def extract_entities(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract entities from the document using specialized models.
        OPTIMIZED: Load/unload models sequentially to avoid GPU OOM.
        
        Args:
            extraction_result: Raw extraction result from PDF
            
        Returns:
            Dictionary of extracted entities by type
        """
        try:
            logger.info("Extraction Team: Starting entity extraction (sequential loading)")
            
            # Log initial GPU state
            self.gpu_manager.log_memory_stats("🔥 GPU before extraction")
            
            # Get text content
            text_content = self._get_text_content(extraction_result)
            if not text_content:
                logger.warning("Extraction Team: No text content found")
                return {}
            
            # Extract entities using different models - ONE AT A TIME
            entities = {}
            
            # 1. Disease/Condition entities (OpenMed DiseaseDetect)
            logger.info("🔄 Model 1/5: DiseaseDetect...")
            biomedical_entities = await self._extract_biomedical_entities(text_content)
            entities['diseases'] = biomedical_entities
            await self._unload_model('disease_detect')
            
            # 2. Chemical/Drug entities (OpenMed PharmaDetect)
            logger.info("🔄 Model 2/5: PharmaDetect...")
            neuro_entities = await self._extract_neuroscience_entities(text_content)
            entities['chemicals'] = neuro_entities
            await self._unload_model('pharma_detect')
            
            # 3. Anatomical entities (OpenMed AnatomyDetect)
            logger.info("🔄 Model 3/5: AnatomyDetect...")
            psych_entities = await self._extract_psychology_entities(text_content)
            entities['anatomy'] = psych_entities
            await self._unload_model('anatomy_detect')
            
            # 4. Gene/Protein entities (OpenMed GenomeDetect)
            logger.info("🔄 Model 4/5: GenomeDetect...")
            clinical_entities = await self._extract_clinical_entities(text_content)
            entities['genes'] = clinical_entities
            await self._unload_model('genome_detect')
            
            # 5. Clinical symptoms/treatments (ClinicalDistilBERT)
            logger.info("🔄 Model 5/5: ClinicalDistilBERT...")
            clinical_symptoms = await self._extract_clinical_symptoms(text_content)
            entities['clinical'] = clinical_symptoms
            await self._unload_model('clinical_distilbert')
            
            # Statistical entities (lightweight, no GPU needed)
            statistical_entities = await self._extract_statistical_entities(text_content)
            entities['statistical'] = statistical_entities
            
            logger.info(f"✅ Extraction Team: Extracted {sum(len(v) for v in entities.values())} entities")
            return entities
            
        except Exception as e:
            logger.error(f"❌ Extraction Team: Entity extraction failed: {e}")
            return {}
    
    async def _unload_model(self, model_key: str):
        """Unload a model from memory and clear GPU cache."""
        try:
            if model_key in self.models:
                logger.info(f"🧹 Unloading {model_key}...")
                del self.models[model_key]
                
                # Use GPU manager for cleanup
                self.gpu_manager.clear_cache()
                self.gpu_manager.log_memory_stats(f"After unloading {model_key}")
                
                logger.info(f"✅ {model_key} unloaded")
        except Exception as e:
            logger.warning(f"Failed to unload {model_key}: {e}")
    
    def _get_text_content(self, extraction_result: Dict[str, Any]) -> str:
        """Extract text content from extraction result"""
        if 'archival_content' in extraction_result:
            return extraction_result['archival_content'].get('full_document_text_markdown', '')
        return ''
    
    async def _extract_biomedical_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract disease/condition entities using OpenMed DiseaseDetect"""
        try:
            # Check GPU memory before loading
            if not self.gpu_manager.check_available_memory(1.0):
                logger.warning("Insufficient GPU memory for disease model, forcing cleanup")
                self.gpu_manager.force_cleanup()
            
            # Load OpenMed DiseaseDetect model
            if 'disease_detect' not in self.models:
                logger.info("Loading OpenMed DiseaseDetect (434M, ~0.85GB)...")
                self.models['disease_detect'] = self.pipeline_loader.get(
                    'disease_detect',
                    "token-classification",
                    "OpenMed/OpenMed-NER-DiseaseDetect-SuperClinical-434M",
                    aggregation_strategy="simple",
                    torch_dtype="float16"
                )
                self.gpu_manager.log_memory_stats("After loading DiseaseDetect")
            
            # Extract entities
            entities = self.models['disease_detect'](text[:2000])  # Limit to first 2000 chars for speed
            
            # Format entities
            biomedical_entities = []
            for entity in entities:
                biomedical_entities.append({
                    'text': entity['word'],
                    'label': 'Disease',
                    'confidence': entity['score'],
                    'start': entity['start'],
                    'end': entity['end']
                })
            
            logger.info(f"✅ DiseaseDetect found {len(biomedical_entities)} disease entities")
            return biomedical_entities
            
        except Exception as e:
            logger.warning(f"⚠️ Disease NER model unavailable (optional), skipping: {str(e)[:100]}")
            return []
    
    async def _extract_neuroscience_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract chemical/neurotransmitter entities using OpenMed PharmaDetect"""
        try:
            # Check GPU memory
            if not self.gpu_manager.check_available_memory(1.0):
                self.gpu_manager.force_cleanup()
            
            # Load OpenMed PharmaDetect model
            if 'pharma_detect' not in self.models:
                logger.info("Loading OpenMed PharmaDetect (434M, ~0.85GB)...")
                self.models['pharma_detect'] = self.pipeline_loader.get(
                    'pharma_detect',
                    "token-classification",
                    "OpenMed/OpenMed-NER-PharmaDetect-SuperClinical-434M",
                    aggregation_strategy="simple",
                    torch_dtype="float16"
                )
                self.gpu_manager.log_memory_stats("After loading PharmaDetect")
            
            # Extract entities
            entities = self.models['pharma_detect'](text[:2000])
            
            # Format entities
            neuro_entities = []
            for entity in entities:
                neuro_entities.append({
                    'text': entity['word'],
                    'label': 'Chemical',
                    'confidence': entity['score'],
                    'start': entity['start'],
                    'end': entity['end']
                })
            
            logger.info(f"✅ PharmaDetect found {len(neuro_entities)} chemical/drug entities")
            return neuro_entities
            
        except Exception as e:
            logger.error(f"Chemical entity extraction failed: {e}")
            return []
    
    async def _extract_psychology_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract anatomical entities using OpenMed AnatomyDetect"""
        try:
            # Check GPU memory
            if not self.gpu_manager.check_available_memory(1.5):
                self.gpu_manager.force_cleanup()
            
            # Load OpenMed AnatomyDetect model (larger: 560M)
            if 'anatomy_detect' not in self.models:
                logger.info("Loading OpenMed AnatomyDetect (560M, ~1.1GB)...")
                self.models['anatomy_detect'] = self.pipeline_loader.get(
                    'anatomy_detect',
                    "token-classification",
                    "OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-560M",
                    aggregation_strategy="simple",
                    torch_dtype="float16"
                )
                self.gpu_manager.log_memory_stats("After loading AnatomyDetect")
            
            # Extract entities
            entities = self.models['anatomy_detect'](text[:2000])
            
            # Format entities
            psych_entities = []
            for entity in entities:
                psych_entities.append({
                    'text': entity['word'],
                    'label': 'Anatomy',
                    'confidence': entity['score'],
                    'start': entity['start'],
                    'end': entity['end']
                })
            
            logger.info(f"✅ AnatomyDetect found {len(psych_entities)} anatomical entities")
            return psych_entities
            
        except Exception as e:
            logger.error(f"Anatomy entity extraction failed: {e}")
            return []
    
    async def _extract_clinical_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract gene/protein entities using OpenMed GenomeDetect"""
        try:
            # Check GPU memory
            if not self.gpu_manager.check_available_memory(1.0):
                self.gpu_manager.force_cleanup()
            
            # Load OpenMed GenomeDetect model
            if 'genome_detect' not in self.models:
                logger.info("Loading OpenMed GenomeDetect (434M, ~0.85GB)...")
                self.models['genome_detect'] = self.pipeline_loader.get(
                    'genome_detect',
                    "token-classification",
                    "OpenMed/OpenMed-NER-GenomeDetect-SuperClinical-434M",
                    aggregation_strategy="simple",
                    torch_dtype="float16"
                )
                self.gpu_manager.log_memory_stats("After loading GenomeDetect")
            
            # Extract entities
            entities = self.models['genome_detect'](text[:2000])
            
            # Format entities
            clinical_entities = []
            for entity in entities:
                clinical_entities.append({
                    'text': entity['word'],
                    'label': 'Gene',
                    'confidence': entity['score'],
                    'start': entity['start'],
                    'end': entity['end']
                })
            
            logger.info(f"✅ GenomeDetect found {len(clinical_entities)} gene/protein entities")
            return clinical_entities
            
        except Exception as e:
            logger.warning(f"⚠️ Gene NER model unavailable (optional), skipping: {str(e)[:100]}")
            return []
    
    async def _extract_clinical_symptoms(self, text: str) -> List[Dict[str, Any]]:
        """Extract clinical symptoms/treatments using ClinicalDistilBERT"""
        try:
            # Check GPU memory
            if not self.gpu_manager.check_available_memory(0.5):
                self.gpu_manager.force_cleanup()
            
            # Load ClinicalDistilBERT i2b2-2010 model
            if 'clinical_distilbert' not in self.models:
                logger.info("Loading ClinicalDistilBERT i2b2-2010 (65M, ~0.13GB)...")
                self.models['clinical_distilbert'] = self.pipeline_loader.get(
                    'clinical_distilbert',
                    "token-classification",
                    "nlpie/clinical-distilbert-i2b2-2010",
                    aggregation_strategy="simple",
                    torch_dtype="float16"
                )
                self.gpu_manager.log_memory_stats("After loading ClinicalDistilBERT")
            
            # Extract entities
            entities = self.models['clinical_distilbert'](text[:2000])
            
            # Format entities (PROBLEM, TREATMENT, TEST)
            clinical_entities = []
            for entity in entities:
                clinical_entities.append({
                    'text': entity['word'],
                    'label': entity.get('entity_group', 'Clinical'),
                    'confidence': entity['score'],
                    'start': entity['start'],
                    'end': entity['end']
                })
            
            logger.info(f"✅ ClinicalDistilBERT found {len(clinical_entities)} clinical entities")
            return clinical_entities
            
        except Exception as e:
            logger.warning(f"⚠️ Clinical NER model unavailable (optional), skipping: {str(e)[:100]}")
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
