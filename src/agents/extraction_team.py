"""Extraction Team - Specialized Entity Recognition with optional deps."""

import asyncio
import logging
import os
import types
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

try:  # pragma: no cover - allow runtime without PyTorch installed
    import torch  # type: ignore
except Exception:  # pragma: no cover - provide minimal stub implementation
    class _CudaStub:
        @staticmethod
        def is_available() -> bool:
            return False

    torch = types.SimpleNamespace(cuda=_CudaStub())  # type: ignore

try:  # pragma: no cover - transformers optional in lightweight environments
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover - provide stub pipeline helper
    def pipeline(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("transformers is not installed")

try:  # pragma: no cover - optional dependency
    from packaging import version
except ImportError:  # pragma: no cover - fallback when packaging is missing
    version = None

from src.extraction.enhanced_extraction_tools import EnhancedExtractionTools
from src.monitoring.error_telemetry import TelemetrySeverity, log_with_telemetry

logger = logging.getLogger(__name__)
TELEMETRY_AGENT = "extraction_team"


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

        self._cache[cache_key] = self._pipeline_factory(
            task,
            model=model_name,
            device=device,
            **pipeline_kwargs,
        )
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
        self.pipeline_loader = LazyPipelineLoader(
            pipeline_factory=pipeline_factory,
            force_cpu=force_cpu,
        )
        self.models = {}
        self.entity_types = {
            'biomedical': ['DISEASE', 'CHEMICAL', 'GENE', 'PROTEIN', 'CELL_LINE'],
            'neuroscience': ['BRAIN_REGION', 'NEURAL_PATHWAY', 'NEUROTRANSMITTER', 'COGNITIVE_FUNCTION'],
            'psychology': ['PSYCHOLOGICAL_CONCEPT', 'BEHAVIOR', 'EMOTION', 'MENTAL_STATE'],
            'clinical': ['SYMPTOM', 'TREATMENT', 'DIAGNOSIS', 'INTERVENTION', 'OUTCOME']
        }
        self.hf_models_enabled = self._resolve_hf_support()
        self._fallback_tools: Optional[EnhancedExtractionTools] = None
        self._heuristic_patterns: Dict[str, Dict[str, Tuple[str, ...]]] = {
            'biomedical': {
                'label': 'BIOMEDICAL_KEYWORD',
                'terms': (
                    'bio',
                    'cell',
                    'molec',
                    'gene',
                    'protein',
                    'enzyme',
                    'biomarker',
                    'mutation',
                    'immune',
                    'disease',
                    'pathogen',
                    'pharma',
                    'drug',
                    'tissue',
                ),
            },
            'neuroscience': {
                'label': 'NEUROSCIENCE_KEYWORD',
                'terms': (
                    'brain',
                    'neuro',
                    'neural',
                    'synap',
                    'cortex',
                    'hippocamp',
                    'dopamine',
                    'serotonin',
                    'glia',
                    'axon',
                    'neur',
                ),
            },
            'psychology': {
                'label': 'PSYCHOLOGY_KEYWORD',
                'terms': (
                    'psych',
                    'behavior',
                    'cognit',
                    'emotion',
                    'affect',
                    'mental',
                ),
            },
            'clinical': {
                'label': 'CLINICAL_KEYWORD',
                'terms': (
                    'clinic',
                    'patient',
                    'treat',
                    'therap',
                    'symptom',
                    'diagnos',
                    'intervention',
                    'trial',
                    'outcome',
                    'care',
                ),
            },
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
                log_with_telemetry(
                    logger.warning,
                    "Extraction Team: No text content found",
                    agent=TELEMETRY_AGENT,
                    severity=TelemetrySeverity.MINOR,
                    impact="Entity extraction skipped due to missing text",
                )
                return {}
            
            # Extract entities using different models
            entities = {}
            
            if self.hf_models_enabled:
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
            else:
                logger.debug("Extraction Team: Hugging Face models disabled; using heuristic extraction")
                entities.update(self._extract_entities_with_heuristics(text_content))
            
            # Statistical entities
            statistical_entities = await self._extract_statistical_entities(text_content)
            entities['statistical'] = statistical_entities
            
            total_entities = sum(
                len(bucket)
                for bucket in entities.values()
                if isinstance(bucket, list)
            )
            entities['total_entities'] = total_entities

            logger.info("Extraction Team: Extracted %d entities", total_entities)
            return entities
            
        except Exception as e:
            log_with_telemetry(
                logger.error,
                "Extraction Team: Entity extraction failed: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Entity extraction failed",
                details={"error": str(e)},
            )
            return {}
    
    def _get_text_content(self, extraction_result: Dict[str, Any]) -> str:
        """Extract text content from extraction result"""
        if 'archival_content' in extraction_result:
            return extraction_result['archival_content'].get('full_document_text_markdown', '')
        return ''
    
    async def _extract_biomedical_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract biomedical entities using BiomedBERT"""
        if not self.hf_models_enabled:
            return []
        try:
            # Load BiomedBERT model
            if 'biomedbert' not in self.models:
                self.models['biomedbert'] = self.pipeline_loader.get(
                    'biomedbert',
                    "ner",
                    "dmis-lab/biobert-base-cased-v1.1",
                    aggregation_strategy="simple",
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
            log_with_telemetry(
                logger.error,
                "Biomedical entity extraction failed: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Biomedical entity extraction failed",
                details={"error": str(e)},
            )
            return []
    
    async def _extract_neuroscience_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract neuroscience entities using NeuroBERT"""
        if not self.hf_models_enabled:
            return []
        try:
            # Load NeuroBERT model
            if 'neurobert' not in self.models:
                self.models['neurobert'] = self.pipeline_loader.get(
                    'neurobert',
                    "ner",
                    "allenai/scibert_scivocab_uncased",
                    aggregation_strategy="simple",
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
            log_with_telemetry(
                logger.error,
                "Neuroscience entity extraction failed: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Neuroscience entity extraction failed",
                details={"error": str(e)},
            )
            return []
    
    async def _extract_psychology_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract psychology entities using PsychBERT"""
        if not self.hf_models_enabled:
            return []
        try:
            # Load PsychBERT model
            if 'psychbert' not in self.models:
                self.models['psychbert'] = self.pipeline_loader.get(
                    'psychbert',
                    "ner",
                    "mental/mental-bert-base-uncased",
                    aggregation_strategy="simple",
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
            log_with_telemetry(
                logger.error,
                "Psychology entity extraction failed: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Psychology entity extraction failed",
                details={"error": str(e)},
            )
            return []
    
    async def _extract_clinical_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract clinical entities using GatorTron"""
        if not self.hf_models_enabled:
            return []
        try:
            # Load GatorTron model
            if 'gator' not in self.models:
                self.models['gator'] = self.pipeline_loader.get(
                    'gator',
                    "ner",
                    "emilyalsentzer/Bio_ClinicalBERT",
                    aggregation_strategy="simple",
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
            log_with_telemetry(
                logger.error,
                "Clinical entity extraction failed: %s",
                e,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MAJOR,
                impact="Clinical entity extraction failed",
                details={"error": str(e)},
            )
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

    def _extract_entities_with_heuristics(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Use lightweight heuristics to approximate specialised entity buckets."""

        buckets: Dict[str, List[Dict[str, Any]]] = {
            'biomedical': [],
            'neuroscience': [],
            'psychology': [],
            'clinical': [],
        }

        keywords: List[Tuple[str, float]] = []
        tools = self._get_fallback_tools()
        if tools is not None:
            try:
                keywords = tools.extract_semantic_keywords(
                    text,
                    keyphrase_ngram_range=(1, 3),
                    top_n=20,
                )
            except Exception as exc:
                log_with_telemetry(
                    logger.warning,
                    "Extraction Team: heuristic keyword extraction failed with EnhancedExtractionTools (%s)",
                    exc,
                    agent=TELEMETRY_AGENT,
                    severity=TelemetrySeverity.MINOR,
                    impact="Heuristic keyword extraction fallback",
                    details={"error": str(exc)},
                )

        if not keywords:
            keywords = self._basic_keyword_fallback(text, top_n=20)

        text_lower = text.lower()
        for keyword, score in keywords:
            if not keyword:
                continue

            keyword_lower = keyword.lower()
            bucket_name = self._classify_keyword(keyword_lower)
            if bucket_name is None:
                continue

            start = text_lower.find(keyword_lower)
            if start == -1:
                start = text_lower.find(keyword_lower.replace('-', ' '))
            end = start + len(keyword) if start != -1 else -1

            label = self._heuristic_patterns[bucket_name]['label']
            confidence = float(score if isinstance(score, (int, float)) else 0.5)
            confidence = max(0.0, min(1.0, confidence))

            buckets[bucket_name].append({
                'text': keyword,
                'label': label,
                'confidence': confidence,
                'start': start,
                'end': end,
            })

        return buckets

    def _get_fallback_tools(self) -> Optional[EnhancedExtractionTools]:
        """Lazily instantiate EnhancedExtractionTools for heuristic extraction."""

        if self._fallback_tools is None:
            try:
                self._fallback_tools = EnhancedExtractionTools(device="cpu")
            except Exception as exc:
                log_with_telemetry(
                    logger.warning,
                    "Extraction Team: unable to initialise EnhancedExtractionTools for heuristics (%s)",
                    exc,
                    agent=TELEMETRY_AGENT,
                    severity=TelemetrySeverity.MINOR,
                    impact="Heuristic tools unavailable",
                    details={"error": str(exc)},
                )
                self._fallback_tools = None
        return self._fallback_tools

    def _classify_keyword(self, keyword_lower: str) -> Optional[str]:
        """Map a keyword to a heuristic entity bucket."""

        for bucket, config in self._heuristic_patterns.items():
            for term in config['terms']:
                if term in keyword_lower or keyword_lower in term:
                    return bucket
        return None

    @staticmethod
    def _basic_keyword_fallback(text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Simplified keyword extraction when EnhancedExtractionTools are unavailable."""

        import re

        words = re.findall(r"\b[a-zA-Z][a-zA-Z-]{1,}\b", text.lower())
        if not words:
            return []

        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'into',
            'from', 'over', 'under', 'after', 'before', 'about', 'within', 'their',
            'them', 'they', 'his', 'her', 'she', 'him', 'you', 'your', 'ours', 'ourselves',
        }
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        if not filtered_words:
            return []

        frequency = Counter(filtered_words)
        results: List[Tuple[str, float]] = []
        for index, (word, _count) in enumerate(frequency.most_common(top_n)):
            score = max(0.0, 1.0 - (index * 0.05))
            results.append((word, score))
        return results

    def _resolve_hf_support(self) -> bool:
        """Determine whether heavyweight HF models should be enabled."""

        toggle = os.getenv("EXTRACTION_ENABLE_HF_MODELS")
        if toggle is not None:
            enabled = toggle.lower() in {"1", "true", "yes"}
            if not enabled:
                logger.info("Extraction Team: HF entity models disabled via environment toggle")
            return enabled

        min_version = "2.6.0"
        torch_version = getattr(torch, "__version__", None)

        if torch_version is None or version is None:
            logger.debug(
                "Extraction Team: Torch version metadata unavailable; enabling HF models with safe defaults",
            )
            return True

        try:
            current_version = version.parse(str(torch_version).split("+")[0])
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "Extraction Team: Unable to parse torch version %s (%s); defaulting to HF models enabled",
                torch_version,
                exc,
            )
            return True

        if current_version < version.parse(min_version):
            log_with_telemetry(
                logger.warning,
                "Extraction Team: Torch %s detected (<%s); skipping HF entity models to avoid load errors",
                torch_version,
                min_version,
                agent=TELEMETRY_AGENT,
                severity=TelemetrySeverity.MINOR,
                impact="HF entity models disabled",
                details={"torch_version": str(torch_version)},
            )
            return False

        return True
