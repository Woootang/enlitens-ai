"""
Confidence Scoring Module - Scores entities/concepts based on mention frequency and context quality.
Higher scores = more confident, lower scores = may need external search.
"""

import logging
import re
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Scores confidence in extracted entities/concepts based on mentions and context."""
    
    def __init__(self, low_threshold: float = 0.5, high_threshold: float = 0.8):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.entity_mentions = defaultdict(list)  # entity -> list of (doc_id, context)
        self.entity_scores = {}  # entity -> confidence score
        
    def add_document(self, doc_id: str, content: Dict[str, Any]):
        """Add a document's content for confidence scoring."""
        # Extract entities from various fields
        entities = self._extract_entities(content)
        
        for entity, context in entities:
            self.entity_mentions[entity].append((doc_id, context))
    
    def _extract_entities(self, content: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Extract entities and their contexts from document content."""
        entities = []
        
        # Extract from research content
        if "research_content" in content:
            research = content["research_content"]
            
            # Findings
            for finding in research.get("findings", []):
                extracted = self._extract_terms(finding)
                entities.extend([(term, finding) for term in extracted])
            
            # Statistics
            for stat in research.get("statistics", []):
                extracted = self._extract_terms(stat)
                entities.extend([(term, stat) for term in extracted])
            
            # Methodologies
            for method in research.get("methodologies", []):
                extracted = self._extract_terms(method)
                entities.extend([(term, method) for term in extracted])
        
        return entities
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract key terms from text (simple approach - can be enhanced)."""
        if not text or len(text) < 3:
            return []
        
        # Common neuroscience/psychology terms to track
        terms_to_track = [
            "ADHD", "autism", "ASD", "neurodivergent", "neurodivergence",
            "executive function", "working memory", "attention", "focus",
            "dopamine", "serotonin", "norepinephrine", "neurotransmitter",
            "prefrontal cortex", "amygdala", "hippocampus", "brain region",
            "anxiety", "depression", "trauma", "PTSD", "OCD",
            "sensory processing", "sensory", "hypersensitivity",
            "emotional regulation", "emotion", "mood",
            "cognitive", "cognition", "memory", "learning",
            "therapy", "treatment", "intervention", "medication",
            "diagnosis", "assessment", "evaluation",
            "child", "adult", "adolescent", "developmental",
            "social", "communication", "language",
            "behavior", "behavioral", "symptom"
        ]
        
        text_lower = text.lower()
        found_terms = []
        
        for term in terms_to_track:
            if term.lower() in text_lower:
                found_terms.append(term)
        
        return found_terms
    
    def calculate_scores(self):
        """Calculate confidence scores for all entities."""
        total_docs = len(set(doc_id for mentions in self.entity_mentions.values() 
                            for doc_id, _ in mentions))
        
        if total_docs == 0:
            logger.warning("No documents to score")
            return
        
        for entity, mentions in self.entity_mentions.items():
            # Factors for confidence:
            # 1. Number of mentions (more = higher confidence)
            # 2. Number of unique documents (spread across docs = higher confidence)
            # 3. Context quality (longer contexts = better)
            
            num_mentions = len(mentions)
            num_docs = len(set(doc_id for doc_id, _ in mentions))
            avg_context_length = sum(len(context) for _, context in mentions) / num_mentions
            
            # Normalize scores (0-1 range)
            mention_score = min(num_mentions / 10.0, 1.0)  # Cap at 10 mentions
            doc_spread_score = min(num_docs / 5.0, 1.0)  # Cap at 5 docs
            context_score = min(avg_context_length / 200.0, 1.0)  # Cap at 200 chars
            
            # Weighted average
            confidence = (
                mention_score * 0.4 +  # 40% weight on mentions
                doc_spread_score * 0.4 +  # 40% weight on doc spread
                context_score * 0.2  # 20% weight on context quality
            )
            
            self.entity_scores[entity] = round(confidence, 3)
        
        logger.info(f"✅ Calculated confidence scores for {len(self.entity_scores)} entities")
    
    def get_low_confidence_entities(self) -> List[Tuple[str, float]]:
        """Get entities with low confidence scores (need external search)."""
        low_conf = [(entity, score) for entity, score in self.entity_scores.items()
                    if score < self.low_threshold]
        
        # Sort by score (lowest first)
        low_conf.sort(key=lambda x: x[1])
        
        return low_conf
    
    def get_high_confidence_entities(self) -> List[Tuple[str, float]]:
        """Get entities with high confidence scores."""
        high_conf = [(entity, score) for entity, score in self.entity_scores.items()
                     if score >= self.high_threshold]
        
        # Sort by score (highest first)
        high_conf.sort(key=lambda x: x[1], reverse=True)
        
        return high_conf
    
    def get_confidence_summary(self) -> Dict[str, Any]:
        """Get a summary of confidence scores."""
        if not self.entity_scores:
            return {}
        
        low_conf = [e for e, s in self.entity_scores.items() if s < self.low_threshold]
        med_conf = [e for e, s in self.entity_scores.items() 
                    if self.low_threshold <= s < self.high_threshold]
        high_conf = [e for e, s in self.entity_scores.items() if s >= self.high_threshold]
        
        return {
            "total_entities": len(self.entity_scores),
            "high_confidence": {
                "count": len(high_conf),
                "percentage": round(len(high_conf) / len(self.entity_scores) * 100, 1),
                "entities": high_conf[:20]  # Top 20
            },
            "medium_confidence": {
                "count": len(med_conf),
                "percentage": round(len(med_conf) / len(self.entity_scores) * 100, 1),
                "entities": med_conf[:20]  # Top 20
            },
            "low_confidence": {
                "count": len(low_conf),
                "percentage": round(len(low_conf) / len(self.entity_scores) * 100, 1),
                "entities": low_conf[:20]  # Top 20 (candidates for external search)
            }
        }
    
    def add_to_knowledge_base(self, knowledge_base: Dict[str, Any]) -> Dict[str, Any]:
        """Add confidence scores to knowledge base."""
        if not self.entity_scores:
            logger.warning("No confidence scores calculated")
            return knowledge_base
        
        summary = self.get_confidence_summary()
        
        knowledge_base["confidence_index"] = {
            "metadata": {
                "low_threshold": self.low_threshold,
                "high_threshold": self.high_threshold,
                "total_entities_tracked": len(self.entity_scores)
            },
            "summary": summary,
            "entity_scores": self.entity_scores
        }
        
        logger.info("✅ Added confidence scores to knowledge base")
        logger.info(f"   - High confidence: {summary['high_confidence']['count']} entities")
        logger.info(f"   - Medium confidence: {summary['medium_confidence']['count']} entities")
        logger.info(f"   - Low confidence: {summary['low_confidence']['count']} entities")
        
        return knowledge_base

