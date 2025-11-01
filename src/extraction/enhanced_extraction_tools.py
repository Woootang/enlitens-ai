"""
Enhanced extraction tools for semantic analysis and content generation.

This module provides advanced NLP tools including KeyBERT for semantic keyword
extraction, VADER for sentiment analysis, and BERTopic for topic modeling.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from statistics import mean, pstdev
try:
    from keybert import KeyBERT
except Exception:  # pragma: no cover - optional dependency
    KeyBERT = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover - optional dependency
    SentimentIntensityAnalyzer = None
try:
    from bertopic import BERTopic
    import cuml
    from cuml.cluster import HDBSCAN
    from cuml.manifold import UMAP
    _GPU_TOPIC_STACK_AVAILABLE = True
except Exception:
    try:
        from bertopic import BERTopic  # type: ignore
        _GPU_TOPIC_STACK_AVAILABLE = False
    except Exception:  # pragma: no cover - optional dependency
        BERTopic = None
        _GPU_TOPIC_STACK_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TopicDiscoveryResult:
    """Structured result returned from topic discovery."""

    topics: List[int] = field(default_factory=list)
    topic_keywords: Dict[int, List[str]] = field(default_factory=dict)
    enhanced_analysis: Dict[str, Any] = field(default_factory=dict)
    probabilities: Optional[List[List[float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a serialisable dictionary."""

        return {
            "topics": self.topics,
            "topic_keywords": self.topic_keywords,
            "enhanced_analysis": self.enhanced_analysis,
            "probabilities": self.probabilities,
            "metadata": self.metadata,
        }

class EnhancedExtractionTools:
    """
    Enhanced extraction tools for semantic analysis and content generation.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        if SentimentIntensityAnalyzer is not None:
            try:
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except Exception as sentiment_error:
                logger.warning(
                    "Failed to initialise VADER sentiment analyzer",
                    extra={"event": "sentiment_initialisation_error", "error": str(sentiment_error)},
                )
                self.sentiment_analyzer = None
        else:
            logger.warning(
                "VADER sentiment analyzer not available, falling back to neutral scores",
                extra={"event": "sentiment_dependency_missing"},
            )
            self.sentiment_analyzer = None
        self.keybert_model = None
        self.sentence_transformer = None
        self.bertopic_model = None
        logger.info(f"Enhanced extraction tools initialized on {device}")

    def _load_keybert(self):
        """Load KeyBERT model if not already loaded."""
        if self.keybert_model is None:
            try:
                if KeyBERT is None or SentenceTransformer is None:
                    raise RuntimeError("KeyBERT dependencies are unavailable")
                # Use a smaller, efficient model for better performance
                self.sentence_transformer = SentenceTransformer('all-mpnet-base-v2', device=self.device)
                self.keybert_model = KeyBERT(model=self.sentence_transformer)
                logger.info("KeyBERT model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load KeyBERT model: {e}")
                raise

    def _load_bertopic(self):
        """Load BERTopic model with GPU acceleration if not already loaded."""
        if self.bertopic_model is None:
            try:
                if BERTopic is None:
                    raise RuntimeError("BERTopic dependency is unavailable")
                if SentenceTransformer is None:
                    raise RuntimeError("SentenceTransformer dependency is unavailable")
                if _GPU_TOPIC_STACK_AVAILABLE and self.device == "cuda":
                    # Configure BERTopic with GPU-accelerated components
                    self.bertopic_model = BERTopic(
                        embedding_model=SentenceTransformer('all-mpnet-base-v2', device=self.device),
                        umap_model=UMAP(n_components=5, random_state=42),
                        hdbscan_model=HDBSCAN(min_cluster_size=10),
                        low_memory=True,
                        calculate_probabilities=False
                    )
                    logger.info("BERTopic model loaded with GPU acceleration")
                else:
                    raise RuntimeError("GPU topic stack unavailable or device not cuda")
            except Exception as e:
                logger.warning(f"GPU BERTopic load failed ({e}); falling back to CPU implementation")
                # Fall back to CPU-friendly BERTopic configuration
                try:
                    self.bertopic_model = BERTopic(
                        embedding_model=SentenceTransformer('all-mpnet-base-v2', device='cpu'),
                        low_memory=True,
                        calculate_probabilities=False
                    )
                    logger.info("BERTopic model loaded with CPU backend")
                except Exception as cpu_error:
                    logger.error(f"Failed to load BERTopic model on CPU fallback: {cpu_error}")
                    raise

    def extract_semantic_keywords(self, text: str,
                                keyphrase_ngram_range: Tuple[int, int] = (1, 3),
                                stop_words: str = "english",
                                top_n: int = 10,
                                candidates: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Extract semantic keywords (fast fallback).
        
        Args:
            text: Input text to analyze
            keyphrase_ngram_range: Range of n-grams to consider
            stop_words: Stop words to filter out
            top_n: Number of top keywords to return
            candidates: Optional list of candidate phrases to score
            
        Returns:
            List of (keyword, score) tuples
        """
        # Fast fallback - return simple keywords based on word frequency
        logger.info("Semantic keyword extraction: using fast fallback")
        return self._fallback_keyword_extraction(text, top_n)

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER with graceful fallback."""

        try:
            if not text or not text.strip():
                logger.info(
                    "Sentiment analysis skipped for empty text",
                    extra={"event": "sentiment_skipped", "text_length": len(text or "")},
                )
                return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}

            if self.sentiment_analyzer is None:
                if SentimentIntensityAnalyzer is None:
                    logger.warning(
                        "Sentiment analyzer dependency missing, returning neutral scores",
                        extra={"event": "sentiment_dependency_missing"},
                    )
                    return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}
                try:
                    self.sentiment_analyzer = SentimentIntensityAnalyzer()
                except Exception as load_error:
                    logger.warning(
                        "Sentiment analyzer unavailable, returning neutral scores",
                        extra={"event": "sentiment_fallback", "error": str(load_error)},
                    )
                    return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}

            scores = self.sentiment_analyzer.polarity_scores(text)
            logger.info(
                "Sentiment analysis completed",
                extra={
                    "event": "sentiment_completed",
                    "compound": scores.get("compound"),
                    "text_length": len(text),
                },
            )
            return scores

        except Exception as e:
            logger.error(
                "Error during sentiment analysis, returning neutral scores",
                extra={"event": "sentiment_error", "error": str(e)},
            )
            return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}

    def discover_topics(
        self,
        texts: List[str],
        min_topic_size: int = 10,
        nr_topics: Optional[int] = None,
    ) -> TopicDiscoveryResult:
        """Discover topics using BERTopic with GPU acceleration."""

        try:
            valid_texts = [t for t in texts if t and t.strip()]
            if not valid_texts:
                logger.warning(
                    "No valid texts provided for topic discovery",
                    extra={"event": "topic_discovery_skipped", "text_count": len(texts)},
                )
                return TopicDiscoveryResult(
                    metadata={
                        "reason": "no_valid_texts",
                        "requested_text_count": len(texts),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            if len(valid_texts) < 2:
                # Lower severity: this is expected when only one sample exists
                logger.info(
                    "Topic discovery requires at least 2 samples, skipping",
                    extra={"event": "topic_discovery_skipped", "text_count": len(valid_texts)},
                )
                return TopicDiscoveryResult(
                    metadata={
                        "reason": "insufficient_samples",
                        "valid_text_count": len(valid_texts),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

            self._load_bertopic()

            topics, probs = self.bertopic_model.fit_transform(valid_texts)

            topic_info = self.bertopic_model.get_topic_info()
            topic_keywords: Dict[int, List[str]] = {}

            for topic_id in topic_info["Topic"]:
                if topic_id != -1:
                    keywords = self.bertopic_model.get_topic(topic_id)
                    topic_keywords[topic_id] = [word for word, _ in keywords[:10]]

            enhanced_analysis = self._enhance_topic_analysis(valid_texts, topics, topic_keywords)

            metadata = {
                "text_count": len(valid_texts),
                "topic_count": len(topic_keywords),
                "timestamp": datetime.utcnow().isoformat(),
                "backend": "gpu" if _GPU_TOPIC_STACK_AVAILABLE and self.device == "cuda" else "cpu",
                "min_topic_size": min_topic_size,
                "nr_topics": nr_topics,
            }

            logger.info(
                "Discovered topics",
                extra={
                    "event": "topic_discovery_completed",
                    "topic_count": metadata["topic_count"],
                    "text_count": metadata["text_count"],
                },
            )

            return TopicDiscoveryResult(
                topics=list(topics),
                topic_keywords=topic_keywords,
                enhanced_analysis=enhanced_analysis,
                probabilities=probs.tolist() if hasattr(probs, "tolist") else probs,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(
                "Error discovering topics",
                extra={"event": "topic_discovery_error", "error": str(e)},
            )
            return TopicDiscoveryResult(
                metadata={
                    "reason": "exception",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

    def _enhance_topic_analysis(self, texts: List[str], topics: List[int],
                              topic_keywords: Dict[int, List[str]]) -> Dict[str, Any]:
        """Enhance topic analysis with additional insights."""
        try:
            analysis = {
                "topic_distribution": {},
                "pain_point_indicators": [],
                "clinical_themes": [],
                "treatment_patterns": []
            }

            # Calculate topic distribution
            total_docs = len(topics)
            for topic_id in topic_keywords.keys():
                count = topics.count(topic_id)
                analysis["topic_distribution"][topic_id] = {
                    "count": count,
                    "percentage": (count / total_docs) * 100,
                    "keywords": topic_keywords[topic_id]
                }

            # Identify pain point indicators
            all_keywords = [keyword for keywords in topic_keywords.values()
                          for keyword in keywords]

            pain_indicators = [
                "anxiety", "depression", "trauma", "stress", "overwhelm",
                "struggle", "difficult", "hard", "overcome", "help"
            ]

            analysis["pain_point_indicators"] = [
                keyword for keyword in all_keywords
                if any(indicator in keyword.lower() for indicator in pain_indicators)
            ]

            # Identify clinical themes
            clinical_terms = [
                "therapy", "treatment", "mental health", "disorder",
                "symptom", "diagnosis", "medication", "coping"
            ]

            analysis["clinical_themes"] = [
                keyword for keyword in all_keywords
                if any(term in keyword.lower() for term in clinical_terms)
            ]

            return analysis

        except Exception as e:
            logger.error(f"Error enhancing topic analysis: {e}")
            return {}

    def analyze_client_intakes(self, intake_texts: List[str]) -> Dict[str, Any]:
        """
        Comprehensive analysis of client intake data.

        Args:
            intake_texts: List of client intake form texts

        Returns:
            Comprehensive analysis including topics, pain points, and themes
        """
        try:
            logger.info(f"Analyzing {len(intake_texts)} client intakes")

            # Discover topics in intakes
            topic_result = self.discover_topics(intake_texts, min_topic_size=3)

            # Analyze sentiment patterns
            sentiment_analysis = self._analyze_sentiment_patterns(intake_texts)

            # Extract key themes
            key_themes = self._extract_key_themes(intake_texts)

            # Identify common pain points
            pain_points = self._identify_pain_points(intake_texts)

            return {
                "topic_modeling": {
                    "topic_assignments": topic_result.topics,
                    "topic_keywords": topic_result.topic_keywords,
                    "topic_distribution": topic_result.enhanced_analysis.get("topic_distribution", {}),
                    "enhanced_analysis": topic_result.enhanced_analysis,
                    "metadata": topic_result.metadata,
                },
                "sentiment_analysis": sentiment_analysis,
                "key_themes": key_themes,
                "pain_points": pain_points,
                "total_intakes": len(intake_texts),
                "analysis_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing client intakes: {e}")
            return {}

    def analyze_founder_transcripts(self, transcript_texts: List[str]) -> Dict[str, Any]:
        """
        Analyze founder transcripts to understand voice and themes.

        Args:
            transcript_texts: List of founder transcript texts

        Returns:
            Analysis of founder voice patterns and themes
        """
        try:
            logger.info(f"Analyzing {len(transcript_texts)} founder transcripts")

            # Discover topics in transcripts
            topic_result = self.discover_topics(transcript_texts, min_topic_size=3)

            # Analyze sentiment and tone
            sentiment_analysis = self._analyze_sentiment_patterns(transcript_texts)

            # Extract founder voice characteristics
            voice_characteristics = self._extract_founder_voice(transcript_texts)

            # Identify key messages and themes
            key_messages = self._extract_key_messages(transcript_texts)

            return {
                "topic_modeling": {
                    "topic_assignments": topic_result.topics,
                    "topic_keywords": topic_result.topic_keywords,
                    "enhanced_analysis": topic_result.enhanced_analysis,
                    "metadata": topic_result.metadata,
                },
                "sentiment_analysis": sentiment_analysis,
                "voice_characteristics": voice_characteristics,
                "key_messages": key_messages,
                "total_transcripts": len(transcript_texts),
                "analysis_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing founder transcripts: {e}")
            return {}

    def _analyze_sentiment_patterns(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze sentiment patterns across multiple texts."""
        try:
            sentiment_scores = []
            for text in texts:
                scores = self.analyze_sentiment(text)
                sentiment_scores.append(scores)

            if not sentiment_scores:
                return {}

            # Calculate aggregate statistics
            compound_scores = [s["compound"] for s in sentiment_scores]

            return {
                "overall_compound": sum(compound_scores) / len(compound_scores),
                "positive_ratio": sum(1 for s in compound_scores if s > 0.1) / len(compound_scores),
                "negative_ratio": sum(1 for s in compound_scores if s < -0.1) / len(compound_scores),
                "neutral_ratio": sum(1 for s in compound_scores if -0.1 <= s <= 0.1) / len(compound_scores),
                "individual_scores": sentiment_scores
            }

        except Exception as e:
            logger.error(f"Error analyzing sentiment patterns: {e}")
            return {}

    def _extract_key_themes(self, texts: List[str]) -> List[str]:
        """Extract key themes from texts."""
        try:
            # Combine all texts
            combined_text = " ".join(texts)

            # Extract semantic keywords
            keywords = self.extract_semantic_keywords(combined_text, top_n=20)

            # Filter for theme-like keywords (longer phrases)
            themes = [keyword for keyword, score in keywords
                     if len(keyword.split()) > 2 and score > 0.3]

            return themes[:10]  # Top 10 themes

        except Exception as e:
            logger.error(f"Error extracting key themes: {e}")
            return []

    def _identify_pain_points(self, texts: List[str]) -> List[str]:
        """Identify common pain points from client texts."""
        try:
            # Common pain point indicators
            pain_indicators = {
                "emotional": ["anxiety", "depression", "trauma", "stress", "overwhelm", "sadness"],
                "functional": ["concentration", "focus", "organization", "time management", "work"],
                "social": ["relationships", "social", "connection", "communication", "isolation"],
                "physical": ["sleep", "energy", "headache", "fatigue", "appetite"],
                "treatment": ["medication", "therapy", "treatment", "help", "support"]
            }

            # Count occurrences
            pain_point_counts = {}
            combined_text = " ".join(texts).lower()

            for category, indicators in pain_indicators.items():
                for indicator in indicators:
                    count = combined_text.count(indicator)
                    if count > 0:
                        pain_point_counts[f"{category}_{indicator}"] = count

            # Sort by frequency
            sorted_pain_points = sorted(pain_point_counts.items(),
                                      key=lambda x: x[1], reverse=True)

            return [point for point, count in sorted_pain_points[:15]]

        except Exception as e:
            logger.error(f"Error identifying pain points: {e}")
            return []

    def _extract_founder_voice(self, texts: List[str]) -> Dict[str, Any]:
        """Extract founder voice characteristics."""
        try:
            combined_text = " ".join(texts).lower()

            # Voice characteristics to analyze
            characteristics = {
                "directness": ["traditional", "instead", "rather", "actually", "real", "honest"],
                "neuroscience_focus": ["brain", "neuroscience", "neuroplasticity", "neural", "biology"],
                "hopeful_tone": ["possible", "can", "will", "hope", "transform", "heal", "change"],
                "rebellious_spirit": ["different", "challenge", "against", "better", "revolutionary"],
                "clinical_expertise": ["therapy", "clinical", "treatment", "evidence", "research"]
            }

            voice_profile = {}
            for characteristic, indicators in characteristics.items():
                count = sum(combined_text.count(indicator) for indicator in indicators)
                voice_profile[characteristic] = count

            return voice_profile

        except Exception as e:
            logger.error(f"Error extracting founder voice: {e}")
            return {}

    def _extract_key_messages(self, texts: List[str]) -> List[str]:
        """Extract key messages from founder transcripts."""
        try:
            # Look for repeated phrases and key concepts
            combined_text = " ".join(texts)

            # Extract longer phrases that might be key messages
            keywords = self.extract_semantic_keywords(combined_text, top_n=30)

            # Filter for message-like phrases
            messages = [keyword for keyword, score in keywords
                       if len(keyword.split()) >= 3 and score > 0.2]

            return messages[:15]  # Top 15 key messages

        except Exception as e:
            logger.error(f"Error extracting key messages: {e}")
            return []

    def extract_client_pain_points(self, intake_texts: List[str]) -> Dict[str, Any]:
        """
        Extract client pain points from intake forms.
        
        Args:
            intake_texts: List of client intake texts
            
        Returns:
            Dictionary with pain point analysis
        """
        try:
            # Analyze sentiment for each intake
            sentiment_scores = []
            for text in intake_texts:
                scores = self.analyze_sentiment(text)
                sentiment_scores.append(scores['compound'])

            if not sentiment_scores:
                logger.info(
                    "No sentiment scores calculated for pain point extraction",
                    extra={"event": "pain_point_skipped"},
                )
                return {}

            mean_score = float(mean(sentiment_scores))
            std_score = float(pstdev(sentiment_scores)) if len(sentiment_scores) > 1 else 0.0

            if std_score < 1e-6:
                if mean_score < -0.2:
                    high_priority_indices = list(range(len(sentiment_scores)))
                else:
                    high_priority_indices = []
            else:
                z_scores = [
                    (score - mean_score) / std_score for score in sentiment_scores
                ]
                high_priority_indices = [
                    idx for idx, z in enumerate(z_scores) if z <= -1.0
                ]

            if not high_priority_indices and len(sentiment_scores) > 0:
                min_score = min(sentiment_scores)
                high_priority_indices = [
                    idx for idx, score in enumerate(sentiment_scores) if score == min_score
                ]

            high_priority_texts = [intake_texts[i] for i in high_priority_indices]

            logger.info(
                "Identified high priority intakes",
                extra={
                    "event": "pain_point_prioritised",
                    "high_priority_count": len(high_priority_indices),
                    "mean_score": mean_score,
                    "std_score": std_score,
                },
            )
            
            # Extract keywords from high-priority intakes
            pain_point_keywords = []
            for text in high_priority_texts:
                keywords = self.extract_semantic_keywords(
                    text, 
                    keyphrase_ngram_range=(1, 4),
                    top_n=5
                )
                pain_point_keywords.extend([kw[0] for kw in keywords])
            
            # Discover topics in high-priority intakes
            topic_result = (
                self.discover_topics(high_priority_texts)
                if high_priority_texts
                else TopicDiscoveryResult()
            )

            return {
                "sentiment_scores": sentiment_scores,
                "high_priority_count": len(high_priority_indices),
                "pain_point_keywords": pain_point_keywords,
                "topic_keywords": topic_result.topic_keywords,
                "topic_metadata": topic_result.metadata,
                "high_priority_texts": high_priority_texts
            }
            
        except Exception as e:
            logger.error(f"Error extracting client pain points: {e}")
            return {}

    def extract_founder_voice_patterns(self, transcript_texts: List[str]) -> Dict[str, Any]:
        """
        Extract voice patterns from founder transcripts.
        
        Args:
            transcript_texts: List of transcript texts
            
        Returns:
            Dictionary with voice pattern analysis
        """
        try:
            # Extract semantic keywords from transcripts
            founder_keywords = []
            for text in transcript_texts:
                keywords = self.extract_semantic_keywords(
                    text,
                    keyphrase_ngram_range=(1, 3),
                    top_n=8
                )
                founder_keywords.extend([kw[0] for kw in keywords])
            
            # Discover topics in transcripts
            topic_result = self.discover_topics(transcript_texts)
            
            # Analyze sentiment patterns
            sentiment_scores = [self.analyze_sentiment(text)['compound'] for text in transcript_texts]
            
            return {
                "founder_keywords": founder_keywords,
                "topic_keywords": topic_result.topic_keywords,
                "topic_metadata": topic_result.metadata,
                "sentiment_scores": sentiment_scores,
                "total_transcripts": len(transcript_texts)
            }
            
        except Exception as e:
            logger.error(f"Error extracting founder voice patterns: {e}")
            return {}

    def generate_content_insights(self, research_text: str, 
                                client_insights: Dict[str, Any],
                                founder_insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate content insights by combining research with client and founder data.
        
        Args:
            research_text: Research paper text
            client_insights: Client pain point analysis
            founder_insights: Founder voice pattern analysis
            
        Returns:
            Dictionary with combined insights
        """
        try:
            # Extract keywords from research
            research_keywords = self.extract_semantic_keywords(
                research_text,
                keyphrase_ngram_range=(1, 4),
                top_n=15
            )
            
            # Find intersections between research and client pain points
            research_terms = [kw[0].lower() for kw in research_keywords]
            pain_point_terms = [term.lower() for term in client_insights.get("pain_point_keywords", [])]
            
            relevant_pain_points = []
            for term in pain_point_terms:
                if any(term in research_term or research_term in term 
                      for research_term in research_terms):
                    relevant_pain_points.append(term)
            
            # Find intersections with founder voice patterns
            founder_terms = [term.lower() for term in founder_insights.get("founder_keywords", [])]
            relevant_founder_patterns = []
            for term in founder_terms:
                if any(term in research_term or research_term in term 
                      for research_term in research_terms):
                    relevant_founder_patterns.append(term)
            
            return {
                "research_keywords": [kw[0] for kw in research_keywords],
                "relevant_pain_points": relevant_pain_points,
                "relevant_founder_patterns": relevant_founder_patterns,
                "content_opportunities": {
                    "pain_point_alignment": len(relevant_pain_points),
                    "voice_alignment": len(relevant_founder_patterns),
                    "total_insights": len(research_keywords)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating content insights: {e}")
            return {}

    def _fallback_keyword_extraction(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Simple fallback keyword extraction using word frequency."""
        import re
        from collections import Counter
        
        # Simple word frequency extraction
        words = re.findall(r'\w+', text.lower())
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Get most common words
        word_counts = Counter(words).most_common(min(top_n, len(words)))
        
        # Return as tuples with fake scores (decreasing from 1.0)
        result = []
        for i, (word, count) in enumerate(word_counts):
            score = 1.0 - (i * 0.1)  # Decreasing scores
            result.append((word, score))
        
        return result

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.bertopic_model:
                del self.bertopic_model
            if self.keybert_model:
                del self.keybert_model
            if self.sentence_transformer:
                del self.sentence_transformer
            logger.info("Enhanced extraction tools cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
