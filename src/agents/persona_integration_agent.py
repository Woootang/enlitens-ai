"""
Persona Integration Agent - Integrates client personas into knowledge base.
Loads 57 personas and extracts insights to enhance research content.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter

logger = logging.getLogger(__name__)


class PersonaIntegrationAgent:
    """Agent that integrates client persona insights into the knowledge base."""
    
    def __init__(self, personas_dir: str = "enlitens_client_profiles/profiles"):
        self.personas_dir = Path(personas_dir)
        self.personas = []
        self.insights = {}
        
    def load_personas(self) -> bool:
        """Load all persona files from the profiles directory."""
        try:
            persona_files = sorted(self.personas_dir.glob("persona_cluster_*.json"))
            
            if not persona_files:
                logger.warning(f"No persona files found in {self.personas_dir}")
                return False
            
            for persona_file in persona_files:
                try:
                    with open(persona_file) as f:
                        persona = json.load(f)
                        self.personas.append(persona)
                except Exception as e:
                    logger.warning(f"Failed to load {persona_file.name}: {e}")
            
            logger.info(f"✅ Loaded {len(self.personas)} personas")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load personas: {e}")
            return False
    
    def extract_insights(self) -> Dict[str, Any]:
        """Extract aggregated insights from all personas."""
        if not self.personas:
            logger.warning("No personas loaded, skipping insight extraction")
            return {}
        
        # Collect data from all personas
        pain_points = []
        therapy_goals = []
        seo_keywords = []
        local_entities = []
        age_ranges = []
        situations = []
        neurodivergence_types = []
        
        for persona in self.personas:
            try:
                # Pain points from goals_barriers
                if "goals_barriers" in persona:
                    goals = persona["goals_barriers"]
                    if "whats_in_the_way" in goals:
                        pain_points.extend(goals["whats_in_the_way"])
                    if "what_they_want_to_change" in goals:
                        therapy_goals.extend(goals["what_they_want_to_change"])
                
                # SEO keywords from marketing_seo
                if "marketing_seo" in persona:
                    seo = persona["marketing_seo"]
                    if "primary_keywords" in seo:
                        seo_keywords.extend(seo["primary_keywords"])
                    if "local_entities" in seo:
                        local_entities.extend(seo["local_entities"])
                
                # Demographics
                if "identity_demographics" in persona:
                    demo = persona["identity_demographics"]
                    if "age_range" in demo and demo["age_range"]:
                        age_ranges.append(demo["age_range"])
                    if "current_life_situation" in demo and demo["current_life_situation"]:
                        situations.append(demo["current_life_situation"])
                
                # Neurodivergence types
                if "neurodivergence_mental_health" in persona:
                    neuro = persona["neurodivergence_mental_health"]
                    if "identities" in neuro:
                        neurodivergence_types.extend(neuro["identities"])
                        
            except Exception as e:
                logger.warning(f"Error extracting from persona: {e}")
        
        # Aggregate and rank
        self.insights = {
            "total_personas": len(self.personas),
            "top_pain_points": self._top_items(pain_points, 20),
            "top_therapy_goals": self._top_items(therapy_goals, 20),
            "top_seo_keywords": self._top_items(seo_keywords, 30),
            "top_local_entities": self._top_items(local_entities, 25),
            "age_distribution": dict(Counter(age_ranges).most_common(15)),
            "common_situations": self._top_items(situations, 20),
            "neurodivergence_types": dict(Counter(neurodivergence_types).most_common(10)),
        }
        
        logger.info(f"✅ Extracted insights from {len(self.personas)} personas")
        logger.info(f"   - {len(self.insights['top_pain_points'])} pain points")
        logger.info(f"   - {len(self.insights['top_therapy_goals'])} therapy goals")
        logger.info(f"   - {len(self.insights['top_seo_keywords'])} SEO keywords")
        logger.info(f"   - {len(self.insights['top_local_entities'])} local entities")
        
        return self.insights
    
    def _top_items(self, items: List[str], limit: int = 20) -> List[str]:
        """Get top N most common items, preserving order."""
        if not items:
            return []
        
        # Count occurrences
        counts = Counter(items)
        
        # Return top N, sorted by count
        return [item for item, count in counts.most_common(limit)]
    
    def get_persona_segments(self) -> List[Dict[str, Any]]:
        """Get simplified persona segments for knowledge base."""
        segments = []
        
        for persona in self.personas:
            try:
                segment = {
                    "persona_id": persona.get("meta", {}).get("profile_id", "unknown"),
                    "persona_name": persona.get("meta", {}).get("persona_name", "Unknown"),
                    "age_range": persona.get("identity_demographics", {}).get("age_range", "Unknown"),
                    "situation": persona.get("identity_demographics", {}).get("current_life_situation", "Unknown"),
                    "neurodivergence": persona.get("neurodivergence_mental_health", {}).get("identities", []),
                    "pain_points": persona.get("goals_barriers", {}).get("whats_in_the_way", [])[:3],
                    "therapy_goals": persona.get("goals_barriers", {}).get("what_they_want_to_change", [])[:3],
                    "locality": persona.get("identity_demographics", {}).get("locality", "St. Louis area"),
                }
                segments.append(segment)
            except Exception as e:
                logger.warning(f"Error creating segment: {e}")
        
        return segments
    
    def integrate_into_knowledge_base(self, knowledge_base: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate persona insights into the knowledge base."""
        if not self.personas:
            logger.warning("No personas loaded, skipping integration")
            return knowledge_base
        
        if not self.insights:
            self.extract_insights()
        
        # Add persona section to knowledge base
        knowledge_base["persona_insights"] = {
            "metadata": {
                "total_personas": len(self.personas),
                "source": "cluster-based generation from 224 client intakes",
                "generated_date": "2025-11-08"
            },
            "aggregated_insights": self.insights,
            "persona_segments": self.get_persona_segments()
        }
        
        logger.info("✅ Integrated persona insights into knowledge base")
        
        return knowledge_base


def load_and_integrate_personas(knowledge_base: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to load personas and integrate them."""
    agent = PersonaIntegrationAgent()
    
    if not agent.load_personas():
        logger.warning("Failed to load personas, continuing without persona integration")
        return knowledge_base
    
    return agent.integrate_into_knowledge_base(knowledge_base)

