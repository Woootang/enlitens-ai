"""
External Search Module - FREE APIs for filling knowledge gaps.
Uses Wikipedia, PubMed, Semantic Scholar, and DuckDuckGo.
"""

import logging
import re
import time
from typing import Dict, List, Any, Optional
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)


class ExternalSearchClient:
    """Client for searching external free APIs to fill knowledge gaps."""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        self.search_count = {"wikipedia": 0, "pubmed": 0, "semantic_scholar": 0, "duckduckgo": 0}
    
    def search_entity(self, entity: str, confidence: float) -> Optional[Dict[str, Any]]:
        """Search for an entity using cascade of free APIs."""
        logger.info(f"Searching for low-confidence entity: {entity} (confidence: {confidence:.2f})")
        
        # Try Wikipedia first (best for general terms)
        result = self._search_wikipedia(entity)
        if result:
            return result
        
        # Try PubMed (best for medical/neuroscience terms)
        result = self._search_pubmed(entity)
        if result:
            return result
        
        # Try Semantic Scholar (best for research terms)
        result = self._search_semantic_scholar(entity)
        if result:
            return result
        
        # Try DuckDuckGo as last resort
        result = self._search_duckduckgo(entity)
        if result:
            return result
        
        logger.warning(f"No results found for: {entity}")
        return None
    
    def _sanitize_wikipedia_entity(self, entity: str) -> str:
        """
        Sanitize and validate an entity for Wikipedia API lookup.
        Returns empty string if entity is unsuitable (too long, contains JSON, etc.)
        """
        # Reject JSON-like structures
        if any(char in entity for char in ["{", "}", "[", "]", '":']):
            return ""
        
        cleaned = re.sub(r"\s+", " ", entity.strip())
        cleaned = re.sub(r"[\"'`]", "", cleaned)
        cleaned = cleaned.strip(".,;:()[]{}").strip()
        
        # Reject overly long entities (likely full sentences)
        if len(cleaned) > 80:
            return ""
        
        # Reject entities with too many words (likely sentences)
        word_count = len(cleaned.split())
        if word_count > 8:
            return ""
        
        if not cleaned:
            return ""
        
        return quote(cleaned.replace(" ", "_"), safe="_")

    def _search_wikipedia(self, entity: str) -> Optional[Dict[str, Any]]:
        """Search Wikipedia API (FREE)."""
        try:
            slug = self._sanitize_wikipedia_entity(entity)
            if not slug:
                return None
            url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + slug
            response = self.client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                self.search_count["wikipedia"] += 1
                
                return {
                    "source": "wikipedia",
                    "entity": entity,
                    "summary": data.get("extract", ""),
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
                }
        except Exception as e:
            logger.debug(f"Wikipedia search failed for {entity}: {e}")
        
        return None
    
    def _search_pubmed(self, entity: str) -> Optional[Dict[str, Any]]:
        """Search PubMed API (FREE)."""
        try:
            # Search for term
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                "db": "pubmed",
                "term": entity,
                "retmax": 1,
                "retmode": "json"
            }
            
            response = self.client.get(search_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                id_list = data.get("esearchresult", {}).get("idlist", [])
                
                if id_list:
                    # Get summary for first result
                    summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                    params = {
                        "db": "pubmed",
                        "id": id_list[0],
                        "retmode": "json"
                    }
                    
                    response = self.client.get(summary_url, params=params)
                    
                    if response.status_code == 200:
                        summary_data = response.json()
                        result = summary_data.get("result", {}).get(id_list[0], {})
                        
                        self.search_count["pubmed"] += 1
                        
                        return {
                            "source": "pubmed",
                            "entity": entity,
                            "summary": result.get("title", "") + ". " + result.get("source", ""),
                            "url": f"https://pubmed.ncbi.nlm.nih.gov/{id_list[0]}/"
                        }
            
            time.sleep(0.34)  # Rate limit: 3 requests/second
            
        except Exception as e:
            logger.debug(f"PubMed search failed for {entity}: {e}")
        
        return None
    
    def _search_semantic_scholar(self, entity: str) -> Optional[Dict[str, Any]]:
        """Search Semantic Scholar API (FREE)."""
        try:
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": entity,
                "limit": 1,
                "fields": "title,abstract,url"
            }
            
            backoff = 1.0
            for attempt in range(3):
                response = self.client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    papers = data.get("data", [])
                    
                    if papers:
                        paper = papers[0]
                        self.search_count["semantic_scholar"] += 1
                        
                        return {
                            "source": "semantic_scholar",
                            "entity": entity,
                            "summary": paper.get("title", "") + ". " + (paper.get("abstract", "") or "")[:200],
                            "url": paper.get("url", "")
                        }
                    break
                if response.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                response.raise_for_status()
            time.sleep(1.2)  # Rate limit: ~1 request/second to be safe
            
        except Exception as e:
            logger.debug(f"Semantic Scholar search failed for {entity}: {e}")
        
        return None
    
    def _search_duckduckgo(self, entity: str) -> Optional[Dict[str, Any]]:
        """Search DuckDuckGo Instant Answer API (FREE)."""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                "q": entity,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            }
            
            response = self.client.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                abstract = data.get("Abstract", "")
                
                if abstract:
                    self.search_count["duckduckgo"] += 1
                    
                    return {
                        "source": "duckduckgo",
                        "entity": entity,
                        "summary": abstract,
                        "url": data.get("AbstractURL", "")
                    }
        except Exception as e:
            logger.debug(f"DuckDuckGo search failed for {entity}: {e}")
        
        return None
    
    def enrich_knowledge_base(self, knowledge_base: Dict[str, Any], 
                            low_confidence_entities: List[tuple]) -> Dict[str, Any]:
        """Enrich knowledge base with external search results."""
        if not low_confidence_entities:
            logger.info("No low-confidence entities to enrich")
            return knowledge_base
        
        logger.info(f"Enriching {len(low_confidence_entities)} low-confidence entities...")
        
        enrichments = []
        
        for entity, confidence in low_confidence_entities[:50]:  # Limit to top 50
            result = self.search_entity(entity, confidence)
            if result:
                enrichments.append(result)
        
        knowledge_base["external_enrichment"] = {
            "total_searches": len(low_confidence_entities),
            "successful_enrichments": len(enrichments),
            "search_counts": self.search_count,
            "enrichments": enrichments
        }
        
        logger.info(f"✅ Enriched {len(enrichments)} entities")
        logger.info(f"   - Wikipedia: {self.search_count['wikipedia']}")
        logger.info(f"   - PubMed: {self.search_count['pubmed']}")
        logger.info(f"   - Semantic Scholar: {self.search_count['semantic_scholar']}")
        logger.info(f"   - DuckDuckGo: {self.search_count['duckduckgo']}")
        
        return knowledge_base
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()

