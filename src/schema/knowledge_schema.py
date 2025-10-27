"""
Knowledge Base Schema for Enlitens AI Pipeline

This module defines the Pydantic models for the JSON knowledge base structure.
The schema is designed to capture all aspects of neuroscience research processing:
- Document metadata and structure
- Entity extraction results
- AI synthesis outputs
- Clinical applications
- Graph hints for future Neo4j integration
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import uuid


class EvidenceStrength(str, Enum):
    """Evidence strength levels"""
    STRONG = "strong"
    MODERATE = "moderate"
    PRELIMINARY = "preliminary"


class ConceptType(str, Enum):
    """Types of neuroscientific concepts"""
    MECHANISM = "mechanism"
    STRUCTURE = "structure"
    PROCESS = "process"
    SYSTEM = "system"
    CIRCUIT = "circuit"
    NEUROTRANSMITTER = "neurotransmitter"


class EntityCategory(str, Enum):
    """Categories of extracted entities"""
    PROTEIN = "protein"
    GENE = "gene"
    CHEMICAL = "chemical"
    DISEASE = "disease"
    CELL_LINE = "cell_line"
    CELL_TYPE = "cell_type"
    ORGAN = "organ"
    ORGANISM = "organism"
    BRAIN_REGION = "brain_region"
    NEUROTRANSMITTER = "neurotransmitter"
    NEURAL_CIRCUIT = "neural_circuit"
    NEURAL_PROCESS = "neural_process"
    METHOD = "method"
    MATERIAL = "material"
    METRIC = "metric"
    TASK = "task"
    OTHER = "other"


class Author(BaseModel):
    """Author information"""
    name: str = Field(..., description="Author name")
    affiliation: Optional[str] = Field(None, description="Author affiliation")
    orcid: Optional[str] = Field(None, description="ORCID identifier")
    email: Optional[str] = Field(None, description="Author email")


class SourceMetadata(BaseModel):
    """Source document metadata"""
    title: str = Field(..., description="Document title")
    authors: List[Author] = Field(..., description="List of authors")
    publication_date: Optional[str] = Field(None, description="Publication date")
    journal: Optional[str] = Field(None, description="Journal name")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    pmid: Optional[str] = Field(None, description="PubMed ID")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    source_filename: str = Field(..., description="Original filename")
    ingestion_timestamp: datetime = Field(default_factory=datetime.now, description="When document was ingested")
    impact_factor: Optional[float] = Field(None, description="Journal impact factor")
    study_type: Optional[str] = Field(None, description="Type of study (e.g., RCT, meta-analysis)")
    sample_size: Optional[int] = Field(None, description="Sample size")


class Section(BaseModel):
    """Document section"""
    title: str = Field(..., description="Section title")
    level: int = Field(1, description="Section hierarchy level")
    content: str = Field(..., description="Section content")
    page_number: Optional[int] = Field(None, description="Page number")


class Table(BaseModel):
    """Extracted table"""
    caption: Optional[str] = Field(None, description="Table caption")
    content: str = Field(..., description="Table content")
    structure: Optional[Dict[str, Any]] = Field(None, description="Table structure")
    cells: List[Dict[str, Any]] = Field(default_factory=list, description="Table cells")
    page_number: Optional[int] = Field(None, description="Page number")


class Figure(BaseModel):
    """Extracted figure"""
    caption: Optional[str] = Field(None, description="Figure caption")
    content: str = Field(..., description="Figure content")
    figure_id: Optional[str] = Field(None, description="Figure identifier")
    page_number: Optional[int] = Field(None, description="Page number")


class ArchivalContent(BaseModel):
    """Archived document content"""
    full_document_text_markdown: str = Field(..., description="Full document text in markdown")
    abstract_markdown: str = Field(..., description="Abstract in markdown")
    sections: List[Section] = Field(default_factory=list, description="Document sections")
    tables: List[Table] = Field(default_factory=list, description="Extracted tables")
    figures: List[Figure] = Field(default_factory=list, description="Extracted figures")
    references: List[str] = Field(default_factory=list, description="Reference list")
    equations: List[Dict[str, Any]] = Field(default_factory=list, description="Mathematical equations")
    extraction_quality_score: float = Field(0.0, description="Quality score of extraction")


class KeyFinding(BaseModel):
    """Key finding from research"""
    finding_text: str = Field(..., description="The specific finding")
    evidence_strength: EvidenceStrength = Field(..., description="Strength of evidence")
    relevance_to_enlitens: str = Field(..., description="How this applies to therapy")


class NeuroscientificConcept(BaseModel):
    """Neuroscientific concept"""
    concept_name: str = Field(..., description="Concept name")
    concept_type: ConceptType = Field(..., description="Type of concept")
    definition_accessible: str = Field(..., description="Accessible definition")
    clinical_relevance: str = Field(..., description="Clinical relevance")


class ClinicalApplication(BaseModel):
    """Clinical application"""
    intervention: str = Field(..., description="Therapeutic intervention")
    mechanism: str = Field(..., description="How it works at neural level")
    evidence_level: EvidenceStrength = Field(..., description="Evidence level")
    timeline: str = Field(..., description="Expected timeline")
    contraindications: str = Field(..., description="Who should avoid this")


class TherapeuticTarget(BaseModel):
    """Therapeutic target"""
    target_name: str = Field(..., description="Target name")
    intervention_type: str = Field(..., description="How to modulate it")
    expected_outcomes: str = Field(..., description="Expected outcomes")
    practical_application: str = Field(..., description="Practical application")


class ClientPresentation(BaseModel):
    """How clients might experience this"""
    symptom_description: str = Field(..., description="How clients describe it")
    neural_basis: str = Field(..., description="Neural basis")
    validation_approach: str = Field(..., description="How to validate")
    hope_message: str = Field(..., description="Hope message")


class InterventionSuggestion(BaseModel):
    """Intervention suggestion"""
    intervention_name: str = Field(..., description="Intervention name")
    how_to_implement: str = Field(..., description="Implementation steps")
    expected_timeline: str = Field(..., description="Expected timeline")
    monitoring_indicators: str = Field(..., description="How to monitor progress")


class AISynthesis(BaseModel):
    """AI synthesis results"""
    enlitens_takeaway: str = Field(..., description="Main takeaway in Enlitens voice")
    eli5_summary: str = Field(..., description="Simple summary")
    key_findings: List[KeyFinding] = Field(default_factory=list, description="Key findings")
    neuroscientific_concepts: List[NeuroscientificConcept] = Field(default_factory=list, description="Neuroscientific concepts")
    clinical_applications: List[ClinicalApplication] = Field(default_factory=list, description="Clinical applications")
    therapeutic_targets: List[TherapeuticTarget] = Field(default_factory=list, description="Therapeutic targets")
    client_presentations: List[ClientPresentation] = Field(default_factory=list, description="Client presentations")
    intervention_suggestions: List[InterventionSuggestion] = Field(default_factory=list, description="Intervention suggestions")
    contraindications: List[str] = Field(default_factory=list, description="Contraindications")
    evidence_strength: EvidenceStrength = Field(..., description="Overall evidence strength")
    powerful_quotes: List[str] = Field(default_factory=list, description="Powerful quotes")
    synthesis_quality_score: float = Field(0.0, description="Quality score of synthesis")


class ExtractedEntity(BaseModel):
    """Extracted entity"""
    text: str = Field(..., description="Entity text")
    label: str = Field(..., description="Entity label")
    confidence: float = Field(..., description="Confidence score")
    start: int = Field(..., description="Start position")
    end: int = Field(..., description="End position")
    category: EntityCategory = Field(..., description="Entity category")
    extraction_method: str = Field(..., description="Extraction method used")


class EntityExtraction(BaseModel):
    """Entity extraction results"""
    biomedical: List[ExtractedEntity] = Field(default_factory=list, description="Biomedical entities")
    scientific: List[ExtractedEntity] = Field(default_factory=list, description="Scientific entities")
    neuroscience_specific: List[ExtractedEntity] = Field(default_factory=list, description="Neuroscience-specific entities")
    brain_regions: List[ExtractedEntity] = Field(default_factory=list, description="Brain regions")
    neurotransmitters: List[ExtractedEntity] = Field(default_factory=list, description="Neurotransmitters")
    neural_circuits: List[ExtractedEntity] = Field(default_factory=list, description="Neural circuits")
    neural_processes: List[ExtractedEntity] = Field(default_factory=list, description="Neural processes")
    extraction_quality_score: float = Field(0.0, description="Quality score of extraction")


class GraphNode(BaseModel):
    """Graph node for Neo4j integration"""
    node_id: str = Field(..., description="Unique node identifier")
    node_label: str = Field(..., description="Node type/label")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Node properties")


class GraphRelationship(BaseModel):
    """Graph relationship for Neo4j integration"""
    source_node_id: str = Field(..., description="Source node ID")
    target_node_id: str = Field(..., description="Target node ID")
    relationship_type: str = Field(..., description="Relationship type")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relationship properties")


class GraphHints(BaseModel):
    """Graph hints for future Neo4j integration"""
    nodes: List[GraphNode] = Field(default_factory=list, description="Graph nodes")
    relationships: List[GraphRelationship] = Field(default_factory=list, description="Graph relationships")


class ProcessingMetadata(BaseModel):
    """Processing metadata"""
    extraction_timestamp: datetime = Field(default_factory=datetime.now, description="When extraction completed")
    synthesis_timestamp: Optional[datetime] = Field(None, description="When synthesis completed")
    processing_duration_seconds: Optional[float] = Field(None, description="Processing duration")
    extraction_method: str = Field(..., description="Extraction method used")
    synthesis_method: str = Field(..., description="Synthesis method used")
    quality_scores: Dict[str, float] = Field(default_factory=dict, description="Quality scores")
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    warnings: List[str] = Field(default_factory=list, description="Processing warnings")


class EnlitensKnowledgeDocument(BaseModel):
    """Complete knowledge document"""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique document identifier")
    source_metadata: SourceMetadata = Field(..., description="Source document metadata")
    archival_content: ArchivalContent = Field(..., description="Archived content")
    entity_extraction: Optional[EntityExtraction] = Field(None, description="Entity extraction results")
    ai_synthesis: Optional[AISynthesis] = Field(None, description="AI synthesis results")
    graph_hints: Optional[GraphHints] = Field(None, description="Graph hints for Neo4j")
    processing_metadata: ProcessingMetadata = Field(..., description="Processing metadata")
    
    @validator('document_id')
    def validate_document_id(cls, v):
        """Validate document ID format"""
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError("Document ID must be a valid UUID")
    
    def get_quality_score(self) -> float:
        """Get overall quality score"""
        scores = []
        
        if self.archival_content.extraction_quality_score > 0:
            scores.append(self.archival_content.extraction_quality_score)
        
        if self.entity_extraction and self.entity_extraction.extraction_quality_score > 0:
            scores.append(self.entity_extraction.extraction_quality_score)
        
        if self.ai_synthesis and self.ai_synthesis.synthesis_quality_score > 0:
            scores.append(self.ai_synthesis.synthesis_quality_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def is_complete(self) -> bool:
        """Check if document processing is complete"""
        return (
            self.archival_content is not None and
            self.entity_extraction is not None and
            self.ai_synthesis is not None
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get document summary"""
        return {
            'document_id': self.document_id,
            'title': self.source_metadata.title,
            'authors': [author.name for author in self.source_metadata.authors],
            'quality_score': self.get_quality_score(),
            'is_complete': self.is_complete(),
            'extraction_quality': self.archival_content.extraction_quality_score,
            'entity_count': sum(len(getattr(self.entity_extraction, field, [])) 
                              for field in ['biomedical', 'scientific', 'neuroscience_specific', 
                                          'brain_regions', 'neurotransmitters', 'neural_circuits', 'neural_processes'])
            if self.entity_extraction else 0,
            'synthesis_quality': self.ai_synthesis.synthesis_quality_score if self.ai_synthesis else 0.0,
            'processing_timestamp': self.processing_metadata.extraction_timestamp.isoformat()
        }


class KnowledgeBase(BaseModel):
    """Complete knowledge base"""
    documents: List[EnlitensKnowledgeDocument] = Field(default_factory=list, description="All documents")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Knowledge base metadata")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    
    def add_document(self, document: EnlitensKnowledgeDocument):
        """Add a document to the knowledge base"""
        self.documents.append(document)
        self.updated_at = datetime.now()
    
    def get_document(self, document_id: str) -> Optional[EnlitensKnowledgeDocument]:
        """Get a document by ID"""
        for doc in self.documents:
            if doc.document_id == document_id:
                return doc
        return None
    
    def get_documents_by_quality(self, min_quality: float = 0.8) -> List[EnlitensKnowledgeDocument]:
        """Get documents above quality threshold"""
        return [doc for doc in self.documents if doc.get_quality_score() >= min_quality]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        if not self.documents:
            return {
                'total_documents': 0,
                'complete_documents': 0,
                'average_quality': 0.0,
                'entity_count': 0,
                'processing_errors': 0
            }
        
        complete_docs = sum(1 for doc in self.documents if doc.is_complete())
        quality_scores = [doc.get_quality_score() for doc in self.documents]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        entity_count = sum(
            doc.get_summary()['entity_count'] 
            for doc in self.documents
        )
        
        error_count = sum(
            len(doc.processing_metadata.errors) 
            for doc in self.documents
        )
        
        return {
            'total_documents': len(self.documents),
            'complete_documents': complete_docs,
            'average_quality': avg_quality,
            'entity_count': entity_count,
            'processing_errors': error_count,
            'high_quality_documents': len([doc for doc in self.documents if doc.get_quality_score() >= 0.8])
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the schema
    from datetime import datetime
    
    # Create a sample document
    sample_author = Author(name="Dr. Jane Smith", affiliation="University of Neuroscience")
    
    sample_metadata = SourceMetadata(
        title="The Role of the Prefrontal Cortex in Executive Function",
        authors=[sample_author],
        journal="Nature Neuroscience",
        doi="10.1038/nn.2024.123",
        source_filename="prefrontal_cortex_study.pdf"
    )
    
    sample_content = ArchivalContent(
        full_document_text_markdown="# The Role of the Prefrontal Cortex...",
        abstract_markdown="This study examines the neural mechanisms...",
        extraction_quality_score=0.95
    )
    
    sample_synthesis = AISynthesis(
        enlitens_takeaway="Of course you feel overwhelmed - your brain is doing exactly what it learned to do...",
        eli5_summary="Your prefrontal cortex is like the CEO of your brain...",
        evidence_strength=EvidenceStrength.STRONG,
        synthesis_quality_score=0.92
    )
    
    sample_document = EnlitensKnowledgeDocument(
        source_metadata=sample_metadata,
        archival_content=sample_content,
        ai_synthesis=sample_synthesis
    )
    
    print("Document created successfully")
    print(f"Quality score: {sample_document.get_quality_score():.2f}")
    print(f"Is complete: {sample_document.is_complete()}")
    print(f"Summary: {sample_document.get_summary()}")
    
    # Test knowledge base
    kb = KnowledgeBase()
    kb.add_document(sample_document)
    
    stats = kb.get_statistics()
    print(f"Knowledge base statistics: {stats}")
