"""
Checkpoint Manager for Resume Capability

This module manages checkpoints to allow resuming processing from failures.
Critical for processing 400+ papers without losing progress.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ProcessingCheckpoint:
    """Checkpoint for document processing"""
    document_id: str
    pdf_path: str
    stage: str  # extraction, entity_extraction, synthesis, completed
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CheckpointManager:
    """
    Manages processing checkpoints for resume capability
    
    Why checkpoints are critical:
    - Processing 400+ papers takes hours
    - Failures can occur at any stage
    - GPU memory issues can cause crashes
    - Network issues can interrupt vLLM
    - Resume from last successful stage
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Checkpoint files
        self.processing_checkpoints_file = self.checkpoint_dir / "processing_checkpoints.json"
        self.failed_documents_file = self.checkpoint_dir / "failed_documents.json"
        self.completed_documents_file = self.checkpoint_dir / "completed_documents.json"
        
        # Load existing checkpoints
        self.processing_checkpoints = self._load_processing_checkpoints()
        self.failed_documents = self._load_failed_documents()
        self.completed_documents = self._load_completed_documents()
        
        logger.info(f"CheckpointManager initialized with {len(self.processing_checkpoints)} checkpoints")
    
    def _load_processing_checkpoints(self) -> Dict[str, ProcessingCheckpoint]:
        """Load processing checkpoints from file"""
        if not self.processing_checkpoints_file.exists():
            return {}
        
        try:
            with open(self.processing_checkpoints_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            checkpoints = {}
            for doc_id, checkpoint_data in data.items():
                # Convert timestamp string back to datetime
                checkpoint_data['timestamp'] = datetime.fromisoformat(checkpoint_data['timestamp'])
                checkpoints[doc_id] = ProcessingCheckpoint(**checkpoint_data)
            
            return checkpoints
        except Exception as e:
            logger.error(f"Failed to load processing checkpoints: {e}")
            return {}
    
    def _load_failed_documents(self) -> Set[str]:
        """Load list of failed documents"""
        if not self.failed_documents_file.exists():
            return set()
        
        try:
            with open(self.failed_documents_file, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except Exception as e:
            logger.error(f"Failed to load failed documents: {e}")
            return set()
    
    def _load_completed_documents(self) -> Set[str]:
        """Load list of completed documents"""
        if not self.completed_documents_file.exists():
            return set()
        
        try:
            with open(self.completed_documents_file, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except Exception as e:
            logger.error(f"Failed to load completed documents: {e}")
            return set()
    
    def _save_processing_checkpoints(self):
        """Save processing checkpoints to file"""
        try:
            # Convert datetime objects to strings for JSON serialization
            data = {}
            for doc_id, checkpoint in self.processing_checkpoints.items():
                checkpoint_dict = asdict(checkpoint)
                checkpoint_dict['timestamp'] = checkpoint.timestamp.isoformat()
                data[doc_id] = checkpoint_dict
            
            with open(self.processing_checkpoints_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save processing checkpoints: {e}")
    
    def _save_failed_documents(self):
        """Save failed documents list"""
        try:
            with open(self.failed_documents_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.failed_documents), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save failed documents: {e}")
    
    def _save_completed_documents(self):
        """Save completed documents list"""
        try:
            with open(self.completed_documents_file, 'w', encoding='utf-8') as f:
                json.dump(list(self.completed_documents), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save completed documents: {e}")
    
    def create_checkpoint(self, document_id: str, pdf_path: str, stage: str, 
                         success: bool, error_message: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> ProcessingCheckpoint:
        """Create a new checkpoint"""
        checkpoint = ProcessingCheckpoint(
            document_id=document_id,
            pdf_path=pdf_path,
            stage=stage,
            timestamp=datetime.now(),
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )
        
        self.processing_checkpoints[document_id] = checkpoint
        
        # Update status sets
        if success and stage == "completed":
            self.completed_documents.add(document_id)
            if document_id in self.failed_documents:
                self.failed_documents.remove(document_id)
        elif not success:
            self.failed_documents.add(document_id)
        
        # Save to disk
        self._save_processing_checkpoints()
        self._save_failed_documents()
        self._save_completed_documents()
        
        logger.info(f"Checkpoint created: {document_id} - {stage} - {'success' if success else 'failed'}")
        return checkpoint
    
    def get_checkpoint(self, document_id: str) -> Optional[ProcessingCheckpoint]:
        """Get checkpoint for a document"""
        return self.processing_checkpoints.get(document_id)
    
    def get_last_successful_stage(self, document_id: str) -> Optional[str]:
        """Get the last successful processing stage for a document"""
        checkpoint = self.get_checkpoint(document_id)
        if checkpoint and checkpoint.success:
            return checkpoint.stage
        return None
    
    def is_document_completed(self, document_id: str) -> bool:
        """Check if document is completed"""
        return document_id in self.completed_documents
    
    def is_document_failed(self, document_id: str) -> bool:
        """Check if document has failed"""
        return document_id in self.failed_documents
    
    def get_resume_point(self, document_id: str) -> Optional[str]:
        """Get the stage to resume from for a document"""
        if self.is_document_completed(document_id):
            return None  # Already completed
        
        last_stage = self.get_last_successful_stage(document_id)
        if last_stage:
            # Resume from next stage
            stage_order = ["extraction", "entity_extraction", "synthesis", "completed"]
            try:
                current_index = stage_order.index(last_stage)
                if current_index < len(stage_order) - 1:
                    return stage_order[current_index + 1]
            except ValueError:
                pass
        
        # Start from beginning
        return "extraction"
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing status"""
        total_checkpoints = len(self.processing_checkpoints)
        completed = len(self.completed_documents)
        failed = len(self.failed_documents)
        in_progress = total_checkpoints - completed - failed
        
        # Count by stage
        stage_counts = {}
        for checkpoint in self.processing_checkpoints.values():
            stage = checkpoint.stage
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        return {
            'total_checkpoints': total_checkpoints,
            'completed_documents': completed,
            'failed_documents': failed,
            'in_progress_documents': in_progress,
            'stage_counts': stage_counts,
            'completion_rate': completed / total_checkpoints if total_checkpoints > 0 else 0.0
        }
    
    def get_failed_documents(self) -> List[Dict[str, Any]]:
        """Get list of failed documents with details"""
        failed_details = []
        for doc_id in self.failed_documents:
            checkpoint = self.get_checkpoint(doc_id)
            if checkpoint:
                failed_details.append({
                    'document_id': doc_id,
                    'pdf_path': checkpoint.pdf_path,
                    'last_stage': checkpoint.stage,
                    'error_message': checkpoint.error_message,
                    'timestamp': checkpoint.timestamp.isoformat()
                })
        return failed_details
    
    def retry_failed_document(self, document_id: str) -> bool:
        """Mark a failed document for retry"""
        if document_id in self.failed_documents:
            self.failed_documents.remove(document_id)
            self._save_failed_documents()
            logger.info(f"Document {document_id} marked for retry")
            return True
        return False
    
    def clear_checkpoint(self, document_id: str):
        """Clear checkpoint for a document"""
        if document_id in self.processing_checkpoints:
            del self.processing_checkpoints[document_id]
            self._save_processing_checkpoints()
        
        if document_id in self.completed_documents:
            self.completed_documents.remove(document_id)
            self._save_completed_documents()
        
        if document_id in self.failed_documents:
            self.failed_documents.remove(document_id)
            self._save_failed_documents()
        
        logger.info(f"Checkpoint cleared for {document_id}")
    
    def cleanup_old_checkpoints(self, days_old: int = 7):
        """Clean up checkpoints older than specified days"""
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        
        to_remove = []
        for doc_id, checkpoint in self.processing_checkpoints.items():
            if checkpoint.timestamp.timestamp() < cutoff_date:
                to_remove.append(doc_id)
        
        for doc_id in to_remove:
            self.clear_checkpoint(doc_id)
        
        logger.info(f"Cleaned up {len(to_remove)} old checkpoints")
    
    def export_checkpoints(self, output_path: str):
        """Export all checkpoints to a file"""
        try:
            export_data = {
                'processing_checkpoints': {
                    doc_id: asdict(checkpoint) for doc_id, checkpoint in self.processing_checkpoints.items()
                },
                'failed_documents': list(self.failed_documents),
                'completed_documents': list(self.completed_documents),
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Checkpoints exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export checkpoints: {e}")
            raise


class ResumeProcessor:
    """
    Handles resuming processing from checkpoints
    
    This class works with the main DocumentProcessor to resume
    processing from the last successful stage.
    """
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
    
    def get_documents_to_process(self, all_pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """Get list of documents that need processing"""
        documents_to_process = []
        
        for pdf_path in all_pdf_paths:
            # Generate document ID from file path
            document_id = self._generate_document_id(pdf_path)
            
            # Check if already completed
            if self.checkpoint_manager.is_document_completed(document_id):
                continue
            
            # Get resume point
            resume_stage = self.checkpoint_manager.get_resume_point(document_id)
            
            documents_to_process.append({
                'document_id': document_id,
                'pdf_path': pdf_path,
                'resume_stage': resume_stage,
                'is_retry': self.checkpoint_manager.is_document_failed(document_id)
            })
        
        return documents_to_process
    
    def _generate_document_id(self, pdf_path: str) -> str:
        """Generate document ID from file path"""
        # Use file path hash for consistent ID
        return hashlib.md5(pdf_path.encode()).hexdigest()[:16]
    
    def get_processing_plan(self, all_pdf_paths: List[str]) -> Dict[str, Any]:
        """Get a processing plan showing what needs to be done"""
        documents_to_process = self.get_documents_to_process(all_pdf_paths)
        
        # Categorize by stage
        by_stage = {}
        for doc in documents_to_process:
            stage = doc['resume_stage']
            if stage not in by_stage:
                by_stage[stage] = []
            by_stage[stage].append(doc)
        
        return {
            'total_documents': len(all_pdf_paths),
            'documents_to_process': len(documents_to_process),
            'by_stage': by_stage,
            'summary': self.checkpoint_manager.get_processing_summary()
        }


# Example usage
if __name__ == "__main__":
    # Test checkpoint manager
    checkpoint_manager = CheckpointManager()
    
    # Create some test checkpoints
    test_doc_id = "test_doc_123"
    checkpoint_manager.create_checkpoint(
        test_doc_id, 
        "/path/to/test.pdf", 
        "extraction", 
        True
    )
    
    # Get processing summary
    summary = checkpoint_manager.get_processing_summary()
    print("Processing summary:", summary)
    
    # Test resume processor
    resume_processor = ResumeProcessor(checkpoint_manager)
    
    # Test with sample PDF paths
    sample_paths = ["/path/to/paper1.pdf", "/path/to/paper2.pdf"]
    plan = resume_processor.get_processing_plan(sample_paths)
    print("Processing plan:", plan)
