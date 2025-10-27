"""
Knowledge Base Manager for Growing Database

This module manages the growing knowledge base, allowing for:
- Adding new PDFs to existing knowledge base
- Incremental updates without reprocessing
- Content deduplication and merging
- Quality validation and scoring
- Export for different use cases
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import logging
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


class KnowledgeBaseManager:
    """
    Manages the growing knowledge base database
    
    Features:
    - Add new PDFs to existing knowledge base
    - Incremental updates without reprocessing
    - Content deduplication and merging
    - Quality validation and scoring
    - Export for different use cases
    """
    
    def __init__(self, knowledge_base_path: str = "./knowledge_base.json", 
                 processed_files_path: str = "./processed_files.json"):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.processed_files_path = Path(processed_files_path)
        self.processed_files = self._load_processed_files()
        self.knowledge_base = self._load_knowledge_base()
        
    def _load_processed_files(self) -> Dict[str, Any]:
        """Load the processed files registry"""
        if self.processed_files_path.exists():
            try:
                with open(self.processed_files_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load processed files: {e}")
                return {}
        return {}
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load the existing knowledge base"""
        if self.knowledge_base_path.exists():
            try:
                with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load knowledge base: {e}")
                return {
                    'papers': [],
                    'metadata': {
                        'total_papers': 0,
                        'last_updated': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                }
        return {
            'papers': [],
            'metadata': {
                'total_papers': 0,
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
        }
    
    def _save_processed_files(self):
        """Save the processed files registry"""
        try:
            with open(self.processed_files_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save processed files: {e}")
    
    def _save_knowledge_base(self):
        """Save the knowledge base"""
        try:
            with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")
    
    def get_unprocessed_files(self, input_dir: str) -> List[str]:
        """Get list of unprocessed PDF files"""
        input_path = Path(input_dir)
        unprocessed = []
        
        if not input_path.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return unprocessed
        
        for pdf_file in input_path.glob("*.pdf"):
            file_hash = self._get_file_hash(str(pdf_file))
            if file_hash not in self.processed_files:
                unprocessed.append(str(pdf_file))
        
        return unprocessed
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file for deduplication"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to get hash for {file_path}: {e}")
            return ""
    
    def add_paper_to_knowledge_base(self, paper_data: Dict[str, Any], 
                                  file_path: str) -> bool:
        """Add a new paper to the knowledge base"""
        try:
            # Get file hash for deduplication
            file_hash = self._get_file_hash(file_path)
            if not file_hash:
                logger.error(f"Failed to get hash for {file_path}")
                return False
            
            # Check if already processed
            if file_hash in self.processed_files:
                logger.info(f"File already processed: {file_path}")
                return True
            
            # Add paper to knowledge base
            paper_entry = {
                'id': len(self.knowledge_base['papers']) + 1,
                'file_path': file_path,
                'file_hash': file_hash,
                'processed_timestamp': datetime.now().isoformat(),
                'data': paper_data
            }
            
            self.knowledge_base['papers'].append(paper_entry)
            self.knowledge_base['metadata']['total_papers'] = len(self.knowledge_base['papers'])
            self.knowledge_base['metadata']['last_updated'] = datetime.now().isoformat()
            
            # Mark as processed
            self.processed_files[file_hash] = {
                'file_path': file_path,
                'processed_timestamp': datetime.now().isoformat(),
                'paper_id': paper_entry['id']
            }
            
            # Save both files
            self._save_knowledge_base()
            self._save_processed_files()
            
            logger.info(f"Added paper to knowledge base: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add paper to knowledge base: {e}")
            return False
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        stats = {
            'total_papers': len(self.knowledge_base['papers']),
            'last_updated': self.knowledge_base['metadata']['last_updated'],
            'version': self.knowledge_base['metadata']['version']
        }
        
        # Content type statistics
        content_stats = {
            'marketing_content': 0,
            'seo_content': 0,
            'blog_content': 0,
            'social_media_content': 0,
            'educational_content': 0,
            'clinical_content': 0,
            'research_content': 0
        }
        
        for paper in self.knowledge_base['papers']:
            data = paper.get('data', {})
            for content_type in content_stats.keys():
                if content_type in data and data[content_type]:
                    content_stats[content_type] += 1
        
        stats['content_stats'] = content_stats
        return stats
    
    def export_knowledge_base(self, export_path: str, 
                            content_types: Optional[List[str]] = None) -> bool:
        """Export knowledge base for specific use cases"""
        try:
            export_data = {
                'metadata': self.knowledge_base['metadata'],
                'papers': []
            }
            
            for paper in self.knowledge_base['papers']:
                paper_data = paper['data']
                
                if content_types:
                    # Filter by content types
                    filtered_data = {}
                    for content_type in content_types:
                        if content_type in paper_data:
                            filtered_data[content_type] = paper_data[content_type]
                    
                    paper_entry = {
                        'id': paper['id'],
                        'file_path': paper['file_path'],
                        'processed_timestamp': paper['processed_timestamp'],
                        'data': filtered_data
                    }
                else:
                    # Export all content
                    paper_entry = paper
                
                export_data['papers'].append(paper_entry)
            
            # Save export
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported knowledge base to: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export knowledge base: {e}")
            return False
    
    def search_knowledge_base(self, query: str, 
                           content_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search the knowledge base"""
        results = []
        query_lower = query.lower()
        
        for paper in self.knowledge_base['papers']:
            paper_data = paper['data']
            matches = []
            
            # Search in specified content types
            if content_types:
                for content_type in content_types:
                    if content_type in paper_data:
                        content = paper_data[content_type]
                        if self._search_in_content(content, query_lower):
                            matches.append(content_type)
            else:
                # Search in all content
                for content_type, content in paper_data.items():
                    if self._search_in_content(content, query_lower):
                        matches.append(content_type)
            
            if matches:
                results.append({
                    'paper_id': paper['id'],
                    'file_path': paper['file_path'],
                    'matches': matches,
                    'data': paper_data
                })
        
        return results
    
    def _search_in_content(self, content: Any, query: str) -> bool:
        """Search for query in content"""
        if isinstance(content, str):
            return query in content.lower()
        elif isinstance(content, list):
            return any(self._search_in_content(item, query) for item in content)
        elif isinstance(content, dict):
            return any(self._search_in_content(value, query) for value in content.values())
        return False
    
    def get_content_by_type(self, content_type: str) -> List[Dict[str, Any]]:
        """Get all content of a specific type"""
        results = []
        
        for paper in self.knowledge_base['papers']:
            paper_data = paper['data']
            if content_type in paper_data and paper_data[content_type]:
                results.append({
                    'paper_id': paper['id'],
                    'file_path': paper['file_path'],
                    'content': paper_data[content_type]
                })
        
        return results
    
    def deduplicate_content(self) -> Dict[str, int]:
        """Remove duplicate content from knowledge base"""
        duplicates_removed = {
            'papers': 0,
            'content_items': 0
        }
        
        # Track seen content
        seen_content = set()
        unique_papers = []
        
        for paper in self.knowledge_base['papers']:
            paper_data = paper['data']
            paper_hash = self._get_paper_hash(paper_data)
            
            if paper_hash not in seen_content:
                seen_content.add(paper_hash)
                unique_papers.append(paper)
            else:
                duplicates_removed['papers'] += 1
        
        # Update knowledge base
        self.knowledge_base['papers'] = unique_papers
        self.knowledge_base['metadata']['total_papers'] = len(unique_papers)
        self.knowledge_base['metadata']['last_updated'] = datetime.now().isoformat()
        
        # Save updated knowledge base
        self._save_knowledge_base()
        
        return duplicates_removed
    
    def _get_paper_hash(self, paper_data: Dict[str, Any]) -> str:
        """Get hash of paper data for deduplication"""
        # Create hash based on key content
        key_content = []
        
        if 'source_metadata' in paper_data:
            metadata = paper_data['source_metadata']
            key_content.append(metadata.get('title', ''))
            key_content.append(metadata.get('doi', ''))
            key_content.append(metadata.get('journal', ''))
        
        content_hash = hashlib.md5(''.join(key_content).encode()).hexdigest()
        return content_hash
    
    def backup_knowledge_base(self, backup_path: str) -> bool:
        """Create backup of knowledge base"""
        try:
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy knowledge base
            shutil.copy2(self.knowledge_base_path, backup_file)
            
            # Copy processed files
            processed_backup = backup_file.parent / "processed_files.json"
            shutil.copy2(self.processed_files_path, processed_backup)
            
            logger.info(f"Backed up knowledge base to: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup knowledge base: {e}")
            return False
    
    def restore_knowledge_base(self, backup_path: str) -> bool:
        """Restore knowledge base from backup"""
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Restore knowledge base
            shutil.copy2(backup_file, self.knowledge_base_path)
            
            # Restore processed files
            processed_backup = backup_file.parent / "processed_files.json"
            if processed_backup.exists():
                shutil.copy2(processed_backup, self.processed_files_path)
            
            # Reload data
            self.processed_files = self._load_processed_files()
            self.knowledge_base = self._load_knowledge_base()
            
            logger.info(f"Restored knowledge base from: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore knowledge base: {e}")
            return False
