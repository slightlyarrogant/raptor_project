"""Recovery system for document processing queue."""

import json
import logging
import time
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from src.processing.queue_manager import DocumentQueueManager, QueuedDocument, QueuePriority

logger = logging.getLogger(__name__)

class QueueRecoverySystem:
    """System for queue state recovery and maintenance."""
    
    def __init__(self, queue_manager: DocumentQueueManager, checkpoint_dir: str):
        """Initialize recovery system."""
        self.queue_manager = queue_manager
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def create_checkpoint(self) -> str:
        """Create a checkpoint of current queue state."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = self.checkpoint_dir / f'queue_checkpoint_{timestamp}.json'
        
        state = {
            'main_queue': self._serialize_queue(self.queue_manager.processing_queue),
            'failed_queue': self._serialize_queue(self.queue_manager.failed_queue),
            'metadata': {
                'timestamp': timestamp,
                'queue_size': self.queue_manager.queue_size(),
                'failed_size': self.queue_manager.failed_queue_size()
            }
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Created checkpoint: {checkpoint_file}")
        return str(checkpoint_file)
    
    def restore_from_checkpoint(self, checkpoint_file: str) -> bool:
        """Restore queue state from checkpoint."""
        try:
            with open(checkpoint_file, 'r') as f:
                state = json.load(f)
            
            # Clear current queues
            self.queue_manager.clear_all_queues()
            
            # Restore main queue
            for doc in state['main_queue']:
                queued_doc = self._deserialize_document(doc)
                self.queue_manager.processing_queue.put(queued_doc)
            
            # Restore failed queue
            for doc in state['failed_queue']:
                queued_doc = self._deserialize_document(doc)
                self.queue_manager.failed_queue.put(queued_doc)
            
            logger.info(f"Restored from checkpoint: {checkpoint_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore from checkpoint: {e}")
            return False
    
    def list_checkpoints(self) -> List[Dict]:
        """List available checkpoints with metadata."""
        checkpoints = []
        for cp_file in self.checkpoint_dir.glob('queue_checkpoint_*.json'):
            try:
                with open(cp_file, 'r') as f:
                    state = json.load(f)
                checkpoints.append({
                    'file': str(cp_file),
                    'timestamp': state['metadata']['timestamp'],
                    'queue_size': state['metadata']['queue_size'],
                    'failed_size': state['metadata']['failed_size']
                })
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {cp_file}: {e}")
        
        return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)
    
    def cleanup_old_checkpoints(self, keep_last: int = 5) -> int:
        """Remove old checkpoints, keeping the specified number of recent ones."""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= keep_last:
            return 0
        
        removed = 0
        for cp in checkpoints[keep_last:]:
            try:
                Path(cp['file']).unlink()
                removed += 1
            except Exception as e:
                logger.error(f"Failed to remove checkpoint {cp['file']}: {e}")
        
        return removed
    
    def _serialize_queue(self, queue) -> List[Dict]:
        """Serialize queue to list of dictionaries."""
        documents = []
        while not queue.empty():
            doc = queue.get()
            documents.append({
                'priority': doc.priority.name,
                'timestamp': doc.timestamp,
                'document': doc.document,
                'retry_count': doc.retry_count
            })
        
        # Put documents back in queue
        for doc in documents:
            queue.put(self._deserialize_document(doc))
        
        return documents
    
    def _deserialize_document(self, doc_dict: Dict) -> QueuedDocument:
        """Deserialize document from dictionary."""
        return QueuedDocument(
            priority=QueuePriority[doc_dict['priority']],
            timestamp=doc_dict['timestamp'],
            document=doc_dict['document'],
            retry_count=doc_dict['retry_count']
        )
