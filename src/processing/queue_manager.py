import time
import json
import logging
from enum import Enum
from typing import Dict, List, Optional
from queue import PriorityQueue
from pathlib import Path
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)

class ProcessingError(Exception):
    """Exception raised for errors during document processing."""
    pass

class QueuePriority(Enum):
    """Priority levels for document processing."""
    HIGH = 1
    NORMAL = 2
    LOW = 3

class RateLimiter:
    """Rate limiter for document processing."""
    
    def __init__(self, docs_per_minute: int = 60, burst_size: int = 10):
        self.docs_per_minute = docs_per_minute
        self.burst_size = burst_size
        self.window_size = 60  # 1 minute in seconds
        self.processing_times = []
        self.lock = Lock()
        
    def allow_processing(self) -> bool:
        """Check if processing is allowed based on rate limits."""
        with self.lock:
            current_time = time.time()
            
            # Remove old timestamps
            window_start = current_time - self.window_size
            self.processing_times = [t for t in self.processing_times if t > window_start]
            
            # Check if we're within rate limits
            if len(self.processing_times) >= self.burst_size:
                if len(self.processing_times) >= self.docs_per_minute:
                    return False
                
                # Check if we're within the rate
                oldest_in_window = self.processing_times[0]
                if current_time - oldest_in_window < self.window_size:
                    return False
            
            # Allow processing and record timestamp
            self.processing_times.append(current_time)
            return True

@dataclass
class QueuedDocument:
    """Document in the processing queue."""
    priority: QueuePriority
    timestamp: float
    document: Dict
    retry_count: int = 0
    
    def __lt__(self, other):
        """Compare documents for priority queue ordering."""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp

class DocumentQueueManager:
    """Manages document processing queue with priority and rate limiting."""
    
    def __init__(self, config: Dict):
        """Initialize the queue manager."""
        self.config = config
        self.processing_queue = PriorityQueue()
        self.failed_queue = PriorityQueue()
        self.rate_limiter = RateLimiter(
            docs_per_minute=config['rate_limit']['docs_per_minute'],
            burst_size=config['rate_limit']['burst_size']
        )
        self.max_retries = config['max_retries']
        self.batch_size = config['batch_size']
        
        # State persistence
        self.state_file = Path("data/queue_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self.queue_lock = Lock()
        
        logger.info(f"Initialized DocumentQueueManager with config: {config}")
    
    def add_document(self, document: Dict, priority: QueuePriority = QueuePriority.NORMAL) -> None:
        """Add a document to the processing queue."""
        with self.queue_lock:
            queued_doc = QueuedDocument(
                priority=priority,
                timestamp=time.time(),
                document=document
            )
            self.processing_queue.put(queued_doc)
            logger.debug(f"Added document {document.get('id')} to queue with priority {priority}")
    
    def get_next_document(self) -> Optional[Dict]:
        """Get the next document from the queue."""
        with self.queue_lock:
            if self.processing_queue.empty():
                return None
            
            if not self.rate_limiter.allow_processing():
                logger.debug("Rate limit reached, waiting...")
                return None
            
            queued_doc = self.processing_queue.get()
            return queued_doc.document
    
    def get_next_batch(self) -> List[Dict]:
        """Get the next batch of documents from the queue."""
        batch = []
        with self.queue_lock:
            while len(batch) < self.batch_size and not self.processing_queue.empty():
                if not self.rate_limiter.allow_processing():
                    break
                
                queued_doc = self.processing_queue.get()
                batch.append(queued_doc.document)
        
        logger.debug(f"Retrieved batch of {len(batch)} documents")
        return batch
    
    def process_next(self) -> bool:
        """Process the next document in the queue."""
        document = self.get_next_document()
        if not document:
            return False
            
        try:
            self._process_document(document)
            return True
        except ProcessingError as e:
            logger.error(f"Error processing document {document.get('id')}: {str(e)}")
            self._handle_processing_error(document)
            return False
    
    def _process_document(self, document: Dict) -> None:
        """Process a single document. Override this method in subclasses."""
        raise NotImplementedError("Document processing not implemented")
    
    def _handle_processing_error(self, document: Dict) -> None:
        """Handle document processing errors."""
        with self.queue_lock:
            # Get or initialize retry count
            retry_count = document.get('retry_count', 0) + 1
            document['retry_count'] = retry_count
            
            queued_doc = QueuedDocument(
                priority=QueuePriority.NORMAL,
                timestamp=time.time(),
                document=document,
                retry_count=retry_count
            )
            
            if retry_count <= self.max_retries:
                logger.info(f"Retrying document {document.get('id')}, attempt {retry_count}")
                self.processing_queue.put(queued_doc)
            else:
                logger.warning(f"Document {document.get('id')} failed after {self.max_retries} retries")
                self.failed_queue.put(queued_doc)
    
    def queue_size(self) -> int:
        """Get the current size of the processing queue."""
        return self.processing_queue.qsize()
    
    def failed_queue_size(self) -> int:
        """Get the current size of the failed queue."""
        return self.failed_queue.qsize()
    
    def clear_all_queues(self) -> None:
        """Clear all queues."""
        with self.queue_lock:
            while not self.processing_queue.empty():
                self.processing_queue.get()
            while not self.failed_queue.empty():
                self.failed_queue.get()
    
    def save_state(self) -> None:
        """Save queue state to disk."""
        with self.queue_lock:
            state = {
                'processing_queue': [],
                'failed_queue': []
            }
            
            # Save processing queue
            temp_queue = PriorityQueue()
            while not self.processing_queue.empty():
                doc = self.processing_queue.get()
                state['processing_queue'].append({
                    'priority': doc.priority.name,
                    'timestamp': doc.timestamp,
                    'document': doc.document,
                    'retry_count': doc.retry_count
                })
                temp_queue.put(doc)
            self.processing_queue = temp_queue
            
            # Save failed queue
            temp_queue = PriorityQueue()
            while not self.failed_queue.empty():
                doc = self.failed_queue.get()
                state['failed_queue'].append({
                    'priority': doc.priority.name,
                    'timestamp': doc.timestamp,
                    'document': doc.document,
                    'retry_count': doc.retry_count
                })
                temp_queue.put(doc)
            self.failed_queue = temp_queue
            
            # Write to file
            with open(self.state_file, 'w') as f:
                json.dump(state, f)
            
            logger.info("Queue state saved to disk")
    
    def load_state(self) -> None:
        """Load queue state from disk."""
        if not self.state_file.exists():
            logger.warning("No saved state found")
            return
            
        with self.queue_lock:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Clear current queues
            self.clear_all_queues()
            
            # Restore processing queue
            for doc_state in state['processing_queue']:
                self.processing_queue.put(QueuedDocument(
                    priority=QueuePriority[doc_state['priority']],
                    timestamp=doc_state['timestamp'],
                    document=doc_state['document'],
                    retry_count=doc_state['retry_count']
                ))
            
            # Restore failed queue
            for doc_state in state['failed_queue']:
                self.failed_queue.put(QueuedDocument(
                    priority=QueuePriority[doc_state['priority']],
                    timestamp=doc_state['timestamp'],
                    document=doc_state['document'],
                    retry_count=doc_state['retry_count']
                ))
            
            logger.info("Queue state loaded from disk")
    
    def delete_saved_state(self) -> None:
        """Delete saved state file."""
        if self.state_file.exists():
            self.state_file.unlink()
            logger.info("Deleted saved state file")
