import unittest
from unittest.mock import Mock, patch
import time
from typing import Dict, List
from queue import PriorityQueue
from src.processing.queue_manager import DocumentQueueManager, QueuePriority, RateLimiter, ProcessingError

class TestDocumentQueueManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = {
            'batch_size': 5,
            'max_retries': 3,
            'rate_limit': {
                'docs_per_minute': 60,
                'burst_size': 10
            }
        }
        self.queue_manager = DocumentQueueManager(self.config)
        
        # Sample test documents
        self.test_docs = [
            {
                'id': f'doc{i}',
                'text': f'Test document {i}',
                'metadata': {'priority': 'normal', 'source': 'test'}
            } for i in range(10)
        ]
        
        # High priority documents
        self.high_priority_docs = [
            {
                'id': f'high_doc{i}',
                'text': f'High priority document {i}',
                'metadata': {'priority': 'high', 'source': 'test'}
            } for i in range(3)
        ]

    def test_add_single_document(self):
        """Test adding a single document to the queue."""
        doc = self.test_docs[0]
        self.queue_manager.add_document(doc)
        
        # Check queue size
        self.assertEqual(self.queue_manager.queue_size(), 1)
        
        # Verify document is retrievable
        queued_doc = self.queue_manager.get_next_document()
        self.assertEqual(queued_doc['id'], doc['id'])

    def test_batch_processing(self):
        """Test processing documents in batches."""
        # Add multiple documents
        for doc in self.test_docs:
            self.queue_manager.add_document(doc)
            
        # Get a batch
        batch = self.queue_manager.get_next_batch()
        
        # Verify batch size
        self.assertEqual(len(batch), min(self.config['batch_size'], len(self.test_docs)))
        
        # Verify remaining queue size
        expected_remaining = max(0, len(self.test_docs) - self.config['batch_size'])
        self.assertEqual(self.queue_manager.queue_size(), expected_remaining)

    def test_priority_handling(self):
        """Test that documents are processed according to priority."""
        # Add normal priority documents
        for doc in self.test_docs[:5]:
            self.queue_manager.add_document(doc)
            
        # Add high priority documents
        for doc in self.high_priority_docs:
            self.queue_manager.add_document(doc, priority=QueuePriority.HIGH)
            
        # Verify high priority documents are processed first
        batch = self.queue_manager.get_next_batch()
        batch_ids = [doc['id'] for doc in batch]
        
        # Check that high priority docs are at the start of the batch
        for high_doc in self.high_priority_docs:
            self.assertIn(high_doc['id'], batch_ids[:len(self.high_priority_docs)])

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Set a very low rate limit for testing
        self.queue_manager.rate_limiter = RateLimiter(docs_per_minute=2, burst_size=1)
        
        # Try to process documents quickly
        processed_count = 0
        
        # First burst should allow one document
        if self.queue_manager.rate_limiter.allow_processing():
            processed_count += 1
                
        # Should only process 1 doc (burst size) immediately
        self.assertEqual(processed_count, 1)
        
        # Immediate second attempt should be blocked
        self.assertFalse(self.queue_manager.rate_limiter.allow_processing())

    def test_error_handling(self):
        """Test handling of processing errors and retries."""
        doc = self.test_docs[0].copy()  # Make a copy to avoid modifying original
        doc['retry_count'] = 0  # Initialize retry count
        
        # Mock a processing error
        with patch.object(self.queue_manager, '_process_document', side_effect=ProcessingError("Test error")):
            self.queue_manager.add_document(doc)
            
            # Process once - should go to retry
            self.queue_manager.process_next()
            self.assertEqual(self.queue_manager.queue_size(), 1)  # Back in queue for retry
            self.assertEqual(self.queue_manager.failed_queue_size(), 0)
            
            # Process again - should go to failed queue after max retries
            for _ in range(self.config['max_retries']):
                self.queue_manager.process_next()
            
            # Verify document is now in failed queue
            self.assertEqual(self.queue_manager.failed_queue_size(), 1)
            self.assertEqual(self.queue_manager.queue_size(), 0)

    def tearDown(self):
        """Clean up after each test method."""
        self.queue_manager.clear_all_queues()

if __name__ == '__main__':
    unittest.main()
