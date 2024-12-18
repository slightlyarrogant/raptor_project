import unittest
from unittest.mock import Mock, patch
import numpy as np
import logging
from pathlib import Path
from src.tree.tree_manager import TreeManager
from src.utils.test_utils import get_mock_openai, get_mock_pinecone
from tests.test_config import DEFAULT_TEST_CONFIG

class TestIncrementalKnowledge(unittest.TestCase):
    """Test suite for incremental knowledge addition capabilities."""
    
    def setUp(self):
        """Set up test environment before each test."""
        self.logger = logging.getLogger(__name__)
        
        # Mock managers with required methods
        embed_manager = Mock()
        embed_manager.embed_documents = Mock()
        storage_manager = Mock()
        storage_manager.store = Mock()
        storage_manager.get = Mock()
        
        # Initialize with test configuration
        self.config = {
            'index_name': 'test-incremental-index',
            'embedding': {
                'dimension': 1536,
                'model': 'text-embedding-3-small',
                'manager': embed_manager
            },
            'clustering': {
                'dimension': 20,
                'threshold': 0.3,
                'max_levels': 3,
                'min_cluster_size': 3,  # Smaller for testing
                'max_cluster_size': 10,  # Smaller for testing
                'min_docs_per_cluster': 2,  # Smaller for testing
                'target_children': 3,
                'force_merge': True,
                'min_similarity': 0.15,
                'decay_rate': 0.7,
                'merge_threshold': 0.25
            },
            'summarization': {
                'model': 'gpt-4-mini',
                'max_tokens': 1000,
                'temperature': 0.3
            },
            'storage': {
                'store': get_mock_pinecone().Index("test-incremental-index"),
                'namespace': 'test-incremental',
                'dimension': 1536,
                'manager': storage_manager
            }
        }
        
        self.tree_manager = TreeManager(self.config)
        
        # Initial test documents about programming languages
        self.initial_docs = [
            {
                'id': 'doc1',
                'text': 'Python is a high-level programming language known for its simplicity and readability.',
                'metadata': {'source': 'test', 'category': 'programming'}
            },
            {
                'id': 'doc2',
                'text': 'Java is a class-based, object-oriented programming language designed for portability.',
                'metadata': {'source': 'test', 'category': 'programming'}
            },
            {
                'id': 'doc3',
                'text': 'JavaScript is a scripting language primarily used for web development.',
                'metadata': {'source': 'test', 'category': 'programming'}
            }
        ]
        
        # New documents to add incrementally
        self.new_docs = [
            {
                'id': 'doc4',
                'text': 'Ruby is a dynamic, object-oriented programming language focused on simplicity.',
                'metadata': {'source': 'test', 'category': 'programming'}
            },
            {
                'id': 'doc5',
                'text': 'Go is a statically typed, compiled programming language designed at Google.',
                'metadata': {'source': 'test', 'category': 'programming'}
            }
        ]
        
        # Mock embeddings for initial docs
        self.mock_initial_embeddings = np.random.rand(len(self.initial_docs), 1536)
        # Mock embeddings for new docs
        self.mock_new_embeddings = np.random.rand(len(self.new_docs), 1536)
    
    @patch('src.tree.tree_manager.TreeManager.process_documents')
    @patch('src.tree.tree_manager.TreeManager.build_tree')
    def test_incremental_addition(self, mock_build_tree, mock_process):
        """Test adding documents incrementally to an existing knowledge base."""
        # Setup mock returns
        mock_root = Mock()
        mock_root.get_all_documents = Mock(return_value=self.initial_docs)
        mock_process.return_value = mock_root
        
        # Process initial documents
        self.tree_manager.process_documents(self.initial_docs)
        
        # Verify initial processing
        self.assertEqual(mock_process.call_count, 1)
        
        # Get initial tree state
        initial_doc_count = len(self.initial_docs)
        
        # Step 2: Add new documents incrementally
        mock_root.get_all_documents = Mock(return_value=self.initial_docs + self.new_docs)
        mock_process.return_value = mock_root
        self.tree_manager.process_documents(self.new_docs)
        
        # Verify incremental processing
        self.assertEqual(mock_process.call_count, 2)  # Called again for new docs
        
        # Get final tree state
        final_doc_count = len(self.initial_docs) + len(self.new_docs)
        
        # Verify document count
        self.assertEqual(len(mock_root.get_all_documents()), final_doc_count)
    
    @patch('src.tree.tree_manager.TreeManager.process_documents')
    @patch('src.tree.tree_manager.TreeManager.build_tree')
    def test_incremental_similar_content(self, mock_build_tree, mock_process):
        """Test adding documents with similar content to existing clusters."""
        # Setup mock returns
        mock_root = Mock()
        mock_root.get_all_documents = Mock(return_value=self.initial_docs)
        mock_process.return_value = mock_root
        
        # Process initial documents
        self.tree_manager.process_documents(self.initial_docs)
        
        # Add similar documents
        similar_docs = [
            {
                'id': 'doc4',
                'text': 'Python has extensive libraries for data science and machine learning.',
                'metadata': {'source': 'test', 'category': 'programming'}
            },
            {
                'id': 'doc5',
                'text': 'Java Virtual Machine ensures write once, run anywhere capability.',
                'metadata': {'source': 'test', 'category': 'programming'}
            }
        ]
        
        mock_root.get_all_documents = Mock(return_value=self.initial_docs + similar_docs)
        mock_process.return_value = mock_root
        self.tree_manager.process_documents(similar_docs)
        
        # Verify processing
        self.assertEqual(mock_process.call_count, 2)
        
        # Verify document count
        self.assertEqual(len(mock_root.get_all_documents()), len(self.initial_docs) + len(similar_docs))
    
    @patch('src.tree.tree_manager.TreeManager.process_documents')
    @patch('src.tree.tree_manager.TreeManager.build_tree')
    def test_empty_incremental_update(self, mock_build_tree, mock_process):
        """Test handling of empty document list in incremental update."""
        # Setup mock returns
        mock_root = Mock()
        mock_root.get_all_documents = Mock(return_value=self.initial_docs)
        mock_process.return_value = mock_root
        
        # Process initial documents first
        self.tree_manager.process_documents(self.initial_docs)
        initial_call_count = mock_process.call_count
        
        # Try to process empty document list
        empty_docs = []
        self.tree_manager.process_documents(empty_docs)
        
        # Verify tree state remained unchanged
        self.assertEqual(mock_process.call_count, initial_call_count + 1)  # Should be called but with empty list
        self.assertEqual(len(mock_root.get_all_documents()), len(self.initial_docs))
    
    def tearDown(self):
        """Clean up after each test."""
        # Clear the test index
        self.config['storage']['store'].delete(deleteAll=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2)
