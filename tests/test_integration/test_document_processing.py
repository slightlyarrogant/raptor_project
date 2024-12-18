import unittest
from unittest.mock import Mock, patch
from src.tree.tree_manager import TreeManager
from src.utils.config import DEFAULT_CONFIG
import numpy as np
from tests.test_config import get_mock_pinecone, DEFAULT_TEST_CONFIG
import logging

class TestDocumentProcessing(unittest.TestCase):
    def setUp(self):
        # Add logger initialization
        self.logger = logging.getLogger(__name__)
        
        self.config = {
            'embedding': {
                'dimension': 1536,
                'model': 'text-embedding-3-small'
            },
            'clustering': {
                'dimension': 20,
                'threshold': 0.3,
                'max_levels': 3,
                'min_cluster_size': 5,
                'max_cluster_size': 20,
                'min_docs_per_cluster': 5,
                'target_children': 3,
                'force_merge': True,
                'min_similarity': 0.15,
                'decay_rate': 0.7,
                'merge_threshold': 0.25
            },
            'summarization': {
                'model': 'gpt-4o-mini',
                'max_tokens': 1000,
                'temperature': 0.3
            },
            'storage': {
                'store': get_mock_pinecone().Index("test-index"),
                'namespace': 'test',
                'dimension': 1536
            }
        }
        self.tree_manager = TreeManager(self.config)
        
    @patch('src.embedding.embed_manager.EmbedManager.embed_documents')
    @patch('src.clustering.cluster_manager.ClusterManager.cluster_embeddings')
    @patch('src.summarization.summary_manager.SummaryManager.summarize_clusters')
    def test_end_to_end_processing(self, mock_summarize, mock_cluster, mock_embed):
        """Test the entire document processing pipeline."""
        # Prepare test data with more realistic dimensions
        test_docs = [
            "Document 1 with some meaningful content",
            "Document 2 with different meaningful content",
            "Document 3 with another topic entirely",
            "Document 4 related to document 1",
            "Document 5 related to document 2"
        ]
        
        # Create well-formed embeddings
        mock_embed.return_value = np.random.rand(len(test_docs), 1536)
        
        # Create realistic cluster assignments with proper structure
        mock_cluster.return_value = {
            'nodes': [
                {
                    'texts': ['Document 1', 'Document 4'],
                    'metadata': {'coherence': 0.9, 'size': 2}
                },
                {
                    'texts': ['Document 2', 'Document 3', 'Document 5'],
                    'metadata': {'coherence': 0.8, 'size': 3}
                }
            ],
            'level': 0,
            'metadata': {'size': 5, 'n_clusters': 2}
        }
        
        # Create meaningful summaries
        mock_summarize.return_value = [
            {'id': 0, 'summary': 'Summary of cluster 1'},
            {'id': 1, 'summary': 'Summary of cluster 2'}
        ]
        
        # Process documents
        try:
            result = self.tree_manager.process_documents(test_docs)
            
            # Verify the processing pipeline
            self.assertTrue(mock_embed.called, "Embedding should be called")
            self.assertTrue(mock_cluster.called, "Clustering should be called")
            self.assertTrue(mock_summarize.called, "Summarization should be called")
            self.assertIsInstance(result, dict, "Result should be a dictionary")
            
            # Log processing details
            self.logger.info(f"Mock summarize call count: {mock_summarize.call_count}")
            self.logger.info(f"Mock cluster call count: {mock_cluster.call_count}")
            self.logger.info(f"Mock embed call count: {mock_embed.call_count}")
            
        except Exception as e:
            self.logger.error(f"Test failed: {str(e)}")
            raise
        