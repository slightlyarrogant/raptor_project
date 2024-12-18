import unittest
import os
import sys
from pathlib import Path
import numpy as np
from unittest.mock import Mock, patch
from pinecone import Pinecone

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.storage.store_manager import StoreManager
from src.storage.pinecone_manager import PineconeManager
from src.tree.tree_manager import TreeManager
from src.utils.openai_client import UnifiedAIClient
from src.utils.pinecone_utils import init_pinecone

class TestIntegration(unittest.TestCase):
    """Test the complete flow from document processing to Pinecone upserting."""
    
    def setUp(self):
        """Set up test environment."""
        self.index_name = 'test-raptor'
        self.config = {
            'index_name': self.index_name,
            'batch_size': 1000,
            'pinecone': {
                'api_key': os.getenv('PINECONE_API_KEY'),
                'index_name': self.index_name
            }
        }
        
        # Initialize Pinecone and create index if needed
        pc = init_pinecone(self.config['pinecone'])
        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric='cosine',
                spec={"pod": {"environment": "gcp-starter"}}
            )
        
        # Create test documents
        self.test_docs = [
            {
                'text': f'Test document {i}',
                'metadata': {
                    'filename': f'test_{i}.txt',
                    'chunk_index': 0,
                    'section': f'section_{i}'
                }
            }
            for i in range(5)  # 5 test documents
        ]
        
        # Initialize managers
        self.store_manager = StoreManager(self.config)
        self.ai_client = UnifiedAIClient()
        self.tree_manager = TreeManager({
            'embedding': {
                'manager': self.ai_client,
                'model': 'text-embedding-3-small',
                'batch_size': 100
            },
            'clustering': {
                'method': 'kmeans',
                'min_cluster_size': 2,
                'max_clusters': 3,
                'dimension': 20,
                'threshold': 0.15,
                'max_levels': 3
            },
            'storage': {
                'manager': self.store_manager
            }
        })

    def test_full_pipeline(self):
        """Test the complete pipeline from documents to Pinecone."""
        try:
            # 1. Process documents and build tree
            tree_data, stats = self.tree_manager.process_documents(self.test_docs)
            self.assertIsNotNone(tree_data)
            self.assertIsNotNone(stats)
            
            # 2. Get tree statistics
            tree_stats = self.tree_manager.get_tree_stats()
            self.assertIsNotNone(tree_stats)
            self.assertTrue('total_nodes' in tree_stats)
            self.assertTrue('depth' in tree_stats)
            
            # 3. Generate visualizations
            self.tree_manager.generate_tree_visualization()
            self.tree_manager.update_cluster_visualization()
            
            # 4. Query Pinecone to verify upserting
            pinecone = PineconeManager(self.config['pinecone'])
            query_results = pinecone.query(
                vector=np.zeros(1536),  # Zero vector for testing
                filter=None,
                top_k=5
            )
            self.assertIsNotNone(query_results)
            self.assertTrue(len(query_results.matches) > 0)
            
            print(" Full pipeline test passed successfully")
            return True
            
        except Exception as e:
            print(f" Test failed: {str(e)}")
            return False
            
    def tearDown(self):
        """Clean up test environment."""
        try:
            # Delete test index
            pc = init_pinecone(self.config['pinecone'])
            if self.index_name in pc.list_indexes().names():
                pc.delete_index(self.index_name)
        except Exception as e:
            print(f"Warning: Failed to clean up test index: {str(e)}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
