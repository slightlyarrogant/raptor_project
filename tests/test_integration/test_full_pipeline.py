import unittest
import logging
import time
import os
from dotenv import load_dotenv
from src.tree.tree_manager import TreeManager
from src.utils.config import DEFAULT_CONFIG
from src.visualization.tree_viz import TreeVisualizer
import numpy as np
from src.utils.test_utils import get_mock_openai, get_mock_pinecone
from tests.test_config import DEFAULT_TEST_CONFIG
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

class TestFullPipeline(unittest.TestCase):
    """Full pipeline test with cost and time tracking."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.logger = logging.getLogger(__name__)
        
        load_dotenv()
        cls.start_time = time.time()
        cls.api_calls = {}
        cls.costs = {
            'embedding': 0.0,
            'chat': 0.0
        }
        
        # Initialize with complete configuration
        cls.config = {
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
                'model': 'gpt-4-mini',
                'max_tokens': 1000,
                'temperature': 0.3
            },
            'storage': {
                'store': get_mock_pinecone().Index("test-index"),
                'namespace': 'test',
                'dimension': 1536
            }
        }
        
        # Initialize components
        try:
            cls.tree_manager = TreeManager(cls.config)
            cls.logger.info("✓ Components initialized")
        except Exception as e:
            cls.logger.error(f"Setup failed: {str(e)}")
            raise

    def setUp(self):
        """Create test data that simulates real documentation."""
        # Create hierarchical test data
        self.n_main_topics = 5  # Main documentation topics
        self.n_sub_topics = 3   # Subtopics per main topic
        self.n_docs_per_topic = 4  # Documents per subtopic
        self.dimension = 20
        
        # Generate structured test data
        self.documents = []
        self.metadata = []
        doc_index = 0
        
        # Create documentation hierarchy
        topics = [
            ("Installation", ["Setup", "Configuration", "Requirements"]),
            ("User Interface", ["Navigation", "Forms", "Reports"]),
            ("Database", ["Schema", "Queries", "Optimization"]),
            ("API", ["Endpoints", "Authentication", "Rate Limits"]),
            ("Security", ["Permissions", "Encryption", "Audit Logs"])
        ]
        
        for main_topic, subtopics in topics:
            for subtopic in subtopics:
                for doc_num in range(self.n_docs_per_topic):
                    # Create realistic document content
                    content = f"""
                    # {main_topic} - {subtopic}
                    
                    ## Document {doc_num + 1}
                    
                    This document covers {subtopic.lower()} aspects of the {main_topic.lower()} module.
                    Key points include:
                    
                    1. Important technical details
                    2. Configuration parameters
                    3. Best practices
                    4. Common issues and solutions
                    
                    ### Technical Details
                    
                    The {subtopic.lower()} system implements various features...
                    """
                    
                    self.documents.append(content)
                    self.metadata.append({
                        'file_name': f"{main_topic}_{subtopic}_{doc_num}.md",
                        'file_index': doc_index,
                        'chunk_index': 0,
                        'total_chunks': 1,
                        'topic': main_topic,
                        'subtopic': subtopic
                    })
                    doc_index += 1
        
        logger.info(f"Created {len(self.documents)} test documents")

        # Add directory verification
        test_dirs = [
            Path("analysis_outputs/test-index/tree_viz"),
            Path("analysis_outputs/test-index/document_analysis")
        ]
        
        for dir_path in test_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            assert dir_path.exists(), f"Failed to create test directory: {dir_path}"

    def test_full_pipeline(self):
        """Test entire pipeline from document processing to visualization."""
        try:
            # Process documents
            result = self.tree_manager.process_documents(self.documents, self.metadata)
            
            # Verify result structure
            self.assertIn('clusters', result)
            self.assertIn('summaries', result)
            self.assertIn('tree_structure', result)
            
            # Verify tree structure
            tree = result['tree_structure']
            self.assertIsInstance(tree, dict)
            self.assertIn('nodes', tree)
            
            # Verify clusters
            clusters = result['clusters']
            self.assertTrue(len(clusters) > 0)
            
            # Verify summaries
            summaries = result['summaries']
            self.assertEqual(len(summaries), len(clusters))
            
            # Log metrics
            self.logger.info("\nPipeline Results:")
            self.logger.info("-" * 50)
            self.logger.info(f"Number of clusters: {len(clusters)}")
            self.logger.info(f"Number of summaries: {len(summaries)}")
            self.logger.info(f"Tree depth: {self._get_tree_depth(tree)}")
            
            # Log costs if available
            if hasattr(self, 'costs'):
                self.logger.info(f"Estimated cost: ${sum(self.costs.values()):.4f}")
                
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
        
    def _get_tree_depth(self, tree: Dict) -> int:
        """Calculate tree depth."""
        if not tree.get('nodes'):
            return 0
        return 1 + max(self._get_tree_depth(node) for node in tree['nodes'])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2) 