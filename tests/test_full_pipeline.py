import unittest
import logging
import time
import os
from dotenv import load_dotenv
from src.tree.tree_manager import TreeManager
from src.utils.config import DEFAULT_CONFIG
from src.visualization.tree_viz import TreeVisualizer
import numpy as np

logger = logging.getLogger(__name__)

class TestFullPipeline(unittest.TestCase):
    """Full pipeline test with cost and time tracking."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        load_dotenv()
        cls.start_time = time.time()
        cls.api_calls = {
            'embedding': 0,
            'chat': 0
        }
        cls.costs = {
            'embedding': 0.0,  # $0.00002 per 1K tokens
            'chat': 0.0       # $0.01 per 1K input tokens, $0.03 per 1K output tokens
        }
        
        # Configuration
        cls.config = DEFAULT_CONFIG.copy()
        cls.config['clustering'].update({
            'dimension': 20,
            'threshold': 0.15,
            'max_levels': 5,
            'min_cluster_size': 5,
            'max_cluster_size': 100
        })
        
        # Initialize components
        try:
            cls.tree_manager = TreeManager(cls.config)
            logger.info("✓ Components initialized")
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
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

    def test_full_pipeline(self):
        """Test entire pipeline from document processing to visualization."""
        start_time = time.time()
        
        try:
            # 1. Process Documents
            logger.info("Processing documents...")
            result = self.tree_manager.process_documents(self.documents, self.metadata)
            self.assertIn('tree_structure', result)
            logger.info("✓ Documents processed")
            
            # 2. Build Tree
            logger.info("Building knowledge tree...")
            tree = self.tree_manager.build_tree()
            self.assertIsNotNone(tree)
            logger.info("✓ Tree built")
            
            # 3. Validate Tree Structure
            self._validate_tree_structure(tree)
            logger.info("✓ Tree structure validated")
            
            # 4. Generate Visualizations
            logger.info("Generating visualizations...")
            viz_dir = "test_outputs/full_pipeline"
            os.makedirs(viz_dir, exist_ok=True)
            
            visualizer = TreeVisualizer(
                output_dir=viz_dir,
                embeddings=self.tree_manager.get_all_embeddings()
            )
            
            # Generate all visualizations
            visualizer.visualize_tree_structure(tree, "tree_structure.html")
            visualizer.visualize_cluster_distribution(tree, "cluster_distribution.png")
            visualizer.visualize_embeddings(tree, "embeddings_3d.html")
            visualizer.visualize_tree_sunburst(tree, "tree_sunburst.html")
            visualizer.visualize_cluster_network(tree, "cluster_network.html")
            logger.info("✓ Visualizations generated")
            
            # 5. Calculate and Log Metrics
            self._log_metrics(tree)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
        finally:
            # Log execution time and costs
            execution_time = time.time() - start_time
            logger.info("\nExecution Summary:")
            logger.info("-" * 50)
            logger.info(f"Total time: {execution_time:.2f}s")
            logger.info(f"Estimated cost: ${sum(self.costs.values()):.4f}")
            logger.info(f"API calls: {sum(self.api_calls.values())}")

    def _validate_tree_structure(self, tree: dict):
        """Validate tree structure meets requirements."""
        # Check basic structure
        self.assertIn('nodes', tree)
        self.assertIn('metadata', tree)
        
        # Validate nodes
        nodes = tree['nodes']
        self.assertGreater(len(nodes), 1, "Tree should have multiple top-level nodes")
        
        # Check clustering
        total_docs = sum(len(node.get('texts', [])) for node in nodes)
        self.assertEqual(total_docs, len(self.documents), "All documents should be included")
        
        # Validate topic grouping
        topics = set(meta['topic'] for meta in self.metadata)
        node_keywords = set(node.get('keyword', '') for node in nodes)
        self.assertTrue(any(topic.lower() in ' '.join(node_keywords).lower() for topic in topics),
                       "Topics should be reflected in node keywords")

    def _log_metrics(self, tree: dict):
        """Log comprehensive tree metrics."""
        logger.info("\nTree Metrics:")
        logger.info("-" * 50)
        
        # Structure metrics
        total_nodes = self._count_nodes(tree)
        leaf_nodes = self._count_leaf_nodes(tree)
        max_depth = self._get_max_depth(tree)
        
        logger.info(f"Structure:")
        logger.info(f"- Total nodes: {total_nodes}")
        logger.info(f"- Leaf nodes: {leaf_nodes}")
        logger.info(f"- Maximum depth: {max_depth}")
        logger.info(f"- Branch nodes: {total_nodes - leaf_nodes}")
        
        # Content metrics
        total_docs = len(self.documents)
        avg_cluster_size = total_docs / max(1, leaf_nodes)
        
        logger.info(f"\nContent:")
        logger.info(f"- Total documents: {total_docs}")
        logger.info(f"- Average cluster size: {avg_cluster_size:.2f}")
        
        # Topic distribution
        topic_counts = {}
        for meta in self.metadata:
            topic_counts[meta['topic']] = topic_counts.get(meta['topic'], 0) + 1
            
        logger.info("\nTopic Distribution:")
        for topic, count in topic_counts.items():
            logger.info(f"- {topic}: {count} documents")

    def _count_nodes(self, tree: dict) -> int:
        """Count total number of nodes."""
        if not tree.get('nodes'):
            return 1
        return 1 + sum(self._count_nodes(node) for node in tree['nodes'])

    def _count_leaf_nodes(self, tree: dict) -> int:
        """Count number of leaf nodes."""
        if not tree.get('nodes'):
            return 1
        return sum(self._count_leaf_nodes(node) for node in tree['nodes'])

    def _get_max_depth(self, tree: dict) -> int:
        """Calculate maximum tree depth."""
        if not tree.get('nodes'):
            return 0
        return 1 + max(self._get_max_depth(node) for node in tree['nodes'])

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2) 