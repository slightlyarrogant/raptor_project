import unittest
import logging
from pathlib import Path
import tempfile
import shutil
import json
import numpy as np
from unittest.mock import Mock, patch
from src.utils.config_validator import ConfigValidator
from src.utils.error_handler import RaptorError, ConfigurationError
from src.monitoring.system_monitor import SystemMonitor
from src.tree.tree_manager import TreeManager
from src.storage.pinecone_manager import PineconeManager

logger = logging.getLogger(__name__)

class TestConfigValidator(unittest.TestCase):
    """Test configuration validation."""
    
    def setUp(self):
        self.validator = ConfigValidator()
        self.test_config = {
            'pinecone': {
                'api_key': 'test_key',
                'dimension': 1536,
                'metric': 'cosine',
                'cloud': 'aws',
                'region': 'us-west-1',
                'index_name': 'test-index'
            },
            'embedding': {
                'model': 'text-embedding-3-small',
                'dimensions': 1536,
                'batch_size': 8
            },
            'clustering': {
                'method': 'kmeans',
                'min_cluster_size': 2,
                'max_clusters': 10,
                'dimension': 20,
                'threshold': 0.15,
                'max_levels': 3
            }
        }
        
    def test_config_validation(self):
        """Test basic configuration validation."""
        validated = self.validator.validate_config(self.test_config)
        self.assertEqual(validated['pinecone']['dimension'], 1536)
        self.assertEqual(validated['embedding']['model'], 'text-embedding-3-small')
        
    def test_missing_required_field(self):
        """Test validation with missing required field."""
        del self.test_config['pinecone']['api_key']
        with self.assertRaises(ConfigurationError):
            self.validator.validate_config(self.test_config)
            
    def test_invalid_value(self):
        """Test validation with invalid value."""
        self.test_config['clustering']['threshold'] = 2.0
        with self.assertRaises(ConfigurationError):
            self.validator.validate_config(self.test_config)
            
    def test_default_values(self):
        """Test default value assignment."""
        del self.test_config['embedding']['batch_size']
        validated = self.validator.validate_config(self.test_config)
        self.assertEqual(validated['embedding']['batch_size'], 8)

class TestSystemMonitor(unittest.TestCase):
    """Test system monitoring functionality."""
    
    def setUp(self):
        self.config = {
            'collection_interval': 1,
            'alert_thresholds': {
                'cpu_percent': 80.0,
                'memory_percent': 85.0,
                'disk_percent': 90.0,
                'error_rate': 0.1
            }
        }
        self.monitor = SystemMonitor(self.config)
        
    def test_metrics_collection(self):
        """Test metrics collection."""
        self.monitor.start_monitoring()
        time.sleep(2)  # Allow time for metrics collection
        self.monitor.process_metrics()
        self.assertGreater(len(self.monitor.metrics_history), 0)
        self.monitor.stop_monitoring()
        
    def test_alert_generation(self):
        """Test alert generation."""
        with patch('psutil.cpu_percent', return_value=90.0):
            self.monitor.start_monitoring()
            time.sleep(2)
            self.monitor.process_metrics()
            self.assertGreater(len(self.monitor.alert_history), 0)
            self.monitor.stop_monitoring()
            
    def test_report_generation(self):
        """Test monitoring report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.monitor.start_monitoring()
            time.sleep(2)
            self.monitor.process_metrics()
            report_path = self.monitor.generate_report(Path(tmpdir))
            self.assertTrue(report_path.exists())
            self.monitor.stop_monitoring()

class TestTreeManager(unittest.TestCase):
    """Test tree management functionality."""
    
    def setUp(self):
        self.config = {
            'embedding': {
                'model': 'text-embedding-3-small',
                'dimensions': 1536,
                'batch_size': 8
            },
            'clustering': {
                'method': 'kmeans',
                'min_cluster_size': 2,
                'max_clusters': 10,
                'dimension': 20,
                'threshold': 0.15,
                'max_levels': 3
            }
        }
        self.tree_manager = TreeManager(self.config)
        
    def test_document_processing(self):
        """Test document processing pipeline."""
        test_docs = [
            "This is test document one.",
            "This is test document two.",
            "This is test document three."
        ]
        metadata = [
            {'filename': f'test_{i}.txt', 'index': i}
            for i in range(len(test_docs))
        ]
        
        # Mock embeddings
        embeddings = np.random.rand(len(test_docs), 1536)
        
        with patch('src.tree.tree_manager.TreeManager._get_embeddings',
                  return_value=embeddings):
            tree_data = self.tree_manager.process_documents(test_docs, metadata)
            self.assertIsNotNone(tree_data)
            self.assertTrue(hasattr(tree_data, 'root'))
            
    def test_tree_building(self):
        """Test tree building logic."""
        # Create test vectors
        num_vectors = 20
        vectors = np.random.rand(num_vectors, 1536)
        metadata = [
            {'filename': f'test_{i}.txt', 'index': i}
            for i in range(num_vectors)
        ]
        
        tree_data = self.tree_manager._build_tree(vectors, metadata)
        self.assertIsNotNone(tree_data)
        self.assertTrue(hasattr(tree_data, 'root'))
        self.assertGreater(len(tree_data.get_all_nodes()), 1)
        
    def test_error_handling(self):
        """Test error handling in tree building."""
        with self.assertRaises(RaptorError):
            self.tree_manager.process_documents([], [])

class TestPineconeManager(unittest.TestCase):
    """Test Pinecone storage functionality."""
    
    def setUp(self):
        self.config = {
            'pinecone': {
                'api_key': 'test_key',
                'dimension': 1536,
                'metric': 'cosine',
                'cloud': 'aws',
                'region': 'us-west-1',
                'index_name': 'test-index'
            }
        }
        
        # Mock Pinecone client
        self.mock_pinecone = Mock()
        with patch('src.storage.pinecone_manager.Pinecone',
                  return_value=self.mock_pinecone):
            self.pinecone_manager = PineconeManager(self.config)
            
    def test_vector_storage(self):
        """Test vector storage operations."""
        vectors = np.random.rand(5, 1536)
        metadata = [
            {'filename': f'test_{i}.txt', 'index': i}
            for i in range(len(vectors))
        ]
        
        # Mock upsert method
        self.mock_pinecone.Index().upsert = Mock()
        
        self.pinecone_manager.store_vectors(vectors, metadata)
        self.mock_pinecone.Index().upsert.assert_called_once()
        
    def test_vector_retrieval(self):
        """Test vector retrieval operations."""
        query_vector = np.random.rand(1536)
        
        # Mock query method
        mock_results = {
            'matches': [
                {
                    'id': 'test_1',
                    'score': 0.9,
                    'metadata': {'filename': 'test_1.txt'}
                }
            ]
        }
        self.mock_pinecone.Index().query = Mock(return_value=mock_results)
        
        results = self.pinecone_manager.query_vectors(query_vector, top_k=1)
        self.assertEqual(len(results['matches']), 1)
        self.mock_pinecone.Index().query.assert_called_once()
        
    def test_error_handling(self):
        """Test error handling in storage operations."""
        # Test with invalid vectors
        with self.assertRaises(RaptorError):
            self.pinecone_manager.store_vectors(
                np.random.rand(5, 100),  # Wrong dimension
                [{'filename': 'test.txt'}]
            )

if __name__ == '__main__':
    unittest.main() 