import unittest
import numpy as np
from unittest.mock import Mock

from tree.tree_manager import TreeManager, Node

class TestTreeManager(unittest.TestCase):
    def setUp(self):
        self.tree_manager = TreeManager()
        self.sample_texts = ["text1", "text2", "text3"]
        self.sample_embeddings = np.array([[1,2,3], [4,5,6], [7,8,9]])
        self.sample_metadata = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
        
    def test_empty_documents(self):
        with self.assertRaises(ValueError):
            self.tree_manager.process_documents([])
            
    def test_invalid_documents(self):
        with self.assertRaises(ValueError):
            self.tree_manager.process_documents([{"text": ""}, {"text": ""}])
            
    def test_single_document(self):
        documents = [{"text": "test", "metadata": {"id": "1"}}]
        
        mock_embed = Mock()
        mock_embed.embed_texts.return_value = np.array([[1,2,3]])
        self.tree_manager.embed_manager = mock_embed
        
        tree_data, _ = self.tree_manager.process_documents(documents)
        
        self.assertIsNotNone(tree_data)
        self.assertEqual(tree_data['texts'], ["test"])
            
    def test_build_tree_small_input(self):
        texts = ["text1"]
        embeddings = np.array([[1,2,3]])
        metadata = [{"id": "1"}]
        
        root = self.tree_manager.build_tree(texts, embeddings, metadata)
        
        self.assertIsNotNone(root)
        self.assertTrue(root.has_data())
        self.assertEqual(len(root.children), 0)
        
    def test_build_tree_clustering(self):
        mock_cluster = Mock()
        mock_cluster.cluster_embeddings.return_value = np.array([0, 1, 1])
        self.tree_manager.cluster_manager = mock_cluster
        
        root = self.tree_manager.build_tree(
            self.sample_texts,
            self.sample_embeddings,
            self.sample_metadata
        )
        
        self.assertIsNotNone(root)
        self.assertTrue(len(root.children) > 0)
        
    def test_build_tree_clustering_failure(self):
        mock_cluster = Mock()
        mock_cluster.cluster_embeddings.return_value = None
        self.tree_manager.cluster_manager = mock_cluster
        
        root = self.tree_manager.build_tree(
            self.sample_texts,
            self.sample_embeddings,
            self.sample_metadata
        )
        
        self.assertIsNotNone(root)
        self.assertTrue(root.has_data())
        self.assertEqual(len(root.children), 0)
            
    def test_node_operations(self):
        # Test node creation and data handling
        data = {
            'texts': ['test'],
            'embedding': [1,2,3],
            'metadata': {'id': '1'}
        }
        node = Node(data)
        
        self.assertTrue(node.has_data())
        self.assertEqual(node.get_data(), data)
        
        # Test child operations
        child = Node({'texts': ['child']})
        success = node.add_child(child)
        
        self.assertTrue(success)
        self.assertEqual(len(node.children), 1)
        self.assertEqual(child.parent, node)
        
        # Test null child
        success = node.add_child(None)
        self.assertFalse(success)
        
    def test_node_serialization(self):
        data = {
            'id': '1',
            'depth': 0,
            'num_docs': 1,
            'texts': ['test'],
            'embeddings': [[1,2,3]],
            'metadata': {'type': 'root'}
        }
        node = Node(data)
        
        # Add a child
        child_data = {
            'id': '2',
            'depth': 1,
            'texts': ['child'],
            'embeddings': [[4,5,6]],
            'metadata': {'type': 'leaf'}
        }
        child = Node(child_data)
        node.add_child(child)
        
        # Test serialization
        node_dict = node.to_dict()
        
        self.assertEqual(node_dict['id'], '1')
        self.assertEqual(node_dict['depth'], 0)
        self.assertEqual(len(node_dict['children']), 1)
        
        # Test deserialization
        restored = Node.from_dict(node_dict)
        
        self.assertEqual(restored.data['id'], '1')
        self.assertEqual(len(restored.children), 1)
