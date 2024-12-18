import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from pathlib import Path
import numpy as np
from src.tree.node import Node

class TestTreeBuildingSteps(unittest.TestCase):
    """Test each step of the tree building process separately."""
    
    def setUp(self):
        """Set up test environment with controlled test data."""
        # Create test dataset with known structure
        self.test_docs = [
            {
                'text': f'Test document {i}.{j}',
                'metadata': {
                    'filename': f'test_{i}.txt',
                    'chunk_index': j,
                    'section': f'section_{i}'
                }
            }
            for i in range(3)  # 3 main sections
            for j in range(4)  # 4 chunks each
        ]
        
        # Create test embeddings
        self.test_embeddings = np.array([
            [0.1 * i + 0.01 * j] * 1536
            for i in range(3)
            for j in range(4)
        ])
        
        # Create minimal config
        self.test_config = {
            'index_name': 'test-index',
            'embedding': {'manager': Mock()},
            'storage': {'manager': Mock()},
            'chunk_size': 512,
            'overlap_size': 50,
            'max_extension': 100,
            'max_children': 4,
            'min_children': 2,
            'balance_threshold': 0.5,
            'max_depth': 3,
            'clustering': {
                'min_cluster_size': 2,
                'max_cluster_size': 6,
                'min_similarity': 0.05
            }
        }
        
        # Mock the embedding manager
        self.test_config['embedding']['manager'].embed_texts = Mock(return_value=self.test_embeddings)

        # Create mock EnhancedRaptorTree
        patcher = patch('src.tree.enhanced_raptor_tree.EnhancedRaptorTree')
        self.mock_tree_class = patcher.start()
        self.addCleanup(patcher.stop)
        
        self.mock_tree = self.mock_tree_class.return_value
        self.mock_tree.max_children = 4
        self.mock_tree.min_children = 2
        self.mock_tree.max_depth = 3
        self.mock_tree.balance_threshold = 0.5

    def test_node_creation(self):
        """Test node creation and properties."""
        texts = ["Test node"]
        metadata = {'type': 'test'}
        node = Node(texts=texts, metadata=metadata)
        
        self.assertEqual(node.texts, texts)
        self.assertEqual(node.metadata['type'], 'test')
        self.assertEqual(node.level, 0)
        self.assertEqual(len(node.children), 0)
        self.assertTrue(node.metadata['is_leaf'])
        self.assertEqual(node.metadata['node_type'], 'leaf')

    def test_node_relationships(self):
        """Test parent-child relationships."""
        parent = Node(texts=["Parent"])
        child1 = Node(texts=["Child 1"])
        child2 = Node(texts=["Child 2"])
        
        parent.add_child(child1)
        parent.add_child(child2)
        
        self.assertEqual(len(parent.children), 2)
        self.assertEqual(child1.parent, parent)
        self.assertEqual(child2.parent, parent)
        self.assertEqual(child1.level, 1)
        self.assertEqual(child2.level, 1)
        self.assertFalse(parent.metadata['is_leaf'])
        self.assertEqual(parent.metadata['node_type'], 'branch')

    def test_node_validation(self):
        """Test node validation rules."""
        parent = Node(texts=["Parent"])
        child = Node(texts=["Child"])
        grandchild = Node(texts=["Grandchild"])
        
        parent.add_child(child)
        child.add_child(grandchild)
        
        # Test parent-child relationships
        self.assertEqual(child.parent, parent)
        self.assertEqual(grandchild.parent, child)
        
        # Test levels
        self.assertEqual(parent.level, 0)
        self.assertEqual(child.level, 1)
        self.assertEqual(grandchild.level, 2)
        
        # Test heights
        self.assertEqual(parent.get_height(), 2)
        self.assertEqual(child.get_height(), 1)
        self.assertEqual(grandchild.get_height(), 0)

    def test_node_balancing(self):
        """Test tree balancing."""
        root = Node(texts=["Root"])
        
        # Create an unbalanced tree
        current = root
        for i in range(4):  # Create 4 nodes to get height 4 (including root)
            node = Node(texts=[f"Node {i}"])
            current.add_child(node)
            current = node
        
        self.assertFalse(root.is_balanced(0.5))
        self.assertEqual(root.get_height(), 4)  # Height is 4 because we have 5 nodes in a chain

    def test_node_serialization(self):
        """Test node serialization/deserialization."""
        original = Node(
            texts=["Test node"],
            metadata={'type': 'test', 'value': 123}
        )
        child = Node(texts=["Child node"])
        original.add_child(child)
        
        # Serialize
        serialized = original.to_dict()
        
        # Deserialize
        deserialized = Node.from_dict(serialized)
        
        self.assertEqual(deserialized.texts, original.texts)
        self.assertEqual(deserialized.metadata['type'], original.metadata['type'])
        self.assertEqual(deserialized.metadata['value'], original.metadata['value'])
        self.assertEqual(len(deserialized.children), len(original.children))
        self.assertEqual(deserialized.children[0].texts, child.texts)

if __name__ == '__main__':
    unittest.main(verbosity=2)