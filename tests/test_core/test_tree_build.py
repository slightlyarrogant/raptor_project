import pytest
import numpy as np
from src.tree.tree_manager import TreeManager

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    texts = [
        "This is document one about ERP systems.",
        "Document two discusses CRM functionality.",
        "Third document about warehouse management.",
        "Fourth document about financial modules.",
        "Fifth document about system integration."
    ]
    
    # Create mock embeddings (5 documents x 1536 dimensions)
    embeddings = np.random.rand(5, 1536)
    
    # Create sample metadata
    metadata = [
        {
            'file_name': f'doc_{i}.txt',
            'file_path': f'/test/docs/doc_{i}.txt',
            'content_length': len(text),
            'processing_date': '2024-01-01'
        }
        for i, text in enumerate(texts)
    ]
    
    return texts, embeddings, metadata

@pytest.fixture
def tree_config():
    """Create test configuration."""
    return {
        'clustering': {
            'min_cluster_size': 2,
            'max_cluster_size': 10,
            'min_docs_per_cluster': 2,
            'target_children': 3,
            'dimension': 1536,
            'threshold': 0.5,
            'max_levels': 3
        },
        'embedding': {
            'model': 'test-model',
            'dimension': 1536
        }
    }

def test_tree_building(sample_data, tree_config):
    """Test tree building functionality."""
    texts, embeddings, metadata = sample_data
    
    # Initialize TreeManager
    tree_manager = TreeManager(tree_config)
    
    # Build tree
    tree = tree_manager.build_tree(
        texts=texts,
        embeddings=embeddings,
        metadata=metadata
    )
    
    # Verify tree structure
    assert isinstance(tree, dict), "Tree should be a dictionary"
    assert 'id' in tree, "Tree should have an ID"
    assert 'metadata' in tree, "Tree should have metadata"
    assert 'texts' in tree or 'nodes' in tree, "Tree should have either texts or nodes"
    
    # Verify metadata
    assert tree['metadata']['level'] == 0, "Root should be at level 0"
    assert tree['metadata']['size'] > 0, "Tree should have non-zero size"
    
    # If it's a branch node, verify children
    if 'nodes' in tree:
        assert isinstance(tree['nodes'], list), "Nodes should be a list"
        assert len(tree['nodes']) > 0, "Branch should have children"
        
        # Verify each child
        for child in tree['nodes']:
            assert isinstance(child, dict), "Child should be a dictionary"
            assert 'id' in child, "Child should have an ID"
            assert 'metadata' in child, "Child should have metadata"
            assert child['metadata']['level'] > 0, "Child level should be greater than root"

def test_tree_manager_integration(sample_data, tree_config):
    """Test TreeManager integration with TreeManager."""
    texts, embeddings, metadata = sample_data
    
    # Initialize TreeManager
    tree_manager = TreeManager(tree_config)
    
    # Process documents
    result = tree_manager.process_documents_locally(
        texts=texts,
        metadata=metadata,
        cache=None  # Mock cache for testing
    )
    
    # Verify result structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert 'tree' in result, "Result should contain tree"
    assert 'summaries' in result, "Result should contain summaries"
    assert 'stats' in result, "Result should contain stats"
    
    # Verify tree stats
    stats = result['stats']
    assert 'total_nodes' in stats, "Stats should include total_nodes"
    assert 'total_documents' in stats, "Stats should include total_documents"
    assert stats['total_documents'] == len(texts), "Document count should match input"

def test_metadata_consistency(sample_data, tree_config):
    """Test metadata consistency through the tree."""
    texts, embeddings, metadata = sample_data
    
    tree_manager = TreeManager(tree_config)
    tree = tree_manager.build_tree(
        texts=texts,
        embeddings=embeddings,
        metadata=metadata
    )
    
    def verify_metadata(node):
        """Recursively verify metadata in tree."""
        assert 'metadata' in node, "Node missing metadata"
        meta = node['metadata']
        
        required_fields = ['id', 'level', 'size', 'timestamp']
        for field in required_fields:
            assert field in meta, f"Metadata missing required field: {field}"
        
        if 'nodes' in node:
            for child in node['nodes']:
                verify_metadata(child)
                assert child['metadata']['level'] == node['metadata']['level'] + 1, \
                    "Child level should be parent level + 1"
    
    verify_metadata(tree) 