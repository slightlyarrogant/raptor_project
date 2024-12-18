"""Tests for tree data models."""
import pytest
from typing import Dict, Any
from unittest.mock import patch, MagicMock
import numpy as np
from src.models.data_models import NodeData, EdgeData, TreeData
from datetime import datetime
from src.models.config import (
    ClusteringConfig,
    EmbeddingConfig,
    StorageConfig,
    AIConfig,
    RaptorConfig
)
from src.tree.tree_manager import TreeManager
from src.embedding.embed_manager import EmbedManager
from src.clustering.cluster_manager import ClusterManager
from src.visualization.tree_viz import TreeVisualizer

def test_node_data_creation():
    """Test creation and validation of NodeData."""
    # Test basic creation
    node = NodeData(
        id="test_node",
        text="Test content",
        level=1,
        metadata={"importance": 0.8}
    )
    
    assert node.id == "test_node"
    assert node.text == "Test content"
    assert node.level == 1
    assert node.metadata["importance"] == 0.8
    
    # Test metadata updates
    node.metadata.update({
        "summary": "Test summary",
        "key_concepts": ["concept1", "concept2"]
    })
    
    assert "summary" in node.metadata
    assert len(node.metadata["key_concepts"]) == 2

def test_tree_data_conversion():
    """Test conversion between TreeData and dictionary formats."""
    # Create sample tree
    nodes = [
        NodeData(id="root", text="Root node", level=0),
        NodeData(id="child1", text="Child 1", level=1),
        NodeData(id="child2", text="Child 2", level=1)
    ]

    edges = [
        EdgeData(source="root", target="child1", weight=0.9),
        EdgeData(source="root", target="child2", weight=0.8)
    ]

    tree = TreeData()
    for node in nodes:
        tree.add_node(node)
    for edge in edges:
        tree.add_edge(edge)
    tree.update_metadata(created_at=datetime.now().isoformat())

    # Test conversion to dict
    tree_dict = tree.to_dict()
    assert "nodes" in tree_dict
    assert "edges" in tree_dict
    assert "metadata" in tree_dict
    assert len(tree_dict["nodes"]) == 3
    assert len(tree_dict["edges"]) == 2

    # Test conversion from dict
    new_tree = TreeData.from_dict(tree_dict)
    assert len(new_tree.nodes) == 3
    assert len(new_tree.edges) == 2
    assert new_tree.nodes[0].id == "root"
    assert new_tree.edges[0].source == "root"
    assert new_tree.edges[0].target == "child1"

def test_config_validation():
    """Test configuration validation and defaults."""
    # Test clustering config
    cluster_config = ClusteringConfig(
        dimension=1536,
        similarity_threshold=0.15
    )
    assert cluster_config.method == "kmeans"  # Check default
    
    # Test embedding config
    embed_config = EmbeddingConfig()
    assert embed_config.model == "text-embedding-3-small"
    assert embed_config.dimension == 1536
    
    # Test complete config
    config = RaptorConfig(
        clustering=cluster_config,
        embedding=embed_config,
        storage=StorageConfig(index_name="test"),
        ai=AIConfig()
    )
    assert config.storage.namespace == "default"  # Check default

@patch('src.embedding.embed_manager.OpenAI')
def test_component_integration(mock_openai):
    """Test data flow between components using standard models."""
    # Mock OpenAI client
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=np.random.rand(1536)) for _ in range(3)]
    )
    mock_openai.return_value = mock_client
    
    # Create configuration
    config = RaptorConfig(
        clustering=ClusteringConfig(dimension=1536),
        embedding=EmbeddingConfig(model="text-embedding-3-small"),
        storage=StorageConfig(index_name="test"),
        ai=AIConfig(model="gpt-4")
    )
    
    # Create test documents
    docs = [
        "This is the first test document.",
        "This is the second test document.",
        "This is the third test document."
    ]
    
    # Initialize components
    tree_manager = TreeManager(config)
    embed_manager = EmbedManager(config.embedding)
    cluster_manager = ClusterManager(config.clustering)
    
    # Get embeddings
    embeddings = embed_manager.get_embeddings(docs)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[1] == config.embedding.dimension
    
    # Perform clustering
    labels, centers = cluster_manager.cluster_documents(embeddings)
    assert len(labels) == len(docs)
    assert centers.shape[1] == config.clustering.dimension
    
    # Build tree
    tree_manager.build_tree(docs, embeddings)
    tree_data = tree_manager.get_tree_data()
    
    assert isinstance(tree_data, TreeData)
    assert len(tree_data.nodes) > 0
    assert len(tree_data.edges) > 0
    assert tree_data.nodes[0].id == "root"
    assert any(node.metadata.get("type") == "cluster" for node in tree_data.nodes)
    assert any(node.metadata.get("type") == "document" for node in tree_data.nodes)
    assert all(node.level in [0, 1, 2] for node in tree_data.nodes)
    assert any(edge.source == "root" for edge in tree_data.edges)
    assert any(edge.source.startswith("cluster_") for edge in tree_data.edges)

def test_error_handling():
    """Test error handling with invalid data."""
    tree = TreeData()
    node1 = NodeData(id="node1", text="Node 1", level=0)
    node2 = NodeData(id="node2", text="Node 2", level=1)
    tree.add_node(node1)
    tree.add_node(node2)
    
    # Test adding edge with non-existent source node
    with pytest.raises(ValueError):
        tree.add_edge(EdgeData(source="nonexistent", target="node2"))
        
    # Test adding edge with non-existent target node
    with pytest.raises(ValueError):
        tree.add_edge(EdgeData(source="node1", target="nonexistent"))
        
    # Test validation with duplicate node IDs
    tree = TreeData()
    node1 = NodeData(id="duplicate", text="Node 1", level=0)
    node2 = NodeData(id="duplicate", text="Node 2", level=1)
    tree.add_node(node1)
    tree.add_node(node2)
    with pytest.raises(ValueError):
        tree.validate()
        
    # Test validation with cycle in tree
    tree = TreeData()
    node1 = NodeData(id="node1", text="Node 1", level=0)
    node2 = NodeData(id="node2", text="Node 2", level=1)
    node3 = NodeData(id="node3", text="Node 3", level=1)
    tree.add_node(node1)
    tree.add_node(node2)
    tree.add_node(node3)
    tree.add_edge(EdgeData(source="node1", target="node2"))
    tree.add_edge(EdgeData(source="node2", target="node3"))
    tree.add_edge(EdgeData(source="node3", target="node1"))
    with pytest.raises(ValueError):
        tree.validate()

if __name__ == "__main__":
    pytest.main([__file__])
