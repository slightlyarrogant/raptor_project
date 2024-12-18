"""Test script for tree building logic using mock embeddings."""
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict
import pickle
import os
from unittest.mock import Mock, MagicMock

from src.clustering.cluster_manager import ClusterManager

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class MockTreeManager:
    """Mock TreeManager that only uses clustering."""
    def __init__(self, config: Dict):
        self.logger = logger
        self.cluster_manager = ClusterManager(config)
        
    def process_documents(self, documents: List[str], metadata: List[Dict]) -> Dict:
        """Process documents using only clustering."""
        try:
            # Use pre-generated embeddings
            embeddings = np.random.randn(len(documents), 1536)
            
            # Create clusters
            clusters = self.cluster_manager.cluster_embeddings(embeddings)
            logger.warning(f"Created {len(clusters)} clusters")
            
            return {
                'tree_structure': clusters,
                'embeddings': embeddings,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error in document processing pipeline: {str(e)}")
            raise

def test_tree_building(n_samples: int = 100):
    """Test tree building with mock data."""
    try:
        logger.warning(f"Testing tree building with {n_samples} samples")
        
        # Generate or load mock embeddings
        cache_file = "mock_embeddings.pkl"
        if os.path.exists(cache_file):
            logger.warning("Loading cached embeddings")
            with open(cache_file, 'rb') as f:
                embeddings = pickle.load(f)
        else:
            logger.warning("Generating mock embeddings")
            embeddings = np.random.randn(n_samples, 1536)
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
                
        # Create mock documents and metadata
        mock_docs = [f"Document {i}" for i in range(n_samples)]
        mock_metadata = [
            {'file_name': f'doc_{i}.txt', 'file_index': i, 'chunk_index': 0, 'total_chunks': 1}
            for i in range(n_samples)
        ]
        
        # Create configuration for clustering only
        config = {
            'clustering': {
                'umap': {
                    'n_neighbors': 10,
                    'n_components': 2,
                    'metric': 'cosine'
                },
                'gmm': {
                    'max_clusters': min(100, n_samples // 10),
                    'threshold': 0.4
                },
                'similarity_threshold': 0.6,
                'min_cluster_size': 3,
                'max_cluster_size': 20
            }
        }
        
        # Initialize mock TreeManager
        tree_manager = MockTreeManager(config)
        
        # Process documents
        logger.warning("Testing clustering pipeline...")
        result = tree_manager.process_documents(mock_docs, mock_metadata)
        
        # Analyze results
        clusters = result['tree_structure']
        leaf_nodes = [c for c in clusters if not c.get('children')]
        branch_nodes = [c for c in clusters if c.get('children')]
        max_depth = max(c.get('level', 0) for c in clusters) + 1
        
        print("\nClustering Results:")
        print(f"Total clusters: {len(clusters)}")
        print(f"Leaf nodes: {len(leaf_nodes)}")
        print(f"Branch nodes: {len(branch_nodes)}")
        print(f"Maximum depth: {max_depth}")
        
        # Analyze cluster sizes
        sizes = [len(c['docs']) for c in leaf_nodes]
        print(f"\nCluster Sizes:")
        print(f"Min: {min(sizes)}")
        print(f"Max: {max(sizes)}")
        print(f"Average: {sum(sizes)/len(sizes):.1f}")
        
        # Print cluster hierarchy
        print("\nCluster Hierarchy:")
        for level in range(max_depth):
            level_clusters = [c for c in clusters if c.get('level') == level]
            print(f"Level {level}: {len(level_clusters)} clusters")
            for cluster in level_clusters:
                print(f"  Cluster {cluster['id']}: {len(cluster['docs'])} docs, "
                      f"{len(cluster.get('children', []))} children")
                if cluster.get('children'):
                    print(f"    Children: {cluster['children']}")
                    
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting tree building test...")
    success = test_tree_building(100)
    print(f"\nTest {'succeeded' if success else 'failed'}")