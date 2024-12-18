import logging
import sys
from pathlib import Path
import json
import click
import os
from src.tree.tree_manager import TreeManager
from src.utils.config import (
    DEFAULT_CONFIG,
    PINECONE_API_KEY,
    PINECONE_REGION,
    PINECONE_CLOUD,
    PINECONE_METRIC,
    PINECONE_DIMENSION
)
from src.embedding.embed_manager import EmbedManager
from src.storage.pinecone_manager import PineconeManager
from src.visualization.tree_viz import TreeVisualizer
from src.storage.db_manager import DatabaseManager
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.command()
@click.option('--index-name', required=True, help='Index name')
def recover_visualization(index_name: str):
    """Recover tree visualization from database."""
    try:
        # Set up paths
        data_dir = Path('data') / index_name
        
        # Initialize managers
        db_manager = DatabaseManager(str(data_dir.parent), index_name)
        pinecone_manager = PineconeManager(index_name)
        tree_visualizer = TreeVisualizer(index_name)
        
        # Load configuration
        tree_config = {
            'index_name': index_name,
            'data_dir': str(data_dir)
        }
        
        # Initialize TreeManager
        tree_manager = TreeManager(tree_config)
        logger.info("Initialized TreeManager")
        
        # Load root node
        root_nodes = db_manager.get_root_nodes()
        if not root_nodes:
            logger.error("No root nodes found in database")
            return
        root_node_data = root_nodes[0]
        
        # Load all nodes
        all_nodes = db_manager.get_tree_nodes()
        logger.info(f"Loaded {len(all_nodes)} nodes from database")
        
        # Load embeddings for nodes with embedding_ids
        embedding_ids = [node['embedding_id'] for node in all_nodes if node['embedding_id']]
        if embedding_ids:
            embeddings = pinecone_manager.fetch_vectors(embedding_ids)
            # Add embeddings to node data
            for node in all_nodes:
                if node['embedding_id'] in embeddings:
                    node['embedding'] = embeddings[node['embedding_id']]
        
        # Construct tree data
        tree_data = {
            'tree': {
                'nodes': all_nodes,
                'metadata': {
                    'root_id': root_node_data['node_id'],
                    'total_nodes': len(all_nodes),
                    'max_depth': max(node['level'] for node in all_nodes),
                    'created_at': datetime.now().isoformat(),
                    'last_modified': datetime.now().isoformat()
                }
            }
        }
        
        # Create visualizations
        tree_visualizer.create_visualizations(tree_data)
        logger.info("Successfully created visualizations")
        
    except Exception as e:
        logger.error(f"Failed to recover visualization: {str(e)}")
        raise

if __name__ == '__main__':
    recover_visualization()
