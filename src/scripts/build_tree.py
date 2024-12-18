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
@click.option('--index-name', default='raptor-cfi', help='Index name')
def build_tree(index_name: str):
    """Build document tree and save to database."""
    try:
        # Initialize base configurations
        db_path = f"cache/{index_name}_cache.db"
        logger.info(f"Using cache database: {db_path}")
        
        # Initialize EmbedManager
        embed_config = {
            'model': 'text-embedding-3-small',
            'dimension': PINECONE_DIMENSION,
            'api_key': os.getenv('OPENAI_API_KEY')
        }
        embed_manager = EmbedManager(embed_config)
        
        # Initialize PineconeManager
        pinecone_config = {
            'api_key': os.getenv('PINECONE_API_KEY'),
            'environment': os.getenv('PINECONE_ENVIRONMENT'),
            'index_name': index_name,
            'dimension': PINECONE_DIMENSION,
            'metric': 'cosine'
        }
        pinecone_manager = PineconeManager(pinecone_config)
        
        # Initialize DatabaseManager
        db_manager = DatabaseManager(db_path)
        
        # Create TreeManager config
        tree_config = {
            'index_name': index_name,
            'embedding': {
                'manager': embed_manager,
                'model': 'text-embedding-3-small',
                'dimension': PINECONE_DIMENSION
            },
            'storage': {
                'manager': pinecone_manager,
                'db_path': db_path
            },
            'clustering': {
                'max_levels': 5,
                'threshold': 0.15,
                'dimension': PINECONE_DIMENSION
            },
            'model_name': 'gpt-4o-mini',
            'cache_db_path': db_path
        }
        
        # Initialize TreeManager
        tree_manager = TreeManager(tree_config)
        logger.info("Initialized TreeManager")

        # Process documents and build tree
        logger.info("Processing documents and building tree...")
        root = tree_manager.process_documents()
        if not root:
            logger.error("Failed to process documents and build tree")
            return
            
        logger.info("Successfully built and saved tree")
        
    except Exception as e:
        logger.error(f"Failed to build tree: {str(e)}")
        raise

if __name__ == '__main__':
    build_tree()
