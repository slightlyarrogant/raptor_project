#!/usr/bin/env python3

import logging
import sys
import os
import json
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(project_root))

from src.tree.tree_manager import TreeManager
from src.embeddings.openai_manager import OpenAIManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to process documents."""
    try:
        # Set up paths
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_dir = os.path.join(project_root, "data")
        index_name = "raptor-technicalbase"
        docs_dir = os.path.join(data_dir, index_name)
        
        logger.info(f"Processing {index_name} documents...")
        
        # Initialize components
        embed_manager = OpenAIManager()
        config = {
            'embedding': {
                'manager': embed_manager,
                'model': 'text-embedding-ada-002',
                'batch_size': 100
            },
            'clustering': {
                'method': 'kmeans',
                'min_cluster_size': 2,
                'max_clusters': 5
            },
            'tree': {
                'max_depth': 5,
                'min_docs_to_split': 3,
                'max_children': 5
            }
        }
        tree_manager = TreeManager(config)
        
        # Get documents from directory
        documents = tree_manager.get_documents_from_directory(docs_dir)
        logger.info(f"Found {len(documents)} documents to process")
        
        # Build document tree
        logger.info("Building document tree...")
        tree_data, stats = tree_manager.process_documents(documents, index_name=index_name)
        
        logger.info(f"Successfully processed {len(documents)} documents")
        logger.info(f"Stats: {json.dumps(stats, indent=2)}")
        
    except Exception as e:
        logger.error(f"Error during document processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
