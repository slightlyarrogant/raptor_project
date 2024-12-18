#!/usr/bin/env python3

import logging
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(str(project_root))

from src.tree.tree_manager import TreeManager
from src.embeddings.openai_manager import OpenAIManager
from src.storage.db_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def migrate_embeddings():
    """Migrate embeddings from cache DB to Pinecone."""
    try:
        # Initialize components
        db_manager = DatabaseManager()
        embed_manager = OpenAIManager()
        
        # Set up configuration
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
        
        # Get documents path
        data_dir = os.path.join(project_root, "data")
        index_name = "raptor-technicalbase"
        docs_dir = os.path.join(data_dir, index_name)
        
        logger.info(f"Starting migration for {index_name}")
        
        # Get existing embeddings from cache
        cached_embeddings = db_manager.get_all_embeddings(index_name)
        logger.info(f"Found {len(cached_embeddings)} cached embeddings")
        
        # Process documents to ensure chunks
        documents = []
        for file_path in Path(docs_dir).rglob("*.*"):
            if file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        doc = {
                            'id': file_path.stem,
                            'name': file_path.name,
                            'path': str(file_path),
                            'text': text
                        }
                        documents.append(doc)
                except Exception as e:
                    logger.error(f"Failed to read {file_path}: {str(e)}")
                    continue
                    
        logger.info(f"Processing {len(documents)} documents")
        
        tree_data, stats = tree_manager.process_documents(documents, index_name=index_name)
        logger.info(f"Document processing stats: {stats}")
        
        # Upsert to Pinecone
        logger.info("Upserting embeddings to Pinecone...")
        tree_manager.upsert_embeddings(cached_embeddings, index_name)
        
        logger.info("Migration completed successfully")
        
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    migrate_embeddings()
