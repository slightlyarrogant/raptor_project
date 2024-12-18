#!/usr/bin/env python3

import logging
import sys
from pathlib import Path
import os

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.storage.db_manager import DatabaseManager
from src.storage.store_manager import StoreManager
from src.storage.pinecone_manager import PineconeManager
from src.utils.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_DIMENSION,
    PINECONE_METRIC,
    PINECONE_CLOUD,
    PINECONE_REGION
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Initialize Pinecone manager
        pinecone_config = {
            'api_key': PINECONE_API_KEY,
            'index_name': PINECONE_INDEX_NAME,
            'dimension': PINECONE_DIMENSION,
            'metric': PINECONE_METRIC,
            'cloud': PINECONE_CLOUD,
            'region': PINECONE_REGION
        }
        pinecone_manager = PineconeManager(pinecone_config)
        
        # Initialize store manager
        store_config = {
            'store': pinecone_manager,
            'namespace': PINECONE_NAMESPACE,
            'batch_size': 100
        }
        store_manager = StoreManager(store_config)
        
        # Start transfer
        logger.info("Starting vector transfer from SQLite to Pinecone")
        store_manager.transfer_vectors_to_pinecone(db_manager)
        logger.info("Transfer completed successfully")
        
    except Exception as e:
        logger.error(f"Error during transfer: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
