import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from src.storage.pinecone_manager import PineconeManager
from src.utils.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_pinecone_status():
    """Check Pinecone index status and contents."""
    try:
        # Initialize Pinecone manager
        config = {
            'api_key': PINECONE_API_KEY,
            'index_name': PINECONE_INDEX_NAME
        }
        pinecone_manager = PineconeManager(config)
        
        # Get index stats
        stats = pinecone_manager.index.describe_index_stats()
        
        logger.info("\n=== Pinecone Index Status ===")
        logger.info(f"Index Name: {PINECONE_INDEX_NAME}")
        logger.info(f"Total Vector Count: {stats.total_vector_count}")
        logger.info("\nNamespace Statistics:")
        
        for namespace, ns_stats in stats.namespaces.items():
            logger.info(f"\nNamespace: {namespace}")
            logger.info(f"Vector Count: {ns_stats.vector_count}")
            
        # Query a few random vectors to verify search
        if stats.total_vector_count > 0:
            logger.info("\nPerforming test query...")
            namespace = next(iter(stats.namespaces))
            result = pinecone_manager.query(
                vector=[0.0] * 1536,  # Zero vector for test
                top_k=1,
                namespace=namespace
            )
            logger.info(f"Query successful, found {len(result['matches'])} matches")
            
    except Exception as e:
        logger.error(f"Error checking Pinecone status: {str(e)}")
        raise

if __name__ == "__main__":
    check_pinecone_status()
