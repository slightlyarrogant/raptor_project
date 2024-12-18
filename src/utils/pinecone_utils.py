import logging
from typing import Optional, Dict
from pinecone import Pinecone
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Global Pinecone instance
_pinecone_instance = None
_current_index = None
_current_index_name = None

def init_pinecone(config: Dict) -> Pinecone:
    """Initialize Pinecone connection (singleton pattern)."""
    global _pinecone_instance
    
    if _pinecone_instance is not None:
        logger.info("Using existing Pinecone instance")
        return _pinecone_instance
    
    # Validate required config - be strict!
    if not config.get('api_key'):
        logger.critical("Missing Pinecone API key")
        raise ValueError("Pinecone API key must be provided")
    
    try:
        # Create Pinecone instance
        _pinecone_instance = Pinecone(
            api_key=config['api_key']
        )
        logger.info("Successfully initialized Pinecone")
        return _pinecone_instance
        
    except Exception as e:
        logger.critical(f"Critical error initializing Pinecone: {str(e)}")
        raise RuntimeError(f"Pinecone initialization failed: {str(e)}")

def get_index(index_name: str) -> Pinecone.Index:
    """Get Pinecone index (singleton pattern)."""
    global _current_index, _pinecone_instance, _current_index_name
    
    if _pinecone_instance is None:
        logger.critical("Attempt to get index without Pinecone initialization")
        raise RuntimeError("Pinecone not initialized. Call init_pinecone first.")
        
    try:
        # Check if index exists
        if index_name not in _pinecone_instance.list_indexes().names():
            logger.critical(f"Index {index_name} does not exist")
            raise ValueError(f"Index {index_name} does not exist")
            
        # Get or create index reference
        if _current_index is None or _current_index_name != index_name:
            _current_index = _pinecone_instance.Index(index_name)
            _current_index_name = index_name
            logger.info(f"Successfully connected to index: {index_name}")
            
        return _current_index
        
    except Exception as e:
        logger.critical(f"Critical error accessing index {index_name}: {str(e)}")
        raise RuntimeError(f"Failed to access Pinecone index: {str(e)}")

def get_stats(index: Pinecone.Index) -> Dict:
    """Get index statistics."""
    try:
        stats = index.describe_index_stats()
        total_vectors = sum(ns.vector_count for ns in stats.namespaces.values())
        return {
            'total_vector_count': total_vectors,
            'dimension': stats.dimension
        }
    except Exception as e:
        logger.error(f"Failed to get index stats: {str(e)}")
        return {'error': str(e)}
