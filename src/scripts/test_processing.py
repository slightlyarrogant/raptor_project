"""Test script for RAPTOR processing pipeline."""
import os
from pathlib import Path
import logging
from typing import List, Dict
from dotenv import load_dotenv
from pinecone import Pinecone

from src.tree.tree_manager import TreeManager
from src.embedding.embed_manager import EmbedManager
from src.summarization.summary_manager import SummaryManager
from src.clustering.cluster_manager import ClusterManager

# Load environment variables first
load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def test_pipeline():
    """Test the complete processing pipeline with a small sample."""
    try:
        # Verify environment variables
        required_vars = ['OPENAI_API_KEY', 'PINECONE_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            logger.error("Please ensure you have a .env file with OPENAI_API_KEY and PINECONE_API_KEY")
            return False
            
        logger.warning("Environment variables loaded successfully")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index("raptor-technicalbase")
        logger.warning(f"Pinecone initialized with index: raptor-technicalbase")
        
        # Test data
        test_docs = [
            "This is a test document about machine learning.",
            "Another document about artificial intelligence.",
            "A third document about data processing."
        ]
        
        # Create test configuration
        config = {
            'embedding': {
                'model_name': "text-embedding-3-small",
                'dimension': 1536,
                'batch_size': 5,
                'api_key': os.getenv('OPENAI_API_KEY')
            },
            'storage': {
                'store': index,  # Pass the initialized index
                'api_key': os.getenv('PINECONE_API_KEY'),
                'index_name': "raptor-technicalbase",
                'namespace': 'test',
                'dimension': 1536,
                'metric': 'cosine'
            },
            'summarization': {
                'model_name': 'gpt-3.5-turbo',
                'max_tokens': 500,
                'temperature': 0.3,
                'api_key': os.getenv('OPENAI_API_KEY')
            },
            'clustering': {
                'umap': {
                    'n_neighbors': 2,
                    'n_components': 2,
                    'metric': 'cosine'
                },
                'gmm': {
                    'max_clusters': 2,
                    'threshold': 0.4
                },
                'similarity_threshold': 0.6,
                'min_cluster_size': 1,
                'max_cluster_size': 3
            }
        }
        
        logger.warning("Configuration prepared")
        
        # Initialize TreeManager (which initializes all other managers)
        tree_manager = TreeManager(config)
        logger.warning("✓ TreeManager initialized")
        
        # Process test documents
        metadata = [
            {'file_name': f'test_doc_{i}.txt', 'file_index': i, 'chunk_index': 0, 'total_chunks': 1}
            for i in range(len(test_docs))
        ]
        
        result = tree_manager.process_documents(test_docs, metadata)
        logger.warning("✓ Pipeline test complete")
        
        # Verify results
        stats = tree_manager.pinecone_manager.get_namespace_stats()
        logger.warning(f"Final stats: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting RAPTOR pipeline test...")
    success = test_pipeline()
    print(f"Test {'succeeded' if success else 'failed'}")