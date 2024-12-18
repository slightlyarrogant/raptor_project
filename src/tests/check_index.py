import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.storage.pinecone_manager import PineconeManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_index():
    """Check index statistics."""
    try:
        # Initialize source index (CFI)
        source_config = {
            'api_key': os.getenv('PINECONE_API_KEY'),
            'index_name': 'raptor-cfi'
        }
        source_index = PineconeManager(source_config)
        
        # Get stats
        stats = source_index.index.describe_index_stats()
        total_vectors = sum(ns.vector_count for ns in stats.namespaces.values())
        
        print(f"\nIndex Statistics:")
        print(f"Total vectors: {total_vectors}")
        print(f"Dimension: {stats.dimension}")
        print(f"Namespaces: {list(stats.namespaces.keys())}")
        
        # Try a test query
        logger.info("\nTesting query...")
        query_text = "Co to jest Vendo?"
        from src.prepare.embeddings import embed_texts
        query_embedding = embed_texts([query_text])[0]
        results = source_index.query(
            vector=query_embedding,
            filter=None,
            top_k=4
        )
        
        print(f"\nQuery Results:")
        if not results.matches:
            print("No matches found!")
        else:
            for i, match in enumerate(results.matches):
                print(f"\nMatch {i+1}:")
                print(f"Score: {match.score}")
                print(f"ID: {match.id}")
                for key, value in match.metadata.items():
                    if key == 'text':
                        print(f"Text: {value[:200]}...")
                    else:
                        print(f"{key}: {value}")
                        
    except Exception as e:
        logger.error(f"Error checking index: {str(e)}")
        raise

if __name__ == "__main__":
    check_index()
