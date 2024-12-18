import os
import logging
from typing import List, Dict
from src.utils.pinecone_utils import init_pinecone, get_index
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_pinecone_config():
    """Get Pinecone configuration from environment variables."""
    return {
        'api_key': os.getenv('PINECONE_API_KEY'),
        'environment': os.getenv('PINECONE_ENVIRONMENT', 'gcp-starter')
    }

def fetch_all_vectors(index, namespace: str, batch_size: int = 100) -> List[Dict]:
    """Fetch all vectors from a namespace using fetch operation."""
    logger.info(f"Fetching vectors from namespace: {namespace}")
    
    # Get total vector count in namespace
    stats = index.describe_index_stats()
    total_vectors = stats.namespaces[namespace].vector_count if namespace in stats.namespaces else 0
    logger.info(f"Total vectors in namespace {namespace}: {total_vectors}")
    
    if total_vectors == 0:
        return []
    
    # Initialize empty list for all vectors
    all_vectors = []
    
    # Use fetch operation to get all vectors
    with tqdm(total=total_vectors, desc=f"Fetching from {namespace}") as pbar:
        try:
            # Fetch all vectors in batches
            for batch in index.fetch(
                ids=[],  # Empty list means fetch all
                namespace=namespace,
                batch_size=batch_size
            ):
                vectors = []
                for id, vector_data in batch.vectors.items():
                    vectors.append({
                        'id': id,
                        'values': vector_data.values,
                        'metadata': vector_data.metadata
                    })
                
                all_vectors.extend(vectors)
                pbar.update(len(vectors))
                
                # Small delay to avoid rate limits
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error fetching vectors: {str(e)}")
    
    logger.info(f"Successfully fetched {len(all_vectors)} vectors from {namespace}")
    return all_vectors

def migrate_vectors(index_name: str = 'raptor-cfi'):
    """Migrate vectors from namespaced to non-namespaced index."""
    try:
        # Initialize Pinecone
        pinecone_config = get_pinecone_config()
        logger.info(f"Initializing Pinecone with config: {pinecone_config}")
        pinecone = init_pinecone(pinecone_config)
        
        # Get index
        index = get_index(index_name)
        
        # Source namespaces
        source_namespaces = [
            f"{index_name}_namespace_chunks",
            f"{index_name}_namespace_summaries"
        ]
        
        # Fetch vectors from each namespace
        all_vectors = []
        for namespace in source_namespaces:
            vectors = fetch_all_vectors(index, namespace)
            all_vectors.extend(vectors)
            
        total_vectors = len(all_vectors)
        logger.info(f"Total vectors to migrate: {total_vectors}")
        
        if total_vectors == 0:
            logger.warning("No vectors found to migrate!")
            return
            
        # Upsert vectors to main namespace (batch size of 100)
        batch_size = 100
        with tqdm(total=total_vectors, desc="Upserting vectors") as pbar:
            for i in range(0, total_vectors, batch_size):
                batch = all_vectors[i:i + batch_size]
                try:
                    # Upsert batch to main namespace (no namespace specified)
                    index.upsert(vectors=batch)
                    pbar.update(len(batch))
                    time.sleep(0.1)  # Small delay to avoid rate limits
                except Exception as e:
                    logger.error(f"Error upserting batch: {str(e)}")
                    continue
        
        # Verify migration
        stats = index.describe_index_stats()
        logger.info("\nFinal index statistics:")
        logger.info("-" * 40)
        logger.info(f"Default namespace: {stats.namespaces.get('', {'vector_count': 0}).vector_count} vectors")
        for ns, ns_stats in stats.namespaces.items():
            if ns:  # Only show non-default namespaces
                logger.info(f"{ns}: {ns_stats.vector_count} vectors")
        
        # Ask for confirmation before deleting old namespaces
        logger.info("\nMigration complete! The old namespaces can now be deleted.")
        logger.info("To delete the old namespaces, run this script with --delete-old-namespaces")
        
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")

def delete_old_namespaces(index_name: str = 'raptor-cfi'):
    """Delete the old namespaces after confirmation."""
    try:
        # Initialize Pinecone
        pinecone_config = get_pinecone_config()
        logger.info(f"Initializing Pinecone with config: {pinecone_config}")
        pinecone = init_pinecone(pinecone_config)
        
        # Get index
        index = get_index(index_name)
        
        # Source namespaces
        source_namespaces = [
            f"{index_name}_namespace_chunks",
            f"{index_name}_namespace_summaries"
        ]
        
        # Delete each namespace
        for namespace in source_namespaces:
            try:
                logger.info(f"Deleting namespace: {namespace}")
                index.delete(delete_all=True, namespace=namespace)
                logger.info(f"✓ Successfully deleted namespace: {namespace}")
            except Exception as e:
                logger.error(f"Error deleting namespace {namespace}: {str(e)}")
        
        # Verify deletion
        stats = index.describe_index_stats()
        logger.info("\nFinal index statistics:")
        logger.info("-" * 40)
        logger.info(f"Default namespace: {stats.namespaces.get('', {'vector_count': 0}).vector_count} vectors")
        for ns, ns_stats in stats.namespaces.items():
            if ns:  # Only show non-default namespaces
                logger.info(f"{ns}: {ns_stats.vector_count} vectors")
                
    except Exception as e:
        logger.error(f"Error during namespace deletion: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrate vectors from namespaced to non-namespaced index")
    parser.add_argument("--index-name", default="raptor-cfi", help="Name of the index to migrate")
    parser.add_argument("--delete-old-namespaces", action="store_true", help="Delete old namespaces after migration")
    args = parser.parse_args()
    
    if args.delete_old_namespaces:
        delete_old_namespaces(args.index_name)
    else:
        migrate_vectors(args.index_name)
