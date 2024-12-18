import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.storage.pinecone_manager import PineconeManager
from src.prepare.embeddings import embed_texts
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_vector_flow():
    """Test the complete vector flow from source to target index."""
    try:
        # Initialize source index (CFI)
        source_config = {
            'api_key': os.getenv('PINECONE_API_KEY'),
            'index_name': 'raptor-cfi'
        }
        source_index = PineconeManager(source_config)
        
        # Initialize test index
        test_config = {
            'api_key': os.getenv('PINECONE_API_KEY'),
            'index_name': 'test'
        }
        test_index = PineconeManager(test_config)
        
        # Query text
        query_text = "Co to jest Vendo?"
        
        # 1. Query source index
        logger.info(f"Querying source index with: {query_text}")
        query_embedding = embed_texts([query_text])[0]
        source_results = source_index.query(
            vector=query_embedding,
            filter=None,
            top_k=4
        )
        
        if not source_results.matches:
            logger.error("No results found in source index")
            return
            
        # 2. Extract chunks and metadata
        chunks = []
        metadata_list = []
        for match in source_results.matches:
            chunks.append(match.metadata.get('text', ''))
            metadata_list.append({
                'text': match.metadata.get('text', ''),
                'score': match.score,
                'source': 'cfi_index'
            })
        
        logger.info(f"Retrieved {len(chunks)} chunks from source index")
        
        # 3. Generate embeddings for chunks
        logger.info("Generating embeddings for chunks")
        chunk_embeddings = embed_texts(chunks)
        
        # 4. Store in test index
        vectors_to_upsert = []
        for i, (embedding, metadata) in enumerate(zip(chunk_embeddings, metadata_list)):
            vectors_to_upsert.append({
                'id': f'test_chunk_{i}',
                'values': embedding,
                'metadata': metadata
            })
        
        logger.info("Upserting vectors to test index")
        test_index.upsert_batch(vectors_to_upsert)
        
        # 5. Query test index with same question
        logger.info("Querying test index")
        test_results = test_index.query(
            vector=query_embedding,
            filter=None,
            top_k=4
        )
        
        # 6. Compare results
        logger.info("\nTest Index Results:")
        if not test_results.matches:
            logger.info("No matches found in test index")
        else:
            for i, match in enumerate(test_results.matches):
                logger.info(f"\nResult {i+1}:")
                logger.info(f"Score: {match.score}")
                logger.info(f"ID: {match.id}")
                for key, value in match.metadata.items():
                    if key == 'text':
                        logger.info(f"Text: {value[:200]}...")
                    else:
                        logger.info(f"{key}: {value}")
            
        return test_results
        
    except Exception as e:
        logger.error(f"Error in vector flow test: {str(e)}")
        raise

if __name__ == "__main__":
    test_vector_flow()
