import os
from dotenv import load_dotenv
from pinecone import Pinecone
import logging
import time

logger = logging.getLogger(__name__)

def setup_test_index():
    """Create and prepare test index in Pinecone."""
    load_dotenv()
    
    try:
        # Initialize Pinecone with your exact configuration
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        
        # Check if index exists
        index_name = os.getenv('PINECONE_INDEX_NAME')  # raptor-cfi
        
        # Delete existing index if it exists
        if index_name in pc.list_indexes().names():
            logger.info(f"Deleting existing index: {index_name}")
            pc.delete_index(index_name)
            logger.info("Waiting for deletion to complete...")
            time.sleep(20)  # Wait for deletion to complete
        
        logger.info(f"Creating new index: {index_name}")
        # Create fresh index with your exact configuration
        pc.create_index(
            name=index_name,
            dimension=int(os.getenv('PINECONE_DIMENSION', '1536')),
            metric=os.getenv('PINECONE_METRIC', 'cosine'),
            spec={
                "serverless": {
                    "cloud": os.getenv('PINECONE_CLOUD'),      # aws
                    "region": os.getenv('PINECONE_REGION')     # us-east-1
                }
            }
        )
        logger.info("Waiting for index to be ready...")
        time.sleep(20)  # Wait for index to be ready
        
        logger.info(f"✓ Index {index_name} is ready")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup test environment: {str(e)}")
        return False

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    setup_test_index()