import schedule
import time
from pathlib import Path
from src.scripts.process_data import process_data_directory
from src.utils.config import PINECONE_INDICES
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='processing.log'
)
logger = logging.getLogger(__name__)

def process_index(index_name: str):
    """Process documents for a specific index."""
    logger.info(f"Starting scheduled processing for index: {index_name}")
    
    index_config = PINECONE_INDICES[index_name]
    data_dir = Path(index_config['data_dir'])
    
    try:
        # Process the documents
        process_data_directory(
            data_dir=str(data_dir),
            index_name=index_config['index_name'],
            namespace=index_config['namespace']
        )
        logger.info(f"Completed processing for index: {index_name}")
        
    except Exception as e:
        logger.error(f"Error processing index {index_name}: {str(e)}")

def setup_schedules():
    """Set up processing schedules for all indices."""
    for index_name, config in PINECONE_INDICES.items():
        schedule.every().day.at(config['schedule']).do(
            process_index, index_name=index_name
        )
        logger.info(f"Scheduled processing for {index_name}: {config['schedule']}")

if __name__ == "__main__":
    print("Starting RAPTOR Scheduled Processor")
    print("\nConfigured Indices:")
    for index_name, config in PINECONE_INDICES.items():
        print(f"- {index_name}: {config['schedule']}")
    
    setup_schedules()
    
    while True:
        schedule.run_pending()
        time.sleep(60) 