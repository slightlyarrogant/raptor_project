import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

def check_environment():
    """Check if all required environment variables are set."""
    load_dotenv()
    
    required_vars = {
        'OPENAI_API_KEY': 'OpenAI API key',
        'PINECONE_API_KEY': 'Pinecone API key',
        'PINECONE_ENVIRONMENT': 'Pinecone environment',
        'PINECONE_INDEX_NAME': 'Pinecone index name'
    }
    
    missing = []
    for var, name in required_vars.items():
        if not os.getenv(var):
            missing.append(name)
    
    if missing:
        logger.error("Missing required environment variables:")
        for item in missing:
            logger.error(f"- {item}")
        raise ValueError("Missing required environment variables")
    
    logger.info("Environment validation successful") 