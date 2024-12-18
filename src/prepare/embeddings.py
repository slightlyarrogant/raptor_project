from src.utils.openai_client import UnifiedAIClient
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    try:
        client = UnifiedAIClient()
        embeddings = client.get_embeddings(texts, model=model)
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        # Return zero vectors as fallback
        return [np.zeros(1536).tolist() for _ in texts]
