import logging
import re
import time
from typing import List, Dict
import numpy as np
from pinecone import Pinecone
from unidecode import unidecode
import uuid
from datetime import datetime
from src.utils.pinecone_utils import init_pinecone, get_index, get_stats
from src.utils.config import (
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_DIMENSION,
    PINECONE_METRIC,
    PINECONE_REGION
)

logger = logging.getLogger(__name__)

# WARNING: DO NOT ADD NAMESPACE FUNCTIONALITY TO THIS CLASS
# The RAPTOR system is designed to work without Pinecone namespaces.
# Adding namespace support will cause failures in the code.
# All vectors should be stored in the default namespace.
class PineconeManager:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.index_name = config['index_name']
        
        # Initialize Pinecone once with simplified config
        init_pinecone({
            'api_key': config['api_key']
        })
        
        # Get index reference
        self.index = get_index(self.index_name)
        self.logger.info(f"Connected to index: {self.index_name}")

    def _chunk_vectors(self, vectors: List[Dict], max_chunk_size: int = 100) -> List[List[Dict]]:
        """Split vectors into smaller chunks to avoid size limits."""
        return [vectors[i:i + max_chunk_size] for i in range(0, len(vectors), max_chunk_size)]

    def check_duplicates(self, vectors: List[Dict]) -> List[str]:
        """Check for existing vectors by ID."""
        try:
            vector_ids = [vec['id'] for vec in vectors]
            # Query in batches to avoid request size limits
            batch_size = 100
            existing_ids = set()
            
            for i in range(0, len(vector_ids), batch_size):
                batch = vector_ids[i:i + batch_size]
                # Use fetch to check existence
                response = self.index.fetch(ids=batch)
                existing_ids.update(response.vectors.keys())
            
            return list(existing_ids)
        except Exception as e:
            self.logger.error(f"Failed to check duplicates: {str(e)}")
            return []

    def vector_exists(self, vector_id: str) -> bool:
        """
        Check if a vector with the given ID exists in the Pinecone index.
        
        Args:
            vector_id (str): The ID of the vector to check
            
        Returns:
            bool: True if the vector exists, False otherwise
        """
        try:
            response = self.index.fetch(ids=[str(vector_id)])
            return bool(response.vectors)
        except Exception as e:
            self.logger.error(f"Failed to check vector existence for ID {vector_id}: {str(e)}")
            return False

    def upsert_batch(self, vectors: List[Dict]) -> None:
        """Upsert a batch of vectors to Pinecone.
        
        Args:
            vectors: List of vector dictionaries with 'id', 'values', and optional 'metadata'
        """
        try:
            # Upsert vectors without namespace
            self.index.upsert(vectors=vectors)
        except Exception as e:
            self.logger.error(f"Failed to upsert batch: {str(e)}")
            raise

    def store_chunks(self, texts: List[str], embeddings: List[List[float]], metadata: List[Dict]):
        """Store document chunks."""
        try:
            vectors = []
            for i, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata)):
                vector_id = f"chunk_{int(time.time())}_{i}"
                meta['text'] = text
                vectors.append({
                    'id': vector_id,
                    'values': embedding,
                    'metadata': meta
                })
            
            self.upsert_batch(vectors)
            
        except Exception as e:
            self.logger.error(f"Failed to store chunks: {str(e)}")
            raise

    def query(self, vector: List[float], filter: Dict = None, top_k: int = 10):
        """Query vectors."""
        try:
            return self.index.query(
                vector=vector,
                filter=filter,
                top_k=top_k,
                include_metadata=True
            )
        except Exception as e:
            self.logger.error(f"Failed to query vectors: {str(e)}")
            raise