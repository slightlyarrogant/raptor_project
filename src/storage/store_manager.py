from typing import Dict, List, Optional, Any
import time
import os
import json
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from src.models.data_models import TreeData
from src.tree.node import LeafNode, SummaryNode
from src.utils.file_manager import FileManager
from src.utils.error_handler import error_handler, RetryableError

logger = logging.getLogger(__name__)

class StoreManager:
    """Manages interactions with the vector store."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize store manager with configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize index name
        self.index_name = config.get('index_name', 'default')
        
        # Initialize file manager
        self.file_manager = FileManager(self.index_name)
        
        # Initialize batch tracking
        self.batch_counters = self._load_or_create_index('batch_counters') or {
            'embeddings': 0,
            'vectors': 0,
            'metadata': 0
        }
        
        # Initialize caches
        self.embedding_batch = []
        self.embedding_index = {}
        self.vector_ids = set()
        
        # Load vector index
        self.vector_index = self._load_or_create_index('vectors')
        
        # Initialize Pinecone if configured
        pinecone_config = config.get('pinecone', {})
        if pinecone_config:
            from src.storage.pinecone_manager import PineconeManager
            self.pinecone_manager = PineconeManager(pinecone_config)
            self.vector_dimension = pinecone_config.get('dimension', 1536)
        else:
            self.pinecone_manager = None
            self.vector_dimension = None
        
        # Set batch size for vector operations
        self.batch_size = config.get('batch_size', 100)
        
        # Initialize node batch and tracking
        self.node_batch = []
        self.last_save_time = time.time()
        self.save_interval = 300
        self.current_batch_num = self._get_next_batch_num()
        self._nodes = {}
        
        self.logger.info(f"StoreManager initialized for index '{self.index_name}'")
        
    def _load_or_create_index(self, index_name: str) -> Dict:
        """Load or create an index file."""
        index_file = self.file_manager.processed_dir / f"{index_name}_index.json"
        
        try:
            data = self.file_manager.safe_read(index_file)
            if data is None:
                return {}
            return data
        except Exception as e:
            self.logger.error(f"Error loading index {index_name}: {str(e)}")
            return {}
        
    def _save_index(self, index_name: str, index_data: Dict):
        """Save index to disk."""
        index_file = self.file_manager.processed_dir / f"{index_name}_index.json"
        self.file_manager.atomic_write(index_data, index_file)
            
    def _get_batch_path(self, data_type: str, batch_num: int) -> Path:
        """Get path for a batch file."""
        return self.file_manager.processed_dir / data_type / f"batch_{batch_num}.json"

    @error_handler
    def save_chunk(self, doc_id: str, chunk_index: int, text: str, metadata: Dict) -> bool:
        """Save a document chunk to storage."""
        chunk_id = f"{doc_id}_{chunk_index}"
        chunk_data = {
            'id': chunk_id,
            'doc_id': doc_id,
            'chunk_index': chunk_index,
            'text': text,
            'metadata': metadata,
            'created_at': time.time()
        }
        
        # Add to batch
        self.chunk_batch.append(chunk_data)
        
        # Write batch if full
        if len(self.chunk_batch) >= self.batch_size:
            self._write_chunk_batch()
            
        return True

    @error_handler
    def save_embedding(self, text: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> bool:
        """Save an embedding to storage."""
        # Generate a unique ID for the text
        text_id = str(hash(text))
        
        # Add to batch
        self.embedding_batch.append({
            'id': text_id,
            'text': text,
            'embedding': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            'metadata': metadata
        })
        
        # Write batch if full
        if len(self.embedding_batch) >= self.batch_size:
            self._write_embedding_batch()
            
        return True

    def _write_chunk_batch(self):
        """Write accumulated chunks to disk."""
        if not self.chunk_batch:
            return
            
        try:
            batch_num = self.batch_counters['chunks']
            batch_file = self._get_batch_path('chunks', batch_num)
            
            # Write chunks atomically
            self.file_manager.atomic_write(self.chunk_batch, batch_file)
            
            # Update counter and clear batch
            self.batch_counters['chunks'] += 1
            self.chunk_batch = []
            
            # Save updated counters
            self._save_index('batch_counters', self.batch_counters)
            
        except Exception as e:
            self.logger.error(f"Error writing chunk batch: {str(e)}")
            raise RetryableError("Failed to write chunk batch", 'CHUNK_WRITE_ERROR', {'batch_num': batch_num})

    def _write_embedding_batch(self):
        """Write accumulated embeddings to disk."""
        if not self.embedding_batch:
            return
            
        try:
            batch_num = self.batch_counters['embeddings']
            batch_file = self._get_batch_path('embeddings', batch_num)
            
            # Write embeddings atomically
            self.file_manager.atomic_write(self.embedding_batch, batch_file)
            
            # Update counter and clear batch
            self.batch_counters['embeddings'] += 1
            self.embedding_batch = []
            
            # Save updated counters
            self._save_index('batch_counters', self.batch_counters)
            
        except Exception as e:
            self.logger.error(f"Error writing embedding batch: {str(e)}")
            raise RetryableError("Failed to write embedding batch", 'EMBEDDING_WRITE_ERROR', {'batch_num': batch_num})

    def get_embedding(self, text_id: str) -> Optional[Dict]:
        """Get an embedding by its ID."""
        try:
            # Check if the embedding is in the current batch
            for item in self.embedding_batch:
                if item['id'] == text_id:
                    return item
            
            # Look up the batch number from the index
            index_entry = self.embedding_index.get(text_id)
            if not index_entry:
                return None
            
            # Read the batch file
            batch_file = self._get_batch_path('embeddings', index_entry['batch_num'])
            batch_data = self.file_manager.safe_read(batch_file)
            
            if not batch_data:
                return None
            
            # Find the embedding in the batch
            for item in batch_data:
                if item['id'] == text_id:
                    return item
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving embedding {text_id}: {str(e)}")
            return None

    def get_chunk(self, chunk_id: str) -> Optional[Dict]:
        """Get a chunk by its ID."""
        try:
            # Check if the chunk is in the current batch
            for item in self.chunk_batch:
                if item['id'] == chunk_id:
                    return item
            
            # Look up the batch number from the index
            index_entry = self.chunk_index.get(chunk_id)
            if not index_entry:
                return None
            
            # Read the batch file
            batch_file = self._get_batch_path('chunks', index_entry['batch_num'])
            batch_data = self.file_manager.safe_read(batch_file)
            
            if not batch_data:
                return None
            
            # Find the chunk in the batch
            for item in batch_data:
                if item['id'] == chunk_id:
                    return item
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving chunk {chunk_id}: {str(e)}")
            return None