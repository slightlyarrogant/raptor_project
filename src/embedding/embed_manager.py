"""Manager for handling text embeddings."""
import logging
from typing import List, Dict, Optional, Any
import numpy as np
import time
from src.utils.openai_client import UnifiedAIClient
from src.utils.config import DEFAULT_CONFIG

class EmbedManager:
    """Manager for document embeddings."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize embedding manager.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
            
        self.unifiedai_client = UnifiedAIClient()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing EmbedManager...")
        
        # Configuration
        embedding_config = self.config.get('embedding', {})
        self.model_name = embedding_config.get('model', 'text-embedding-3-small')
        self.dimension = self.config['embedding']['dimensions']
        self.batch_size = embedding_config.get('batch_size', 8)
        self.max_tokens = self.config.get('max_tokens', 8000)
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to get embeddings for
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if not texts:
            return np.array([])
            
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                batch_embeddings = self.unifiedai_client.get_embeddings(
                    texts=batch,
                    model=self.model_name
                )
                embeddings.extend(batch_embeddings)
            except Exception as e:
                self.logger.error(f"Error getting embeddings for batch: {e}")
                # Return zero embeddings for failed batch
                batch_embeddings = np.zeros((len(batch), self.config['embedding']['dimensions']))
                embeddings.extend(batch_embeddings)
                
        return np.array(embeddings)

    def embed_texts(self, texts: List[str], metadata: Optional[List[Dict]] = None) -> List[List[float]]:
        """Generate embeddings for a list of texts with metadata tracking."""
        try:
            if not texts:
                self.logger.warning("No texts provided for embedding")
                return []
                
            # Log embedding request
            self.logger.info(f"Embedding Request: {len(texts)} texts")
            start_time = time.time()
            
            # Process in batches
            batch_size = self.batch_size
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size] if metadata else None
                
                try:
                    # Get embeddings for batch using get_embeddings method
                    batch_embeddings = self.get_embeddings(batch_texts)
                    
                    # Log batch completion
                    batch_time = time.time() - start_time
                    self.logger.debug(f"Batch {i//batch_size + 1} completed: {len(batch_embeddings)} embeddings")
                    
                    # Track metadata if provided
                    if batch_metadata:
                        for j, (emb, meta) in enumerate(zip(batch_embeddings, batch_metadata)):
                            meta.update({
                                'embedding_model': self.model_name,
                                'embedding_time': batch_time,
                                'embedding_batch': i//batch_size + 1,
                                'embedding_position': i + j
                            })
                    
                    all_embeddings.extend(batch_embeddings)
                    
                except Exception as e:
                    self.logger.error(f"Error embedding batch {i//batch_size + 1}: {str(e)}")
                    # Continue with next batch instead of failing completely
                    continue
            
            # Validate results
            if len(all_embeddings) != len(texts):
                self.logger.error(f"Embedding count mismatch: got {len(all_embeddings)}, expected {len(texts)}")
                raise ValueError("Embedding generation failed: count mismatch")
            
            duration = time.time() - start_time
            self.logger.info(f"Embedding Request: {len(texts)} texts completed in {duration:.2f}s")
            
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {str(e)}")
            raise

    def _split_batch(self, texts: List[str]) -> List[List[str]]:
        """Split large batch into smaller ones."""
        sub_batches = []
        current_batch = []
        current_tokens = 0
        
        for text in texts:
            text_tokens = len(text)
            if current_tokens + text_tokens > self.max_tokens:
                if current_batch:
                    sub_batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens
        
        if current_batch:
            sub_batches.append(current_batch)
            
        return sub_batches

    def _test_initialization(self):
        """Test embedding functionality with a simple example."""
        try:
            test_text = "Test initialization of embedding system."
            test_embedding = self.get_embeddings([test_text])
            
            # Check if we got a valid numpy array with the right shape
            if not isinstance(test_embedding, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(test_embedding)}")
                
            if test_embedding.shape != (1, self.dimension):
                raise ValueError(f"Embedding dimension mismatch: got shape {test_embedding.shape}, expected (1, {self.dimension})")
                
            self.logger.info("✓ Embedding test successful")
            
        except Exception as e:
            self.logger.error(f"Embedding test failed: {str(e)}")
            raise ValueError("Failed to initialize embedding system") from e
