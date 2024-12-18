import logging
import time
from typing import Dict

import tiktoken

class RaptorTree:
    """Core tree operations and data processing."""
    
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self._validate_config(config)
        
        # Store configuration
        self.config = config
        self.index_name = config['index_name']
        
        # Store managers
        self.pinecone_manager = config['pinecone_manager']  # Use existing manager
        self.embed_manager = config['embed_manager']
        self.cluster_manager = config['cluster_manager']
        self.summary_manager = config['summary_manager']
        
        # Initialize other parameters
        self.batch_size = config.get('batch_size', 100)
        self.namespace = f"{self.index_name}_namespace"
        
        # Initialize tracking
        self.usage_stats = self._init_usage_stats()
        self.token_counter = tiktoken.get_encoding("cl100k_base")
        
        self.logger.info(f"RaptorTree initialized for index: {self.index_name}")

    def _validate_config(self, config: Dict):
        """Validate required configuration."""
        required = ['pinecone_manager', 'embed_manager', 'cluster_manager', 'summary_manager', 'index_name']
        missing = [item for item in required if item not in config]
        if missing:
            raise ValueError(f"Missing required config items: {', '.join(missing)}")

    def store_tree(self, tree_data: Dict) -> str:
        """Store tree in vector database."""
        try:
            tree_id = f"tree_{int(time.time())}"
            vectors = self._prepare_vectors(tree_data['tree'], tree_id)
            
            # Store in batches using existing PineconeManager
            for batch in self._batch_vectors(vectors):
                self.pinecone_manager.upsert_batch(
                    vectors=batch,
                    namespace=f"{self.index_name}_tree"
                )
            
            return tree_id
            
        except Exception as e:
            self.logger.error(f"Failed to store tree: {str(e)}")
            raise 