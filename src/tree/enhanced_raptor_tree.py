"""DEPRECATED: This module has been superseded by tree_manager.py

The tree building functionality has been moved to the TreeManager class in tree_manager.py.
This file is kept only for historical reference and will be removed in a future version.

For new code, please use:
from src.tree.tree_manager import TreeManager
"""

import time
from typing import Dict, List, Tuple, Optional, Iterator
import logging
import numpy as np
from pathlib import Path
import gc
import json
import psutil
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from src.tree.node import Node
from src.utils.chunking import SmartChunker
from src.embedding.embed_manager import EmbedManager
from src.summarization.summary_manager import SummaryManager
from src.storage.pinecone_manager import PineconeManager
from src.clustering.cluster_manager import ClusterManager
from src.utils.cache_manager import CacheManager
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class EnhancedRaptorTree:
    """DEPRECATED: This class has been superseded by TreeManager in tree_manager.py.
    
    Please use TreeManager for all tree-related operations. This class will be removed
    in a future version.
    """
    
    def __init__(self, config: Dict):
        """Initialize RaptorTree with configuration."""
        import warnings
        warnings.warn(
            "EnhancedRaptorTree is deprecated and will be removed in a future version. "
            "Use TreeManager from tree_manager.py instead.",
            DeprecationWarning
        )
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.embed_manager = config.get('embedding', {}).get('manager')
        self.pinecone_manager = config.get('storage', {}).get('manager')
        self.summary_manager = SummaryManager(config)
        self.cluster_manager = ClusterManager(config)
        self.cache_manager = CacheManager(config.get('index_name', 'default'))
        
        # Initialize chunker
        self.chunker = SmartChunker(
            target_chunk_size=config.get('chunk_size', 512),
            overlap_size=config.get('overlap_size', 50),
            max_extension=config.get('max_extension', 100)
        )
        
        # Tree-specific configurations
        self.max_children = config.get('max_children', 10)
        self.min_children = config.get('min_children', 2)
        self.balance_threshold = config.get('balance_threshold', 0.5)
        self.max_depth = config.get('max_depth', 5)
        
        # Initialize timing and memory tracking
        self.start_time = time.time()
        self.processing_times = {
            'chunking': 0,
            'embedding': 0,
            'clustering': 0,
            'summarization': 0,
            'total': 0
        }
        
        self.logger.info("RaptorTree initialized with components and configs:")
        self.logger.info(f"- Chunker: SmartChunker (target size: {config.get('chunk_size', 512)})")
        self.logger.info(f"- Max children per node: {self.max_children}")
        self.logger.info(f"- Min children per node: {self.min_children}")
        self.logger.info(f"- Balance threshold: {self.balance_threshold}")
        self.logger.info(f"- Max depth: {self.max_depth}")
        self.logger.info(f"- Memory usage at start: {self._get_memory_usage():.2f} MB")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def _log_processing_times(self):
        """Log detailed processing times and memory usage."""
        self.logger.info("\nProcessing Times:")
        for stage, duration in self.processing_times.items():
            self.logger.info(f"- {stage.capitalize()}: {duration:.2f} seconds")
        self.logger.info(f"Memory usage: {self._get_memory_usage():.2f} MB")

    def process_documents(self, documents: List[Dict]) -> Node:
        """Process documents and build a robust tree structure."""
        try:
            self.logger.info(f"Starting document processing for {len(documents)} documents")
            start_time = time.time()
            
            # Step 1: Chunk documents
            chunk_start = time.time()
            self.logger.info("Starting document chunking...")
            chunks = self._chunk_documents(documents)
            self.processing_times['chunking'] = time.time() - chunk_start
            self.logger.info(f"Chunking completed: {len(chunks)} chunks created")
            
            # Step 2: Generate embeddings
            embed_start = time.time()
            self.logger.info("Generating embeddings...")
            embeddings = self.embed_manager.embed_texts([chunk['text'] for chunk in chunks])
            self.processing_times['embedding'] = time.time() - embed_start
            self.logger.info(f"Embeddings generated: {len(embeddings)} vectors")
            
            # Step 3: Build initial tree
            tree_start = time.time()
            self.logger.info("Building initial tree structure...")
            root = self._build_initial_tree(chunks, embeddings)
            self.processing_times['clustering'] = time.time() - tree_start
            
            # Step 4: Generate summaries
            summary_start = time.time()
            self.logger.info("Generating node summaries...")
            self._generate_summaries(root)
            self.processing_times['summarization'] = time.time() - summary_start
            
            # Update total processing time
            self.processing_times['total'] = time.time() - start_time
            
            # Log processing times and stats
            self._log_processing_times()
            self._log_tree_stats(root)
            
            return root
            
        except Exception as e:
            self.logger.error(f"Error during document processing: {str(e)}", exc_info=True)
            raise RaptorTreeError(f"Document processing failed: {str(e)}")

    def _log_tree_stats(self, root: Node):
        """Log comprehensive tree statistics."""
        stats = self._calculate_tree_stats(root)
        self.logger.info("\nTree Statistics:")
        self.logger.info(f"- Total nodes: {stats['total_nodes']}")
        self.logger.info(f"- Max depth: {stats['max_depth']}")
        self.logger.info(f"- Avg children per node: {stats['avg_children']:.2f}")
        self.logger.info(f"- Leaf nodes: {stats['leaf_nodes']}")
        self.logger.info(f"- Branching factor: {stats['branching_factor']:.2f}")

    def save_tree(self, root: Node, filepath: str):
        """Save tree structure to file with error handling."""
        try:
            self.logger.info(f"Saving tree to {filepath}")
            tree_data = root.to_dict()
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(tree_data, f, indent=2)
            
            self.logger.info(f"Tree successfully saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save tree: {str(e)}", exc_info=True)
            raise RaptorTreeError(f"Tree save failed: {str(e)}")

    def load_tree(self, filepath: str) -> Node:
        """Load tree structure from file with validation."""
        try:
            self.logger.info(f"Loading tree from {filepath}")
            
            if not Path(filepath).exists():
                raise FileNotFoundError(f"Tree file not found: {filepath}")
                
            with open(filepath, 'r') as f:
                tree_data = json.load(f)
                
            root = Node.from_dict(tree_data)
            self.logger.info("Tree successfully loaded")
            self._log_tree_stats(root)
            return root
            
        except Exception as e:
            self.logger.error(f"Failed to load tree: {str(e)}", exc_info=True)
            raise RaptorTreeError(f"Tree load failed: {str(e)}")

    def find_similar_nodes(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find similar nodes using vector similarity search."""
        try:
            self.logger.info(f"Searching for nodes similar to query: {query[:100]}...")
            
            # Generate query embedding
            query_embedding = self.embed_manager.embed_texts([query])[0]
            
            # Search in vector store
            results = self.pinecone_manager.similarity_search(
                query_embedding,
                top_k=top_k
            )
            
            self.logger.info(f"Found {len(results)} similar nodes")
            return results
            
        except Exception as e:
            self.logger.error(f"Similarity search failed: {str(e)}", exc_info=True)
            raise RaptorTreeError(f"Similarity search failed: {str(e)}")

    def _chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Chunk documents into smaller pieces."""
        try:
            total_docs = len(documents)
            self.logger.info(f"Starting document chunking for {total_docs} documents")
            chunks = []
            
            for i, doc in enumerate(documents, 1):
                self.logger.info(f"Chunking document {i}/{total_docs}: {doc.get('metadata', {}).get('name', 'unnamed')}")
                doc_chunks = self.chunker.chunk_text(doc['text'])
                self.logger.info(f"Created {len(doc_chunks)} chunks for document {i}")
                
                for chunk in doc_chunks:
                    chunks.append({
                        'text': chunk,
                        'metadata': {
                            **doc.get('metadata', {}),
                            'chunk_size': len(chunk),
                            'created_at': int(time.time())
                        }
                    })
                    
            self.logger.info(f"Chunking completed. Total chunks created: {len(chunks)}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error during document chunking: {str(e)}", exc_info=True)
            raise RaptorTreeError(f"Document chunking failed: {str(e)}")

    def _build_initial_tree(self, chunks: List[Dict], embeddings: np.ndarray) -> Node:
        """Build initial tree structure from chunks."""
        try:
            self.logger.info(f"Building initial tree from {len(chunks)} chunks")
            start_time = time.time()
            
            # Create root node
            self.logger.info("Creating root node...")
            root = Node(
                texts=[chunk['text'] for chunk in chunks],
                embeddings=embeddings,
                metadata={'level': 0, 'node_type': 'root'}
            )
            
            # Build tree recursively
            self.logger.info("Starting recursive tree building from root...")
            self._build_subtree(root, level=0)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Initial tree building completed in {elapsed:.2f} seconds")
            self._log_tree_stats(root)
            
            return root
            
        except Exception as e:
            self.logger.error(f"Error during initial tree building: {str(e)}", exc_info=True)
            raise RaptorTreeError(f"Initial tree building failed: {str(e)}")
        
    def _build_subtree(self, node: Node, level: int) -> None:
        """Recursively build subtree."""
        try:
            self.logger.info(f"Building subtree at level {level} with {len(node.texts)} texts")
            start_time = time.time()
            
            if level >= self.max_depth:
                self.logger.info(f"Reached maximum depth {level}, stopping recursion")
                return
                
            if len(node.texts) <= self.min_children:
                self.logger.info(f"Node has {len(node.texts)} texts (≤ {self.min_children} min_children), making it a leaf")
                return
            
            # Cluster the node's texts
            self.logger.info(f"Clustering {len(node.texts)} texts at level {level}...")
            clusters = self.cluster_manager.cluster_texts(
                texts=node.texts,
                embeddings=node.embeddings,
                min_clusters=self.min_children,
                max_clusters=self.max_children
            )
            
            self.logger.info(f"Created {len(clusters)} clusters at level {level}")
            
            # Create child nodes for each cluster
            for cluster_idx, indices in clusters.items():
                child_texts = [node.texts[i] for i in indices]
                child_embeddings = node.embeddings[indices]
                
                self.logger.info(f"Creating child node for cluster {cluster_idx} with {len(child_texts)} texts")
                
                child = Node(
                    texts=child_texts,
                    embeddings=child_embeddings,
                    metadata={
                        'cluster_id': cluster_idx,
                        'level': level + 1,
                        'parent_id': node.id,
                        'size': len(child_texts)
                    }
                )
                node.add_child(child)
                
                # Log memory usage periodically
                if cluster_idx % 10 == 0:
                    self.logger.info(f"Memory usage after cluster {cluster_idx}: {self._get_memory_usage():.2f} MB")
                
                # Recursively build subtree
                self.logger.info(f"Recursing into cluster {cluster_idx} at level {level + 1}")
                self._build_subtree(child, level + 1)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Completed subtree at level {level} in {elapsed:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error building subtree at level {level}: {str(e)}", exc_info=True)
            raise RaptorTreeError(f"Subtree building failed at level {level}: {str(e)}")
            
    def _generate_summaries(self, node: Node) -> None:
        """Generate summaries for each node."""
        try:
            self.logger.info(f"Generating summaries for node at level {node.metadata.get('level', 'unknown')}")
            start_time = time.time()
            
            # Generate summary for current node
            if node.texts:
                self.logger.info(f"Generating summary for node with {len(node.texts)} texts")
                summary = self.summary_manager.generate_summary(node.texts)
                node.metadata['summary'] = summary
                self.logger.info("Summary generation completed")
            
            # Recursively generate summaries for children
            if node.children:
                self.logger.info(f"Generating summaries for {len(node.children)} child nodes")
                for i, child in enumerate(node.children, 1):
                    self.logger.info(f"Processing child {i}/{len(node.children)}")
                    self._generate_summaries(child)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Completed summary generation for node and children in {elapsed:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error generating summaries: {str(e)}", exc_info=True)
            raise RaptorTreeError(f"Summary generation failed: {str(e)}")

    def _calculate_tree_stats(self, root: Node) -> Dict:
        """Calculate comprehensive tree statistics."""
        stats = {
            'total_nodes': 1,
            'leaf_nodes': 1 if not root.children else 0,
            'max_depth': root.get_depth(),
            'total_texts': len(root.texts),
            'avg_texts_per_node': 0,
            'avg_children_per_node': 0,
            'branching_factor': 0
        }
        
        # Collect stats from all nodes
        nodes = [root]
        total_texts = len(root.texts)
        total_children = len(root.children)
        
        while nodes:
            node = nodes.pop(0)
            nodes.extend(node.children)
            
            if node.children:
                stats['total_nodes'] += len(node.children)
                total_children += sum(len(child.children) for child in node.children)
            else:
                stats['leaf_nodes'] += 1
                
            total_texts += sum(len(child.texts) for child in node.children)
            
        # Calculate averages
        if stats['total_nodes'] > 0:
            stats['avg_texts_per_node'] = total_texts / stats['total_nodes']
            stats['avg_children_per_node'] = total_children / stats['total_nodes']
            stats['branching_factor'] = total_children / stats['total_nodes']
            
        return stats

    def get_tree_stats(self, root: Node) -> Dict:
        """Get comprehensive statistics about the tree."""
        stats = {
            'total_nodes': 1,
            'leaf_nodes': 1 if not root.children else 0,
            'max_depth': root.get_depth(),
            'total_texts': len(root.texts),
            'avg_texts_per_node': 0,
            'avg_children_per_node': 0,
            'is_balanced': root.is_balanced(self.balance_threshold)
        }
        
        # Collect stats from all nodes
        nodes = [root]
        total_texts = len(root.texts)
        total_children = len(root.children)
        
        while nodes:
            node = nodes.pop(0)
            nodes.extend(node.children)
            
            if node.children:
                stats['total_nodes'] += len(node.children)
                total_children += sum(len(child.children) for child in node.children)
            else:
                stats['leaf_nodes'] += 1
                
            total_texts += sum(len(child.texts) for child in node.children)
            
        # Calculate averages
        if stats['total_nodes'] > 0:
            stats['avg_texts_per_node'] = total_texts / stats['total_nodes']
            stats['avg_children_per_node'] = total_children / stats['total_nodes']
            
        return stats

class RaptorTreeError(Exception):
    """Base exception class for RaptorTree errors."""
    pass
