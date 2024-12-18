"""Tree update strategy for handling incremental changes to the document tree."""

import logging
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
from pathlib import Path

from src.tree.tree_manager import TreeManager
from src.storage.vector_manager import VectorManager

logger = logging.getLogger(__name__)

class TreeUpdater:
    def __init__(self, tree_manager: TreeManager, vector_manager: VectorManager, namespace: str):
        self.tree_manager = tree_manager
        self.vector_manager = vector_manager
        self.base_namespace = namespace
        self.chunks_ns = f"{namespace}_chunks"
        self.summaries_ns = f"{namespace}_summaries"

    def update_tree(self, 
                   new_chunks: List[str], 
                   new_embeddings: List[np.ndarray],
                   removed_chunk_ids: Optional[List[str]] = None,
                   modified_chunk_ids: Optional[List[str]] = None) -> Dict:
        """
        Update existing tree structure with new, modified, or removed chunks.
        
        Args:
            new_chunks: List of new chunk texts to add
            new_embeddings: Corresponding embeddings for new chunks
            removed_chunk_ids: List of chunk IDs to remove
            modified_chunk_ids: List of chunk IDs that were modified
            
        Returns:
            Dict containing updated tree structure information
        """
        try:
            # Step 1: Handle removals first
            if removed_chunk_ids:
                self._remove_chunks(removed_chunk_ids)
            
            # Step 2: Get affected internal nodes for modifications
            affected_nodes = set()
            if modified_chunk_ids:
                affected_nodes.update(self._get_affected_nodes(modified_chunk_ids))
                
            # Step 3: Process new chunks
            if new_chunks and new_embeddings:
                new_node_data = self.tree_manager.build_subtree(new_chunks, new_embeddings)
                if not new_node_data:
                    raise ValueError("Failed to build subtree for new chunks")
                affected_nodes.update(self._integrate_subtree(new_node_data))
            
            # Step 4: Rebalance affected portions of tree
            if affected_nodes:
                updated_tree = self._rebalance_nodes(affected_nodes)
                return updated_tree
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to update tree: {str(e)}")
            return {}

    def _remove_chunks(self, chunk_ids: List[str]) -> None:
        """Remove chunks and their associated internal nodes."""
        try:
            # Check if tree is locked
            if hasattr(self.tree_manager, '_tree_locked') and self.tree_manager._tree_locked:
                raise RuntimeError("Cannot remove chunks after tree is locked")
                
            # Get internal nodes referencing these chunks
            affected_nodes = self._get_affected_nodes(chunk_ids)
            
            # Delete chunks from vector store
            self.vector_manager.delete_vectors(chunk_ids, self.chunks_ns)
            logger.info(f"Removed {len(chunk_ids)} chunks from vector store")
            
            # Delete affected internal nodes
            if affected_nodes:
                internal_ids = [f"summary_{node_id}" for node_id in affected_nodes]
                self.vector_manager.delete_vectors(internal_ids, self.summaries_ns)
                logger.info(f"Removed {len(internal_ids)} internal nodes from vector store")
                
        except Exception as e:
            logger.error(f"Failed to remove chunks: {str(e)}")

    def _get_affected_nodes(self, chunk_ids: List[str]) -> Set[str]:
        """Find all internal nodes affected by changes to given chunks."""
        affected_nodes = set()
        try:
            # Query vector store for chunks to get their tree node IDs
            chunk_vectors = self.vector_manager.get_vectors(chunk_ids, self.chunks_ns)
            
            # Collect tree node IDs
            tree_node_ids = set()
            for vector in chunk_vectors:
                if vector and vector.get('metadata', {}).get('tree_node_id'):
                    tree_node_ids.add(vector['metadata']['tree_node_id'])
            
            # Find all internal nodes that reference these tree nodes
            if tree_node_ids:
                query = {
                    "node_type": "internal",
                    "$or": [{"children_ids": {"$in": list(tree_node_ids)}}]
                }
                internal_vectors = self.vector_manager.query_vectors(query, self.summaries_ns)
                
                for vector in internal_vectors:
                    if vector and vector.get('metadata', {}).get('tree_node_id'):
                        affected_nodes.add(vector['metadata']['tree_node_id'])
                        
        except Exception as e:
            logger.error(f"Failed to get affected nodes: {str(e)}")
            
        return affected_nodes

    def _integrate_subtree(self, subtree_data: Dict) -> Set[str]:
        """Integrate a new subtree into the existing tree structure."""
        affected_nodes = set()
        try:
            # Check if tree is locked
            if hasattr(self.tree_manager, '_tree_locked') and self.tree_manager._tree_locked:
                raise RuntimeError("Cannot integrate subtree after tree is locked")
                
            # Store new leaf nodes
            chunk_vectors = []
            for node_id, node_data in subtree_data.get('leaf_nodes', {}).items():
                if node_data.get('embedding') is not None:
                    vector_id = f"chunk_{int(time.time())}_{node_id}"
                    metadata = {
                        'text': node_data.get('text', ''),
                        'vector_id': vector_id,
                        'node_type': 'leaf',
                        'tree_node_id': node_id,
                        'filename': node_data.get('filename', 'unknown')
                    }
                    chunk_vectors.append({
                        'id': vector_id,
                        'values': node_data['embedding'].tolist() if isinstance(node_data['embedding'], np.ndarray) else node_data['embedding'],
                        'metadata': metadata
                    })
            
            if chunk_vectors:
                self.vector_manager.upsert_vectors(chunk_vectors, self.chunks_ns)
                logger.info(f"Integrated {len(chunk_vectors)} new leaf nodes")
            
            # Find attachment points for the subtree
            attachment_points = self._find_attachment_points(subtree_data)
            affected_nodes.update(attachment_points)
            
            # Store new internal nodes
            summary_vectors = []
            for node_id, node_data in subtree_data.get('internal_nodes', {}).items():
                if node_data.get('embedding') is not None:
                    vector_id = f"summary_{int(time.time())}_{node_id}"
                    metadata = {
                        'text': node_data.get('summary', ''),
                        'vector_id': vector_id,
                        'node_type': 'internal',
                        'tree_node_id': node_id,
                        'depth': node_data.get('depth', 0),
                        'children_ids': node_data.get('children', [])
                    }
                    summary_vectors.append({
                        'id': vector_id,
                        'values': node_data['embedding'].tolist() if isinstance(node_data['embedding'], np.ndarray) else node_data['embedding'],
                        'metadata': metadata
                    })
            
            if summary_vectors:
                self.vector_manager.upsert_vectors(summary_vectors, self.summaries_ns)
                logger.info(f"Integrated {len(summary_vectors)} new internal nodes")
                
        except Exception as e:
            logger.error(f"Failed to integrate subtree: {str(e)}")
            
        return affected_nodes

    def _find_attachment_points(self, subtree_data: Dict) -> Set[str]:
        """Find optimal points to attach new subtree to existing tree."""
        attachment_points = set()
        try:
            # Get root embedding of subtree
            root_id = subtree_data.get('root_id')
            if not root_id:
                return attachment_points
                
            root_embedding = subtree_data.get('internal_nodes', {}).get(root_id, {}).get('embedding')
            if root_embedding is None:
                return attachment_points
            
            # Query for nearest internal nodes
            query_vector = root_embedding.tolist() if isinstance(root_embedding, np.ndarray) else root_embedding
            nearest_nodes = self.vector_manager.query_vectors(
                {"node_type": "internal"},
                self.summaries_ns,
                vector=query_vector,
                top_k=3
            )
            
            # Add nearest nodes and their parents to attachment points
            for node in nearest_nodes:
                if node and node.get('metadata', {}).get('tree_node_id'):
                    attachment_points.add(node['metadata']['tree_node_id'])
                    
        except Exception as e:
            logger.error(f"Failed to find attachment points: {str(e)}")
            
        return attachment_points

    def _rebalance_nodes(self, affected_nodes: Set[str]) -> Dict:
        """Rebalance tree structure for affected nodes."""
        try:
            # Check if tree is locked
            if hasattr(self.tree_manager, '_tree_locked') and self.tree_manager._tree_locked:
                raise RuntimeError("Cannot rebalance nodes after tree is locked")
                
            # Get all vectors for affected nodes and their children
            affected_vectors = []
            for node_id in affected_nodes:
                # Get the node and its children
                node_vector = self.vector_manager.get_vectors([f"summary_{node_id}"], self.summaries_ns)
                if node_vector:
                    affected_vectors.extend(node_vector)
                    
                    # Get children if they exist
                    children_ids = node_vector[0].get('metadata', {}).get('children_ids', [])
                    if children_ids:
                        child_vectors = self.vector_manager.get_vectors(
                            [f"chunk_{id}" for id in children_ids],
                            self.chunks_ns
                        )
                        affected_vectors.extend(child_vectors)
            
            # Extract texts and embeddings
            texts = []
            embeddings = []
            for vector in affected_vectors:
                if vector:
                    texts.append(vector.get('metadata', {}).get('text', ''))
                    embeddings.append(vector.get('values', []))
            
            # Rebuild subtree for affected portion
            if texts and embeddings:
                updated_tree = self.tree_manager.build_subtree(texts, embeddings)
                
                # Update vector store with new structure
                self._integrate_subtree(updated_tree)
                
                return updated_tree
                
            return {}
            
        except Exception as e:
            logger.error(f"Failed to rebalance nodes: {str(e)}")
            return {}
