"""Tree manager for building and maintaining document trees."""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import uuid
import time
from datetime import datetime
import os
from pathlib import Path
from sklearn.cluster import KMeans
import hdbscan
from src.models.data_models import TreeData, NodeData, EdgeData
from src.embedding.embed_manager import EmbedManager
from src.clustering.cluster_manager import ClusterManager
from src.storage.store_manager import StoreManager
from src.storage.db_manager import DatabaseManager
from src.utils.openai_client import UnifiedAIClient
from src.text.chunk_manager import ChunkManager
from src.summarization.summary_manager import SummaryManager
from src.tree.node import Node, LeafNode, SummaryNode
import json
import math
from collections import defaultdict, Counter
from sklearn.cluster import AgglomerativeClustering

class TreeManager:
    """Manager for building and maintaining document trees."""

    def __init__(self, config: Dict = None):
        """Initialize TreeManager with configuration."""
        if not isinstance(config, dict):
            raise ValueError(f"Expected dict config, got {type(config)}")
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("Initializing TreeManager...")
        
        # Initialize all managers as None - they will be set externally
        self.embed_manager: Optional[EmbedManager] = None
        self.cluster_manager: Optional[ClusterManager] = None
        self.store_manager: Optional[StoreManager] = None
        self.db_manager: Optional[DatabaseManager] = None
        self.unifiedai_client: Optional[UnifiedAIClient] = None
        self.chunk_manager: Optional[ChunkManager] = None
        self.summary_manager: Optional[SummaryManager] = None
        
        # Tree configuration
        self.balance_threshold: float = self.config.get('balance_threshold', 0.3)
        self.min_cluster_size: int = self.config.get('min_cluster_size', 3)
        self.max_cluster_size: int = self.config.get('max_cluster_size', 10)
        
        # Add debug logging
        self.logger.info(f"Tree configuration: max_cluster_size={self.max_cluster_size}, "
                        f"min_cluster_size={self.min_cluster_size}, "
                        f"balance_threshold={self.balance_threshold}")
        
        # Calculate tree depth based on leaf count (log10)
        self._leaf_count = 0
        self._tree_depth = 1  # Default to 1 level for very small datasets
        
        # Initialize cluster counts
        self._init_cluster_counts()
        
        # Initialize data structures
        self.chunks: List[Dict] = []
        self.embeddings: np.ndarray = np.array([])
        self.clusters: List[List[int]] = []
        self.tree_data: TreeData = TreeData()
        self.root: Optional[Node] = None
        
        # Setup directories
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'test_outputs',
            'tree_viz'
        )
        self.cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'test_outputs',
            'cache'
        )
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Tree locking mechanism - initialize as unlocked
        self._is_locked: bool = False
        self._lock_time: Optional[float] = None
        self._lock_id: Optional[str] = None
        
        # Setup lock file path
        self._lock_file = os.path.join(
            self.cache_dir,
            f"{self.config.get('index_name', 'default')}_tree.lock"
        )
        
        # Check for existing lock file and remove it during initialization
        if os.path.exists(self._lock_file):
            try:
                os.remove(self._lock_file)
                self.logger.info("Removed existing lock file during initialization")
            except Exception as e:
                self.logger.warning(f"Could not remove existing lock file: {str(e)}")

    def _initialize_managers(self):
        """Initialize all required managers."""
        try:
            # All managers should be set externally by the caller
            if not all([
                self.unifiedai_client,
                self.embed_manager,
                self.cluster_manager,
                self.store_manager,
                self.chunk_manager,
                self.summary_manager
            ]):
                missing = []
                if not self.unifiedai_client: missing.append("unifiedai_client")
                if not self.embed_manager: missing.append("embed_manager")
                if not self.cluster_manager: missing.append("cluster_manager")
                if not self.store_manager: missing.append("store_manager")
                if not self.chunk_manager: missing.append("chunk_manager")
                if not self.summary_manager: missing.append("summary_manager")
                
                self.logger.error(f"Missing required managers: {', '.join(missing)}")
                raise ValueError(f"Required managers not set: {', '.join(missing)}")
                
            self.logger.info("✓ All managers initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing managers: {str(e)}")
            raise

    def _init_cluster_counts(self):
        """Initialize cluster counts based on current leaf count."""
        # Calculate tree depth based on log10 of leaf count
        # Add 1 to handle the case of 0 leaves initially
        self._tree_depth = max(1, int(math.log10(self._leaf_count + 1))) + 1  # Fixed parentheses
        
        # Ensure we have at least one cluster if we have any leaves
        min_clusters = 1 if self._leaf_count > 0 else 0
        
        # Adjust cluster counts based on tree depth
        if self._tree_depth == 1:
            # For very small datasets (<10 leaves), use minimal clustering
            self.twig_cluster_count = min_clusters
            self.branch_cluster_count = 0  # No branches needed for single level
        elif self._tree_depth == 2:
            # For small datasets (10-100 leaves), use root -> branch -> leaves
            self.twig_cluster_count = min(
                max(min_clusters, self._leaf_count // self.min_cluster_size),
                self.max_cluster_size
            )
            self.branch_cluster_count = min_clusters
        else:
            # For larger datasets, calculate balanced cluster counts
            leaves_per_twig = max(self.min_cluster_size, 
                               min(self._leaf_count // 10, self.max_cluster_size))
            self.twig_cluster_count = max(
                min_clusters,
                min(self._leaf_count // leaves_per_twig, self.max_cluster_size)
            )
            
            twigs_per_branch = max(self.min_cluster_size,
                                min(self.twig_cluster_count // 3, self.max_cluster_size))
            self.branch_cluster_count = max(
                min_clusters,
                min(self.twig_cluster_count // twigs_per_branch, self.max_cluster_size)
            )
        
        self.logger.info(
            f"Tree structure initialized: depth={self._tree_depth}, "
            f"branches={self.branch_cluster_count}, twigs={self.twig_cluster_count}, "
            f"total_leaves={self._leaf_count}"
        )

    def process_documents_with_embeddings(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]] = None
    ) -> Tuple[TreeData, Dict[str, Any]]:
        """Process documents with pre-computed embeddings."""
        try:
            # Group chunks by file
            chunks_by_file = defaultdict(list)
            for i, (text, meta) in enumerate(zip(texts, metadata or [{}] * len(texts))):
                filename = meta.get('filename', f'doc_{i}')
                chunks_by_file[filename].append((i, text, meta))
            
            self.logger.info(f"Processing {len(texts)} documents from {len(chunks_by_file)} files")
            
            # Create leaf nodes
            leaf_nodes = []
            node_embeddings = []
            for filename, file_chunks in chunks_by_file.items():
                total_chunks = len(file_chunks)
                self.logger.info(f"Creating {total_chunks} leaf nodes for file: {filename}")
                
                for chunk_idx, (i, text, meta) in enumerate(file_chunks, 1):
                    file_metadata = {
                        'filename': filename,
                        'path': meta.get('path', ''),
                        'size': meta.get('size', len(text.encode())),
                        'modified': meta.get('modified', time.time()),
                        'type': meta.get('type', 'txt'),
                        'chunk_index': chunk_idx,
                        'total_chunks': total_chunks,
                        'embedding': embeddings[i]  # Add embedding to metadata
                    }
                    
                    leaf_node = self.create_leaf_node(text, file_metadata)
                    self.logger.debug(f"Created leaf node {leaf_node.id} for chunk {chunk_idx}")
                    leaf_nodes.append(leaf_node)
                    node_embeddings.append(embeddings[i])
            
            self.logger.info(f"Created {len(leaf_nodes)} leaf nodes with embeddings")
            
            # Convert to numpy array
            node_embeddings = np.array(node_embeddings)
            
            # Build tree from leaf nodes
            self._build_tree_from_texts(leaf_nodes, node_embeddings)
            
            # Verify tree was built
            if not self.root:
                raise ValueError("Tree building failed - no root node created")
            
            # Create tree data
            tree_data = self._create_tree_data()
            if not tree_data or not tree_data.nodes:
                raise ValueError("Tree data creation failed - no nodes in tree data")
            
            # Store tree data in Pinecone
            if self.store_manager:
                try:
                    self.logger.info("Storing tree data in Pinecone...")
                    self.store_manager.store_tree_data(tree_data)
                    self.logger.info("✓ Tree data stored in Pinecone")
                except Exception as e:
                    self.logger.error(f"Failed to store tree data: {str(e)}")
                    # Don't raise - storage failure shouldn't stop processing
            
            # Collect statistics
            stats = {
                'tree_stats': {
                    'total_nodes': len(tree_data.nodes),
                    'leaf_nodes': len([n for n in tree_data.nodes if isinstance(n, LeafNode)]),
                    'max_depth': max(n.level for n in tree_data.nodes) if tree_data.nodes else 0,
                    'files_processed': len(chunks_by_file)
                },
                'processing_stats': {
                    'total_documents': len(texts),
                    'total_embeddings': len(embeddings),
                    'embedding_dimensions': embeddings.shape[1] if len(embeddings) > 0 else 0
                }
            }
            
            self.logger.info(f"Tree processing complete. Stats: {stats}")
            return tree_data, stats
            
        except Exception as e:
            self.logger.error(f"Failed to process documents: {str(e)}")
            raise

    def get_tree_data(self) -> TreeData:
        """Get tree data in standard format.
        
        Returns:
            TreeData object containing nodes and edges
        """
        return self.tree_data

    def _validate_managers(self):
        """Check that all required managers are initialized."""
        missing = []
        for name, manager in zip(['store_manager', 'embed_manager', 'cluster_manager', 'unifiedai_client', 'summary_manager'], [self.store_manager, self.embed_manager, self.cluster_manager, self.unifiedai_client, self.summary_manager]):
            if not manager:
                missing.append(name)
        if missing:
            raise ValueError(f"Missing required managers: {', '.join(missing)}")

    def _log_error(self, message):
        """Centralized error logging method."""
        self.logger.error(message, extra={'method': '_log_error'})

    def load_tree(self, tree_data: Dict) -> None:
        """Load a tree from standardized dictionary format.
        
        Args:
            tree_data: Tree data in the format returned by Node.to_dict()
        """
        if not tree_data:
            raise ValueError("No tree data provided")
            
        try:
            # Create root node
            self.root = Node.from_dict(tree_data)
            self.logger.info(f"Loaded tree with {len(self.get_all_nodes())} nodes", extra={'method': 'load_tree'})
            
        except Exception as e:
            self.logger.error(f"Failed to load tree: {str(e)}", extra={'method': 'load_tree'})
            raise
            
    def save_tree(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Save tree data to JSON file and return the data."""
        try:
            tree_data = self._serialize_tree()
            
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                try:
                    with open(output_path, 'w') as f:
                        json.dump(tree_data, f, indent=2)
                    self.logger.info(f"Tree data saved to {output_path}")
                except (TypeError, ValueError, OSError) as e:
                    # Log error but don't raise since saving to file is not mission critical
                    self.logger.error(f"Failed to save tree data to file: {str(e)}")
            
            return tree_data
            
        except Exception as e:
            self.logger.error(f"Failed to serialize tree: {str(e)}")
            # Return empty dict as fallback
            return {}

    def get_all_nodes(self, start_node=None) -> set:
        """Get all nodes in the tree or subtree."""
        if start_node is None:
            start_node = self.root
        if not start_node:
            return set()
        
        nodes = {start_node}
        
        # Handle both old Node class and new LeafNode/SummaryNode classes
        if isinstance(start_node, (LeafNode, SummaryNode)):
            # New node types
            for child in start_node.children:
                nodes.update(self.get_all_nodes(child))
        elif hasattr(start_node, 'children'):
            # Old Node class
            for child in start_node.children:
                nodes.update(self.get_all_nodes(child))
            
        return nodes

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID.
        
        Args:
            node_id: ID of the node to find
            
        Returns:
            Node if found, None otherwise
        """
        for node in self.get_all_nodes():
            if node.id == node_id:
                return node
        return None
        
    def update_node(self, node: Node, text: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update a node with new text or metadata."""
        self._check_tree_locked()
        if text is not None:
            node.data['text'] = text
        if metadata is not None:
            node.data['metadata'].update(metadata)
            
    def get_node_subtree_stats(self, node: Node) -> Dict:
        """Get statistics about a specific node and its subtree."""
        if not node:
            return {}
        
        # Get metadata from data dictionary
        metadata = node.data.get('metadata', {})
        if isinstance(metadata, list):
            # If metadata is a list (for non-leaf nodes), use first item or empty dict
            metadata = metadata[0] if metadata else {}
        
        stats = {
            'total_nodes': 1,
            'leaf_nodes': len(node.children) == 0,
            'max_depth': 0,
            'total_depth': node.get_depth(),
            'total_children': len(node.children),
            'node_types': {metadata.get('node_type', 'unknown'): 1}
        }
        
        # Recursively get stats from children
        for child in node.children:
            child_stats = self.get_node_subtree_stats(child)
            
            # Update counts
            stats['total_nodes'] += child_stats.get('total_nodes', 0)
            stats['leaf_nodes'] += child_stats.get('leaf_nodes', 0)
            stats['max_depth'] = max(stats['max_depth'], child_stats.get('max_depth', 0) + 1)  # Fixed parenthesis
            stats['total_depth'] += child_stats.get('total_depth', 0)
            stats['total_children'] += child_stats.get('total_children', 0)
            
            # Merge node type counts
            for node_type, count in child_stats.get('node_types', {}).items():
                stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + count
                
        return stats

    def get_tree_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the tree structure."""
        if not self.root:
            return {}
            
        # Count nodes by type
        total_nodes = 0
        leaf_nodes = 0
        internal_nodes = 0
        level_distribution = {}
        
        def count_nodes(node):
            nonlocal total_nodes, leaf_nodes, internal_nodes
            total_nodes += 1
            
            # Count by level
            level = node.data.level
            level_distribution[level] = level_distribution.get(level, 0) + 1
            
            # Count by type
            if not node.children:
                leaf_nodes += 1
            else:
                internal_nodes += 1
                
            for child in node.children:
                count_nodes(child)
        
        count_nodes(self.root)
        
        return {
            'total_nodes': total_nodes,
            'leaf_nodes': leaf_nodes,
            'internal_nodes': internal_nodes,
            'level_distribution': level_distribution,
            'max_depth': self._calculate_tree_depth(self.root)
        }

    def _optimize_tree(self, node: Node) -> Node:
        """Optimize tree structure through post-processing."""
        try:
            # Balance tree if needed
            if self.balance_threshold > 0:
                node = self._balance_tree(node)
                
            # Merge small clusters
            if self.min_cluster_size > 0:
                node = self._merge_small_clusters(node)
                
            # Split large clusters
            if self.max_cluster_size > 0:
                node = self._split_large_clusters(node)
                
            return node
            
        except Exception as e:
            self.logger.error(f"Failed to optimize tree: {str(e)}", extra={'method': '_optimize_tree'})
            raise

    def _balance_tree(self, node: Node) -> Node:
        """Balance the tree by redistributing nodes to maintain a balanced structure.
        
        Args:
            node: Root node of the tree/subtree to balance
            
        Returns:
            Balanced node
        """
        if not node or not node.children:
            return node
            
        # Calculate average children per node at this level
        total_children = sum(len(child.children) for child in node.children)
        avg_children = total_children / len(node.children)
        
        # Check if any node deviates too much from average
        for child in node.children:
            if len(child.children) > 0:
                deviation = abs(len(child.children) - avg_children) / avg_children
                if deviation > self.balance_threshold:
                    # Redistribute children
                    self._redistribute_children(child, avg_children)
            
            # Recursively balance child's subtree
            self._balance_tree(child)
            
        return node
        
    def _redistribute_children(self, node: Node, target_size: float) -> None:
        """Redistribute children of a node to achieve better balance.
        
        Args:
            node: Node whose children need redistribution
            target_size: Target number of children per node
        """
        if not node or len(node.children) <= 1:
            return
            
        # Sort children by similarity
        children = self._organize_nodes_by_similarity(node.children)
        
        # Create new nodes to hold redistributed children
        new_nodes = []
        current_node = Node(
            data={
                'text': f"Redistributed {len(new_nodes) + 1}",
                'level': node.level + 1,
                'metadata': {
                    'node_type': node.metadata.get('node_type', 'branch'),
                    'is_leaf': False
                }
            },
            db_manager=self.db_manager
        )
        current_children = []
        
        # Distribute children
        for child in children:
            current_children.append(child)
            if len(current_children) >= target_size:
                # Set children for current node
                for c in current_children:
                    c.set_parent(current_node)
                new_nodes.append(current_node)
                
                # Create new node for next batch
                current_node = Node(
                    data={
                        'text': f"Branch node for {node.text[:50]}...",
                        'level': node.level + 1,
                        'metadata': {
                            'node_type': node.metadata.get('node_type', 'branch'),
                            'is_leaf': False
                        }
                    },
                    db_manager=self.db_manager
                )
                current_children = []
                
        # Handle remaining children
        if current_children:
            for c in current_children:
                c.set_parent(current_node)
            new_nodes.append(current_node)
            
        # Replace original node's children with new nodes
        node.children = new_nodes

    def _merge_small_clusters(self, tree: Node) -> Node:
        # TO DO: implement merging small clusters logic
        return tree

    def _split_large_clusters(self, tree: Node) -> Node:
        # TO DO: implement splitting large clusters logic
        return tree

    def resume_from_cache(self, index_name: str, cache_path: Optional[str] = None) -> Tuple[Dict, Dict]:
        """Resume tree building from cached embeddings and texts."""
        try:
            if cache_path is None:
                cache_path = str(Path('cache') / f'{index_name}_cache.db')
                
            self.logger.info(f"Resuming from cache: {cache_path}", extra={'method': 'resume_from_cache'})
            
            # Connect to cache database
            conn = sqlite3.connect(cache_path)
            cursor = conn.cursor()
            
            # Get all embeddings ordered by file and chunk index
            cursor.execute("""
                SELECT doc_hash, embedding, metadata 
                FROM embeddings
                ORDER BY json_extract(metadata, '$.filename'), 
                         CAST(json_extract(metadata, '$.chunk_index') AS INTEGER)
            """)
            rows = cursor.fetchall()
            
            if not rows:
                raise ValueError("No embeddings found in cache")
                
            self.logger.info(f"Found {len(rows)} cached embeddings", extra={'method': 'resume_from_cache'})
            
            # Extract data
            texts = []
            embeddings = []
            metadata_list = []
            
            # Process each chunk directly without grouping
            for doc_hash, embedding_bytes, metadata_json in rows:
                try:
                    # Load metadata
                    metadata = json.loads(metadata_json)
                    
                    # Get text from metadata
                    text = metadata.get('text', '')
                    if not text:
                        self.logger.warning(f"No text found for doc_hash {doc_hash}", extra={'method': 'resume_from_cache'})
                        continue
                    
                    # Convert embedding from bytes to array
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                    
                    # Add to lists
                    texts.append(text)
                    embeddings.append(embedding)
                    metadata_list.append(metadata)
                    
                except Exception as e:
                    self.logger.error(f"Error processing cached item {doc_hash}: {str(e)}", extra={'method': 'resume_from_cache'})
                    continue
            
            if not texts:
                raise ValueError("No valid texts found in cache")
                
            self.logger.info(f"Successfully loaded {len(texts)} chunks from cache", extra={'method': 'resume_from_cache'})
            self.logger.info("Building tree from cached data...", extra={'method': 'resume_from_cache'})
            
            # Build tree from cached data
            tree_data, stats = self.process_documents_with_embeddings(texts, embeddings, metadata_list)
            
            self.logger.info("\n=== Tree Building Complete ===", extra={'method': 'resume_from_cache'})
            self.logger.info(f"Total Nodes: {stats['tree_stats']['total_nodes']}", extra={'method': 'resume_from_cache'})
            self.logger.info(f"Leaf Nodes: {stats['tree_stats']['leaf_nodes']}", extra={'method': 'resume_from_cache'})
            self.logger.info(f"Tree Depth: {stats['tree_stats']['depth']}", extra={'method': 'resume_from_cache'})
            
            return tree_data, stats
            
        except Exception as e:
            self.logger.error(f"Failed to resume from cache: {str(e)}", extra={'method': 'resume_from_cache'})
            raise

    def create_visualizations(self, output_dir: str) -> None:
        """Create visualizations of the tree structure."""
        try:
            # Validate and repair tree structure first
            self._validate_and_repair_tree()
            
            # Import TreeVisualizer here to avoid circular import
            from src.visualization.tree_viz import TreeVisualizer
            from src.models.data_models import TreeData, NodeData, EdgeData
            
            # Create tree visualizer
            visualizer = TreeVisualizer(output_dir=self.output_dir)
            
            # Validate root node exists
            if self.root is None:
                self.logger.error("Cannot create visualizations: Root node is None")
                return
                
            # Convert tree to dictionary format
            tree_dict = self.root.to_tree_format()
            
            # Create proper TreeData structure
            tree_data = TreeData()
            
            # First pass: Create all nodes and find the highest level
            nodes_by_id = {}
            max_level = -1
            root_node = None
            
            for node_data in tree_dict.get('tree', {}).get('nodes', []):
                level = node_data.get('level', 0)
                if level > max_level:
                    max_level = level
                
                node = NodeData(
                    id=node_data['id'],
                    text=node_data.get('text', ''),
                    level=node_data.get('level', 0),
                    parent_id=node_data.get('parent_id'),
                    metadata=node_data.get('metadata', {}),
                )
                nodes_by_id[node.id] = node
                
                # Track potential root node (highest level)
                if level == max_level:
                    root_node = node
                    node.parent_id = None  # Ensure root has no parent
            
            # Second pass: Adjust parent-child relationships
            if root_node:
                self.logger.info(f"Found root node at level {max_level}")
                # Add nodes to tree_data in hierarchical order
                tree_data.add_node(root_node)
                
                # Add remaining nodes
                for node in nodes_by_id.values():
                    if node.id != root_node.id:
                        # If node has no parent or parent would create cycle, connect to root
                        if not node.parent_id or node.parent_id == node.id:
                            node.parent_id = root_node.id
                        tree_data.add_node(node)
                
                # Create edges based on parent-child relationships
                for node in tree_data.nodes:
                    if node.parent_id:
                        edge = EdgeData(
                            source=node.parent_id,
                            target=node.id,
                            weight=1.0,
                            metadata={'type': 'parent-child'}
                        )
                        tree_data.add_edge(edge)
                
                # Validate the tree structure
                try:
                    tree_data.validate()
                    self.logger.info(f"Tree validation successful:")
                    self.logger.info(f"- Total nodes: {len(tree_data.nodes)}")
                    self.logger.info(f"- Max depth: {tree_data.calculate_depth()}")
                    self.logger.info(f"- Root node ID: {root_node.id}")
                    
                    # Create visualizations
                    self.logger.info("Creating tree visualizations...")
                    visualizer.create_tree_visualization(tree_data)
                    self.logger.info("✓ Tree visualization created")
                    
                except ValueError as e:
                    self.logger.error(f"Tree validation failed: {str(e)}")
                    return
            else:
                self.logger.error("No valid root node found in tree data")
                return
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}", extra={'method': 'create_visualizations'})
            raise

    def _validate_and_repair_tree(self):
        """Validate and repair the tree structure to ensure single root and proper hierarchy."""
        if not self.root:
            self.logger.error("No root node found", extra={'method': '_validate_and_repair_tree'})
            return
        
        # Get all nodes and create node map
        all_nodes = self.get_all_nodes()
        node_map = {node.id: node for node in all_nodes}
        
        # First pass - identify issues
        orphaned_nodes = []
        incorrect_parent_nodes = []
        multiple_roots = []
        
        for node in all_nodes:
            if node != self.root:
                if not node.parent:
                    orphaned_nodes.append(node)
                elif node.parent.id not in node_map:
                    incorrect_parent_nodes.append(node)
            elif node.parent is not None:
                multiple_roots.append(node)
                
        # Log repair statistics
        if orphaned_nodes or incorrect_parent_nodes or multiple_roots:
            self.logger.info(f"Found issues in tree structure:", extra={'method': '_validate_and_repair_tree'})
            self.logger.info(f"- Orphaned nodes: {len(orphaned_nodes)}", extra={'method': '_validate_and_repair_tree'})
            self.logger.info(f"- Incorrect parent nodes: {len(incorrect_parent_nodes)}", extra={'method': '_validate_and_repair_tree'})
            self.logger.info(f"- Multiple roots: {len(multiple_roots)}", extra={'method': '_validate_and_repair_tree'})
            
            # Fix orphaned nodes by attaching to root
            for node in orphaned_nodes:
                self.root.add_child(node)
                
            # Fix incorrect parent nodes by attaching to root
            for node in incorrect_parent_nodes:
                self.root.add_child(node)
                
            # Fix multiple roots by making them children of the main root
            for node in multiple_roots:
                if node != self.root:
                    node.parent = None
                    self.root.add_child(node)
                    
            self.logger.info("Tree structure repaired", extra={'method': '_validate_and_repair_tree'})
        else:
            self.logger.info("Tree structure validation passed", extra={'method': '_validate_and_repair_tree'})
            
        # Repair node levels
        self.repair_node_levels(self.root, 0)

    def repair_node_levels(self, node: Node, level: int) -> None:
        """Recursively repair node levels starting from a given node.
        
        Args:
            node: Node to start level repair from
            level: Expected level for this node
        """
        if not node:
            return
            
        # Update node level in NodeData
        node.data.level = level
        
        # Recursively update children
        for child in node.children:
            self.repair_node_levels(child, level + 1)
        
    def _serialize_tree(self) -> Dict[str, Any]:
        """Serialize the tree structure for storage, excluding embeddings."""
        try:
            # Get all nodes and edges
            all_nodes = self.get_all_nodes()
            
            # Convert nodes to NodeData format
            nodes_data = []
            edges_data = []
            
            for node in all_nodes:
                try:
                    # Create NodeData
                    node_data = node.to_dict()
                    
                    # Remove embedding from metadata to avoid JSON serialization issues
                    if 'embedding' in node_data['metadata']:
                        del node_data['metadata']['embedding']
                    
                    nodes_data.append(node_data)
                    
                    # Create EdgeData for each child
                    for child in node.children:
                        edge_data = {
                            'source': node.id,
                            'target': child.id,
                            'weight': 1.0,
                            'metadata': {}
                        }
                        edges_data.append(edge_data)
                except Exception as e:
                    self.logger.error(f"Failed to serialize node {node.id}: {str(e)}")
                    continue
            
            # Create tree data dictionary
            tree_data = {
                'nodes': nodes_data,
                'edges': edges_data,
                'metadata': {
                    'created_at': time.time(),
                    'last_modified': time.time(),
                    'version': '1.0',
                    'stats': self.get_tree_stats()
                }
            }
            
            # Prepare data for JSON serialization
            tree_data = self._prepare_tree_data_for_json(tree_data)
            
            return tree_data
            
        except Exception as e:
            self.logger.error(f"Failed to serialize tree: {str(e)}")
            # Return empty dict as fallback
            return {}

    def _parse_summary(self, summary):
        """Parse the summary response into a consistent dictionary format.
        
        Args:
            summary: Raw summary response from the summarizer (str or dict)
            
        Returns:
            dict: Parsed summary with consistent structure
        """
        if isinstance(summary, str):
            try:
                import json
                summary_dict = json.loads(summary)
                self.logger.info("Successfully parsed summary from JSON string")
            except json.JSONDecodeError:
                self.logger.info("Summary is plain text, creating default structure")
                summary_dict = {
                    'main_topic': summary,
                    'title': '',
                    'key_concepts': [],
                    'technical_details': {},
                    'clusters': [{'unifying_theme': summary, 'distinguishing_features': []}]
                }
        else:
            summary_dict = summary

        # Ensure all expected fields exist
        summary_dict.setdefault('main_topic', '')
        summary_dict.setdefault('title', '')
        summary_dict.setdefault('key_concepts', [])
        summary_dict.setdefault('technical_details', {})
        summary_dict.setdefault('clusters', [{'unifying_theme': '', 'distinguishing_features': []}])

        # Generate title from main topic if empty
        if not summary_dict['title']:
            main_topic = summary_dict['main_topic']
            summary_dict['title'] = main_topic[:50] + '...' if len(main_topic) > 50 else main_topic

        return summary_dict

    def _organize_nodes_by_similarity(self, nodes: List[Node]) -> List[Node]:
        """Organize nodes by similarity using their embeddings.
        
        Args:
            nodes: List of nodes to organize
            
        Returns:
            List of nodes organized by similarity
        """
        import numpy as np
        from sklearn.cluster import KMeans
        
        # Separate nodes with and without embeddings
        nodes_with_embeddings = [n for n in nodes if n.data.get_embedding() is not None]
        nodes_without_embeddings = [n for n in nodes if n.data.get_embedding() is None]
        
        if not nodes_with_embeddings:
            self.logger.warning("No nodes with embeddings found", extra={'method': '_organize_nodes_by_similarity'})
            return nodes
            
        # Convert embeddings to numpy array
        embeddings = np.array([n.data.get_embedding() for n in nodes_with_embeddings])
        
        # Determine number of clusters based on number of nodes
        n_clusters = min(len(nodes_with_embeddings), max(2, int(np.sqrt(len(nodes_with_embeddings)))))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Sort nodes by cluster and within clusters by similarity to cluster center
        organized_nodes = []
        for i in range(n_clusters):
            # Get nodes in current cluster
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_nodes = [nodes_with_embeddings[idx] for idx in cluster_indices]
            
            # Sort by distance to cluster center
            cluster_center = kmeans.cluster_centers_[i]
            distances = [np.linalg.norm(n.data.get_embedding() - cluster_center) for n in cluster_nodes]
            sorted_indices = np.argsort(distances)
            
            organized_nodes.extend([cluster_nodes[idx] for idx in sorted_indices])
            
        # Add nodes without embeddings at the end
        organized_nodes.extend(nodes_without_embeddings)
        
        return organized_nodes

    def root_score(self, node):
        """Calculate a score for how suitable a node is to be the root.
        
        Args:
            node: The node to score
            
        Returns:
            float: A score between 0 and 1, higher is better
        """
        # Prefer nodes at level 0
        level_score = 1.0 if node.get_depth() == 0 else 0.0
        
        # Prefer nodes with more children (normalized by max children in candidates)
        max_children = max(len(n.children) for n in self.get_all_nodes()) or 1
        children_score = len(node.children) / max_children
        
        # Prefer nodes with text that indicates a root role
        text_score = 0.5 if any(x in node.text.lower() for x in ['root', 'collection', 'main']) else 0.0
        
        return level_score + children_score + text_score

    def create_node(self, text: str, depth: int = 0, metadata: Optional[Dict[str, Any]] = None) -> Node:
        """Create a new node.
        
        Args:
            text (str): Node text content
            depth (int): Node depth in tree
            metadata (Optional[Dict[str, Any]]): Additional metadata
            
        Returns:
            Node: Created node
        """
        if metadata is None:
            metadata = {}
            
        # Ensure is_document_chunk is set
        if 'is_document_chunk' not in metadata:
            metadata['is_document_chunk'] = False  # Default to non-document chunk
        
        # Create NodeData object
        node_data = NodeData(
            id=str(uuid.uuid4()),
            text=text,
            level=depth,
            metadata=metadata,
            parent_id=None,
            children_ids=[]
        )
        
        return Node(data=node_data)

    def _prepare_tree_data_for_json(self, data: Any) -> Any:
        """Prepare data for JSON serialization by converting numpy arrays and other special types.
        
        Args:
            data: Data to prepare for JSON
            
        Returns:
            JSON-serializable data
        """
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._prepare_tree_data_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_tree_data_for_json(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._prepare_tree_data_for_json(item) for item in data)
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            # Convert any other types to string representation
            return str(data)

    def _create_cluster_node(self, cluster_nodes: List[Node], level: int, node_type: str) -> Optional[Node]:
        """Create a cluster node (twig or branch) from child nodes."""
        try:
            if not self.summary_manager:
                # Initialize summary manager if not already done
                summary_config = {
                    'model': self.config.get('summary_model', 'gpt-4'),
                    'embedding': {
                        'model': 'text-embedding-ada-002',
                        'dimensions': 1536,
                        'batch_size': 8
                    },
                    'max_tokens': 8000
                }
                self.summary_manager = SummaryManager(summary_config)
                
            # Log the input data structure
            self.logger.debug(f"Creating {node_type} node with {len(cluster_nodes)} nodes")
            
            # Process text from each node
            processed_texts = []
            for i, node in enumerate(cluster_nodes):
                if not isinstance(node.data, NodeData):
                    self.logger.error(f"Node {i} data is not NodeData: {type(node.data)}")
                    continue
                    
                text = node.data.text
                if text:
                    preview = text[:100] + '...' if len(text) > 100 else text
                    self.logger.debug(f"Node {i} text preview: {preview}")
                    processed_texts.append(text)
                else:
                    self.logger.warning(f"No text content found for node {i}")
            
            # Only proceed with summarization if we have texts
            if not processed_texts:
                self.logger.warning("No valid text content found in cluster nodes")
                return None
                
            # Combine texts for summary
            combined_text = "\n\n".join(processed_texts)
            
            # Generate summary using AI
            summary_data = self.summary_manager.generate_summary(combined_text)
            self.logger.debug("Summary data keys: %s", list(summary_data.keys()))
            
            # Create node data
            node_data = NodeData(
                id=str(uuid.uuid4()),
                text=summary_data.get('summary', ''),
                level=level,
                parent_id=None,
                children_ids=[node.data.id for node in cluster_nodes],
                metadata={
                    'created_at': time.time(),
                    'last_modified': time.time(),
                    'is_leaf': False,
                    'node_type': node_type,
                    'title': summary_data.get('title', ''),
                    'cluster_score': 0.0,
                    'importance_score': 0.0,
                    'summary': summary_data.get('summary', ''),
                    'main_topics': [concept['concept'] for concept in summary_data.get('key_concepts', [])],
                    'key_concepts': summary_data.get('key_concepts', []),
                    'technical_analysis': summary_data.get('technical_analysis', {}),
                    'relationships': summary_data.get('relationships', []),
                    'clusters': summary_data.get('clusters', [])
                }
            )
            
            # Calculate and set embedding for the cluster node
            if not self.embed_manager:
                self._initialize_managers()
                
            try:
                # Get embeddings from the summary text
                root_embedding = self.embed_manager.get_embeddings([summary_data.get('summary', '')])[0]
                # Use NodeData helper method to set embedding
                node_data.set_embedding(root_embedding)
            except Exception as e:
                self.logger.error(f"Failed to compute embedding for {node_type} node: {str(e)}")
                # Try averaging child embeddings as fallback
                try:
                    child_embeddings = []
                    for node in cluster_nodes:
                        # Use NodeData helper method to get embeddings
                        emb = node.data.get_embedding()
                        if emb is not None:
                            child_embeddings.append(emb)
                    
                    if child_embeddings:
                        # Average the numpy arrays
                        avg_embedding = np.mean(child_embeddings, axis=0)
                        # Use NodeData helper method to set averaged embedding
                        node_data.set_embedding(avg_embedding)
                        self.logger.info(f"Used average of {len(child_embeddings)} child embeddings as fallback")
                    else:
                        self.logger.error("Could not compute embedding from children either")
                except Exception as e2:
                    self.logger.error(f"Failed to compute average embedding: {str(e2)}")
            
            # Debug logging
            self.logger.debug("Created node data:")
            self.logger.debug(f"  - metadata keys: {list(node_data.metadata.keys())}")
            self.logger.debug(f"  - text type: {type(node_data.text)}")
            self.logger.debug(f"  - children count: {len(node_data.children_ids)}")
            
            # Create and return the node
            return Node(data=node_data, children=cluster_nodes)
            
        except Exception as e:
            self.logger.error(f"Failed to create {node_type} node: {str(e)}")
            self.logger.error("Error details:", exc_info=True)
            return None

    def _extract_title_from_text(self, text: str) -> str:
        """Extract a meaningful title from text.
        
        Args:
            text: Text to extract title from
            
        Returns:
            str: Extracted title
        """
        if not text:
            return "Untitled Document Group"
            
        # Try to get first sentence
        sentences = text.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            # If first sentence is too long, take first phrase
            if len(first_sentence) > 50:
                phrases = first_sentence.split(',')
                if phrases:
                    return phrases[0].strip()[:50] + '...' if len(phrases[0]) > 50 else phrases[0].strip()
            return first_sentence[:50] + '...' if len(first_sentence) > 50 else first_sentence
            
        # Fallback to first 50 chars if no sentences
        return text[:50] + '...' if len(text) > 50 else text

    def _create_leaf_nodes(self, texts: List[str], embeddings: List[np.ndarray], metadata: List[Dict] = None) -> List[Node]:
        """Create leaf nodes from texts and embeddings."""
        if not texts:
            return []
            
        # Initialize managers if needed
        if not self.embed_manager:
            self._initialize_managers()
            
        # Create nodes
        nodes = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            try:
                # Extract text from dictionary if needed
                if isinstance(text, dict):
                    text = text.get('text', '')
                    
                if not isinstance(text, str):
                    self.logger.error(f"Invalid text type for node {i}: {type(text)}")
                    continue
                    
                # Create node data
                node_data = NodeData(
                    id=str(uuid.uuid4()),
                    text=text,
                    level=0,
                    parent_id=None,
                    children_ids=[],
                    metadata={
                        'created_at': time.time(),
                        'last_modified': time.time(),
                        'is_leaf': True,
                        'node_type': 'leaf',
                        'title': self._extract_title_from_text(text),
                        'cluster_score': 0.0,
                        'importance_score': 0.0,
                        'summary': text,
                        'main_topics': [],
                        # Add any provided metadata
                        **(metadata[i] if metadata else {})
                    }
                )
                
                # Set embedding using helper method
                node_data.set_embedding(embedding)
                
                # Create and add node
                node = Node(data=node_data)
                nodes.append(node)
                self.logger.debug(f"Created leaf node {i} with text: {text[:100]}...")
                
            except Exception as e:
                self.logger.error(f"Error creating leaf node {i}: {str(e)}")
                continue
                
        return nodes

    def _create_twig_nodes(self, leaf_nodes: List[Node]) -> List[Node]:
        """Create twig nodes from leaf nodes."""
        if not leaf_nodes:
            self.logger.warning("No leaf nodes provided, skipping twig node creation")
            return []
        
        self.logger.info(f"Creating twig nodes from {len(leaf_nodes)} leaf nodes")
        
        # Extract embeddings for clustering
        leaf_embeddings = []
        valid_nodes = []
        for node in leaf_nodes:
            embedding = node.data.get_embedding()
            if embedding is not None:
                leaf_embeddings.append(embedding)
                valid_nodes.append(node)
            else:
                self.logger.warning(f"Node {node.data.id} missing valid embedding, skipping")
                
        if not valid_nodes:
            self.logger.error("No valid nodes with embeddings found")
            return []
        
        # Convert to numpy array
        leaf_embeddings = np.array(leaf_embeddings)
        
        # Calculate dynamic twig count based on dataset size
        min_nodes_per_cluster = max(self.min_cluster_size, 2)
        max_nodes_per_cluster = min(self.max_cluster_size, len(valid_nodes))
        adjusted_twig_count = max(1, len(valid_nodes) // min_nodes_per_cluster)
        adjusted_twig_count = min(adjusted_twig_count, len(valid_nodes) // max_nodes_per_cluster)
        
        self.logger.info(f"Adjusted twig cluster count to {adjusted_twig_count} based on {len(valid_nodes)} nodes")
        
        try:
            # Cluster leaf nodes
            twig_assignments = self._cluster_documents(leaf_embeddings, adjusted_twig_count)
            cluster_distribution = {i: list(twig_assignments).count(i) for i in set(twig_assignments)}
            
            # Create twig nodes for each cluster
            twig_nodes = []
            for cluster_id in range(max(twig_assignments) + 1):
                # Get nodes in this cluster
                cluster_nodes = [node for i, node in enumerate(valid_nodes) if twig_assignments[i] == cluster_id]
                
                try:
                    # Create twig node with dynamic level
                    twig_node = self._create_cluster_node(
                        cluster_nodes, 
                        level=1,  # Start at level 1, will be adjusted later
                        node_type='twig'
                    )
                    if twig_node:
                        twig_nodes.append(twig_node)
                        
                except Exception as e:
                    self.logger.error(f"Failed to create twig node for cluster {cluster_id}: {str(e)}")
                    continue
            
            return twig_nodes
            
        except Exception as e:
            self.logger.error(f"Failed to create twig nodes: {str(e)}")
            return []

    def _create_branch_nodes(self, twig_nodes: List[Node]) -> Optional[Node]:
        """Create branch nodes from twig nodes."""
        if not twig_nodes:
            return None
            
        try:
            # For very small trees, return single branch node
            if len(twig_nodes) <= self.max_cluster_size:
                branch_node = self._create_cluster_node(
                    twig_nodes, 
                    level=2,  # Will be adjusted in repair_node_levels
                    node_type='branch'
                )
                return branch_node
            
            # Extract embeddings for clustering
            twig_embeddings = []
            valid_nodes = []
            for node in twig_nodes:
                embedding = node.data.get_embedding()
                if embedding is not None:
                    twig_embeddings.append(embedding)
                    valid_nodes.append(node)
                    
            if not valid_nodes:
                return None
                
            # Calculate dynamic branch count
            min_clusters_per_branch = max(self.min_cluster_size, 2)
            max_clusters_per_branch = min(self.max_cluster_size, len(valid_nodes))
            adjusted_branch_count = max(1, len(valid_nodes) // min_clusters_per_branch)
            adjusted_branch_count = min(adjusted_branch_count, len(valid_nodes) // max_clusters_per_branch)
            
            # Cluster twig nodes
            branch_assignments = self._cluster_documents(np.array(twig_embeddings), adjusted_branch_count)
            
            # Create branch nodes
            branch_nodes = []
            for branch_id in range(max(branch_assignments) + 1):
                branch_cluster = [node for i, node in enumerate(valid_nodes) if branch_assignments[i] == branch_id]
                if branch_cluster:
                    branch_node = self._create_cluster_node(
                        branch_cluster, 
                        level=2,  # Will be adjusted in repair_node_levels
                        node_type='branch'
                    )
                    if branch_node:
                        branch_nodes.append(branch_node)
            
            # If multiple branch nodes, create root node
            if len(branch_nodes) > 1:
                root_node = self._create_cluster_node(
                    branch_nodes,
                    level=3,  # Will be adjusted in repair_node_levels
                    node_type='root'
                )
                return root_node
            elif len(branch_nodes) == 1:
                return branch_nodes[0]
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create branch nodes: {str(e)}")
            return None

    def _build_tree_from_texts(self, leaf_nodes: List[LeafNode], embeddings: np.ndarray) -> None:
        """Build tree structure from leaf nodes and their embeddings using HDBSCAN."""
        try:
            n_leaves = len(leaf_nodes)
            
            # For very small sets, create simpler tree
            if n_leaves <= 4:
                self.logger.info(f"Small dataset ({n_leaves} nodes), creating simple tree")
                # Create single root node containing all leaves
                summary = self._generate_hierarchical_summary(leaf_nodes)
                self.root = SummaryNode(
                    id=str(uuid.uuid4()),
                    text=summary,
                    level=1,
                    metadata={
                        'node_type': 'root',
                        'cluster_size': len(leaf_nodes)
                    }
                )
                # Set relationships
                for leaf in leaf_nodes:
                    leaf.metadata['parent_id'] = self.root.id
                    self.root.children.append(leaf)
                return
            
            # Determine tree depth based on number of leaves
            if n_leaves < 200:
                depth = 3  # leaves -> branches -> root
                level_names = ['leaves', 'branches', 'root']
            elif n_leaves < 2000:
                depth = 4  # leaves -> twigs -> branches -> root
                level_names = ['leaves', 'twigs', 'branches', 'root']
            else:
                depth = 5  # leaves -> twigs -> branches -> boughs -> root
                level_names = ['leaves', 'twigs', 'branches', 'boughs', 'root']
                
            self.logger.info(f"Building {depth}-level tree with {n_leaves} leaf nodes")
            self.logger.info(f"Level structure: {level_names}")
            
            # Start with leaf nodes at level 0
            current_nodes = leaf_nodes
            current_embeddings = embeddings
            
            # Build each level bottom-up
            for level in range(1, depth):
                level_name = level_names[level]
                
                # Special case for small node sets
                if len(current_nodes) <= 2:
                    self.logger.info(f"Small node set at {level_name} level, creating single node")
                    summary = self._generate_hierarchical_summary(current_nodes)
                    node = SummaryNode(
                        id=str(uuid.uuid4()),
                        text=summary,
                        level=level,
                        metadata={
                            'node_type': level_name[:-1],
                            'cluster_size': len(current_nodes)
                        }
                    )
                    for child in current_nodes:
                        child.metadata['parent_id'] = node.id
                        node.children.append(child)
                    self.root = node
                    return
                
                # Calculate target cluster size based on total nodes and level
                total_nodes = len(current_nodes)
                max_cluster_size = {
                    'twigs': 50,      # Smaller clusters at lower levels
                    'branches': 100,   # Medium clusters at middle levels
                    'boughs': 200,     # Larger clusters near root
                    'root': total_nodes  # Root can contain all
                }.get(level_name[:-1], 50)  # Default to 50 if level not found
                
                target_clusters = max(1, total_nodes // max_cluster_size)
                if level == depth - 1:  # Root level
                    target_clusters = 1
                
                self.logger.info(f"Level {level_name}: targeting {target_clusters} clusters with max size {max_cluster_size}")
                
                # Configure HDBSCAN parameters based on level
                if level == 1:  # First level above leaves (twigs)
                    min_cluster_size = min(20, max(5, total_nodes // target_clusters))
                    min_samples = max(3, min_cluster_size // 3)
                    cluster_selection_epsilon = 0.1  # More granular clustering
                elif level == depth - 1:  # Root level
                    min_cluster_size = 2
                    min_samples = 1
                    cluster_selection_epsilon = 0.5
                else:  # Intermediate levels
                    min_cluster_size = min(30, max(5, total_nodes // target_clusters))
                    min_samples = max(3, min_cluster_size // 3)
                    cluster_selection_epsilon = 0.2
                
                self.logger.info(
                    f"HDBSCAN parameters for {level_name}:"
                    f" min_cluster_size={min_cluster_size},"
                    f" min_samples={min_samples},"
                    f" epsilon={cluster_selection_epsilon}"
                )
                
                # Try HDBSCAN first
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    metric='euclidean',
                    cluster_selection_epsilon=cluster_selection_epsilon,
                    cluster_selection_method='eom',
                    prediction_data=True
                )
                
                # Normalize embeddings
                norms = np.linalg.norm(current_embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                normalized_embeddings = current_embeddings / norms
                
                # Perform clustering
                cluster_labels = clusterer.fit_predict(normalized_embeddings)
                
                # Check cluster sizes and subdivide if needed
                unique_labels = np.unique(cluster_labels[cluster_labels != -1])
                cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
                
                # If any cluster is too large, use KMeans to subdivide it
                new_labels = cluster_labels.copy()
                next_label = max(unique_labels) + 1 if len(unique_labels) > 0 else 0
                
                for idx, size in enumerate(cluster_sizes):
                    if size > max_cluster_size:
                        label = unique_labels[idx]
                        cluster_mask = (cluster_labels == label)
                        cluster_embeddings = normalized_embeddings[cluster_mask]
                        
                        # Calculate number of subclusters needed
                        n_subclusters = max(2, size // (max_cluster_size // 2))
                        self.logger.info(f"Subdividing cluster of size {size} into {n_subclusters} subclusters")
                        
                        # Use KMeans for subdivision
                        kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
                        subcluster_labels = kmeans.fit_predict(cluster_embeddings)
                        
                        # Update labels
                        new_labels[cluster_mask] = subcluster_labels + next_label
                        next_label += n_subclusters
                
                cluster_labels = new_labels
                
                # Get final cluster statistics
                unique_labels = np.unique(cluster_labels)
                cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
                self.logger.info(f"Final clusters: {len(unique_labels)}")
                self.logger.info(f"Cluster sizes: {cluster_sizes}")
                
                # Create summary nodes for each cluster
                new_nodes = []
                new_embeddings = []
                
                # Process each cluster
                for cluster_label in unique_labels:
                    cluster_indices = np.where(cluster_labels == cluster_label)[0]
                    cluster_nodes = [current_nodes[i] for i in cluster_indices]
                    
                    # Generate hierarchical summary
                    summary = self._generate_hierarchical_summary(cluster_nodes)
                    
                    # Create summary node
                    summary_node = SummaryNode(
                        id=str(uuid.uuid4()),
                        text=summary,
                        level=level,
                        metadata={
                            'node_type': level_name[:-1],
                            'cluster_size': len(cluster_nodes)
                        }
                    )
                    
                    # Set relationships
                    for child in cluster_nodes:
                        child.metadata['parent_id'] = summary_node.id
                        summary_node.children.append(child)
                    
                    # Get embedding for summary node
                    summary_embedding = self.embed_manager.embed_texts([summary])[0]
                    summary_node.metadata['embedding'] = summary_embedding
                    
                    new_nodes.append(summary_node)
                    new_embeddings.append(summary_embedding)
                
                # Update for next level
                current_nodes = new_nodes
                current_embeddings = np.array(new_embeddings)
                
                self.logger.info(f"Created {len(current_nodes)} {level_name} nodes")
            
            # Ensure we have at least one node at the end
            if not current_nodes:
                raise ValueError("No nodes created during tree building")
                
            # Set root node
            self.root = current_nodes[0]
            self.logger.info(f"Tree building complete. Root node: {self.root.id}")
            
            # Verify level structure
            node_counts = Counter(node.level for node in self.get_all_nodes())
            self.logger.info(f"Node counts by level: {dict(node_counts)}")
            
        except Exception as e:
            self.logger.error(f"Failed to build tree: {str(e)}")
            raise

    def _create_tree_data(self) -> TreeData:
        """Convert current tree structure to TreeData format."""
        if not self.root:
            self.logger.warning("Creating empty TreeData - no root node found")
            return TreeData(nodes=[], edges=[], metadata={})
        
        # Get all nodes in tree
        all_nodes = list(self.get_all_nodes())  # Convert set to list
        
        # Debug log node counts by level
        level_counts = {}
        for node in all_nodes:
            level = node.level
            level_counts[level] = level_counts.get(level, 0) + 1
        self.logger.info(f"Node counts by level: {level_counts}")
        
        # Debug log nodes with embeddings
        nodes_with_embeddings = [n for n in all_nodes if 'embedding' in n.metadata]
        self.logger.info(f"Nodes with embeddings: {len(nodes_with_embeddings)} out of {len(all_nodes)}")
        
        # Create edges based on parent-child relationships
        edges = []
        for node in all_nodes:
            if isinstance(node, (LeafNode, SummaryNode)):
                parent_id = node.metadata.get('parent_id')
                if parent_id:
                    edge = EdgeData(source=parent_id, target=node.id)
                    edges.append(edge)
        
        return TreeData(nodes=all_nodes, edges=edges)

    def _summarize_texts(self, texts: List[str]) -> str:
        """Generate a summary from multiple texts."""
        if not self.summary_manager:
            # Simple fallback if no summary manager
            return "\n\n".join(text[:100] + "..." for text in texts)
        
        try:
            return self.summary_manager.summarize("\n\n".join(texts))
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {str(e)}")
            return "\n\n".join(text[:100] + "..." for text in texts)

    def _check_tree_locked(self) -> None:
        """Check if tree is locked and raise error if it is."""
        if self._is_locked:
            raise ValueError(
                f"Tree is locked (Lock ID: {self._lock_id}). "
                f"Locked since: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self._lock_time))}. "
                "Unlock the existing tree before making modifications."
            )

    def _lock_tree(self) -> None:
        """Lock the tree to prevent modifications."""
        if self._is_locked:
            raise ValueError(f"Tree is already locked (Lock ID: {self._lock_id})")
        
        self._is_locked = True
        self._lock_time = time.time()
        self._lock_id = str(uuid.uuid4())
        
        # Save lock state to file
        lock_data = {
            'is_locked': True,
            'lock_id': self._lock_id,
            'lock_time': self._lock_time,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self._lock_file, 'w') as f:
            json.dump(lock_data, f, indent=2)
        
        self.logger.info(f"Tree locked (ID: {self._lock_id})")

    def _unlock_tree(self) -> None:
        """Unlock the tree to allow modifications."""
        if not self._is_locked:
            self.logger.warning("Tree is not locked")
            return
        
        self._is_locked = False
        self._lock_time = None
        self._lock_id = None
        
        # Remove lock file if it exists
        if os.path.exists(self._lock_file):
            os.remove(self._lock_file)
        
        self.logger.info("Tree unlocked")

    def _cluster_documents(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Cluster documents using HDBSCAN.
        
        Args:
            embeddings: Document embeddings array
            n_clusters: Target number of clusters
            
        Returns:
            Array of cluster assignments
        """
        min_cluster_size = self.min_cluster_size
        if min_cluster_size > len(embeddings):
            min_cluster_size = 2  # Minimum possible value
            self.logger.warning(f"Adjusting min_cluster_size to {min_cluster_size} due to small dataset")
            
        try:
            # Perform clustering
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
                cluster_selection_epsilon=0.15,
                prediction_data=True
            )
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # Handle outliers (-1) and ensure balanced clusters
            if -1 in cluster_labels or len(set(cluster_labels)) != n_clusters:
                self.logger.info("Using KMeans as fallback for more balanced clustering")
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(embeddings)
            
            # Log cluster distribution
            unique, counts = np.unique(cluster_labels, return_counts=True)
            distribution = dict(zip(unique, counts))
            self.logger.info(f"Cluster distribution: {distribution}")
            
            return cluster_labels
            
        except Exception as e:
            self.logger.error(f"Clustering error: {str(e)}, falling back to KMeans")
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            return kmeans.fit_predict(embeddings)

    def _calculate_tree_depth(self, node: Node) -> int:
        """Calculate the maximum depth of the tree from the given node."""
        if not node:
            return 0
        if not node.children:
            return 1
        return 1 + max(self._calculate_tree_depth(child) for child in node.children)

    def _prepare_tree_data_for_json(self, data: Any) -> Any:
        """Prepare data for JSON serialization by converting numpy arrays and other special types.
        
        Args:
            data: Data to prepare for JSON
            
        Returns:
            JSON-serializable data
        """
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._prepare_tree_data_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_tree_data_for_json(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._prepare_tree_data_for_json(item) for item in data)
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            # Convert any other types to string representation
            return str(data)

    def _extract_cluster_insights(self, summary_dict: Dict) -> Dict:
        """Extract cluster insights from a summary dictionary.
        
        Args:
            summary_dict (Dict): Summary dictionary containing cluster information
            
        Returns:
            Dict: Extracted cluster insights
        """
        insights = {
            'unifying_themes': [],
            'distinguishing_features': [],
            'key_patterns': []
        }
        
        # Extract insights from clusters if present
        if 'clusters' in summary_dict:
            for cluster in summary_dict['clusters']:
                if 'unifying_theme' in cluster:
                    insights['unifying_themes'].append(cluster['unifying_theme'])
                if 'distinguishing_features' in cluster:
                    insights['distinguishing_features'].extend(cluster['distinguishing_features'])
                    
        # Extract patterns from technical details
        if 'technical_details' in summary_dict:
            tech_details = summary_dict['technical_details']
            if 'cluster_patterns' in tech_details:
                insights['key_patterns'].extend(tech_details['cluster_patterns'])
                
        # Add key concepts as distinguishing features if none found
        if not insights['distinguishing_features'] and 'key_concepts' in summary_dict:
            insights['distinguishing_features'] = [
                f"{concept['concept']}: {concept['description']}"
                for concept in summary_dict['key_concepts']
                if 'concept' in concept and 'description' in concept
            ]
            
        return insights

    def _group_documents_by_subcluster(self, docs: List[str], embeddings: np.ndarray) -> List[List[str]]:
        """Group documents by subcluster."""
        if not self.cluster_manager:
            raise ValueError("Cluster manager not initialized")
        return self.cluster_manager.group_by_similarity(docs, embeddings)
        
    def _build_tree_recursive(self, chunks: List[Dict], depth: int = 0) -> Optional[Node]:
        """Build tree recursively from text chunks."""
        if not chunks:
            return None
            
        # Create root node from first chunk
        root = self._create_node_from_chunk(chunks[0], depth)
        
        # Process remaining chunks
        for chunk in chunks[1:]:
            node = self._create_node_from_chunk(chunk, depth + 1)
            root.add_child(node)
            
        return root

    def _create_node_from_chunk(self, chunk: Dict, depth: int = 0) -> Node:
        """Create a node from a text chunk.
        
        Args:
            chunk (Dict): Chunk data dictionary
            depth (int): Node depth in tree
            
        Returns:
            Node: Created node
        """
        # Create base node data
        node_data = NodeData(
            text=chunk.get('text', ''),
            level=depth
        )
        
        # Set basic metadata
        node_data.metadata.update({
            'is_leaf': True,
            'node_type': 'leaf',
            'title': self._extract_title_from_text(chunk.get('text', '')),
            'creation_time': time.time(),
            'last_modified': time.time()
        })
        
        # Handle embeddings
        if 'embedding' in chunk:
            embedding = chunk['embedding']
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            node_data.metadata['embedding'] = embedding
            node_data.metadata['embedding_id'] = str(uuid.uuid4())
        
        # Set leaf information (for document chunks)
        node_data.leaf = {
            "filename": chunk.get('filename', ''),
            "total_chunks": int(chunk.get('total_chunks', 0)),  # Ensure integer
            "chunk_index": int(chunk.get('chunk_index', 0))    # Ensure integer
        }
        
        # Set document processing information
        node_data.document_info.update({
            "path": chunk.get('path', ''),
            "size": chunk.get('size', 0),
            "modified": chunk.get('modified', 0.0),
            "type": chunk.get('type', ''),
            "token_count": chunk.get('token_count', 0),
            "char_count": chunk.get('char_count', 0),
            "is_first_chunk": chunk.get('is_first_chunk', False),
            "is_final_chunk": chunk.get('is_final_chunk', False),
            "embedding_model": chunk.get('embedding_model', ''),
            "embedding_time": chunk.get('embedding_time', 0.0),
            "embedding_batch": chunk.get('embedding_batch', 0),
            "embedding_position": chunk.get('embedding_position', 0),
            "is_document_chunk": True
        })
        
        return Node(data=node_data)

    def _get_context_metadata(self, chunk: Dict) -> Dict[str, Any]:
        """Get context metadata for a chunk."""
        metadata = chunk.get('metadata', {}).copy() if chunk.get('metadata') else {}
        
        # Add context information
        metadata['previous_header'] = metadata.get('previous_header', '')
        metadata['next_header'] = metadata.get('next_header', '')
        metadata['section_title'] = metadata.get('section_title', '')
        metadata['document_title'] = metadata.get('document_title', '')
            
        return metadata

    def _get_concept_metadata(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata from concept."""
        metadata = {}
        if isinstance(concept, dict):
            metadata['concept'] = concept.get('concept', '')
        return metadata

    def _process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Process text into chunks."""
        if not self.unifiedai_client:
            raise ValueError("UnifiedAI client not initialized")
            
        # Check token count
        token_count = self.unifiedai_client.count_tokens(text)
        if token_count > self.config.get('unifiedai', {}).get('max_tokens', 2048):
            # Split into chunks
            if not self.chunk_manager:
                self._initialize_managers()
                if not self.chunk_manager:
                    raise RuntimeError("Failed to initialize chunk manager")
            chunks = self.chunk_manager.chunk_text(text, metadata or {})
        else:
            # Single chunk
            chunks = [{'text': text, 'metadata': metadata or {}}]
            
        return chunks
        
    def _merge_chunks(self, chunks: List[Dict]) -> str:
        """Merge multiple text chunks into a single coherent text.
        
        Args:
            chunks: List of dictionaries containing text and metadata, or Node objects
        
        Returns:
            str: Merged text
        """
        if not chunks:
            return ""
        
        # Extract texts from chunks
        texts = []
        for chunk in chunks:
            # Handle both dict and Node inputs
            if isinstance(chunk, dict):
                text = chunk.get('text', '').strip()
            else:  # Assume Node object
                text = chunk.data.text.strip()
                
            if text:
                texts.append(text)
        
        # Join texts with newlines, removing duplicate whitespace
        merged_text = '\n\n'.join(texts)
        merged_text = ' '.join(merged_text.split())
        
        # Truncate if too long (prevent memory issues)
        max_length = 10000  # Reasonable limit for merged text
        if len(merged_text) > max_length:
            merged_text = merged_text[:max_length] + "..."
        
        return merged_text

    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text using UnifiedAI."""
        if not self.unifiedai_client:
            raise ValueError("UnifiedAI client not initialized")
            
        token_count = self.unifiedai_client.count_tokens(text)
        if token_count > self.config.get('unifiedai', {}).get('max_tokens', 2048):
            raise ValueError(f"Text too long: {token_count} tokens > {self.config.get('unifiedai', {}).get('max_tokens', 2048)}")
            
        # Generate analysis using UnifiedAI
        prompt = f"Analyze the following text:\n\n{text}"
        response = self.unifiedai_client.analyze_text(prompt)
        try:
            return response
        except Exception as e:
            return {"error": str(e), "raw_response": response}

    def _create_node(self, data: Dict[str, Any]) -> Node:
        """Create a new node."""
        node = Node(data=data)
        if self.db_manager:
            node.db_manager = self.db_manager
        return node
        
    def _group_documents_by_subcluster(self, docs: List[str], embeddings: np.ndarray) -> List[List[str]]:
        """Group documents by subcluster."""
        if not self.cluster_manager:
            raise ValueError("Cluster manager not initialized")
        return self.cluster_manager.group_by_similarity(docs, embeddings)
        
    def _collect_metadata_from_nodes(self, nodes: List[Union[LeafNode, SummaryNode]]) -> Dict:
        """Collect and merge metadata from a list of nodes."""
        if not nodes:
            return {}
        
        try:
            # Get metadata directly from node (new style) or from data.metadata (old style)
            def get_node_metadata(node):
                if hasattr(node, 'metadata'):
                    return node.metadata
                elif hasattr(node, 'data') and hasattr(node.data, 'metadata'):
                    return node.data.metadata
                else:
                    self.logger.warning(f"Node {node.id} has no metadata")
                    return {}
            
            # Start with first node's metadata
            merged_metadata = dict(get_node_metadata(nodes[0]))
            
            # Merge in metadata from remaining nodes
            for node in nodes[1:]:
                node_metadata = get_node_metadata(node)
                
                # Update counts and lists
                for key, value in node_metadata.items():
                    if key in merged_metadata:
                        if isinstance(value, (int, float)):
                            # Sum numeric values
                            merged_metadata[key] += value
                        elif isinstance(value, list):
                            # Combine lists
                            merged_metadata[key].extend(value)
                        elif isinstance(value, dict):
                            # Merge dictionaries
                            merged_metadata[key].update(value)
                        else:
                            # Keep most recent value for other types
                            merged_metadata[key] = value
                    else:
                        # Add new keys
                        merged_metadata[key] = value
            
            # Add aggregated metadata
            merged_metadata.update({
                'node_count': len(nodes),
                'total_text_length': sum(len(node.text) for node in nodes),
                'creation_time': time.time(),
                'last_modified': time.time()
            })
            
            return merged_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to collect metadata: {str(e)}", exc_info=True)
            raise

    def create_leaf_node(self, text: str, file_metadata: Dict) -> LeafNode:
        """Create a leaf node from document chunk."""
        node_id = str(uuid.uuid4())
        
        # Clean metadata - only technical info
        metadata = {
            # File info
            "filename": file_metadata["filename"],
            "path": file_metadata["path"],
            "size": file_metadata["size"],
            "modified": file_metadata["modified"],
            "type": file_metadata["type"],
            
            # Chunk info
            "chunk_index": file_metadata["chunk_index"],
            "total_chunks": file_metadata["total_chunks"],
            "tokens": len(text.split()),
            
            # Node properties
            "is_leaf": True,
            "node_type": "leaf",
            "level": 0,
            
            # Processing info
            "creation_time": time.time()
        }
        
        # Add embedding directly from input embeddings
        if 'embedding' in file_metadata:
            metadata['embedding'] = file_metadata['embedding']
        
        return LeafNode(
            id=node_id,
            text=text,
            level=0,
            metadata=metadata
        )

    def create_summary_node(self, child_nodes: List[Union[LeafNode, SummaryNode]], 
                          level: int, node_type: str) -> SummaryNode:
        """Create a summary node from child nodes.
        
        Args:
            child_nodes: List of child nodes to summarize
            level: Level in the tree (1=twigs, 2=branches, etc)
            node_type: Type of node ('twigs', 'branches', etc)
        """
        node_id = str(uuid.uuid4())
        
        # Combine child texts for summarization
        combined_text = "\n\n".join(node.text for node in child_nodes)
        
        # Generate summary
        try:
            summary = self.summary_manager.summarize(combined_text)
        except Exception as e:
            self.logger.error(f"Failed to generate summary: {str(e)}")
            summary = combined_text[:1000] + "..."  # Fallback
        
        # Collect metadata from children
        metadata = self._collect_metadata_from_nodes(child_nodes)
        metadata.update({
            'node_type': node_type,
            'level': level,
            'is_leaf': False,
            'creation_time': time.time()
        })
        
        return SummaryNode(
            id=node_id,
            text=summary,
            level=level,
            metadata=metadata
        )

    def calculate_tree_structure(self, num_chunks: int) -> Dict:
        """Calculate optimal tree structure based on number of chunks.
        
        Tree depth based on log20:
        1: root only (1-2 chunks)
        2: root + leaves (3-20 chunks)
        3: root + twigs + leaves (21-400 chunks)
        4: root + branches + twigs + leaves (401-8000 chunks)
        5: root + boughs + branches + twigs + leaves (8001+ chunks)
        
        Returns:
            Dict containing:
            - depth: total tree depth
            - levels: list of level names
            - target_children: children per node at each level
        """
        import math
        
        # Calculate base depth from log20
        log20 = math.log(num_chunks, 20) if num_chunks > 0 else 0
        depth = min(5, max(1, math.ceil(log20) + 1))  # +1 for root level
        
        # Define level names from bottom up
        all_levels = ['leaves', 'twigs', 'branches', 'boughs', 'root']
        levels = all_levels[-depth:]
        
        # Calculate target children per level
        if depth <= 1:
            target_children = [num_chunks]
        else:
            # Calculate geometric progression for balanced tree
            ratio = math.pow(num_chunks, 1/(depth-1))
            target_children = [min(20, max(2, round(ratio)))] * (depth-1)
        
        # Add root (always has all direct children)
        target_children.append(num_chunks if depth == 1 else target_children[0])
        
        return {
            'depth': depth,
            'levels': levels,
            'target_children': target_children
        }

    def _handle_noise_points(self, noise_points, current_nodes, current_embeddings, new_nodes, new_embeddings):
        """Handle noise points from HDBSCAN clustering."""
        if len(noise_points) <= 5:
            # For very few noise points, assign to nearest non-noise cluster
            for point in noise_points:
                nearest_node = self._find_nearest_summary_node(
                    current_nodes[point['index']], 
                    current_embeddings[point['index']], 
                    new_nodes, 
                    new_embeddings
                )
                if nearest_node:
                    nearest_node.children.append(current_nodes[point['index']])
                    current_nodes[point['index']].metadata['parent_id'] = nearest_node.id
        else:
            # Create new small clusters from noise points
            noise_embeddings = current_embeddings[[p['index'] for p in noise_points]]
            noise_nodes = [current_nodes[p['index']] for p in noise_points]
            
            # Use agglomerative clustering for noise points
            from sklearn.cluster import AgglomerativeClustering
            n_clusters = max(1, len(noise_points) // 5)
            agg = AgglomerativeClustering(n_clusters=n_clusters)
            noise_labels = agg.fit_predict(noise_embeddings)
            
            # Create summary nodes for noise clusters
            for label in range(n_clusters):
                cluster_indices = [i for i, l in enumerate(noise_labels) if l == label]
                cluster_nodes = [noise_nodes[i] for i in cluster_indices]
                
                summary = self._generate_hierarchical_summary(cluster_nodes)
                summary_node = SummaryNode(
                    id=str(uuid.uuid4()),
                    text=summary,
                    level=new_nodes[0].level if new_nodes else 1,
                    metadata={'node_type': 'noise_cluster'}
                )
                
                for child in cluster_nodes:
                    child.metadata['parent_id'] = summary_node.id
                    summary_node.children.append(child)
                
                # Use embed_texts instead of embed_text
                summary_embedding = self.embed_manager.embed_texts([summary])[0]
                summary_node.metadata['embedding'] = summary_embedding
                
                new_nodes.append(summary_node)
                new_embeddings.append(summary_embedding)

    def _generate_hierarchical_summary(self, nodes: List[Union[LeafNode, SummaryNode]]) -> str:
        """Generate hierarchical summary for a cluster of nodes."""
        try:
            # Sanitize text before summarization
            def sanitize_text(text: str) -> str:
                if not isinstance(text, str):
                    return ""
                # Remove or replace problematic characters
                text = text.encode('ascii', errors='ignore').decode('ascii')
                # Normalize whitespace
                text = ' '.join(text.split())
                return text
            
            if len(nodes) <= 15:
                texts = [sanitize_text(node.text) for node in nodes if node.text]
                if not texts:
                    return "No valid text to summarize"
                try:
                    return self.summary_manager.summarize("\n\n".join(texts))
                except Exception as e:
                    self.logger.error(f"Failed to summarize texts: {str(e)}")
                    return texts[0][:500] + "..."  # Fallback to first text snippet
            
            # For larger clusters, summarize in hierarchical fashion
            summaries = []
            for i in range(0, len(nodes), 10):
                chunk = nodes[i:i+10]
                texts = [sanitize_text(node.text) for node in chunk if node.text]
                if texts:
                    try:
                        sub_summary = self.summary_manager.summarize("\n\n".join(texts))
                        summaries.append(sub_summary)
                    except Exception as e:
                        self.logger.error(f"Failed to generate sub-summary: {str(e)}")
                        summaries.append(texts[0][:200] + "...")  # Fallback
            
            if not summaries:
                return "Failed to generate summaries"
                
            # Final summary of sub-summaries
            try:
                return self.summary_manager.summarize("\n\n".join(summaries))
            except Exception as e:
                self.logger.error(f"Failed to generate final summary: {str(e)}")
                return "\n\n".join(s[:200] + "..." for s in summaries[:3])  # Fallback
                
        except Exception as e:
            self.logger.error(f"Error in hierarchical summary generation: {str(e)}")
            return "Error generating summary"

    def _find_nearest_summary_node(self, node, node_embedding, summary_nodes, summary_embeddings):
        """Find the nearest summary node for a given node based on embedding similarity.
        
        Args:
            node: The node to find nearest neighbor for
            node_embedding: The embedding of the node
            summary_nodes: List of summary nodes to search in
            summary_embeddings: List of embeddings for summary nodes
            
        Returns:
            The nearest summary node or None if no valid match found
        """
        if not summary_nodes or not summary_embeddings:
            return None
            
        try:
            # Convert to numpy array if needed
            if isinstance(node_embedding, list):
                node_embedding = np.array(node_embedding)
            summary_embeddings = np.array(summary_embeddings)
            
            # Calculate cosine similarities
            similarities = np.dot(summary_embeddings, node_embedding) / (
                np.linalg.norm(summary_embeddings, axis=1) * np.linalg.norm(node_embedding)
            )
            
            # Find most similar summary node
            most_similar_idx = np.argmax(similarities)
            similarity_score = similarities[most_similar_idx]
            
            # Only return if similarity is above threshold
            if similarity_score > 0.5:  # Adjust threshold as needed
                return summary_nodes[most_similar_idx]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding nearest summary node: {str(e)}")
            return None