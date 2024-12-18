import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Union
from src.utils.config import DEFAULT_CONFIG
from sklearn.mixture import GaussianMixture
import umap.umap_ as umap
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import time
import hdbscan
from src.tree.node import LeafNode, SummaryNode

logger = logging.getLogger(__name__)

class ClusterManager:
    """Manager for clustering documents."""

    def __init__(self, config: Dict):
        """Initialize the cluster manager.
        
        Args:
            config: Configuration for clustering
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ClusterManager...")
        
        if not isinstance(config, dict):
            raise ValueError(f"Expected dict, got {type(config)}")
            
        self.config = config
        
        # Initialize parameters with better defaults for large datasets
        self.dimension = config.get('dimension', 2)
        self.threshold = config.get('threshold', 0.5)
        self.max_levels = config.get('max_levels', 4)  # Fixed at 4 levels
        self.min_cluster_size = config.get('min_cluster_size', 10)  # Adjusted for better granularity
        self.target_children = config.get('target_children', 4)  # Optimized for 4-level structure
        
        # Parameters for hierarchical clustering
        self.base_threshold = config.get('base_threshold', 0.7)  # Increased for more selective merging
        self.min_threshold = config.get('min_threshold', 0.3)   # Adjusted minimum threshold
        self.decay_rate = config.get('decay_rate', 0.8)      # Faster decay for 4 levels
        
        # New parameters for balanced clustering
        self.max_cluster_ratio = config.get('max_cluster_ratio', 0.5)   # Reduced for more uniform sizes
        self.min_cluster_ratio = config.get('min_cluster_ratio', 0.1)  # Increased minimum ratio
        self.target_cluster_size = config.get('target_cluster_size', 20)  # Optimized for 4-level structure
        self.rebalance_threshold = config.get('rebalance_threshold', 0.2)  # More aggressive rebalancing
        
        self.logger.info(f"Using dimension: {self.dimension}")
        self.logger.info(f"Using threshold: {self.threshold}")
        self.logger.info(f"Using max_levels: {self.max_levels}")
        self.logger.info(f"Using min_cluster_size: {self.min_cluster_size}")
        self.logger.info(f"Using target_children: {self.target_children}")
        self.logger.info("✓ ClusterManager initialized")

    def cluster_embeddings(self, embeddings, n_clusters, return_centers=False):
        """Cluster embeddings using KMeans.
        
        Args:
            embeddings: List of embeddings to cluster
            n_clusters: Number of clusters to create
            return_centers: Whether to return cluster centers
            
        Returns:
            If return_centers is False: List of cluster labels
            If return_centers is True: Tuple of (labels, centers)
        """
        try:
            # Convert embeddings to numpy array if needed
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            
            # Ensure embeddings are 2D
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # Check if we have enough samples to cluster
            if len(embeddings) < n_clusters:
                self.logger.warning(f"Not enough samples ({len(embeddings)}) for {n_clusters} clusters")
                if return_centers:
                    return np.zeros(len(embeddings)), np.array([np.mean(embeddings, axis=0)])
                return np.zeros(len(embeddings))
                
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            
            # Check if any clusters are empty
            unique_labels = np.unique(labels)
            if len(unique_labels) < n_clusters:
                self.logger.warning("Some clusters are empty, trying again with different initialization")
                kmeans = KMeans(n_clusters=n_clusters, random_state=None)  # Random initialization
                labels = kmeans.fit_predict(embeddings)
            
            if return_centers:
                return labels, kmeans.cluster_centers_
            return labels
            
        except Exception as e:
            self.logger.error(f"Error during clustering: {str(e)}")
            if return_centers:
                return np.zeros(len(embeddings)), np.array([np.mean(embeddings, axis=0)])
            return np.zeros(len(embeddings))

    def _calculate_optimal_clusters(self, n_samples: int, min_clusters: int, max_clusters: int) -> int:
        """Calculate optimal number of clusters based on dataset size."""
        if n_samples >= 10000:
            target_size = 75  # For very large datasets (10k+)
        elif n_samples >= 5000:
            target_size = 50  # For large datasets (5k-10k)
        elif n_samples >= 1000:
            target_size = 35  # For medium datasets (1k-5k)
        else:
            target_size = 20  # For small datasets (<1k)
            
        optimal_clusters = min(
            max(min_clusters, n_samples // target_size),
            min(max_clusters, n_samples // 8)  # Allow up to 1/8th of samples
        )
        
        self.logger.info(f"Target size per cluster: ~{target_size} documents")
        return optimal_clusters
        
    def _perform_clustering(self, embeddings: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, int]:
        """Perform KMeans clustering with given parameters."""
        if n_clusters >= len(embeddings):
            self.logger.warning(f"Number of clusters ({n_clusters}) >= number of samples ({len(embeddings)})")
            # Return single cluster containing all points
            return np.zeros(len(embeddings)), 1
            
        clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,  # Increased from 5
            max_iter=300
        )
        
        try:
            self.logger.info(f"Fitting KMeans model on {len(embeddings)} samples into {n_clusters} clusters...")
            start_time = time.time()
            labels = clusterer.fit_predict(embeddings)
            
            # Check for empty clusters and adjust if needed
            unique_labels, counts = np.unique(labels, return_counts=True)
            empty_clusters = n_clusters - len(unique_labels)
            
            if empty_clusters > 0:
                self.logger.warning(f"Found {empty_clusters} empty clusters, adjusting...")
                # Reassign points from largest clusters to empty ones
                for empty_idx in range(n_clusters):
                    if empty_idx not in unique_labels:
                        # Find largest cluster
                        largest_cluster = unique_labels[np.argmax(counts)]
                        largest_cluster_points = np.where(labels == largest_cluster)[0]
                        
                        # Take half of points from largest cluster
                        points_to_move = largest_cluster_points[:len(largest_cluster_points)//2]
                        labels[points_to_move] = empty_idx
                        
                        # Update counts
                        unique_labels, counts = np.unique(labels, return_counts=True)
                
            self.logger.info(f"Clustering completed in {time.time() - start_time:.2f} seconds")
            self.logger.info(f"Final cluster sizes: {dict(zip(unique_labels, counts))}")
            return labels, len(unique_labels)
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {str(e)}", exc_info=True)
            return np.zeros(len(embeddings)), 1

    def _create_base_clusters(self, embeddings: np.ndarray, texts: List[str]) -> List[Dict]:
        """Create initial clusters with proper tracking."""
        try:
            # Track processed nodes to avoid duplicates
            processed_nodes = set()
            clusters = []
            
            # Get cluster assignments
            assignments = self._get_cluster_assignments(embeddings)
            
            # Group by cluster
            cluster_groups = {}
            for idx, (text, embedding, assignment) in enumerate(zip(texts, embeddings, assignments)):
                if idx not in processed_nodes:
                    cluster_id = f"cluster_0_{assignment}"
                    if cluster_id not in cluster_groups:
                        cluster_groups[cluster_id] = {
                            'texts': [],
                            'embeddings': [],
                            'indices': []
                        }
                    cluster_groups[cluster_id]['texts'].append(text)
                    cluster_groups[cluster_id]['embeddings'].append(embedding)
                    cluster_groups[cluster_id]['indices'].append(idx)
                    processed_nodes.add(idx)
            
            # Create cluster objects
            for cluster_id, group in cluster_groups.items():
                cluster = {
                    'id': cluster_id,
                    'level': 0,
                    'texts': group['texts'],
                    'embeddings': group['embeddings'],
                    'indices': group['indices'],
                    'centroid': np.mean(group['embeddings'], axis=0),
                    'size': len(group['texts']),
                    'children': [],
                    'parent_id': None
                }
                clusters.append(cluster)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error creating base clusters: {str(e)}")
            raise
            
    def _build_hierarchy(self, base_clusters: List[Dict], embeddings: np.ndarray) -> List[Dict]:
        """Build hierarchical structure from base clusters."""
        try:
            all_nodes = base_clusters.copy()
            current_level = 0
            next_cluster_id = len(base_clusters)
            
            while len([c for c in all_nodes if c['level'] == current_level]) > 1:
                level_clusters = [c for c in all_nodes if c['level'] == current_level]
                logger.warning(f"Building level {current_level + 1} from {len(level_clusters)} clusters")
                
                # Calculate similarities between all clusters at this level
                centroids = np.array([c['centroid'] for c in level_clusters])
                similarities = cosine_similarity(centroids)
                
                # Adaptive threshold based on level and similarity distribution
                sim_threshold = max(
                    self.base_threshold * (self.decay_rate ** current_level),
                    self.min_threshold
                )
                
                logger.warning(f"Level {current_level} threshold: {sim_threshold:.3f}")
                
                # Group clusters into parent nodes
                parent_nodes = []
                used_clusters = set()
                
                # First pass: group most similar clusters
                for i in range(len(level_clusters)):
                    if i in used_clusters:
                        continue
                        
                    # Find most similar clusters
                    similar_indices = []
                    for j in range(len(level_clusters)):
                        if j not in used_clusters and i != j:
                            if similarities[i, j] > sim_threshold:
                                similar_indices.append((j, similarities[i, j]))
                    
                    # Sort by similarity
                    similar_indices.sort(key=lambda x: x[1], reverse=True)
                    
                    # Take top N most similar clusters, but don't exceed target_children
                    group_size = min(self.target_children, len(similar_indices) + 1)
                    group_indices = [i] + [idx for idx, _ in similar_indices[:group_size-1]]
                    
                    if len(group_indices) > 1:
                        used_clusters.update(group_indices)
                        child_clusters = [level_clusters[idx] for idx in group_indices]
                        
                        # Collect all documents
                        all_docs = []
                        for c in child_clusters:
                            all_docs.extend(c['docs'])
                        
                        # Create parent node
                        parent = {
                            'id': f"cluster_{current_level+1}_{next_cluster_id}",
                            'level': current_level + 1,
                            'docs': all_docs,
                            'centroid': np.mean([c['centroid'] for c in child_clusters], axis=0),
                            'children': [c['id'] for c in child_clusters],
                            'parent_id': None,
                            'similarity_score': np.mean([similarities[i, j] for i, j in zip(group_indices[:-1], group_indices[1:])])
                        }
                        
                        # Update child references
                        for c in child_clusters:
                            c['parent_id'] = parent['id']
                        
                        parent_nodes.append(parent)
                        next_cluster_id += 1
                
                # Second pass: handle remaining clusters
                remaining = [c for i, c in enumerate(level_clusters) if i not in used_clusters]
                if remaining:
                    # If we have just a few remaining clusters, merge them into one parent
                    if len(remaining) <= self.target_children:
                        all_docs = []
                        for c in remaining:
                            all_docs.extend(c['docs'])
                        
                        parent = {
                            'id': f"cluster_{current_level+1}_{next_cluster_id}",
                            'level': current_level + 1,
                            'docs': all_docs,
                            'centroid': np.mean([c['centroid'] for c in remaining], axis=0),
                            'children': [c['id'] for c in remaining],
                            'parent_id': None,
                            'similarity_score': self._calculate_similarity(embeddings[all_docs])
                        }
                        
                        for c in remaining:
                            c['parent_id'] = parent['id']
                        
                        parent_nodes.append(parent)
                        next_cluster_id += 1
                    else:
                        # Otherwise, recursively cluster the remaining nodes
                        remaining_embeddings = np.array([c['centroid'] for c in remaining])
                        remaining_labels, n_remaining_clusters = self.cluster_embeddings(
                            remaining_embeddings,
                            min_clusters=max(2, len(remaining) // self.target_children),
                            max_clusters=max(3, len(remaining) // 2)
                        )
                        
                        # Group by cluster
                        remaining_groups = {}
                        for i, label in enumerate(remaining_labels):
                            if label not in remaining_groups:
                                remaining_groups[label] = []
                            remaining_groups[label].append(remaining[i])
                        
                        # Create parent nodes for each group
                        for group in remaining_groups.values():
                            if len(group) > 0:
                                all_docs = []
                                for c in group:
                                    all_docs.extend(c['docs'])
                                
                                parent = {
                                    'id': f"cluster_{current_level+1}_{next_cluster_id}",
                                    'level': current_level + 1,
                                    'docs': all_docs,
                                    'centroid': np.mean([c['centroid'] for c in group], axis=0),
                                    'children': [c['id'] for c in group],
                                    'parent_id': None,
                                    'similarity_score': self._calculate_similarity(embeddings[all_docs])
                                }
                                
                                for c in group:
                                    c['parent_id'] = parent['id']
                                
                                parent_nodes.append(parent)
                                next_cluster_id += 1
                
                if parent_nodes:
                    all_nodes.extend(parent_nodes)
                    current_level += 1
                    
                    # Log the new level structure
                    level_sizes = {}
                    for node in all_nodes:
                        level = node['level']
                        if level not in level_sizes:
                            level_sizes[level] = 0
                        level_sizes[level] += 1
                    
                    self.logger.info("\nCurrent tree structure:")
                    for level, size in sorted(level_sizes.items()):
                        self.logger.info(f"Level {level}: {size} nodes")
                else:
                    # If we couldn't create any parent nodes, we're done
                    break
                
                # Check if we've reached the maximum number of levels
                if current_level >= self.max_levels - 1:
                    self.logger.info(f"Reached maximum number of levels ({self.max_levels})")
                    break
            
            return all_nodes
            
        except Exception as e:
            logger.error(f"Error building hierarchy: {str(e)}")
            raise

    def cluster_texts(self, texts: List[str], embeddings: np.ndarray, min_clusters: int = 2, max_clusters: int = 3) -> Dict[int, List[int]]:
        """Cluster texts using their embeddings."""
        try:
            self.logger.info(f"Starting text clustering for {len(texts)} texts")
            start_time = time.time()
            
            # Input validation
            if len(texts) != embeddings.shape[0]:
                error_msg = f"Number of texts ({len(texts)}) doesn't match number of embeddings ({embeddings.shape[0]})"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Get cluster assignments
            self.logger.info("Getting cluster assignments...")
            labels, n_clusters = self.cluster_embeddings(embeddings, min_clusters, max_clusters)
            
            # Group indices by cluster
            self.logger.info("Grouping texts by cluster...")
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)
            
            # Log cluster sizes
            for cluster_id, indices in clusters.items():
                self.logger.info(f"Cluster {cluster_id}: {len(indices)} texts")
            
            elapsed = time.time() - start_time
            self.logger.info(f"Text clustering completed in {elapsed:.2f} seconds")
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error during text clustering: {str(e)}", exc_info=True)
            raise

    def _get_cluster_assignments(self, embeddings: np.ndarray) -> np.ndarray:
        """Get cluster assignments for embeddings."""
        try:
            self.logger.info(f"Getting cluster assignments for {len(embeddings)} embeddings")
            start_time = time.time()
            
            # Preprocess embeddings
            self.logger.info("Preprocessing embeddings...")
            preprocessed = self._preprocess_embeddings(embeddings)
            
            # Reduce dimensionality if needed
            if preprocessed.shape[1] > self.dimension:
                self.logger.info(f"Reducing dimensionality from {preprocessed.shape[1]} to {self.dimension}")
                preprocessed = self._reduce_dimensions(preprocessed, n_components=self.dimension)
            
            # Get optimal number of clusters
            n_clusters = self._estimate_optimal_clusters(preprocessed)
            self.logger.info(f"Using {n_clusters} clusters")
            
            # Perform clustering
            self.logger.info("Performing clustering...")
            labels = self._perform_clustering(preprocessed, n_clusters)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Cluster assignment completed in {elapsed:.2f} seconds")
            
            # Validate clustering
            self._validate_clustering(preprocessed, labels)
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Error getting cluster assignments: {str(e)}", exc_info=True)
            raise

    def cluster_nodes(self, nodes: List[Union['LeafNode', 'SummaryNode']], embeddings: np.ndarray, target_clusters: int = None) -> List[List[int]]:
        """Cluster nodes based on their embeddings."""
        try:
            if target_clusters is None:
                target_clusters = max(2, len(nodes) // 20)  # Aim for ~20 nodes per cluster
                
            self.logger.info(f"Clustering {len(nodes)} nodes into {target_clusters} clusters")
            
            # Check for NaN values
            if np.isnan(embeddings).any():
                self.logger.warning("Found NaN values in embeddings, cleaning data...")
                # Get indices of rows without NaN
                valid_indices = ~np.isnan(embeddings).any(axis=1)
                clean_embeddings = embeddings[valid_indices]
                
                if len(clean_embeddings) < len(embeddings):
                    self.logger.warning(f"Removed {len(embeddings) - len(clean_embeddings)} embeddings with NaN values")
                    
                if len(clean_embeddings) < target_clusters:
                    self.logger.error("Too few valid embeddings for requested number of clusters")
                    # Return single cluster with all indices as fallback
                    return [list(range(len(nodes)))]
            else:
                clean_embeddings = embeddings
                valid_indices = np.ones(len(embeddings), dtype=bool)
            
            # Normalize embeddings
            norms = np.linalg.norm(clean_embeddings, axis=1)[:, np.newaxis]
            norms[norms == 0] = 1  # Avoid division by zero
            normalized_embeddings = clean_embeddings / norms
            
            # Use KMeans for clustering
            kmeans = KMeans(
                n_clusters=min(target_clusters, len(clean_embeddings)),
                random_state=42,
                n_init=10
            )
            
            cluster_labels = kmeans.fit_predict(normalized_embeddings)
            
            # Map back to original indices
            full_labels = np.zeros(len(embeddings), dtype=int)
            full_labels[valid_indices] = cluster_labels
            
            # Group node indices by cluster
            clusters = [[] for _ in range(max(cluster_labels) + 1)]
            for i, label in enumerate(full_labels):
                clusters[label].append(i)
            
            # Log cluster sizes
            sizes = [len(c) for c in clusters]
            self.logger.info(f"Cluster sizes: {sizes}")
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Clustering failed: {str(e)}")
            # Return single cluster as fallback
            return [list(range(len(nodes)))]

    def cluster_documents(self, docs: List[str], embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Cluster documents into groups with balanced sizes.
        
        Args:
            docs: List of document texts
            embeddings: Document embeddings
            n_clusters: Target number of clusters
            
        Returns:
            Array of cluster labels for each document
        """
        if len(docs) < n_clusters:
            self.logger.warning(f"Number of documents ({len(docs)}) less than requested clusters ({n_clusters})")
            n_clusters = max(2, len(docs) // 2)
        
        # Initialize KMeans
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        
        # Get initial cluster assignments
        labels = kmeans.fit_predict(embeddings)
        unique_labels = np.unique(labels)
        
        # Check cluster sizes
        cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
        min_size = self.min_cluster_size
        max_size = len(docs) // 2  # No cluster should have more than half the documents
        
        # Identify clusters that need rebalancing
        small_clusters = [l for l, size in cluster_sizes.items() if size < min_size]
        large_clusters = [l for l, size in cluster_sizes.items() if size > max_size]
        
        if small_clusters or large_clusters:
            self.logger.info("Rebalancing clusters...")
            
            # Handle small clusters first
            for small_label in small_clusters:
                small_indices = np.where(labels == small_label)[0]
                
                # Find closest large cluster for each document
                for idx in small_indices:
                    doc_embedding = embeddings[idx]
                    
                    # Calculate similarity to cluster centers
                    similarities = []
                    for label in unique_labels:
                        if label not in small_clusters:
                            cluster_docs = embeddings[labels == label]
                            center = np.mean(cluster_docs, axis=0)
                            sim = np.dot(doc_embedding, center)
                            similarities.append((label, sim))
                    
                    # Assign to most similar cluster
                    if similarities:
                        best_label = max(similarities, key=lambda x: x[1])[0]
                        labels[idx] = best_label
            
            # Handle large clusters
            for large_label in large_clusters:
                large_indices = np.where(labels == large_label)[0]
                cluster_docs = embeddings[large_indices]
                
                # Perform sub-clustering
                sub_n_clusters = len(large_indices) // min_size
                sub_kmeans = KMeans(n_clusters=sub_n_clusters, random_state=42)
                sub_labels = sub_kmeans.fit_predict(cluster_docs)
                
                # Assign new cluster labels
                next_label = max(unique_labels) + 1
                for i, sub_label in enumerate(sub_labels):
                    if sub_label > 0:  # Keep first subcluster with original label
                        labels[large_indices[i]] = next_label + sub_label - 1
        
        return labels

    def store_clusters(self, store_manager, clusters: List[Dict]) -> None:
        """Store clusters using the store manager."""
        try:
            self.logger.info(f"Storing {len(clusters)} clusters")
            store_manager.store_clusters(clusters)
            self.logger.info("✓ Clusters stored successfully")
        except Exception as e:
            self.logger.error(f"Error storing clusters: {str(e)}")
            raise

    def build_tree(self, embeddings: List[List[float]]) -> Dict:
        """Build tree structure from embeddings."""
        try:
            self.logger.info(f"Building tree from {len(embeddings)} embeddings")
            
            # Convert embeddings to numpy array if needed
            embeddings_array = np.array(embeddings)
            
            # Get cluster assignments
            cluster_labels, n_clusters = self.cluster_embeddings(embeddings_array)
            
            # Initialize tree data structure in the format expected by TreeVisualizer
            tree_data = {
                'tree': {
                    'nodes': [],
                    'clusters': {i: label for i, label in enumerate(cluster_labels)}
                }
            }
            
            # Process each cluster
            for cluster_id in range(n_clusters):
                # Get indices for this cluster
                indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                # Get embeddings for this cluster
                cluster_embeddings = [embeddings[i] for i in indices]
                
                # Create cluster node in the format expected by TreeVisualizer
                node = {
                    'id': f"cluster_{cluster_id}",
                    'level': 0,
                    'texts': [str(i) for i in indices],  # Placeholder texts, will be replaced by actual texts
                    'metadata': {
                        'cluster_id': cluster_id,
                        'level': 0,
                        'size': len(indices),
                        'is_leaf': True,  # All nodes are leaves in flat clustering
                        'embeddings': cluster_embeddings,
                        'indices': indices
                    }
                }
                
                tree_data['tree']['nodes'].append(node)
            
            if not tree_data['tree']['nodes']:
                raise ValueError("No nodes created during tree building")
                
            return tree_data
            
        except Exception as e:
            self.logger.error(f"Failed to build tree: {str(e)}")
            raise  # Re-raise the exception to crash the code

    def _reduce_dimensions(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Reduce dimensions of embeddings using UMAP with thread-safe configuration."""
        try:
            # Configure UMAP for thread safety
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine',
                random_state=42,
                n_jobs=1,  # Force single-threaded operation
                low_memory=True
            )
            
            # Perform reduction with progress logging
            self.logger.info(f"Reducing {len(embeddings)} embeddings to {n_components} dimensions...")
            reduced = reducer.fit_transform(embeddings)
            self.logger.info("✓ Dimension reduction complete")
            
            return reduced
            
        except Exception as e:
            self.logger.error(f"Error in dimension reduction: {str(e)}")
            raise

    def group_by_similarity(self, embeddings: np.ndarray, threshold: float = 0.5) -> List[List[int]]:
        """Group documents by similarity."""
        # TODO: Implement similarity-based grouping
        return [[0]]  # Placeholder implementation
        
    def cluster_documents(self, embeddings: np.ndarray, n_clusters: int) -> List[List[int]]:
        """Cluster documents into groups."""
        # TODO: Implement document clustering
        return [[i] for i in range(min(len(embeddings), n_clusters))]