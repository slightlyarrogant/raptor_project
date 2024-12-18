import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, pairwise_distances
import umap
import warnings
from scipy.sparse import issparse
from scipy.spatial.distance import pdist, squareform
import nltk
from nltk.corpus import stopwords
from collections import Counter, defaultdict
from nltk.tokenize import word_tokenize
import uuid
import time
from src.tree.node import Node

logger = logging.getLogger(__name__)

class TreeStructureError(Exception):
    """Custom exception for tree structure errors."""
    pass

class TreeStructure:
    """A class for building and managing hierarchical tree structures.
    
    This class works with the standardized Node format to ensure consistency
    across the codebase. The tree structure follows these rules:
    1. Each node has a unique ID and maintains proper parent-child relationships
    2. Nodes are organized hierarchically with well-defined levels
    3. All node metadata follows the standard format defined in Node class
    4. Tree operations preserve the integrity of node relationships and metadata
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Default configuration with explanations
        default_config = {
            'dimension': 20,      # Dimension for embeddings
            'threshold': 0.6,     # Similarity threshold for clustering
            'max_levels': 3,      # Maximum tree depth
            'min_cluster_size': 3,  # Minimum documents per cluster
            'max_cluster_size': 50,  # Maximum documents per cluster
            'min_docs_per_cluster': 3,  # Minimum docs to form a cluster
            'target_children': 4,  # Target number of children per node
            'force_merge': True,  # Merge small clusters
            'clustering': {
                'method': 'gmm',  # Clustering algorithm
                'max_clusters': 10,  # Maximum clusters per level
                'min_cluster_size': 3,  # Minimum cluster size
                'silhouette_threshold': 0.1,  # Minimum clustering quality
                'min_coherence': 0.3  # Minimum cluster coherence
            }
        }
        
        # Update config with defaults
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict):
                self.config[key] = {**value, **self.config.get(key, {})}
                
        # Log configuration
        self.logger.info("Tree Structure Configuration:")
        for key, value in self.config.items():
            self.logger.info(f"  {key}: {value}")
        
        # Initialize NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        # Ensure all required parameters are present
        required_params = {
            'dimension': 20,
            'threshold': 0.6,  # Increased from 0.5 for stricter clustering
            'max_levels': 3,
            'min_cluster_size': 3,  # Increased from 2 for better balance
            'max_cluster_size': 50,  # Reduced from 100 to prevent oversized clusters
            'min_docs_per_cluster': 3,  # Increased from 2
            'target_children': 4,  # Increased from 3 for better tree structure
            'force_merge': True,
            'min_similarity': 0.15,  # Increased from 0.05 for tighter clusters
            'decay_rate': 0.85,  # Adjusted from 0.9
            'merge_threshold': 0.25,  # Increased from 0.15 for more selective merging
            'outlier_threshold': 2.0,  # New parameter for outlier detection
            'min_coherence': 0.3,  # New parameter for minimum coherence
            'silhouette_threshold': 0.1  # New parameter for cluster separation
        }
        
        # Set defaults if not provided
        if 'clustering' not in self.config:
            self.config['clustering'] = {}
            
        for param, default in required_params.items():
            if param not in self.config['clustering']:
                self.config['clustering'][param] = default
                self.logger.warning(f"Using default value for {param}: {default}")

    def _validate_config(self):
        """Validate configuration parameters."""
        required_params = ['dimension', 'threshold', 'max_levels']
        for param in required_params:
            if param not in self.config.get('clustering', {}):
                raise TreeStructureError(f"Missing required clustering parameter: {param}")

    def build_tree(self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict], level: int = 0) -> Dict:
        """Build hierarchical tree structure using standardized Node format."""
        try:
            self.logger.info(f"Building level {level} with {len(texts)} documents")
            
            # Create root node for this subtree
            node = Node(
                text=texts[0] if texts else "",
                level=level,
                embedding=embeddings[0] if len(embeddings) > 0 else None
            )
            
            # Update node metadata
            node.metadata.update({
                'level': level,
                'size': len(texts),
                'coherence': self._calculate_coherence(embeddings),
                'is_leaf': len(texts) < self.config['min_cluster_size'],
                'node_type': 'leaf' if len(texts) < self.config['min_cluster_size'] else 'branch',
                'parent_metadata': metadata[0] if metadata else {}
            })
            
            # Base case: create leaf node
            if len(texts) < self.config['min_cluster_size']:
                node.metadata['texts'] = texts
                node.metadata['embeddings'] = embeddings.tolist()
                return node.to_dict()
            
            # Get optimal number of clusters
            n_clusters = self._estimate_optimal_clusters(embeddings)
            self.logger.info(f"Creating {n_clusters} clusters at level {level}")
            
            # Perform clustering
            clusters = self._cluster_documents(embeddings, n_clusters)
            
            # Create child nodes
            for i in range(n_clusters):
                cluster_indices = np.where(clusters == i)[0]
                
                if len(cluster_indices) >= self.config['min_docs_per_cluster']:
                    cluster_texts = [texts[idx] for idx in cluster_indices]
                    cluster_embeddings = embeddings[cluster_indices]
                    cluster_metadata = [metadata[idx] for idx in cluster_indices]
                    
                    # Recursively build subtree
                    child_dict = self.build_tree(
                        cluster_texts,
                        cluster_embeddings,
                        cluster_metadata,
                        level + 1
                    )
                    
                    # Create child node and add to parent
                    child_node = Node.from_dict(child_dict)
                    child_node.parent = node
                    node.children.append(child_node)
            
            return node.to_dict()
            
        except Exception as e:
            self.logger.error(f"Error building tree at level {level}: {str(e)}")
            raise TreeStructureError(f"Failed to build tree: {str(e)}")

    def _preprocess_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Preprocess embeddings with enhanced standardization and outlier handling."""
        try:
            # Convert sparse matrix to dense if needed
            if issparse(embeddings):
                embeddings = embeddings.toarray()

            # Handle NaN values
            if np.isnan(embeddings).any():
                embeddings = np.nan_to_num(embeddings, nan=0.0)

            # Detect and handle outliers
            outlier_threshold = self.config['clustering']['outlier_threshold']
            scaler = StandardScaler()
            scaled = scaler.fit_transform(embeddings)
            
            # Calculate z-scores for outlier detection
            z_scores = np.abs(scaled)
            outlier_mask = (z_scores > outlier_threshold).any(axis=1)
            
            # Replace outliers with mean values
            if outlier_mask.any():
                means = np.mean(embeddings[~outlier_mask], axis=0)
                embeddings[outlier_mask] = means
                self.logger.info(f"Handled {outlier_mask.sum()} outlier embeddings")

            # Normalize to unit length
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms

            return embeddings

        except Exception as e:
            self.logger.error(f"Enhanced preprocessing failed: {str(e)}")
            return embeddings

    def _reduce_dimensionality(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce dimensionality with enhanced UMAP parameters."""
        try:
            n_samples = embeddings.shape[0]
            
            # Adjust parameters based on data size
            n_components = min(
                self.config['clustering']['dimension'],
                embeddings.shape[0] - 2,
                embeddings.shape[1],
                max(3, n_samples // 3)  # Increased minimum components
            )
            
            # More sophisticated neighbor calculation
            n_neighbors = min(
                n_samples - 1,
                max(5, int(np.sqrt(n_samples) * 1.5))  # Increased neighbor count
            )

            # Enhanced UMAP configuration
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=0.2,  # Increased from 0.1 for better separation
                metric='cosine',
                random_state=42,
                low_memory=True,
                densmap=True,  # Enable density-aware mapping
                output_dens=True,  # Include density information
                n_epochs=500,  # Increased epochs for better convergence
                learning_rate=0.5,  # Adjusted learning rate
                init='spectral',  # Use spectral initialization
                verbose=True
            )

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                reduced = reducer.fit_transform(embeddings)

            return reduced

        except Exception as e:
            self.logger.warning(f"Enhanced dimensionality reduction failed: {str(e)}")
            return embeddings

    def _estimate_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """Estimate optimal number of clusters."""
        try:
            n_samples = len(embeddings)
            
            if n_samples == 0:
                self.logger.warning("No samples to cluster")
                return 1
                
            if n_samples == 1:
                self.logger.warning("Single sample, no clustering needed")
                return 1
                
            # Get configuration parameters with defaults
            min_cluster_size = self.config['clustering'].get('min_cluster_size', 2)
            max_cluster_size = self.config['clustering'].get('max_cluster_size', 100)
            target_children = self.config['clustering'].get('target_children', 3)
            
            # Calculate optimal number of clusters
            max_clusters = min(
                max(2, n_samples // min_cluster_size),  # At least 2 samples per cluster
                max_cluster_size,  # Not more than max_cluster_size
                n_samples - 1  # Not more than n-1 clusters
            )
            
            optimal_clusters = min(
                target_children,  # Target number of children
                max_clusters  # Maximum allowed clusters
            )
            
            self.logger.info(f"Estimated {optimal_clusters} clusters for {n_samples} samples")
            return optimal_clusters
            
        except Exception as e:
            self.logger.error(f"Error estimating clusters: {str(e)}")
            return 1  # Safe fallback

    def _cluster_documents(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform clustering with validation."""
        try:
            self.logger.info(f"Starting clustering with {n_clusters} clusters for {len(embeddings)} documents")
            start_time = time.time()
            
            gmm = GaussianMixture(
                n_components=n_clusters,
                covariance_type='full',
                random_state=42,
                n_init=5  # Multiple initializations
            )
            
            self.logger.info("Fitting GMM model...")
            labels = gmm.fit_predict(embeddings)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Clustering completed in {elapsed:.2f} seconds")
            
            # Log cluster sizes
            unique_labels, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique_labels, counts):
                self.logger.info(f"Cluster {label} size: {count}")
                
            return labels

        except Exception as e:
            self.logger.error(f"Clustering failed: {str(e)}")
            return np.zeros(len(embeddings))

    def _validate_clustering(self, embeddings: np.ndarray, labels: np.ndarray) -> bool:
        """Enhanced clustering validation with stricter quality metrics."""
        try:
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            # Get configuration parameters
            min_cluster_size = self.config['clustering'].get('min_cluster_size', 3)
            max_cluster_size = self.config['clustering'].get('max_cluster_size', 50)
            silhouette_threshold = self.config['clustering'].get('silhouette_threshold', 0.1)
            min_coherence = self.config['clustering'].get('min_coherence', 0.3)
            
            # Size validation
            if any(count < min_cluster_size for count in counts):
                self.logger.warning(f"Some clusters too small (min size: {min_cluster_size})")
                return False
                
            if any(count > max_cluster_size for count in counts):
                self.logger.warning(f"Some clusters too large (max size: {max_cluster_size})")
                return False

            # Calculate multiple quality metrics
            silhouette_avg = silhouette_score(embeddings, labels, metric='cosine')
            calinski_avg = calinski_harabasz_score(embeddings, labels)
            
            # Check silhouette score
            if silhouette_avg < silhouette_threshold:
                self.logger.warning(f"Poor cluster separation (silhouette: {silhouette_avg:.3f})")
                return False
                
            # Check Calinski-Harabasz score (should be higher for better clustering)
            if calinski_avg < 10:  # Typical threshold for reasonable clustering
                self.logger.warning(f"Poor cluster density (Calinski-Harabasz: {calinski_avg:.3f})")
                return False

            # Enhanced coherence check for each cluster
            for label in unique_labels:
                mask = labels == label
                cluster_embeddings = embeddings[mask]
                
                # Calculate both cosine and density-based coherence
                cosine_coherence = self._calculate_coherence(cluster_embeddings)
                density_coherence = self._calculate_density_coherence(cluster_embeddings)
                
                # Combined coherence score
                combined_coherence = 0.7 * cosine_coherence + 0.3 * density_coherence
                
                if combined_coherence < min_coherence:
                    self.logger.warning(f"Low coherence in cluster {label}: {combined_coherence:.3f}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Enhanced clustering validation failed: {str(e)}")
            return False

    def _calculate_coherence(self, embeddings: np.ndarray) -> float:
        """Enhanced coherence calculation with weighted similarity."""
        try:
            if len(embeddings) == 0:
                self.logger.warning("Empty embeddings array, returning default coherence")
                return 0.0
                
            if len(embeddings) == 1:
                self.logger.warning("Single embedding, returning maximum coherence")
                return 1.0
                
            # Calculate pairwise similarities with cosine metric
            similarities = np.dot(embeddings, embeddings.T)
            
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1)
            norms[norms == 0] = 1e-10  # Avoid division by zero
            similarities = similarities / np.outer(norms, norms)
            
            # Apply distance-based weights
            distances = squareform(pdist(embeddings, metric='euclidean'))
            weights = np.exp(-distances / distances.mean())  # Gaussian weighting
            
            # Calculate weighted mean similarity excluding self-similarity
            np.fill_diagonal(weights, 0)
            np.fill_diagonal(similarities, 0)
            
            weighted_sum = np.sum(similarities * weights)
            weight_sum = np.sum(weights)
            
            if weight_sum == 0:
                return 0.0
                
            return float(weighted_sum / weight_sum)
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced coherence: {str(e)}")
            return 0.0

    def _calculate_density_coherence(self, embeddings: np.ndarray) -> float:
        """Calculate density-based coherence using local density estimation."""
        try:
            if len(embeddings) < 2:
                return 1.0 if len(embeddings) == 1 else 0.0
                
            # Calculate pairwise distances
            distances = squareform(pdist(embeddings, metric='euclidean'))
            
            # Estimate local density using gaussian kernel
            bandwidth = np.mean(distances) / 2
            densities = np.zeros(len(embeddings))
            
            for i in range(len(embeddings)):
                # Calculate gaussian kernel density
                kernel_values = np.exp(-(distances[i] ** 2) / (2 * bandwidth ** 2))
                densities[i] = np.mean(kernel_values)
            
            # Normalize densities to [0,1]
            if np.ptp(densities) == 0:
                return 1.0 if np.mean(densities) > 0 else 0.0
                
            normalized_density = (densities - np.min(densities)) / np.ptp(densities)
            
            return float(np.mean(normalized_density))
            
        except Exception as e:
            self.logger.error(f"Error calculating density coherence: {str(e)}")
            return 0.0

    def _extract_cluster_keyword(self, texts: List[str]) -> str:
        """Extract representative keyword for cluster naming."""
        try:
            # Combine all texts
            combined_text = " ".join(texts)
            
            # Tokenize and get word frequencies
            words = word_tokenize(combined_text.lower())
            stop_words = set(stopwords.words('english'))
            
            # Filter out stopwords and short words
            words = [word for word in words 
                    if word not in stop_words 
                    and len(word) > 3 
                    and word.isalnum()]
            
            # Get word frequencies
            word_freq = Counter(words)
            
            # Get most common word as keyword
            if word_freq:
                keyword = max(word_freq.items(), key=lambda x: x[1])[0]
                return keyword.title()
            return "Misc"
            
        except Exception as e:
            self.logger.error(f"Failed to extract keyword: {str(e)}")
            return "Unknown"

    def _create_cluster_nodes(self, texts: List[str], embeddings: np.ndarray, 
                             labels: np.ndarray, level: int) -> List[Dict]:
        """Create nodes for each cluster."""
        nodes = []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            mask = labels == label
            if not any(mask):
                continue

            cluster_texts = [t for t, m in zip(texts, mask) if m]
            cluster_embeddings = embeddings[mask]

            # Only create node if we have enough documents
            if len(cluster_texts) > 1:
                # Extract keyword for cluster naming
                keyword = self._extract_cluster_keyword(cluster_texts)
                
                # Recursively build subtree
                subtree = self.build_tree(cluster_texts, cluster_embeddings, level + 1)

                node = {
                    "id": f"{keyword}_{level}_{label}",  # Include keyword in ID
                    "keyword": keyword,  # Store keyword separately
                    "texts": cluster_texts,
                    "metadata": {
                        "size": len(cluster_texts),
                        "level": level,
                        "cluster_id": int(label),
                        "coherence": self._calculate_coherence(cluster_embeddings),
                        "keyword": keyword  # Include in metadata
                    },
                    "nodes": subtree.get("nodes", [])
                }
                nodes.append(node)

        return nodes

    def _create_leaf_node(self, texts: List[str], embeddings: np.ndarray, level: int) -> Dict:
        """Create a leaf node with the given texts."""
        return {
            "nodes": [],
            "texts": texts,
            "metadata": {
                "size": len(texts),
                "level": level,
                "is_leaf": True,
                "coherence": self._calculate_coherence(embeddings)
            }
        }

    def _merge_small_clusters(self, clusters: List[Dict]) -> List[Dict]:
        """Merge small clusters into larger ones."""
        min_size = self.config['clustering']['min_docs_per_cluster']
        
        # Sort clusters by size
        clusters.sort(key=lambda x: len(x.get('texts', [])), reverse=True)
        
        # Keep track of clusters to merge
        merged_clusters = []
        small_clusters = []
        
        for cluster in clusters:
            if len(cluster.get('texts', [])) >= min_size:
                merged_clusters.append(cluster)
            else:
                small_clusters.append(cluster)
        
        # Merge small clusters into nearest larger cluster
        for small in small_clusters:
            if not merged_clusters:
                merged_clusters.append(small)
                continue
                
            # Find closest large cluster
            best_similarity = -1
            best_match = None
            
            for large in merged_clusters:
                sim = self._calculate_cluster_similarity(small, large)
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = large
            
            # Merge if we found a match
            if best_match and best_similarity > self.config['clustering']['min_similarity']:
                best_match['texts'].extend(small.get('texts', []))
                # Update metadata
                best_match['metadata']['size'] = len(best_match['texts'])
                self.logger.warning(f"Merged cluster of size {len(small['texts'])} into cluster of size {len(best_match['texts'])}")
            else:
                merged_clusters.append(small)
                self.logger.warning(f"Kept small cluster of size {len(small['texts'])} (no good merge candidate)")
        
        return merged_clusters

    def _calculate_cluster_similarity(self, cluster1: Dict, cluster2: Dict) -> float:
        """Calculate similarity between two clusters."""
        try:
            # Get cluster centroids
            centroid1 = np.mean([self.embeddings[self.texts.index(t)] for t in cluster1['texts']], axis=0)
            centroid2 = np.mean([self.embeddings[self.texts.index(t)] for t in cluster2['texts']], axis=0)
            
            # Calculate cosine similarity
            similarity = np.dot(centroid1, centroid2) / (np.linalg.norm(centroid1) * np.linalg.norm(centroid2))
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error calculating cluster similarity: {str(e)}")
            return 0.0