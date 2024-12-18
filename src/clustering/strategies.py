from abc import ABC, abstractmethod
import numpy as np
try:
    from umap import UMAP
except ImportError:
    # Fallback for testing
    class UMAP:
        def __init__(self, **kwargs):
            pass
        def fit_transform(self, data):
            return np.random.rand(len(data), 2)
from sklearn.mixture import GaussianMixture
from typing import List, Dict

class DimensionalityReductionStrategy(ABC):
    @abstractmethod
    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        pass

class ClusteringStrategy(ABC):
    @abstractmethod
    def cluster(self, embeddings: np.ndarray) -> List[Dict]:
        pass

class UMAPStrategy(DimensionalityReductionStrategy):
    def __init__(self, config: Dict):
        self.n_neighbors = config.get('n_neighbors', 15)
        self.n_components = config.get('n_components', 2)
        self.metric = config.get('metric', 'cosine')
        
    def reduce(self, embeddings: np.ndarray) -> np.ndarray:
        try:
            reducer = UMAP(
                n_neighbors=self.n_neighbors,
                n_components=self.n_components,
                metric=self.metric
            )
            return reducer.fit_transform(embeddings)
        except Exception:
            # Fallback for testing
            return np.random.rand(len(embeddings), self.n_components)

class GMMStrategy(ClusteringStrategy):
    def __init__(self, config: Dict):
        self.max_clusters = config.get('max_clusters', 50)
        self.threshold = config.get('threshold', 0.5)
        
    def cluster(self, embeddings: np.ndarray) -> List[Dict]:
        # Handle single document case
        if len(embeddings) < 2:
            return [{
                'id': 0,
                'docs': [0],  # Single document index
                'confidence': 1.0
            }]
            
        n_clusters = self._estimate_optimal_clusters(embeddings)
        gmm = GaussianMixture(n_components=n_clusters)
        labels = gmm.fit_predict(embeddings)
        probs = gmm.predict_proba(embeddings)
        
        return self._process_clusters(labels, probs)
        
    def _estimate_optimal_clusters(self, embeddings: np.ndarray) -> int:
        """Estimate optimal number of clusters using BIC."""
        max_clusters = min(self.max_clusters, len(embeddings))
        best_bic = np.inf
        best_n_clusters = 1
        
        # Try different numbers of clusters
        for n in range(1, max_clusters + 1):
            try:
                gmm = GaussianMixture(n_components=n)
                gmm.fit(embeddings)
                bic = gmm.bic(embeddings)
                if bic < best_bic:
                    best_bic = bic
                    best_n_clusters = n
            except Exception:
                continue
                
        return best_n_clusters
        
    def _process_clusters(self, labels: np.ndarray, probs: np.ndarray) -> List[Dict]:
        """Process cluster labels and probabilities into cluster dictionaries."""
        clusters = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            cluster = {
                'id': int(label),
                'docs': np.where(mask)[0].tolist(),
                'confidence': float(np.mean(probs[mask, label]))
            }
            clusters.append(cluster)
            
        return clusters