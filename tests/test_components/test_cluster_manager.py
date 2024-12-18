import unittest
from src.clustering.cluster_manager import ClusterManager

class TestClusterManager(unittest.TestCase):
    def setUp(self):
        self.config = {
            'umap': {'n_neighbors': 5},
            'gmm': {'max_clusters': 10}
        }
        self.manager = ClusterManager(self.config)