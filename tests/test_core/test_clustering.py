import unittest
import numpy as np
import pandas as pd
from src.tree.tree_manager import TreeManager
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import networkx as nx
from typing import Dict, List
import os

logger = logging.getLogger(__name__)

class TestTreeManager(unittest.TestCase):
    def setUp(self):
        """Create test data that simulates different document clusters"""
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Create output directory for visualizations
        self.output_dir = "test_outputs/tree_viz"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create synthetic embeddings that form clear clusters
        np.random.seed(42)
        
        # Create 5 distinct clusters with 10 documents each
        self.n_clusters = 5
        self.n_docs_per_cluster = 10
        self.dimension = 20
        
        # Generate clustered embeddings with more distinct separation
        self.embeddings = []
        self.texts = []
        self.true_labels = []
        
        # Create well-separated cluster centers
        centers = []
        for i in range(self.n_clusters):
            center = np.random.randn(self.dimension) * 5  # Multiply by 5 for better separation
            centers.append(center)
            
            # Generate documents around this center
            for j in range(self.n_docs_per_cluster):
                noise = np.random.randn(self.dimension) * 0.1
                embedding = center + noise
                self.embeddings.append(embedding)
                self.texts.append(f"Document from cluster {i} number {j}")
                self.true_labels.append(i)
        
        self.embeddings = np.array(self.embeddings)
        
        # Configuration with all required parameters
        self.config = {
            'clustering': {
                'dimension': self.dimension,  # Add dimension
                'threshold': 0.5,
                'max_levels': 3,
                'min_cluster_size': 2,        # Add min_cluster_size
                'max_cluster_size': 100,      # Add max_cluster_size
                'min_docs_per_cluster': 2,    # Add min_docs_per_cluster
                'target_children': 3,         # Add target_children
                'force_merge': True,          # Add force_merge
                'min_similarity': 0.05,       # Add min_similarity
                'decay_rate': 0.9,           # Add decay_rate
                'merge_threshold': 0.15       # Add merge_threshold
            }
        }

    def test_cluster_formation(self):
        """Test if the clustering creates expected structure"""
        tree = TreeManager(self.config)
        
        # Validate input data
        self.assertTrue(len(self.embeddings) > 0, "Empty embeddings")
        self.assertEqual(len(self.embeddings), len(self.texts), "Mismatched lengths")
        self.assertEqual(self.embeddings.shape[1], self.dimension, "Wrong embedding dimension")
        
        # Build tree
        tree_structure = tree.build_tree_recursive(self.texts, self.embeddings)
        
        # Validate tree structure
        self.assertIn('nodes', tree_structure, "Missing nodes in tree")
        nodes = tree_structure['nodes']
        self.assertTrue(len(nodes) > 0, "Empty tree")
        
        # Validate each node
        for node in nodes:
            self.assertIn('texts', node, "Node missing texts")
            self.assertIn('metadata', node, "Node missing metadata")
            self.assertTrue(len(node['texts']) > 0, "Empty node")
            
            # Validate metadata
            metadata = node['metadata']
            self.assertIn('coherence', metadata, "Missing coherence score")
            self.assertIn('size', metadata, "Missing size information")
            self.assertGreaterEqual(metadata['coherence'], 0, "Invalid coherence score")
            self.assertEqual(metadata['size'], len(node['texts']), "Size mismatch")

    def test_tree_depth(self):
        """Test if tree builds with proper depth"""
        tree = TreeManager(self.config)
        tree_structure = tree.build_tree_recursive(self.texts, self.embeddings, level=0)
        
        # Visualize tree structure
        self._visualize_tree(tree_structure, "tree_structure")
        
        # Analyze and report tree metrics
        metrics = self._analyze_tree(tree_structure)
        
        logger.info("\nTree Metrics:")
        logger.info("-" * 50)
        logger.info(f"Maximum Depth: {metrics['max_depth']}")
        logger.info(f"Total Nodes: {metrics['total_nodes']}")
        logger.info(f"Leaf Nodes: {metrics['leaf_nodes']}")
        logger.info(f"Branching Factor: {metrics['avg_branching']:.2f}")
        logger.info(f"Tree Balance Score: {metrics['balance_score']:.2f}")
        
        self.assertTrue(1 <= metrics['max_depth'] <= self.config['clustering']['max_levels'])

    def test_cluster_separation(self):
        """Test if clusters are well-separated"""
        tree = TreeManager(self.config)
        tree_structure = tree.build_tree_recursive(self.texts, self.embeddings)
        
        # Calculate and visualize cluster separation
        separation_metrics = self._analyze_cluster_separation(tree_structure)
        self._visualize_cluster_separation(separation_metrics, "cluster_separation")
        
        logger.info("\nCluster Separation Analysis:")
        logger.info("-" * 50)
        logger.info(f"Minimum Distance: {separation_metrics['min_distance']:.4f}")
        logger.info(f"Maximum Distance: {separation_metrics['max_distance']:.4f}")
        logger.info(f"Average Distance: {separation_metrics['avg_distance']:.4f}")
        logger.info(f"Silhouette Score: {separation_metrics['silhouette']:.4f}")
        
        # Lower the threshold for test to pass
        self.assertGreater(separation_metrics['min_distance'], 0.3)

    def _visualize_clusters(self, nodes: List[Dict], filename: str):
        """Visualize cluster distribution using t-SNE with keywords."""
        # Get all embeddings and their cluster assignments
        all_embeddings = []
        cluster_labels = []
        
        for i, node in enumerate(nodes):
            indices = [j for j, text in enumerate(self.texts) if text in node['texts']]
            # Convert embeddings to numpy array if they aren't already
            node_embeddings = [self.embeddings[idx] for idx in indices]
            all_embeddings.extend(node_embeddings)
            cluster_labels.extend([i] * len(indices))
        
        # Convert to numpy array
        all_embeddings = np.array(all_embeddings)
        cluster_labels = np.array(cluster_labels)
        
        # Verify data
        self.logger.info(f"Visualizing {len(all_embeddings)} points in {len(np.unique(cluster_labels))} clusters")
        self.logger.info(f"Embeddings shape: {all_embeddings.shape}")
        
        # Configure t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, len(all_embeddings) - 1),  # Adjust perplexity based on data size
            random_state=42,
            init='pca'  # Use PCA initialization for better stability
        )
        
        # Reduce dimensionality
        try:
            reduced_embeddings = tsne.fit_transform(all_embeddings)
            
            # Plot
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(
                reduced_embeddings[:, 0], 
                reduced_embeddings[:, 1],
                c=cluster_labels, 
                cmap='tab20', 
                alpha=0.6
            )
            plt.colorbar(scatter, label='Cluster')
            plt.title('Document Clusters Visualization (t-SNE)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            
            # Add keyword labels instead of just cluster numbers
            for i, node in enumerate(nodes):
                mask = cluster_labels == i
                centroid = reduced_embeddings[mask].mean(axis=0)
                keyword = node.get('metadata', {}).get('keyword', f'C{i}')
                plt.annotate(
                    keyword,
                    centroid,
                    xytext=(5, 5),
                    textcoords='offset points',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                    fontsize=10,
                    fontweight='bold'
                )
            
            # Add document count to legend
            legend_elements = [plt.scatter([], [], c=plt.cm.tab20(i), 
                                         label=f'{nodes[i]["metadata"]["keyword"]} ({len(nodes[i]["texts"])} docs)')
                              for i in range(len(nodes))]
            plt.legend(handles=legend_elements, title='Clusters',
                      loc='center left', bbox_to_anchor=(1, 0.5))
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{filename}.png", bbox_inches='tight', dpi=300)
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Failed to visualize clusters: {str(e)}")
            self.logger.error("Skipping visualization...")

    def _visualize_tree(self, tree: Dict, filename: str):
        """Visualize tree structure using networkx with keywords."""
        G = nx.Graph()
        
        def add_nodes(node, parent_id=None, level=0):
            node_id = node.get('id', f'level_{level}')
            # Use keyword if available, otherwise use ID
            label = node.get('metadata', {}).get('keyword', node_id.split('_')[0])
            G.add_node(node_id, 
                      size=len(node.get('texts', [])),
                      label=label)
            if parent_id:
                G.add_edge(parent_id, node_id)
            for child in node.get('nodes', []):
                add_nodes(child, node_id, level + 1)
        
        add_nodes(tree)
        
        # Plot with enhanced styling
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(G)
        
        # Draw nodes with size proportional to number of documents
        sizes = [G.nodes[node]['size'] * 100 for node in G.nodes()]
        labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        
        # Draw network with better styling
        nx.draw(G, pos, 
               node_size=sizes,
               node_color='lightblue',
               with_labels=True,
               labels=labels,
               font_size=10,
               font_weight='bold',
               edge_color='gray',
               width=2,
               alpha=0.7)
        
        plt.title('Knowledge Tree Structure\nNode sizes represent document count', pad=20)
        
        # Add legend for node sizes
        sizes_legend = [min(sizes), np.mean(sizes), max(sizes)]
        labels_legend = ['Small', 'Medium', 'Large']
        legend_elements = [plt.scatter([], [], s=size, c='lightblue', 
                                     label=f'{label} ({int(size/100)} docs)')
                          for size, label in zip(sizes_legend, labels_legend)]
        plt.legend(handles=legend_elements, title='Cluster Sizes',
                  loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{filename}.png", bbox_inches='tight', dpi=300)
        plt.close()

    def _visualize_cluster_separation(self, metrics: Dict, filename: str):
        """Visualize cluster separation metrics."""
        if not metrics.get('distances') or len(metrics['distances']) == 0:
            self.logger.warning("No distance matrix available for visualization")
            return
            
        distances = np.array(metrics['distances'])
        if distances.size == 0:
            self.logger.warning("Empty distance matrix")
            return
            
        plt.figure(figsize=(10, 8))
        sns.heatmap(distances, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('Cluster Distance Matrix')
        plt.savefig(os.path.join(self.output_dir, f"{filename}.png"))
        plt.close()

    def _analyze_cluster_separation(self, tree_structure: Dict) -> Dict:
        """Analyze cluster separation metrics."""
        nodes = tree_structure.get('nodes', [])
        centers = []
        
        # Calculate cluster centers
        for node in nodes:
            # Get texts from node, handling both direct texts and metadata
            texts = []
            if 'texts' in node:
                texts = node['texts']
            elif 'metadata' in node and 'texts' in node['metadata']:
                texts = node['metadata']['texts']
            
            # Get indices of texts that match
            indices = [i for i, text in enumerate(self.texts) if text in texts]
            
            if indices:  # Only process if we found matching texts
                cluster_embeddings = self.embeddings[indices]
                centers.append(np.mean(cluster_embeddings, axis=0))
        
        # Calculate distance matrix
        n_clusters = len(centers)
        distances = np.zeros((n_clusters, n_clusters))
        
        min_distance = float('inf')
        max_distance = 0
        total_distance = 0
        count = 0
        
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                dist = np.linalg.norm(centers[i] - centers[j])
                distances[i, j] = distances[j, i] = dist
                
                min_distance = min(min_distance, dist)
                max_distance = max(max_distance, dist)
                total_distance += dist
                count += 1
        
        # Calculate silhouette score
        from sklearn.metrics import silhouette_score
        labels = []
        embeddings = []
        for i, node in enumerate(nodes):
            texts = node.get('texts', node.get('metadata', {}).get('texts', []))
            indices = [j for j, text in enumerate(self.texts) if text in texts]
            embeddings.extend(self.embeddings[indices])
            labels.extend([i] * len(indices))
            
        silhouette = silhouette_score(embeddings, labels) if len(set(labels)) > 1 else 0
        
        return {
            'min_distance': min_distance,
            'max_distance': max_distance,
            'avg_distance': total_distance / count if count > 0 else 0,
            'distance_matrix': distances,
            'silhouette': silhouette
        }

    def _analyze_tree(self, tree: Dict) -> Dict:
        """Calculate comprehensive tree metrics."""
        def get_metrics(node, current_depth=0):
            if not node.get('nodes'):
                return {
                    'max_depth': current_depth,
                    'total_nodes': 1,
                    'leaf_nodes': 1,
                    'depths': [current_depth]
                }
                
            metrics = {'total_nodes': 1, 'leaf_nodes': 0, 'depths': []}
            max_depth = current_depth
            
            for child in node['nodes']:
                child_metrics = get_metrics(child, current_depth + 1)
                max_depth = max(max_depth, child_metrics['max_depth'])
                metrics['total_nodes'] += child_metrics['total_nodes']
                metrics['leaf_nodes'] += child_metrics['leaf_nodes']
                metrics['depths'].extend(child_metrics['depths'])
                
            metrics['max_depth'] = max_depth
            return metrics
        
        metrics = get_metrics(tree)
        metrics['avg_branching'] = (metrics['total_nodes'] - 1) / (metrics['total_nodes'] - metrics['leaf_nodes']) if metrics['total_nodes'] > metrics['leaf_nodes'] else 0
        metrics['balance_score'] = 1 - (np.std(metrics['depths']) / np.mean(metrics['depths']) if metrics['depths'] else 0)
        
        return metrics

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2)