import logging
from typing import Dict, List
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import time
import psutil
import json
from pathlib import Path

class TreeEvaluator:
    """Evaluate RAPTOR tree quality metrics."""
    
    def __init__(self, output_dir: str):
        self.logger = logging.getLogger(__name__)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_tree(self, tree_data: Dict) -> Dict:
        """Evaluate comprehensive tree metrics."""
        metrics = {}
        
        # 1. Structural Metrics
        metrics['structural'] = self._evaluate_structure(tree_data)
        
        # 2. Computational Efficiency
        metrics['computational'] = self._evaluate_efficiency(tree_data)
        
        # 3. Clustering Quality
        metrics['clustering'] = self._evaluate_clustering(tree_data)
        
        # 4. Compression Analysis
        metrics['compression'] = self._evaluate_compression(tree_data)
        
        # Save detailed metrics
        self._save_metrics(metrics)
        
        return metrics
        
    def _evaluate_structure(self, tree_data: Dict) -> Dict:
        """Evaluate structural metrics."""
        nodes = tree_data['tree']['nodes']
        clusters = tree_data['tree']['clusters']
        
        # Clean up file attribution
        file_mapping = {}
        for node in nodes:
            filename = node.get('metadata', {}).get('filename', '')
            if filename and filename not in ['unknown', 'doc']:
                if filename not in file_mapping:
                    file_mapping[filename] = {
                        'total_chunks': 0,
                        'clusters': set(),
                        'chunk_ids': []
                    }
                file_mapping[filename]['total_chunks'] += 1
                file_mapping[filename]['chunk_ids'].append(node['id'])

        # Calculate clean cluster metrics
        cluster_metrics = {}
        for cluster in clusters:
            cluster_id = cluster['id']
            cluster_files = set()
            for node in cluster['nodes']:
                filename = node.get('metadata', {}).get('filename', '')
                if filename and filename not in ['unknown', 'doc']:
                    cluster_files.add(filename)
                    if filename in file_mapping:
                        file_mapping[filename]['clusters'].add(cluster_id)
            
            cluster_metrics[cluster_id] = {
                'size': len(cluster['nodes']),
                'unique_files': list(cluster_files),
                'file_count': len(cluster_files)
            }

        metrics = {
            'total_nodes': len(nodes),
            'total_clusters': len(clusters),
            'avg_nodes_per_cluster': len(nodes) / len(clusters),
            'cluster_size_distribution': {
                'min': min(len(c['nodes']) for c in clusters),
                'max': max(len(c['nodes']) for c in clusters),
                'mean': np.mean([len(c['nodes']) for c in clusters]),
                'std': np.std([len(c['nodes']) for c in clusters])
            },
            'file_distribution': {
                filename: {
                    'total_chunks': stats['total_chunks'],
                    'clusters': list(stats['clusters']),
                    'chunk_distribution': self._analyze_chunk_distribution(stats['chunk_ids'])
                }
                for filename, stats in file_mapping.items()
            },
            'cluster_details': cluster_metrics,
            'cross_cluster_stats': {
                'files_per_cluster': np.mean([len(c['unique_files']) for c in cluster_metrics.values()]),
                'clusters_per_file': np.mean([len(stats['clusters']) for stats in file_mapping.values()])
            }
        }

        # Print detailed analysis
        print("\n=== Tree Structure Analysis ===")
        print(f"\nOverall Statistics:")
        print(f"- Total Nodes: {metrics['total_nodes']}")
        print(f"- Total Clusters: {metrics['total_clusters']}")
        print(f"- Average Nodes per Cluster: {metrics['avg_nodes_per_cluster']:.2f}")
        
        print("\nFile Distribution:")
        for filename, stats in metrics['file_distribution'].items():
            print(f"\n{filename}:")
            print(f"- Total Chunks: {stats['total_chunks']}")
            print(f"- Appears in {len(stats['clusters'])} clusters: {stats['clusters']}")
            
        print("\nCluster Analysis:")
        for cluster_id, stats in metrics['cluster_details'].items():
            print(f"\nCluster {cluster_id}:")
            print(f"- Size: {stats['size']} nodes")
            print(f"- Contains {stats['file_count']} unique files")
            print(f"- Files: {', '.join(stats['unique_files'])}")
        
        return metrics
        
    def _evaluate_efficiency(self, tree_data: Dict) -> Dict:
        """Evaluate computational efficiency metrics."""
        metrics = {
            'memory_usage': {
                'total_mb': psutil.Process().memory_info().rss / (1024 * 1024),
                'per_node_kb': (psutil.Process().memory_info().rss / len(tree_data['tree']['nodes'])) / 1024
            },
            'storage_efficiency': {
                'total_text_size': sum(len(n['text']) for n in tree_data['tree']['nodes']),
                'total_embedding_size': sum(len(str(n['embedding'])) for n in tree_data['tree']['nodes']),
                'compression_ratio': None  # Will be calculated
            }
        }
        
        # Calculate compression ratio
        total_size = metrics['storage_efficiency']['total_text_size']
        compressed_size = metrics['storage_efficiency']['total_embedding_size']
        metrics['storage_efficiency']['compression_ratio'] = (total_size - compressed_size) / total_size
        
        self.logger.info("\n=== Efficiency Metrics ===")
        self.logger.info(f"Memory Usage: {metrics['memory_usage']['total_mb']:.2f} MB")
        self.logger.info(f"Memory per Node: {metrics['memory_usage']['per_node_kb']:.2f} KB")
        self.logger.info(f"Compression Ratio: {metrics['storage_efficiency']['compression_ratio']:.2%}")
        
        return metrics
        
    def _evaluate_clustering(self, tree_data: Dict) -> Dict:
        """Evaluate clustering quality metrics."""
        nodes = tree_data['tree']['nodes']
        clusters = tree_data['tree']['clusters']
        
        # Get embeddings and cluster assignments
        embeddings = np.array([n['embedding'] for n in nodes])
        cluster_labels = np.array([n.get('cluster_id', -1) for n in nodes])
        
        metrics = {
            'silhouette_score': silhouette_score(embeddings, cluster_labels) if len(set(cluster_labels)) > 1 else 0,
            'intra_cluster_similarity': self._calculate_intra_cluster_similarity(clusters),
            'inter_cluster_similarity': self._calculate_inter_cluster_similarity(clusters)
        }
        
        self.logger.info("\n=== Clustering Quality ===")
        self.logger.info(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
        self.logger.info(f"Avg Intra-cluster Similarity: {metrics['intra_cluster_similarity']:.3f}")
        
        return metrics
        
    def _evaluate_compression(self, tree_data: Dict) -> Dict:
        """Evaluate compression and summarization metrics."""
        clusters = tree_data['tree']['clusters']
        
        metrics = {
            'summary_stats': {
                'avg_summary_length': np.mean([len(c.get('summary', '')) for c in clusters]),
                'compression_ratio': self._calculate_compression_ratio(clusters)
            },
            'content_preservation': self._evaluate_content_preservation(clusters)
        }
        
        self.logger.info("\n=== Compression Analysis ===")
        self.logger.info(f"Avg Summary Length: {metrics['summary_stats']['avg_summary_length']:.1f}")
        self.logger.info(f"Compression Ratio: {metrics['summary_stats']['compression_ratio']:.2%}")
        
        return metrics
        
    def _calculate_intra_cluster_similarity(self, clusters: List[Dict]) -> float:
        """Calculate average similarity within clusters."""
        similarities = []
        for cluster in clusters:
            embeddings = np.array([n['embedding'] for n in cluster['nodes']])
            if len(embeddings) > 1:
                sim_matrix = cosine_similarity(embeddings)
                similarities.append(np.mean(sim_matrix))
        return np.mean(similarities) if similarities else 0
        
    def _calculate_inter_cluster_similarity(self, clusters: List[Dict]) -> float:
        """Calculate average similarity between clusters."""
        similarities = []
        for i, c1 in enumerate(clusters):
            for j, c2 in enumerate(clusters[i+1:], i+1):
                emb1 = np.mean([n['embedding'] for n in c1['nodes']], axis=0)
                emb2 = np.mean([n['embedding'] for n in c2['nodes']], axis=0)
                sim = cosine_similarity([emb1], [emb2])[0][0]
                similarities.append(sim)
        return np.mean(similarities) if similarities else 0
        
    def _save_metrics(self, metrics: Dict):
        """Save metrics to file."""
        output_path = self.output_dir / 'tree_quality_metrics.json'
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"\nDetailed metrics saved to: {output_path}") 
        
    def _analyze_chunk_distribution(self, chunk_ids: List[str]) -> Dict:
        """Analyze the distribution of chunks."""
        # Extract chunk indices
        indices = [int(cid.split('_')[-1]) for cid in chunk_ids if '_' in cid]
        if not indices:
            return {}
            
        return {
            'min_index': min(indices),
            'max_index': max(indices),
            'total_chunks': len(indices),
            'has_gaps': not all(i in indices for i in range(min(indices), max(indices) + 1))
        }