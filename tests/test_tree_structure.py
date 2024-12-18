import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import List, Dict

class TreeManager:
    def _visualize_clusters(self, nodes: List[Dict], filename: str):
        """Visualize cluster distribution using t-SNE with keywords."""
        # ... (previous code) ...
        
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