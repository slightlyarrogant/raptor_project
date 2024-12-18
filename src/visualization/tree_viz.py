# Standard library imports
import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import math
import warnings
import uuid
import pickle
import json
from pathlib import Path
import colorsys

# Configure matplotlib for non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Third-party imports
import numpy as np
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from collections import Counter
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import dendrogram, linkage
import umap

# Local imports
from src.models.data_models import TreeData, NodeData, EdgeData
from src.tree.node import Node, LeafNode, SummaryNode

# Configure UMAP for thread safety
import umap.umap_ as umap
umap.UMAP(n_jobs=1)  # Force single-threaded operation

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)): 
            return None
        return super().default(obj)

def convert_numpy_to_list(obj):
    """Recursively convert numpy arrays to lists in a nested structure."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    return obj

class TreeVisualizer:
    """Creates visualizations for tree data."""
    
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.data_dir = Path('data') / index_name
        self.viz_dir = self.data_dir / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different visualization types
        self.network_dir = self.viz_dir / 'network'
        self.sunburst_dir = self.viz_dir / 'sunburst'
        self.stats_dir = self.viz_dir / 'stats'
        
        for dir in [self.network_dir, self.sunburst_dir, self.stats_dir]:
            dir.mkdir(exist_ok=True)
            
        self.logger = logging.getLogger(__name__)

    def create_visualizations(self, tree_data: TreeData):
        """Create all visualizations for the tree."""
        try:
            # Create network visualization
            network_file = self.network_dir / f"tree_network_{self.index_name}.html"
            self.create_network(tree_data, str(network_file))
            
            # Create sunburst visualization
            sunburst_file = self.sunburst_dir / f"tree_sunburst_{self.index_name}.html"
            self.create_sunburst(tree_data, str(sunburst_file))
            
            # Save tree statistics
            stats_file = self.stats_dir / f"tree_stats_{self.index_name}.json"
            self._save_tree_stats(tree_data, stats_file)
            
        except Exception as e:
            self.logger.error(f"Failed to create visualizations: {str(e)}")
            raise

    def create_network(self, tree_data: TreeData, output_file: str):
        """Create network visualization of the tree."""
        try:
            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            
            for node in tree_data.nodes.values():
                pos = node.position if hasattr(node, 'position') else [0, 0]
                node_x.append(pos[0])
                node_y.append(pos[1])
                node_text.append(node.title if hasattr(node, 'title') else str(node.id))
                node_size.append(len(node.children) * 10 + 20 if hasattr(node, 'children') else 20)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    size=node_size,
                    line_width=2
                )
            )
            
            # Create edge trace
            edge_x = []
            edge_y = []
            
            for node in tree_data.nodes.values():
                if hasattr(node, 'children'):
                    for child in node.children:
                        if hasattr(child, 'position'):
                            edge_x.extend([node.position[0], child.position[0], None])
                            edge_y.extend([node.position[1], child.position[1], None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Create figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="Tree Network",
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
            )
            
            # Save visualization
            fig.write_html(
                output_file,
                include_plotlyjs=True,
                full_html=True,
                include_mathjax=False,
                config={'displayModeBar': False}
            )
            
            self.logger.info(f"Network visualization saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create network visualization: {str(e)}")
            raise

    def create_sunburst(self, tree_data: TreeData, output_file: str):
        """Create sunburst visualization of the tree."""
        try:
            # Prepare data
            ids = []
            labels = []
            parents = []
            hovers = []
            colors = []
            
            # Add root node
            root = tree_data.root
            ids.append(str(root.id))
            labels.append(root.title if hasattr(root, 'title') else 'Root')
            parents.append('')
            hovers.append(self._create_hover_text(root))
            colors.append('#1f77b4')  # Root color
            
            # Process all nodes
            for node in tree_data.nodes.values():
                if node == root:
                    continue
                    
                ids.append(str(node.id))
                labels.append(node.title if hasattr(node, 'title') else str(node.id))
                
                # Find parent
                parent_id = ''
                for potential_parent in tree_data.nodes.values():
                    if hasattr(potential_parent, 'children') and node in potential_parent.children:
                        parent_id = str(potential_parent.id)
                        break
                parents.append(parent_id)
                
                # Create hover text
                hovers.append(self._create_hover_text(node))
                
                # Assign color based on level
                if hasattr(node, 'level'):
                    level_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                    colors.append(level_colors[node.level % len(level_colors)])
                else:
                    colors.append('#95A5A6')  # Default gray if no color inheritance possible
            
            # Create figure
            fig = go.Figure(
                go.Sunburst(
                    ids=ids,
                    labels=labels,
                    parents=parents,
                    marker=dict(colors=colors),
                    hovertemplate='%{customdata}<extra></extra>',
                    customdata=hovers
                )
            )
            
            # Update layout
            fig.update_layout(
                title="Tree Structure",
                width=1200,
                height=1200
            )
            
            # Write HTML file directly
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Tree Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
    <div id="plot"></div>
    <script>
        var data = {json.dumps(fig.to_dict()['data'])};
        var layout = {json.dumps(fig.to_dict()['layout'])};
        Plotly.newPlot('plot', data, layout);
    </script>
</body>
</html>
"""
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            file_size = os.path.getsize(output_file)
            self.logger.info(f"Sunburst visualization saved to {output_file} ({file_size:,} bytes)")
            
        except Exception as e:
            self.logger.error(f"Failed to create sunburst visualization: {str(e)}")
            raise

    def _create_hover_text(self, node) -> str:
        """Create hover text for a node."""
        hover_text = []
        
        if hasattr(node, 'title'):
            hover_text.append(f"Title: {node.title}")
        
        if hasattr(node, 'id'):
            hover_text.append(f"ID: {node.id}")
        
        if hasattr(node, 'level'):
            hover_text.append(f"Level: {node.level}")
        
        if hasattr(node, 'children'):
            hover_text.append(f"Children: {len(node.children)}")
        
        if hasattr(node, 'metadata'):
            for key, value in node.metadata.items():
                if isinstance(value, (str, int, float)):
                    hover_text.append(f"{key}: {value}")
        
        return "<br>".join(hover_text)

    def _save_tree_stats(self, tree_data: TreeData, output_file: Path):
        """Save tree statistics to JSON file."""
        try:
            stats = {
                'total_nodes': len(tree_data.nodes),
                'max_depth': max(node.level for node in tree_data.nodes.values() if hasattr(node, 'level')),
                'nodes_by_level': {},
                'avg_children': sum(len(node.children) for node in tree_data.nodes.values() if hasattr(node, 'children')) / len(tree_data.nodes),
                'timestamp': str(datetime.now())
            }
            
            # Count nodes by level
            for node in tree_data.nodes.values():
                if hasattr(node, 'level'):
                    level = str(node.level)
                    if level not in stats['nodes_by_level']:
                        stats['nodes_by_level'][level] = 0
                    stats['nodes_by_level'][level] += 1
            
            with open(output_file, 'w') as f:
                json.dump(stats, f, indent=2)
                
            self.logger.info(f"Tree statistics saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save tree statistics: {str(e)}")
            raise
