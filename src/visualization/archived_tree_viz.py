# Standard library imports
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import math
import warnings

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
from pathlib import Path
from collections import Counter
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import dendrogram, linkage
import json
import umap
from textblob import TextBlob

# Configure UMAP for thread safety
import umap.umap_ as umap
umap.UMAP(n_jobs=1)  # Force single-threaded operation

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

class TreeVisualizer:
    """Visualization tools for the standardized tree structure.
    
    This class creates various visualizations of the tree structure while maintaining
    compatibility with the standardized Node format. All visualizations expect
    tree data in the format returned by Node.to_dict() or Node.to_tree_format().
    """
    
    def __init__(self, output_dir: Path):
        """Initialize TreeVisualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Visualization settings
        self.color_scale = px.colors.sequential.Viridis
        self.node_size_range = (20, 100)  # Min/max node sizes
        self.level_height = 1.0  # Vertical spacing between levels
        self.node_spacing = 1.5  # Horizontal spacing between nodes
        
        # Initialize plotly template
        import plotly.io as pio
        pio.templates.default = "plotly_white"
        
        # Visualization parameters optimized for 4-level hierarchy
        self.level_colors = {
            0: "#2E86C1",  # Root level - blue
            1: "#27AE60",  # Level 1 - green
            2: "#F39C12",  # Level 2 - orange
            3: "#C0392B",  # Level 3 - red
            4: "#8E44AD"   # Level 4 - purple
        }
        
        # Node size parameters for better hierarchy visualization
        self.node_size_range = {
            0: (50, 70),   # Root level
            1: (40, 60),   # Level 1
            2: (30, 50),   # Level 2
            3: (20, 40),   # Level 3
            4: (15, 30)    # Level 4
        }
        
        # Layout parameters for balanced tree display
        self.layout_params = {
            'hierarchical': {
                'levelSeparation': 200,
                'nodeSpacing': 150,
                'treeSpacing': 200
            }
        }
        
        # Ensure required directories exist
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        # Configure logging
        self.logger.info(f"TreeVisualizer initialized with output directory: {self.output_dir}")

    def _validate_tree_data(self, tree_data: Dict) -> bool:
        """Validate tree data structure."""
        try:
            if not isinstance(tree_data, dict):
                self.logger.error("Tree data must be a dictionary")
                return False
                
            required_fields = ['id', 'level', 'text', 'metadata']
            if not all(field in tree_data for field in required_fields):
                self.logger.error(f"Missing required fields: {required_fields}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating tree data: {str(e)}")
            return False

    def _ensure_output_path(self, output_file: str) -> Path:
        """Ensure output path exists and return full path."""
        output_path = self.output_dir / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _debug_print_structure(self, data: Any, max_items: int = 3, current_depth: int = 0, max_depth: int = 3) -> str:
        """Helper method to create a readable string representation of nested data structures.
        
        Args:
            data: The data structure to print
            max_items: Maximum number of items to show for lists/dicts
            current_depth: Current recursion depth
            max_depth: Maximum depth to traverse
        """
        indent = "  " * current_depth
        
        if current_depth >= max_depth:
            return f"{indent}... (max depth reached)"
            
        if isinstance(data, dict):
            if not data:
                return f"{indent}{{}}"
            
            items = list(data.items())[:max_items]
            lines = [f"{indent}{{"]
            for k, v in items:
                lines.append(f"{indent}  {k}: {self._debug_print_structure(v, max_items, current_depth + 1, max_depth)}")
            if len(data) > max_items:
                lines.append(f"{indent}  ... ({len(data) - max_items} more items)")
            lines.append(f"{indent}}}")
            return "\n".join(lines)
            
        elif isinstance(data, (list, tuple)):
            if not data:
                return f"{indent}[]"
                
            items = data[:max_items]
            lines = [f"{indent}["]
            for item in items:
                lines.append(f"{indent}  {self._debug_print_structure(item, max_items, current_depth + 1, max_depth)}")
            if len(data) > max_items:
                lines.append(f"{indent}  ... ({len(data) - max_items} more items)")
            lines.append(f"{indent}]")
            return "\n".join(lines)
            
        elif isinstance(data, (int, float, str, bool, type(None))):
            return str(data)
        else:
            return f"{type(data).__name__}({str(data)})"

    def _log_error_context(self, error: Exception, tree_data: Dict, context: str = ""):
        """Log detailed error context including data structure."""
        self.logger.error(f"\n{'='*50}")
        self.logger.error(f"Error in {context}: {str(error)}")
        self.logger.error("\nData Structure:")
        self.logger.error(self._debug_print_structure(tree_data))
        self.logger.error(f"{'='*50}\n")

    def create_tree_visualization(self, tree_data: Dict, output_file: str = None) -> go.Figure:
        """Create a hierarchical tree visualization.
        
        Args:
            tree_data: Tree data in Node.to_dict() format
            output_file: Optional path to save the visualization
            
        Returns:
            Plotly figure object
        """
        if not self._validate_tree_data(tree_data):
            self.logger.warning("Invalid tree data format")
            return go.Figure()  # Return empty figure
            
        # Create node map for easy lookup
        node_map = {tree_data['id']: tree_data}
        for child_id in tree_data.get('children', []):
            node_map[child_id] = child_id  # Placeholder for child nodes
            
        # Create figure
        fig = go.Figure()
        
        def add_node_to_plot(node_id: str, x: float, y: float, dx: float):
            """Recursively add nodes and their children to the plot."""
            node = node_map[node_id]
            children = node.get('children', [])
            num_children = len(children)
            
            # Calculate node size based on content
            text_length = len(node.get('text', ''))
            node_size = 20 + min(30, math.log2(1 + text_length))
            
            # Get node color based on level
            level = node.get('level', 0)
            color = self.level_colors.get(level, "#808080")  # Default gray if level not found
            
            # Get the best available label for the node
            metadata = node.get('metadata', {})
            is_leaf = metadata.get('is_leaf', False)
            
            if level == 0:
                # For root node, always use 'root'
                node_title = 'root'
            elif is_leaf:
                # For leaf nodes, prioritize doc_name with chunk info
                doc_name = metadata.get('doc_name')
                chunk_num = metadata.get('chunk_num')
                if doc_name:
                    if chunk_num is not None:
                        node_title = f"{doc_name} (chunk {chunk_num})"
                    else:
                        node_title = doc_name
                else:
                    # Fallback for leaf nodes without doc_name
                    node_title = (
                        metadata.get('title') or
                        (node.get('texts', [''])[0][:50] + '...') if node.get('texts') else None or
                        node_id
                    )
            else:
                # For non-leaf nodes, prioritize title
                node_title = (
                    metadata.get('title') or
                    metadata.get('doc_name') or
                    (node.get('texts', [''])[0][:50] + '...') if node.get('texts') else None or
                    node_id
                )
            
            # Create node text
            hover_text = f"""
            <b>{node_title}</b><br>
            Level: {level}<br>
            Type: {metadata.get('node_type', 'unknown')}<br>
            Children: {num_children}
            """
            
            # Add node
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(
                    size=node_size,
                    color=color,
                    line=dict(color='darkgray', width=1)
                ),
                text=[node_title],
                textposition="top center",
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=False
            ))
            
            # Add children
            if children:
                child_dx = dx / num_children
                for i, child_id in enumerate(children):
                    child_x = x - dx/2 + child_dx/2 + i*child_dx
                    child_y = y - 1
                    
                    # Add edge
                    fig.add_trace(go.Scatter(
                        x=[x, child_x],
                        y=[y, child_y],
                        mode='lines',
                        line=dict(color='gray', width=1),
                        hoverinfo='none',
                        showlegend=False
                    ))
                    
                    # Recursively add child
                    add_node_to_plot(child_id, child_x, child_y, child_dx)
        
        # Start plotting from root node
        root_id = tree_data['id']  # Root node is the input node
        add_node_to_plot(root_id, 0, 0, 2)
        
        # Update layout
        fig.update_layout(
            title='Tree Structure Visualization',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        if output_file:
            fig.write_html(self.output_dir / output_file)
        
        return fig

    def create_cluster_distribution(self, tree_data: Dict, output_file: str):
        """Create visualization of cluster size distribution."""
        try:
            # Extract cluster sizes
            cluster_sizes = []
            for node in tree_data['tree']['nodes']:
                if node.get('texts'):
                    cluster_sizes.append(len(node['texts']))
            
            if not cluster_sizes:
                self.logger.warning("No cluster sizes available")
                return
            
            # Create distribution plot
            plt.figure(figsize=(12, 6))
            sns.histplot(cluster_sizes, bins=range(min(cluster_sizes), max(cluster_sizes) + 2, 1))
            plt.title('Cluster Size Distribution')
            plt.xlabel('Number of Documents in Cluster')
            plt.ylabel('Frequency')
            
            # Save plot
            output_path = self._ensure_output_path(output_file)
            plt.savefig(str(output_path))
            plt.close()
            
            # Save statistics
            stats = {
                'sizes': cluster_sizes,
                'statistics': {
                    'total': len(cluster_sizes),
                    'mean': np.mean(cluster_sizes),
                    'median': np.median(cluster_sizes),
                    'min': min(cluster_sizes),
                    'max': max(cluster_sizes),
                    'std': np.std(cluster_sizes)
                }
            }
            stats_file = self.output_dir / 'cluster_distribution_stats.json'
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            self.logger.info(f"Cluster distribution saved to {output_path}")
            self.logger.info(f"Distribution statistics saved to {stats_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create cluster distribution: {str(e)}")
            raise
            
    def create_embedding_visualization(self, tree_data: Dict, output_file: str):
        """Create 3D embedding visualization with improved clustering display."""
        try:
            # Extract embeddings and metadata
            embeddings = []  
            metadata = []    
            
            # Process vectors
            if 'vectors' in tree_data:
                vectors = tree_data['vectors']
                for vector in vectors:
                    if isinstance(vector, dict):
                        embeddings.append(vector['values'])
                        metadata.append(vector['metadata'])
                    else:
                        # Fallback for object-style access
                        embeddings.append(vector.values)
                        metadata.append(vector.metadata)
            
            # Convert to numpy array and check for NaN values
            embeddings_array = np.array(embeddings)
            self.logger.info(f"Processing {len(embeddings)} embeddings...")
            
            # UMAP reduction with thread-safe configuration
            reducer = umap.UMAP(
                n_components=3,
                random_state=42,
                n_jobs=1,  # Force single-threaded
                n_neighbors=15,
                min_dist=0.1,
                metric='euclidean'
            )
            
            # Disable tkinter warnings
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reduced_embeddings = reducer.fit_transform(embeddings_array)
            
            self.logger.info("UMAP reduction complete")
            
            # Prepare visualization data
            self.logger.info("\nPreparing visualization data...")
            
            # Create scatter plot
            fig = go.Figure()
            
            # Add points colored by cluster
            unique_clusters = list(set(m.get('cluster_id', 'unknown') for m in metadata))
            colors = px.colors.qualitative.Set3[:len(unique_clusters)]  # Get enough colors
            
            for cluster_id, color in zip(unique_clusters, colors):
                cluster_mask = [m.get('cluster_id', 'unknown') == cluster_id for m in metadata]
                if any(cluster_mask):
                    points = reduced_embeddings[cluster_mask]
                    cluster_meta = [m for m, is_cluster in zip(metadata, cluster_mask) if is_cluster]
                    
                    # Add trace with improved styling
                    fig.add_trace(go.Scatter3d(
                        x=points[:, 0],
                        y=points[:, 1],
                        z=points[:, 2],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=color,
                            opacity=0.7,
                            line=dict(width=0.5, color='white')
                        ),
                        text=[f"Title: {m.get('title', '')}<br>Cluster: {m.get('cluster_id', 'unknown')}<br>{m.get('text', '')}" for m in cluster_meta],
                        name=f"Cluster {cluster_id}",
                        hoverinfo='text'
                    ))
            
            # Update layout with improved styling
            fig.update_layout(
                title={
                    'text': 'Document Embedding Visualization',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=20)
                },
                scene=dict(
                    xaxis_title='UMAP 1',
                    yaxis_title='UMAP 2',
                    zaxis_title='UMAP 3',
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5)
                    ),
                    aspectmode='cube'
                ),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255, 255, 255, 0.9)'
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                paper_bgcolor='white',
                plot_bgcolor='white',
                width=1200,
                height=800
            )
            
            # Save visualization
            output_path = self._ensure_output_path(output_file)
            fig.write_html(str(output_path))
            self.logger.info(f"Embedding visualization saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create embedding visualization: {str(e)}")
            raise
            
    def create_sunburst(self, tree_data: Dict, output_file: str):
        """Create sunburst visualization of tree hierarchy."""
        try:
            import plotly.graph_objects as go
            from collections import defaultdict
            
            # Group nodes by level
            nodes_by_level = defaultdict(list)
            for node in tree_data['tree']['nodes']:
                level = node.get('level', 0)
                nodes_by_level[level].append(node)
            
            print("\nDEBUG: Tree Data Structure")
            print(f"Total nodes in tree: {len(tree_data['tree']['nodes'])}")
            print(f"Levels: {sorted(nodes_by_level.keys())}")
            for level, nodes in sorted(nodes_by_level.items()):
                print(f"Level {level}: {len(nodes)} nodes")
            
            # Prepare data for sunburst
            data = {
                'ids': [],
                'labels': [],
                'parents': [],
                'values': [],
                'customdata': []
            }
            
            # First pass - collect all titles and create cluster mapping
            cluster_titles = {}
            cluster_nodes = defaultdict(list)
            
            # Process all nodes first
            for level, nodes in sorted(nodes_by_level.items()):
                for node in nodes:
                    # Get cluster info from metadata or cluster_info
                    metadata = node.get('metadata', {})
                    cluster_info = node.get('cluster_info', {})
                    
                    # Try to get cluster ID from different possible locations
                    cluster_id = str(metadata.get('cluster_id') or 
                                  cluster_info.get('cluster_id') or 
                                  cluster_info.get('cluster_name') or 
                                  node.get('cluster_id', '0'))
                    
                    # Add node to its cluster
                    cluster_nodes[cluster_id].append(node)
                    
                    # Get the best title for this node
                    node_title = (
                        metadata.get('title') or
                        cluster_info.get('cluster_name') or
                        metadata.get('doc_name')
                    )
                    
                    # If node has a meaningful title, use it for cluster title
                    if node_title and cluster_id not in cluster_titles:
                        cluster_titles[cluster_id] = node_title
            
            print("\nDEBUG: Updated Clusters")
            print(f"Found {len(cluster_nodes)} clusters")
            for cluster_id, nodes in cluster_nodes.items():
                print(f"Cluster {cluster_id}: {len(nodes)} nodes")
                print(f"Title: {cluster_titles.get(cluster_id, 'No Title')}")
            
            # Add root node
            data['ids'].append('root')
            data['labels'].append('root')  # Always use 'root' as the label
            data['parents'].append('')
            data['values'].append(len(tree_data['tree']['nodes']))
            data['customdata'].append(f"Root Node<br>Total Documents: {len(tree_data['tree']['nodes'])}")
            
            # Add nodes by cluster
            for cluster_id, nodes in cluster_nodes.items():
                if not nodes:
                    continue
                
                # Get or create cluster title
                cluster_title = cluster_titles.get(cluster_id, f"Cluster {cluster_id}")
                
                # Add cluster
                cluster_id_str = f"cluster_{cluster_id}"
                data['ids'].append(cluster_id_str)
                data['labels'].append(cluster_title)
                data['parents'].append('root')
                data['values'].append(len(nodes))
                data['customdata'].append(f"{cluster_title}<br>Documents: {len(nodes)}")
                
                # Add individual nodes
                for node in nodes:
                    node_id = str(node['id'])
                    metadata = node.get('metadata', {})
                    is_leaf = metadata.get('is_leaf', False)
                    
                    # Get the best available label
                    if is_leaf:
                        # For leaf nodes, prioritize doc_name with chunk info
                        doc_name = metadata.get('doc_name')
                        chunk_num = metadata.get('chunk_num')
                        if doc_name:
                            if chunk_num is not None:
                                node_title = f"{doc_name} (chunk {chunk_num})"
                            else:
                                node_title = doc_name
                        else:
                            # Fallback for leaf nodes without doc_name
                            node_title = (
                                metadata.get('title') or
                                (node.get('texts', [''])[0][:50] + '...') if node.get('texts') else None or
                                node_id
                            )
                    else:
                        # For non-leaf nodes, prioritize title
                        node_title = (
                            metadata.get('title') or
                            metadata.get('doc_name') or
                            (node.get('texts', [''])[0][:50] + '...') if node.get('texts') else None or
                            node_id
                        )
                    
                    data['ids'].append(node_id)
                    data['labels'].append(node_title)
                    data['parents'].append(cluster_id_str)
                    data['values'].append(1)  # Each document counts as 1
                    
                    # Create informative hover text
                    hover_text = [
                        f"Title: {metadata.get('title', 'N/A')}",
                        f"Cluster: {cluster_title}",
                        f"Type: {'Document Chunk' if is_leaf else 'Summary'}"
                    ]
                    if metadata.get('doc_name'):
                        hover_text.append(f"Document: {metadata['doc_name']}")
                        if metadata.get('chunk_num') is not None:
                            hover_text.append(f"Chunk: {metadata['chunk_num']}")
                    if node.get('texts'):
                        hover_text.append(f"Text Length: {len(node['texts'][0])} chars")
                    
                    data['customdata'].append('<br>'.join(hover_text))
            
            print("\nDEBUG: Final Data")
            print(f"Total entries: {len(data['ids'])}")
            print(f"Root node: {data['labels'][0]}")
            if len(data['ids']) > 1:
                print(f"First cluster: {data['labels'][1]}")
                if len(data['ids']) > 2:
                    print(f"First document: {data['labels'][2]}")
            
            # Create figure
            fig = go.Figure(go.Sunburst(
                ids=data['ids'],
                labels=data['labels'],
                parents=data['parents'],
                values=data['values'],
                customdata=data['customdata'],
                hovertemplate='%{customdata}<br>Size: %{value}<extra></extra>',
                branchvalues='total'
            ))
            
            # Update layout with better styling
            fig.update_layout(
                title={
                    'text': "Tree Hierarchy Visualization",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24)
                },
                width=1000,
                height=1000,
                template="plotly_white",
                margin=dict(t=100, l=0, r=0, b=0)
            )
            
            # Save visualization
            output_path = self._ensure_output_path(output_file)
            fig.write_html(str(output_path))
            
            # Save data for debugging
            data_path = self.output_dir / 'data' / 'sunburst_data.json'
            data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(data_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Sunburst visualization saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create sunburst visualization: {str(e)}")
            raise

    def create_network(self, tree_data: Dict, output_file: str):
        """Create network visualization of tree nodes and levels."""
        try:
            # Create network graph
            import networkx as nx
            import plotly.graph_objects as go
            import numpy as np
            
            G = nx.Graph()
            
            # Group nodes by level to create clusters
            nodes_by_level = {}
            for node in tree_data['tree']['nodes']:
                level = node['level']
                if level not in nodes_by_level:
                    nodes_by_level[level] = []
                nodes_by_level[level].append(node)
            
            # Calculate min and max sizes for scaling
            all_sizes = [len(nodes) for nodes in nodes_by_level.values()]
            min_size = min(all_sizes)
            max_size = max(all_sizes)
            
            # Add cluster nodes (one per level)
            for level, nodes in nodes_by_level.items():
                size = len(nodes)
                # Use logarithmic scaling with base adjustment for better visualization
                # Add 1 to avoid log(0), multiply by factor for visibility
                log_size = np.log(size + 1) * 20
                G.add_node(
                    f"level_{level}", 
                    size=size,
                    display_size=log_size,  # Store display size separately
                    title=f"Level {level}",
                    nodes=nodes
                )
            
            # Add edges between adjacent levels
            levels = sorted(nodes_by_level.keys())
            for i in range(len(levels)-1):
                current_level = levels[i]
                next_level = levels[i+1]
                
                # Count parent-child relationships between levels
                connections = 0
                for node in nodes_by_level[next_level]:
                    if node.get('parent_id') and any(p['id'] == node['parent_id'] for p in nodes_by_level[current_level]):
                        connections += 1
                
                if connections > 0:
                    # Scale edge weight logarithmically too
                    log_weight = np.log(connections + 1) * 2
                    G.add_edge(
                        f"level_{current_level}",
                        f"level_{next_level}",
                        weight=connections,
                        display_weight=log_weight  # Store display weight separately
                    )
            
            # Calculate layout with more space
            pos = nx.spring_layout(G, k=2.0, iterations=50)
            
            # Create edge trace
            edge_x = []
            edge_y = []
            edge_text = []
            edge_width = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                weight = G.edges[edge]['weight']
                edge_text.extend([f"Connections: {weight}", "", None])
                edge_width.extend([G.edges[edge]['display_weight']] * 3)
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='text',
                text=edge_text,
                mode='lines'
            )
            
            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Use logarithmically scaled size
                actual_size = G.nodes[node]['size']
                display_size = G.nodes[node]['display_size']
                node_size.append(display_size)
                node_text.append(f"{G.nodes[node]['title']}<br>Nodes: {actual_size}")
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    showscale=True,
                    size=node_size,
                    colorscale='YlGnBu',
                    line_width=2
                )
            )
            
            # Create figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='Tree Level Network',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
            )
            
            # Save figure
            output_path = self._ensure_output_path(output_file)
            fig.write_html(str(output_path))
            self.logger.info(f"Network visualization saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create network visualization: {str(e)}")
            raise

    def create_network_d3(self, tree_data: Dict, output_file: str, start_level: int = 1):
        """Create D3.js network visualization of tree hierarchy."""
        try:
            # Extract title from metadata if available
            title = "Tree Network Visualization"
            if 'metadata' in tree_data:
                title = tree_data['metadata'].get('title', title)
                if 'file_path' in tree_data['metadata']:
                    title += f" - {Path(tree_data['metadata']['file_path']).name}"

            # Filter nodes by level and prepare data
            nodes_data = []
            links_data = []
            node_map = {}
            max_texts_length = 1  # For node size scaling

            # First pass: collect nodes and find max text length for scaling
            for node in tree_data['tree']['nodes']:
                level = node.get('level', 0)
                if level < start_level:
                    continue
                
                # Debug: Print the first node's complete structure
                if len(nodes_data) == 0:  # Only for the first node
                    print("\nDEBUG: Example Node Structure:")
                    print("Node ID:", node.get('id'))
                    print("Level:", level)
                    print("Cluster ID:", node.get('cluster_id'))
                    print("Complete node data:", json.dumps(node, indent=2))
                    print("\n")
                
                node_id = str(node['id'])
                cluster_id = node.get('cluster_id', 0)
                texts = node.get('texts', [])
                
                if texts:
                    max_texts_length = max(max_texts_length, len(texts[0]) if texts else 0)
                
                # Truncate text for display
                display_text = texts[0][:100] + '...' if texts and len(texts[0]) > 100 else texts[0] if texts else ''
                
                # Get the best available label for the node
                metadata = node.get('metadata', {})
                
                # Use title from metadata as the primary label
                is_leaf = metadata.get('is_leaf', False)
                
                if is_leaf:
                    # For leaf nodes, prioritize doc_name with chunk info
                    doc_name = metadata.get('doc_name')
                    chunk_num = metadata.get('chunk_num')
                    if doc_name:
                        if chunk_num is not None:
                            label = f"{doc_name} (chunk {chunk_num})"
                        else:
                            label = doc_name
                    else:
                        # Fallback for leaf nodes without doc_name
                        label = (
                            metadata.get('title') or  # First try title
                            (node.get('texts', [''])[0][:50] + '...') if node.get('texts') else None or  # Then try truncated text
                            node_id  # Finally fall back to node ID
                        )
                else:
                    # For non-leaf nodes, prioritize title
                    label = (
                        metadata.get('title') or  # First try title
                        metadata.get('doc_name') or  # Then try document name
                        (node.get('texts', [''])[0][:50] + '...') if node.get('texts') else None or  # Then try truncated text
                        node_id  # Finally fall back to node ID
                    )
                
                node_data = {
                    'id': node_id,
                    'level': level,
                    'cluster': cluster_id,
                    'text': display_text,
                    'concept': label,  # Use the title from metadata as the label
                    'size': len(texts[0]) if texts else 0,  # Will be scaled later
                    'num_texts': len(texts)
                }
                nodes_data.append(node_data)
                node_map[node_id] = node_data

            # Scale node sizes
            for node in nodes_data:
                # Log scale for better size distribution
                node['size'] = 5 + (math.log2(1 + node['size']) / math.log2(1 + max_texts_length)) * 15

            # Second pass: collect links
            for node in tree_data['tree']['nodes']:
                if str(node['id']) not in node_map:
                    continue
                for child_id in node.get('children', []):
                    child_id_str = str(child_id)
                    if child_id_str in node_map:
                        links_data.append({
                            'source': str(node['id']),
                            'target': child_id_str
                        })

            # Generate the HTML with embedded D3.js
            html_content = f'''
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>{title}</title>
                <script src="https://d3js.org/d3.v7.min.js"></script>
                <style>
                    body {{ margin: 0; overflow: hidden; }}
                    #viz {{ width: 100vw; height: 100vh; }}
                    .node {{ stroke: #fff; stroke-width: 1.5px; }}
                    .link {{ stroke: #999; stroke-opacity: 0.6; }}
                    .node text {{ font-family: Arial; font-size: 10px; }}
                    .title {{ 
                        position: absolute; 
                        top: 20px; 
                        left: 50%; 
                        transform: translateX(-50%);
                        font-family: Arial;
                        font-size: 24px;
                        font-weight: bold;
                        background: rgba(255, 255, 255, 0.9);
                        padding: 10px;
                        border-radius: 5px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }}
                    .tooltip {{
                        position: absolute;
                        background: white;
                        padding: 10px;
                        border-radius: 5px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                        font-family: Arial;
                        font-size: 12px;
                        pointer-events: none;
                        max-width: 300px;
                        white-space: pre-wrap;
                    }}
                </style>
            </head>
            <body>
                <div id="title" class="title">{title}</div>
                <div id="viz"></div>
                <script>
                    const width = window.innerWidth;
                    const height = window.innerHeight;
                    const color = d3.scaleOrdinal(d3.schemeCategory10);
                    
                    // Create zoom behavior
                    const zoom = d3.zoom()
                        .scaleExtent([0.1, 4])
                        .on("zoom", zoomed);
                    
                    const svg = d3.select("#viz")
                        .append("svg")
                        .attr("width", width)
                        .attr("height", height)
                        .call(zoom);
                    
                    const g = svg.append("g");
                    
                    // Create tooltip
                    const tooltip = d3.select("body").append("div")
                        .attr("class", "tooltip")
                        .style("opacity", 0);
                    
                    // Create force simulation
                    const simulation = d3.forceSimulation()
                        .force("link", d3.forceLink().id(d => d.id).distance(100))
                        .force("charge", d3.forceManyBody().strength(-500))
                        .force("center", d3.forceCenter(width / 2, height / 2))
                        .force("collision", d3.forceCollide().radius(d => d.size + 10))
                        .force("x", d3.forceX(width / 2).strength(0.1))
                        .force("y", d3.forceY(height / 2).strength(0.1));
                    
                    // Load the data
                    const graph = {{
                        nodes: {json.dumps(nodes_data)},
                        links: {json.dumps(links_data)}
                    }};
                    
                    const link = g.append("g")
                        .selectAll("line")
                        .data(graph.links)
                        .enter().append("line")
                        .attr("class", "link");
                    
                    const node = g.append("g")
                        .selectAll("g")
                        .data(graph.nodes)
                        .enter().append("g")
                        .call(d3.drag()
                            .on("start", dragstarted)
                            .on("drag", dragged)
                            .on("end", dragended));
                    
                    node.append("circle")
                        .attr("class", "node")
                        .attr("r", d => d.size)
                        .style("fill", d => color(d.cluster));
                    
                    node.append("text")
                        .attr("dx", d => d.size + 5)
                        .attr("dy", ".35em")
                        .text(d => d.concept);  // Display concept from key_concepts
                    
                    // Add hover effects
                    node.on("mouseover", function(event, d) {{
                        tooltip.transition()
                            .duration(200)
                            .style("opacity", .9);
                        tooltip.html(`<b>${{d.concept}}</b><br>Level: ${{d.level}}<br>Cluster: ${{d.cluster}}<br>Texts: ${{d.num_texts}}<br><br>${{d.text}}`)
                            .style("left", (event.pageX + 10) + "px")
                            .style("top", (event.pageY - 10) + "px");
                    }})
                    .on("mouseout", function(d) {{
                        tooltip.transition()
                            .duration(500)
                            .style("opacity", 0);
                    }});
                    
                    simulation
                        .nodes(graph.nodes)
                        .on("tick", ticked);
                    
                    simulation.force("link")
                        .links(graph.links);
                    
                    // Add zoom functionality
                    function zoomed(event) {{
                        g.attr("transform", event.transform);
                    }}
                    
                    function ticked() {{
                        link
                            .attr("x1", d => d.source.x)
                            .attr("y1", d => d.source.y)
                            .attr("x2", d => d.target.x)
                            .attr("y2", d => d.target.y);
                        
                        node
                            .attr("transform", d => `translate(${{d.x}},${{d.y}})`);
                    }}
                    
                    function dragstarted(event, d) {{
                        if (!event.active) simulation.alphaTarget(0.3).restart();
                        d.fx = d.x;
                        d.fy = d.y;
                    }}
                    
                    function dragged(event, d) {{
                        d.fx = event.x;
                        d.fy = event.y;
                    }}
                    
                    function dragended(event, d) {{
                        if (!event.active) simulation.alphaTarget(0);
                        d.fx = null;
                        d.fy = null;
                    }}
                    
                    // Initial zoom to fit
                    svg.call(zoom.transform, d3.zoomIdentity
                        .translate(width/2, height/2)
                        .scale(0.5)
                        .translate(-width/2, -height/2));
                </script>
            </body>
            </html>
            '''

            # Save visualization
            output_path = self._ensure_output_path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            self.logger.info(f"Network visualization saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to create network visualization: {str(e)}")
            raise

    def create_interactive_tree_explorer(self, tree_data: Dict, output_file: str):
        """Create interactive tree exploration visualization."""
        try:
            # Create network graph
            G = nx.Graph()
            
            # Add nodes with metadata (only leaf nodes)
            leaf_nodes = [node for node in tree_data['tree']['nodes'] if node['metadata'].get('is_leaf', False)]
            for node in leaf_nodes:
                # Get title from metadata or first text
                node_title = node['metadata'].get('title', '')
                if not node_title and node['texts']:
                    node_title = node['texts'][0][:100] + '...' if len(node['texts'][0]) > 100 else node['texts'][0]
                if not node_title:
                    node_title = f'Node {node["id"]}'

                G.add_node(
                    node['id'],
                    level=node['level'],
                    size=len(node['texts']),
                    type=node['metadata'].get('node_type', 'unknown'),
                    texts=node['texts'],
                    title=node_title
                )
            
            # Add edges (only between leaf nodes)
            for node in leaf_nodes:
                if node.get('parent'):
                    parent = next((n for n in leaf_nodes if n['id'] == node['parent']), None)
                    if parent:
                        G.add_edge(node['id'], parent['id'])
            
            # Calculate layout with more space
            pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig.add_trace(
                go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
            )
            
            # Add nodes
            node_x = []
            node_y = []
            node_colors = []
            node_sizes = []
            node_text = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Create content preview
                content_preview = []
                for text in G.nodes[node]['texts'][:2]:
                    if len(text) > 100:
                        content_preview.append(text[:100] + '...')
                    else:
                        content_preview.append(text)
                content_text = '\n'.join(content_preview)
                
                # Create hover text
                hover_text = f"""
                <b>{G.nodes[node]['title']}</b><br>
                Level: {G.nodes[node]['level']}<br>
                Size: {G.nodes[node]['size']}<br>
                Type: {G.nodes[node]['type']}<br>
                Content Preview:<br>{content_text}
                """
                node_text.append(hover_text)
                node_colors.append(G.nodes[node]['level'])
                node_sizes.append(np.sqrt(G.nodes[node]['size']) * 20)  # Scale node sizes
            
            fig.add_trace(
                go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    text=node_text,
                    marker=dict(
                        showscale=True,
                        colorscale='YlGnBu',
                        color=node_colors,
                        size=node_sizes,
                        line_width=2,
                        line=dict(color='white'),
                        colorbar=dict(
                            title='Tree Level',
                            thickness=15,
                            len=0.5
                        )
                    )
                )
            )
            
            # Update layout
            fig.update_layout(
                title="Interactive Tree Explorer",
                title_x=0.5,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=800,
                width=1200,
                template="plotly_white",
                annotations=[
                    dict(
                        text=(
                            "Node Size: Document Length | "
                            "Color: Tree Level | "
                            "Hover for Details"
                        ),
                        showarrow=False,
                        x=0.5,
                        y=1.05,
                        xref="paper",
                        yref="paper"
                    )
                ]
            )
            
            # Save
            output_path = self._ensure_output_path(output_file)
            fig.write_html(str(output_path))
            self.logger.info(f"Interactive tree explorer saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create interactive tree explorer: {str(e)}")
            raise

    def generate_tree_statistics(self, tree_data: Dict, output_file: str = None):
        """Generate and save comprehensive tree statistics."""
        try:
            # Handle both flattened and nested tree structures
            nodes = tree_data.get('tree', {}).get('nodes', [])
            if not nodes and isinstance(tree_data, dict):
                # Try to find nodes at the root level
                nodes = tree_data.get('nodes', [])
            
            if not nodes:
                self.logger.error("No nodes found in tree data")
                return None

            stats = {
                'total_nodes': len(nodes),
                'max_depth': 0,
                'avg_text_length': 0,
                'nodes_by_level': {},
                'nodes_by_type': {}
            }
            
            total_text_length = 0
            for node in nodes:
                # Track depth/level statistics
                level = node.get('level', 0)
                stats['max_depth'] = max(stats['max_depth'], level)
                stats['nodes_by_level'][level] = stats['nodes_by_level'].get(level, 0) + 1
                
                # Track node type statistics
                node_type = node.get('metadata', {}).get('type', 'unknown')
                stats['nodes_by_type'][node_type] = stats['nodes_by_type'].get(node_type, 0) + 1
                
                # Track text length
                text = node.get('text', '')
                if text:
                    total_text_length += len(text)
            
            if stats['total_nodes'] > 0:
                stats['avg_text_length'] = total_text_length / stats['total_nodes']
            
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(stats, f, indent=2)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to generate tree statistics: {str(e)}")
            return None

    def save_visualization_data(self, tree_data: Dict, output_path: str):
        """Save visualization data for debugging."""
        try:
            viz_data = {
                'tree_structure': {
                    'nodes': [{
                        'id': node['id'],
                        'text': node['text'][:200],  # Truncate for readability
                        'filename': node.get('metadata', {}).get('filename', 'unknown'),
                        'cluster_id': node.get('cluster_id', 'unknown'),
                        'embedding_size': len(node.get('embedding', [])),
                        'metadata': node.get('metadata', {})
                    } for node in tree_data['tree']['nodes']],
                    'clusters': [{
                        'id': cluster['id'],
                        'size': cluster['size'],
                        'title': cluster.get('title', ''),
                        'summary': cluster.get('summary', '')[:200],  # Truncate for readability
                        'files': cluster.get('files', []),
                        'node_count': len(cluster.get('nodes', [])),
                        'metadata': cluster.get('metadata', {})
                    } for cluster in tree_data['tree']['clusters']],
                    'stats': tree_data.get('stats', {})
                },
                'relationships': {
                    'file_clusters': {},
                    'cluster_connections': []
                }
            }
            
            # Calculate file distribution across clusters
            for cluster in tree_data['tree']['clusters']:
                for file in cluster.get('files', []):
                    if file not in viz_data['relationships']['file_clusters']:
                        viz_data['relationships']['file_clusters'][file] = []
                    viz_data['relationships']['file_clusters'][file].append(cluster['id'])
            
            # Calculate cluster connections based on shared files
            for i, c1 in enumerate(tree_data['tree']['clusters']):
                for c2 in tree_data['tree']['clusters'][i+1:]:
                    shared_files = set(c1.get('files', [])) & set(c2.get('files', []))
                    if shared_files:
                        viz_data['relationships']['cluster_connections'].append({
                            'source': c1['id'],
                            'target': c2['id'],
                            'shared_files': list(shared_files),
                            'strength': len(shared_files)
                        })
            
            # Save data
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(viz_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Visualization data saved to: {output_path}")
            return viz_data
            
        except Exception as e:
            self.logger.error(f"Failed to save visualization data: {str(e)}")
            raise

    def print_visualization_summary(self, viz_data: Dict):
        """Print summary of visualization data."""
        print("\n=== Tree Visualization Summary ===")
        print(f"\nStructure:")
        print(f"- Total Nodes: {len(viz_data['tree_structure']['nodes'])}")
        print(f"- Total Clusters: {len(viz_data['tree_structure']['clusters'])}")
        
        print("\nCluster Details:")
        for cluster in viz_data['tree_structure']['clusters']:
            print(f"\nCluster {cluster['id']}:")
            print(f"- Size: {cluster['size']} nodes")
            print(f"- Title: {cluster['title']}")
            print(f"- Files: {', '.join(cluster['files'])}")
        
        print("\nFile Distribution:")
        for file, clusters in viz_data['relationships']['file_clusters'].items():
            print(f"\n{file}:")
            print(f"- Appears in clusters: {clusters}")
        
        print("\nCluster Connections:")
        for conn in viz_data['relationships']['cluster_connections']:
            print(f"\nConnection between clusters {conn['source']} and {conn['target']}:")
            print(f"- Shared files: {len(conn['shared_files'])}")
            print(f"- Connection strength: {conn['strength']}")
            
    def _calculate_balance_metrics(self, tree_data: Dict) -> Dict:
        """Calculate comprehensive tree balance metrics."""
        try:
            nodes = tree_data['tree']['nodes']
            if not nodes:
                return {
                    'level_distribution': {},
                    'balance_score': 0.0,
                    'branching_factor': 0.0,
                    'depth_ratio': 0.0
                }
                
            # Calculate level distribution
            level_dist = {}
            for node in nodes:
                level = node.get('level', 0)
                level_dist[level] = level_dist.get(level, 0) + 1
            
            # Calculate balance metrics
            max_level = max(level_dist.keys()) if level_dist else 0
            total_nodes = len(nodes)
            
            # For single node trees
            if total_nodes == 1:
                return {
                    'level_distribution': level_dist,
                    'balance_score': 1.0,
                    'branching_factor': 0.0,
                    'depth_ratio': 1.0
                }
            
            # Calculate metrics
            avg_nodes_per_level = total_nodes / (max_level + 1)
            
            # Calculate normalized variance
            squared_diffs = []
            for level, count in level_dist.items():
                diff = (count - avg_nodes_per_level) / total_nodes
                squared_diffs.append(diff * diff)
            
            if squared_diffs:
                level_variance = sum(squared_diffs) / len(squared_diffs)
                # Convert variance to a score between 0 and 1
                # Lower variance means better balance
                balance_score = 1.0 / (1.0 + math.sqrt(level_variance))
            else:
                balance_score = 0.0
            
            # Calculate branching factor
            branching_factors = []
            for node in nodes:
                children = [n for n in nodes if n.get('parent_id') == node['id']]
                if children:
                    branching_factors.append(len(children))
            avg_branching = sum(branching_factors) / len(branching_factors) if branching_factors else 0
            
            # Calculate depth ratio (actual/ideal)
            ideal_depth = max(1, math.log2(total_nodes + 1))  # Ideal depth for a balanced binary tree
            depth_ratio = (max_level + 1) / ideal_depth
            
            return {
                'level_distribution': level_dist,
                'balance_score': balance_score,
                'branching_factor': avg_branching,
                'depth_ratio': depth_ratio,
                'metrics': {
                    'total_nodes': total_nodes,
                    'max_level': max_level,
                    'avg_nodes_per_level': avg_nodes_per_level,
                    'ideal_depth': ideal_depth
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating balance metrics: {str(e)}")
            return {
                'level_distribution': {},
                'balance_score': 0.0,
                'branching_factor': 0.0,
                'depth_ratio': 0.0
            }

    def _get_level_distribution(self, tree_data: Dict) -> Dict:
        """Get node distribution by level."""
        try:
            nodes = tree_data['tree']['nodes']
            if isinstance(nodes, dict):  # Handle single node dict
                nodes = [nodes]  # Convert to list with single node
                
            level_dist = Counter()
            for node in nodes:
                level = node.get('level', 0)
                level_dist[level] += 1
                
            return level_dist
            
        except Exception as e:
            self.logger.error(f"Error getting level distribution: {str(e)}")
            return Counter()
            
    def _analyze_content_coverage(self, tree_data: Dict) -> Dict:
        """Analyze content coverage across the tree."""
        try:
            nodes = tree_data['tree']['nodes']
            if isinstance(nodes, dict):
                nodes = [nodes]
                
            total_texts = 0
            covered_texts = 0
            coverage_by_level = {}
            
            for node in nodes:
                level = node.get('level', 0)
                texts = node.get('texts', [])
                
                if texts:
                    total_texts += len(texts)
                    covered_texts += len(texts)
                    
                if level not in coverage_by_level:
                    coverage_by_level[level] = {'total': 0, 'covered': 0}
                coverage_by_level[level]['total'] += 1
                if texts:
                    coverage_by_level[level]['covered'] += 1
            
            coverage_scores = {
                level: data['covered'] / data['total'] if data['total'] > 0 else 0
                for level, data in coverage_by_level.items()
            }
            
            return {
                'total_texts': total_texts,
                'covered_texts': covered_texts,
                'coverage_by_level': coverage_scores,
                'overall_coverage': covered_texts / total_texts if total_texts > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing content coverage: {str(e)}")
            return {
                'total_texts': 0,
                'covered_texts': 0,
                'coverage_by_level': {},
                'overall_coverage': 0
            }
    
    def _calculate_doc_quality_metrics(self, tree_data: Dict) -> Dict:
        """Calculate document quality metrics."""
        quality_metrics = {
            'readability_scores': [],
            'completeness_scores': [],
            'structure_scores': [],
            'freshness_scores': [],
            'overall_scores': [],
            'document_ids': [],  # Added to track documents
            'quality_issues': []  # Added to track specific issues
        }
        
        for node in tree_data['tree']['nodes']:
            if not node.get('texts') or not node.get('metadata'):
                continue
                
            # Calculate readability (Flesch reading ease)
            try:
                readability = np.mean([TextBlob(text).sentiment.polarity + 1 for text in node['texts']]) * 50 + 50
            except:
                readability = 0
                quality_metrics['quality_issues'].append(f"Failed to calculate readability for node {node.get('id')}")
            
            # Calculate completeness (based on text length and metadata)
            text_length = len('\n'.join(node['texts']))
            completeness = min(100, text_length / 500 * 100)
            if text_length < 100:
                quality_metrics['quality_issues'].append(f"Very short text content in node {node.get('id')}")
            
            # Calculate structure score (based on metadata presence)
            expected_metadata = ['id', 'level', 'title', 'last_modified', 'file_path']
            structure = len([1 for field in expected_metadata if field in node['metadata']]) / len(expected_metadata) * 100
            if structure < 60:
                quality_metrics['quality_issues'].append(f"Missing important metadata in node {node.get('id')}")
            
            # Calculate freshness (based on last_modified timestamp)
            if 'last_modified' in node['metadata']:
                age_days = (datetime.now().timestamp() - node['metadata']['last_modified']) / (24 * 3600)
                freshness = max(0, 100 - (age_days / 30) * 100)  # Scale based on 30 days
                if age_days > 90:
                    quality_metrics['quality_issues'].append(f"Content is over 90 days old in node {node.get('id')}")
            else:
                freshness = 0
                quality_metrics['quality_issues'].append(f"Missing last_modified timestamp in node {node.get('id')}")
            
            # Calculate overall score (weighted average)
            overall = np.mean([
                readability * 0.3,
                completeness * 0.3,
                structure * 0.2,
                freshness * 0.2
            ])
            
            quality_metrics['readability_scores'].append(readability)
            quality_metrics['completeness_scores'].append(completeness)
            quality_metrics['structure_scores'].append(structure)
            quality_metrics['freshness_scores'].append(freshness)
            quality_metrics['overall_scores'].append(overall)
            quality_metrics['document_ids'].append(node.get('id', 'unknown'))
        
        # Store quality metrics in tree data for persistence
        tree_data['quality_metrics'] = quality_metrics
        
        return quality_metrics

    def create_tree_health_dashboard(self, tree_data: Dict, output_file: str):
        """Create comprehensive tree health visualization dashboard."""
        try:
            # Calculate metrics
            balance_metrics = self._calculate_balance_metrics(tree_data)
            level_dist = self._get_level_distribution(tree_data)
            coverage = self._analyze_content_coverage(tree_data)
            quality = self._calculate_doc_quality_metrics(tree_data)
            
            # Create subplots with proper specs
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Tree Health Score', 'Level Distribution',
                    'Content Coverage', 'Document Quality',
                    'Cluster Balance', 'Node Distribution'
                ),
                specs=[
                    [{"type": "indicator"}, {"type": "xy"}],
                    [{"type": "xy"}, {"type": "xy"}],
                    [{"type": "xy"}, {"type": "xy"}]
                ],
                vertical_spacing=0.3,
                horizontal_spacing=0.2
            )
            
            # Main health score indicator (top)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=balance_metrics['balance_score'],
                    title={'text': "Overall Tree Health"},
                    gauge={'axis': {'range': [0, 1]}},  
                    domain={'row': 0, 'column': 0}
                ),
                row=1, col=1
            )
            
            # Supporting visualizations (below)
            fig.add_trace(
                go.Bar(x=list(level_dist.keys()), y=list(level_dist.values()), name="Level Distribution"),
                row=1, col=2
            )
            
            # Convert coverage matrix to bar chart for better visibility
            coverage_by_level = coverage['coverage_by_level']
            fig.add_trace(
                go.Bar(
                    x=list(coverage_by_level.keys()),
                    y=list(coverage_by_level.values()),
                    name="Content Coverage by Level"
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(x=list(quality.keys()), y=list(quality.values()), name="Document Quality"),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=['Balance Score', 'Depth Score', 'Distribution Score'],
                    y=[balance_metrics['balance_score'], balance_metrics['depth_ratio'], balance_metrics['branching_factor']],
                    name="Balance Metrics"
                ),
                row=3, col=1
            )
            
            # Node distribution
            node_counts = Counter(node['level'] for node in tree_data['tree']['nodes'])
            fig.add_trace(
                go.Bar(
                    x=list(node_counts.keys()),
                    y=list(node_counts.values()),
                    name="Nodes per Level"
                ),
                row=3, col=2
            )
            
            # Update layout
            fig.update_layout(
                height=1200,
                width=1000,
                showlegend=False,
                title_text="Document Tree Health Analysis"
            )
            
            # Save dashboard
            output_path = self._ensure_output_path(output_file)
            fig.write_html(str(output_path))
            self.logger.info(f"Tree health dashboard saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create tree health dashboard: {str(e)}")
            raise
            
    def create_document_quality_report(self, tree_data: Dict, output_file: str):
        """Create document quality analysis visualization."""
        try:
            # Calculate quality metrics if not already present
            quality_metrics = tree_data.get('quality_metrics')
            if not quality_metrics:
                quality_metrics = self._calculate_doc_quality_metrics(tree_data)
            
            # 1. Create parallel coordinates plot
            fig_parallel = go.Figure()
            fig_parallel.add_trace(
                go.Parcoords(
                    line=dict(
                        color=quality_metrics['overall_scores'],
                        colorscale='Viridis',
                        showscale=True,
                        cmin=0,
                        cmax=100
                    ),
                    dimensions=[
                        dict(range=[0, 100], label='Readability', values=quality_metrics['readability_scores']),
                        dict(range=[0, 100], label='Completeness', values=quality_metrics['completeness_scores']),
                        dict(range=[0, 100], label='Structure', values=quality_metrics['structure_scores']),
                        dict(range=[0, 100], label='Freshness', values=quality_metrics['freshness_scores']),
                        dict(range=[0, 100], label='Overall', values=quality_metrics['overall_scores'])
                    ]
                )
            )
            fig_parallel.update_layout(
                title="Document Quality Overview",
                height=500
            )
            
            # 2. Create distribution plot
            fig_dist = go.Figure()
            for metric in ['readability_scores', 'completeness_scores', 'structure_scores', 'freshness_scores']:
                fig_dist.add_trace(
                    go.Violin(
                        y=quality_metrics[metric],
                        name=metric.replace('_scores', '').title(),
                        box_visible=True,
                        meanline_visible=True
                    )
                )
            fig_dist.update_layout(
                title="Quality Metrics Distribution",
                height=500,
                showlegend=True
            )
            
            # 3. Create poor documents table
            poor_indices = np.argsort(quality_metrics['overall_scores'])[:10]
            poor_docs_data = []
            for idx in poor_indices:
                poor_docs_data.append([
                    quality_metrics['document_ids'][idx],
                    f"{quality_metrics['overall_scores'][idx]:.1f}",
                    f"{quality_metrics['readability_scores'][idx]:.1f}",
                    f"{quality_metrics['completeness_scores'][idx]:.1f}",
                    f"{quality_metrics['structure_scores'][idx]:.1f}",
                    f"{quality_metrics['freshness_scores'][idx]:.1f}"
                ])
            
            fig_table = go.Figure(data=[go.Table(
                header=dict(
                    values=['Document ID', 'Overall', 'Readability', 'Completeness', 'Structure', 'Freshness'],
                    font=dict(size=12),
                    align="left"
                ),
                cells=dict(
                    values=list(zip(*poor_docs_data)),
                    font=dict(size=11),
                    align="left"
                )
            )])
            fig_table.update_layout(
                title="Documents Needing Attention",
                height=400
            )
            
            # 4. Create issues bar chart
            issues_summary = Counter(quality_metrics['quality_issues'])
            fig_issues = go.Figure(data=[
                go.Bar(
                    x=list(issues_summary.values()),
                    y=list(issues_summary.keys()),
                    orientation='h'
                )
            ])
            fig_issues.update_layout(
                title="Quality Issues Summary",
                height=500,
                margin=dict(l=300)  # Add margin for long issue descriptions
            )
            
            # Save visualizations to separate files
            output_path = self._ensure_output_path(output_file)
            parallel_path = output_path.parent / 'quality_overview.html'
            dist_path = output_path.parent / 'quality_distribution.html'
            table_path = output_path.parent / 'quality_poor_docs.html'
            issues_path = output_path.parent / 'quality_issues.html'
            
            fig_parallel.write_html(str(parallel_path))
            fig_dist.write_html(str(dist_path))
            fig_table.write_html(str(table_path))
            fig_issues.write_html(str(issues_path))
            
            # Create combined HTML report
            report_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Document Quality Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .summary {{ background: #f5f5f5; padding: 20px; margin-bottom: 20px; border-radius: 5px; }}
                    .viz-container {{ margin-bottom: 30px; }}
                    iframe {{ width: 100%; border: none; }}
                </style>
            </head>
            <body>
                <h1>Document Quality Analysis Report</h1>
                
                <div class="summary">
                    <h2>Summary Statistics</h2>
                    <p>Total Documents: {len(quality_metrics['overall_scores'])}</p>
                    <p>Average Quality Score: {np.mean(quality_metrics['overall_scores']):.1f}</p>
                    <p>Documents Needing Attention: {sum(1 for s in quality_metrics['overall_scores'] if s < 60)}</p>
                    <p>High Quality Documents: {sum(1 for s in quality_metrics['overall_scores'] if s >= 90)}</p>
                </div>
                
                <div class="viz-container">
                    <h2>Document Quality Overview</h2>
                    <iframe src="quality_overview.html" height="550px"></iframe>
                </div>
                
                <div class="viz-container">
                    <h2>Quality Metrics Distribution</h2>
                    <iframe src="quality_distribution.html" height="550px"></iframe>
                </div>
                
                <div class="viz-container">
                    <h2>Documents Needing Attention</h2>
                    <iframe src="quality_poor_docs.html" height="450px"></iframe>
                </div>
                
                <div class="viz-container">
                    <h2>Quality Issues Summary</h2>
                    <iframe src="quality_issues.html" height="550px"></iframe>
                </div>
            </body>
            </html>
            """
            
            with open(output_path, 'w') as f:
                f.write(report_html)
            
            # Save detailed report data
            report_data = {
                'summary': {
                    'total_documents': len(quality_metrics['overall_scores']),
                    'average_quality': float(np.mean(quality_metrics['overall_scores'])),
                    'poor_quality_count': sum(1 for s in quality_metrics['overall_scores'] if s < 60),
                    'excellent_quality_count': sum(1 for s in quality_metrics['overall_scores'] if s >= 90)
                },
                'quality_metrics': quality_metrics,
                'poor_documents': poor_docs_data,
                'quality_issues': dict(issues_summary)
            }
            
            report_path = self.output_dir / 'document_quality_report.json'
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            self.logger.info(f"Document quality report saved to {output_path}")
            self.logger.info(f"Detailed quality data saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create document quality report: {str(e)}")
            raise

    def create_network_visualization(self, tree_data: Dict, output_file: str = None) -> go.Figure:
        """Create an interactive network visualization of the tree.
        
        Args:
            tree_data: Tree data in standardized format (from Node.to_dict())
            output_file: Optional path to save the visualization
            
        Returns:
            Plotly figure object
        """
        # Extract nodes and build parent-child relationships
        nodes = tree_data.get('nodes', [])
        node_map = {node['id']: node for node in nodes}
        edges = []
        
        # Create edges from parent-child relationships
        for node in nodes:
            if 'children' in node:
                parent_id = node['id']
                for child_id in node['children']:
                    edges.append((parent_id, child_id))
        
        # Create networkx graph
        G = nx.Graph()
        
        # Add nodes with metadata
        for node in nodes:
            node_id = node['id']
            level = node['metadata']['level']
            size = node['metadata']['size']
            is_leaf = node['metadata']['is_leaf']
            
            # Calculate node color based on level
            color_idx = min(level, len(self.color_scale)-1)
            color = self.color_scale[color_idx]
            
            # Calculate node size based on content
            base_size = self.node_size_range[0]
            size_scale = (self.node_size_range[1] - self.node_size_range[0])
            node_size = base_size + min(size_scale, math.log2(1 + size) * 10)
            
            G.add_node(
                node_id,
                level=level,
                color=color,
                size=node_size,
                is_leaf=is_leaf,
                label=node['text'][:50] + '...' if len(node['text']) > 50 else node['text']
            )
        
        # Add edges
        G.add_edges_from(edges)
        
        # Create layout
        pos = nx.spring_layout(G, k=1/math.sqrt(G.number_of_nodes()))
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Create content preview
            content_preview = []
            for text in G.nodes[node]['texts'][:2]:
                if len(text) > 100:
                    content_preview.append(text[:100] + '...')
                else:
                    content_preview.append(text)
            content_text = '\n'.join(content_preview)
            
            # Create hover text
            hover_text = f"""
            <b>{G.nodes[node]['title']}</b><br>
            Level: {G.nodes[node]['level']}<br>
            Size: {G.nodes[node]['size']}<br>
            Type: {G.nodes[node]['type']}<br>
            Content Preview:<br>{content_text}
            """
            node_text.append(hover_text)
            node_colors.append(G.nodes[node]['level'])
            node_sizes.append(G.nodes[node]['size'])
        
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                color=node_colors,
                size=node_sizes,
                line_width=2,
                line=dict(color='white'),
                colorbar=dict(
                    title='Tree Level',
                    thickness=15,
                    len=0.5
                )
            )
        ))
        
        # Update layout
        fig.update_layout(
            title='Tree Structure Visualization',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        # Save if output file specified
        if output_file:
            fig.write_html(self.output_dir / output_file)
        
        return fig

    def create_tree_visualization(self, tree_data: Dict, output_file: str = None) -> go.Figure:
        """Create a hierarchical tree visualization.
        
        Args:
            tree_data: Tree data in standardized format (from Node.to_dict())
            output_file: Optional path to save the visualization
            
        Returns:
            Plotly figure object
        """
        # Extract nodes and build hierarchy
        nodes = tree_data.get('nodes', [])
        if not nodes:
            self.logger.warning("No nodes found in tree data")
            return go.Figure()  # Return empty figure
            
        node_map = {node['id']: node for node in nodes}
        
        # Create figure
        fig = go.Figure()
        
        def add_node_to_plot(node_id: str, x: float, y: float, level_width: float = 4.0):
            """Recursively add nodes and their children to the plot."""
            node = node_map[node_id]
            children = node.get('children', [])
            num_children = len(children)
            
            # Calculate node size based on content
            text_length = len(node.get('text', ''))
            node_size = 20 + min(30, math.log2(1 + text_length))
            
            # Get node color based on level
            level = node.get('level', 0)
            color = self.level_colors.get(level, "#808080")  # Default gray if level not found
            
            # Get the best available label for the node
            metadata = node.get('metadata', {})
            is_leaf = metadata.get('is_leaf', False)
            
            if level == 0:
                # For root node, always use 'root'
                node_title = 'root'
            elif is_leaf:
                # For leaf nodes, prioritize doc_name with chunk info
                doc_name = metadata.get('doc_name')
                chunk_num = metadata.get('chunk_num')
                if doc_name:
                    if chunk_num is not None:
                        node_title = f"{doc_name} (chunk {chunk_num})"
                    else:
                        node_title = doc_name
                else:
                    # Fallback for leaf nodes without doc_name
                    node_title = (
                        metadata.get('title') or
                        (node.get('texts', [''])[0][:50] + '...') if node.get('texts') else None or
                        node_id
                    )
            else:
                # For non-leaf nodes, prioritize title
                node_title = (
                    metadata.get('title') or
                    metadata.get('doc_name') or
                    (node.get('texts', [''])[0][:50] + '...') if node.get('texts') else None or
                    node_id
                )
            
            # Create node text
            hover_text = f"""
            <b>{node_title}</b><br>
            Level: {level}<br>
            Type: {metadata.get('node_type', 'unknown')}<br>
            Children: {num_children}
            """
            
            # Add node
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(
                    size=node_size,
                    color=color,
                    line=dict(color='darkgray', width=1)
                ),
                text=[node_title],
                textposition="top center",
                hoverinfo='text',
                hovertext=hover_text,
                showlegend=False
            ))
            
            # Add children
            if children:
                child_dx = level_width / num_children
                for i, child_id in enumerate(children):
                    child_x = x - level_width/2 + child_dx/2 + i*child_dx
                    child_y = y - 1
                    
                    # Add edge
                    fig.add_trace(go.Scatter(
                        x=[x, child_x],
                        y=[y, child_y],
                        mode='lines',
                        line=dict(color='gray', width=1),
                        hoverinfo='none',
                        showlegend=False
                    ))
                    
                    # Recursively add child
                    add_node_to_plot(child_id, child_x, child_y, child_dx)
        
        # Find root node - try different methods
        root_id = None
        
        # Method 1: Look for level 0
        try:
            root_id = next(node['id'] for node in nodes if node.get('metadata', {}).get('level') == 0)
        except StopIteration:
            # Method 2: Look for node without parent
            try:
                root_id = next(node['id'] for node in nodes if not node.get('parent_id'))
            except StopIteration:
                # Method 3: Just take the first node
                if nodes:
                    root_id = nodes[0]['id']
                    self.logger.warning("No root node found, using first node as root")
                else:
                    self.logger.error("No nodes found in tree data")
                    return fig
        
        if root_id:
            add_node_to_plot(root_id, 0, 0)
        
        # Update layout
        fig.update_layout(
            title='Hierarchical Tree Visualization',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        # Save if output file specified
        if output_file:
            fig.write_html(self.output_dir / output_file)
        
        return fig