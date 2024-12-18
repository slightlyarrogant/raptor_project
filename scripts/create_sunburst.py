#!/usr/bin/env python3

import json
from pathlib import Path
import plotly.graph_objects as go
from collections import defaultdict
import logging
import click
from typing import Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_node_values(ids, parents, levels):
    """Calculate values to ensure even distribution and better visualization."""
    values = [1] * len(ids)  # Initialize all values to 1
    
    # Create parent-child mapping
    parent_children = defaultdict(list)
    for i, (id_, parent) in enumerate(zip(ids, parents)):
        if parent in ids:
            parent_children[parent].append(i)
    
    # Count total descendants for each node
    def count_descendants(node_idx):
        children = parent_children[ids[node_idx]]
        if not children:
            return 1
        return 1 + sum(count_descendants(child) for child in children)
    
    # Calculate values based on number of descendants
    for i in range(len(ids)):
        values[i] = count_descendants(i)
    
    return values

def create_parent_map(ids):
    """Create a mapping of branch numbers to their full IDs."""
    branch_map = {}
    for node_id in ids:
        if node_id.startswith('branch_'):
            parts = node_id.split('_')
            branch_num = parts[1]
            branch_map[branch_num] = node_id
    return branch_map

def parse_node_id(node_id, branch_map):
    """Parse node ID to determine its level and parent."""
    if node_id == 'root':
        return 0, ''
    
    parts = node_id.split('_')
    
    if parts[0] == 'branch':
        return 1, 'root'
    elif parts[0] == 'twig':
        # twig_X_Y_timestamp -> parent is branch_X with its original timestamp
        branch_num = parts[1]
        if branch_num in branch_map:
            return 2, branch_map[branch_num]
        logger.warning(f"Could not find branch {branch_num} for twig {node_id}")
        return 2, 'root'
    elif parts[0] == 'leaf':
        # leaf_X_Y_Z_timestamp -> parent is twig_X_Y_timestamp
        branch_num = parts[1]
        twig_num = parts[2]
        timestamp = parts[-1]
        return 3, f'twig_{branch_num}_{twig_num}_{timestamp}'
    return 0, 'root'

def validate_node_data(ids, labels, parents, values, levels):
    """Validate node data and print detailed debugging information."""
    logger.info("\n=== Data Validation ===")
    
    # 1. Check lengths
    lengths = {
        'ids': len(ids),
        'labels': len(labels),
        'parents': len(parents),
        'values': len(values),
        'levels': len(levels)
    }
    logger.info(f"Array lengths: {lengths}")
    assert all(l == lengths['ids'] for l in lengths.values()), "All arrays must have the same length!"
    
    # 2. Check for empty or None values
    for i, (id_, label, parent, value, level) in enumerate(zip(ids, labels, parents, values, levels)):
        assert id_, f"Empty ID at index {i}"
        assert label is not None, f"None label at index {i}"
        assert value > 0, f"Invalid value {value} at index {i}"
        assert level >= 0, f"Invalid level {level} at index {i}"
    
    # 3. Print hierarchy details
    logger.info("\n=== Hierarchy Analysis ===")
    by_level = {0: [], 1: [], 2: [], 3: []}
    for i, (id_, level, parent) in enumerate(zip(ids, levels, parents)):
        by_level[level].append((id_, parent))
    
    for level, nodes in by_level.items():
        logger.info(f"\nLevel {level} nodes:")
        for id_, parent in nodes:
            logger.info(f"  {id_} -> parent: {parent}")
    
    # 4. Validate parent-child relationships
    logger.info("\n=== Parent-Child Validation ===")
    id_set = set(ids)
    for i, (id_, parent) in enumerate(zip(ids, parents)):
        if parent and parent not in id_set:
            raise ValueError(f"Node {id_} has non-existent parent {parent}")
    
    # 5. Check for circular references
    def find_path_to_root(node_id, path=None):
        if path is None:
            path = []
        if node_id in path:
            raise ValueError(f"Circular reference detected: {' -> '.join(path + [node_id])}")
        if not node_id or node_id == 'root':
            return
        parent = parents[ids.index(node_id)]
        find_path_to_root(parent, path + [node_id])
    
    for id_ in ids:
        find_path_to_root(id_)
    
    # 6. Print sample paths
    logger.info("\n=== Sample Paths to Root ===")
    for level in [3, 2, 1]:  # Print a sample path from each level
        for id_ in [n[0] for n in by_level[level]][:1]:  # Take first node of each level
            path = []
            current = id_
            while current:
                path.append(current)
                current = parents[ids.index(current)] if current != 'root' else ''
            logger.info(f"Path for {id_}: {' -> '.join(path)}")

def process_tree_data(tree_data: Dict) -> Dict:
    """Transform the original tree data into a format suitable for visualization."""
    try:
        if not isinstance(tree_data, dict):
            raise ValueError("Input must be a dictionary")
        
        # Initialize the processed data structure
        processed_data = {
            'tree': {
                'nodes': []
            }
        }
        
        def process_node(node: Dict) -> Dict:
            """Process a single node and its metadata."""
            node_id = node.get('id', '')
            metadata = node.get('metadata', {})
            
            # Extract or generate title
            title = metadata.get('title', node_id)
            if not title:
                title = node_id
            
            # Get parent ID and texts
            parent_id = metadata.get('parent_id', '')
            texts = node.get('texts', [])
            description = ' '.join(texts) if texts else ''
            
            # Create processed node
            processed_node = {
                'id': node_id,
                'metadata': {
                    'title': title,
                    'parent_id': parent_id
                },
                'texts': [description] if description else []
            }
            
            return processed_node
        
        # Process all nodes in the tree
        if 'tree' in tree_data and 'nodes' in tree_data['tree']:
            nodes = tree_data['tree']['nodes']
            processed_data['tree']['nodes'] = [process_node(node) for node in nodes]
            
            # Log sample of processed data
            if processed_data['tree']['nodes']:
                logger.info("\n=== Sample Processed Node ===")
                logger.info(json.dumps(processed_data['tree']['nodes'][0], indent=2))
        
        return processed_data
    
    except Exception as e:
        logger.error(f"Error processing tree data: {str(e)}")
        raise

def create_sunburst_from_tree(tree_data: Dict, output_file: str):
    """Create a sunburst visualization from tree data."""
    try:
        # Process the tree data first
        processed_data = process_tree_data(tree_data)
        nodes = processed_data['tree']['nodes']
        
        # Extract data for visualization
        ids = []
        labels = []
        parents = []
        descriptions = []
        
        # Process each node
        for node in nodes:
            node_id = node.get('id')
            if not node_id:
                logger.warning(f"Node missing ID: {node}")
                continue
            
            metadata = node.get('metadata', {})
            label = metadata.get('title', node_id)
            parent_id = metadata.get('parent_id', '')
            description = ' '.join(node.get('texts', []))
            
            ids.append(node_id)
            labels.append(label)
            parents.append(parent_id)
            descriptions.append(description)
        
        # Create hover text
        hover_text = [f"{label}<br>{desc}" if desc else label 
                     for label, desc in zip(labels, descriptions)]
        
        # Create sunburst figure
        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels,
            parents=parents,
            hovertext=hover_text,
            hoverinfo="text",
            insidetextorientation='horizontal',
            textfont=dict(
                size=12,
                family="Arial"
            ),
            maxdepth=-1,
            marker=dict(
                line=dict(color='white', width=0.5)
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Code Structure Visualization",
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=24)
            ),
            width=1200,
            height=1000,
            margin=dict(t=100, l=10, r=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            uniformtext=dict(
                mode='show',
                minsize=10
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'y': 1.1,
                'x': 0.1,
                'buttons': [{
                    'method': 'relayout',
                    'label': 'Reset View',
                    'args': ['sunburstcolorway', None]
                }]
            }]
        )
        
        # Save visualization
        fig.write_html(
            output_file,
            include_plotlyjs=True,
            full_html=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['zoom', 'pan', 'select', 'lasso2d'],
                'scrollZoom': False,
                'responsive': True,
                'doubleClick': 'reset'
            }
        )
        
        logger.info(f"Sunburst visualization saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error creating sunburst visualization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def create_sunburst_from_debug(debug_file: str, output_file: str):
    """Create a sunburst visualization from debug data."""
    try:
        # Load debug data
        with open(debug_file, 'r') as f:
            debug_data = json.load(f)
        
        # Extract sunburst data
        data = debug_data['sunburst_data']
        
        # Create branch mapping
        branch_map = create_parent_map(data['ids'])
        
        # Recalculate parents based on node IDs
        parents = []
        levels = []
        for node_id in data['ids']:
            level, parent = parse_node_id(node_id, branch_map)
            parents.append(parent)
            levels.append(level)
        
        # Calculate values based on level
        values = calculate_node_values(data['ids'], parents, levels)
        
        # Validate and debug data
        validate_node_data(data['ids'], data['labels'], parents, values, levels)
        
        # Print exact data being passed to plotly
        logger.info("\n=== Plotly Input Data ===")
        sample_size = 10
        logger.info(f"First {sample_size} nodes of plotly input:")
        for i in range(min(sample_size, len(data['ids']))):
            logger.info(f"\nNode {i}:")
            logger.info(f"  id: {data['ids'][i]}")
            logger.info(f"  label: {data['labels'][i]}")
            logger.info(f"  parent: {parents[i]}")
            logger.info(f"  value: {values[i]}")
            logger.info(f"  level: {levels[i]}")
            logger.info(f"  color: {('#2E86C1' if levels[i] == 0 else '#27AE60' if levels[i] == 1 else '#F39C12' if levels[i] == 2 else '#E74C3C')}")
        
        # Create figure
        fig = go.Figure(go.Sunburst(
            ids=data['ids'],
            labels=data['labels'],
            parents=parents,
            values=values,
            branchvalues='total',  
            customdata=data['customdata'],
            hovertemplate='<b>%{label}</b><br>%{customdata}<extra></extra>',
            marker=dict(
                colors=[
                    '#2E86C1' if level == 0 else  # Root - Blue
                    '#27AE60' if level == 1 else  # Branch - Green
                    '#F39C12' if level == 2 else  # Twig - Orange
                    '#E74C3C'                     # Leaf - Red
                    for level in levels
                ],
                line=dict(color='white', width=1)
            ),
            maxdepth=None,  
            insidetextorientation='radial',
            textfont=dict(size=12, color='white'),
            textinfo='label',
            leaf=dict(opacity=0.7)
        ))
        
        # Debug the figure data
        logger.info("\n=== Figure Data ===")
        figure_data = fig.data[0]
        logger.info(f"Figure data type: {type(figure_data)}")
        logger.info(f"Number of nodes in figure: {len(figure_data.ids)}")
        logger.info(f"Data fields present: {[field for field in dir(figure_data) if not field.startswith('_')]}")
        
        # Update layout
        fig.update_layout(
            title={
                'text': "Document Hierarchy",
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24)
            },
            width=1000,  
            height=1000,
            margin=dict(t=50, l=0, r=0, b=0),
            showlegend=False,
            template="plotly_white",
            uniformtext=dict(minsize=8, mode='hide'),
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        # Save figure
        fig.write_html(output_file)
        logger.info(f"\nSunburst visualization saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error creating sunburst visualization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
def main(input_file: str, output_file: str):
    """Create a sunburst visualization from a tree data JSON file."""
    try:
        # Read input file
        with open(input_file, 'r') as f:
            tree_data = json.load(f)
        
        # Create visualization
        if 'tree' in tree_data:
            create_sunburst_from_tree(tree_data, output_file)
        else:
            create_sunburst_from_debug(input_file, output_file)
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
