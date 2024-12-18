#!/usr/bin/env python3

import pickle
import json
from pathlib import Path
import logging

# Set up logging with a more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simplified format for readability
)
logger = logging.getLogger(__name__)

def print_node_details(node, prefix="", is_last=True):
    """Print detailed information about a node with proper indentation."""
    # Create the branch prefix
    branch = "└── " if is_last else "├── "
    
    # Get metadata
    metadata = node.metadata if hasattr(node, 'metadata') else {}
    
    # Print node information
    logger.info(f"{prefix}{branch}Node ID: {node.id}")
    logger.info(f"{prefix}    Level: {node.level}")
    
    # Print all metadata with better structure
    if hasattr(node, 'metadata'):
        logger.info(f"{prefix}    Metadata:")
        for key, value in node.metadata.items():
            if isinstance(value, dict):
                logger.info(f"{prefix}        {key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, str):
                        subvalue = subvalue[:100] + "..." if len(subvalue) > 100 else subvalue
                        subvalue = subvalue.replace('\n', ' ')
                    logger.info(f"{prefix}            {subkey}: {subvalue}")
            else:
                if isinstance(value, str):
                    value = value[:100] + "..." if len(value) > 100 else value
                    value = value.replace('\n', ' ')
                logger.info(f"{prefix}        {key}: {value}")
    
    # Print text content if available
    if hasattr(node, 'text'):
        text_preview = node.text[:100] + "..." if len(node.text) > 100 else node.text
        text_preview = text_preview.replace('\n', ' ')
        logger.info(f"{prefix}    Text: {text_preview}")
    
    logger.info("")  # Empty line for readability

def find_example_path(tree_data):
    """Find an example path from root to leaf."""
    # Create parent-child map
    parent_child_map = {}
    for edge in tree_data.edges:
        if edge.source not in parent_child_map:
            parent_child_map[edge.source] = []
        parent_child_map[edge.source].append(edge.target)
    
    # Find root node (node that's not a target in any edge)
    target_nodes = set(edge.target for edge in tree_data.edges)
    source_nodes = set(edge.source for edge in tree_data.edges)
    root_candidates = source_nodes - target_nodes
    root_id = next(iter(root_candidates)) if root_candidates else tree_data.nodes[0].id
    
    # Find a path from root to leaf
    path = []
    current_id = root_id
    node_dict = {node.id: node for node in tree_data.nodes}
    
    while current_id:
        path.append(node_dict[current_id])
        children = parent_child_map.get(current_id, [])
        current_id = children[0] if children else None
    
    return path

def main():
    # Load tree data
    tree_data_path = "data/test/visualizations/data/tree_data.pkl"
    with open(tree_data_path, 'rb') as f:
        tree_data = pickle.load(f)
    
    # Find an example path
    path = find_example_path(tree_data)
    
    # Print path details
    logger.info("=== Example Path from Root to Leaf ===\n")
    for i, node in enumerate(path):
        is_last = (i == len(path) - 1)
        prefix = "    " * i
        print_node_details(node, prefix, is_last)
        
    # Print summary statistics
    logger.info("=== Tree Statistics ===")
    logger.info(f"Total nodes: {len(tree_data.nodes)}")
    logger.info(f"Total edges: {len(tree_data.edges)}")
    logger.info(f"Example path length: {len(path)}")

if __name__ == "__main__":
    main()
