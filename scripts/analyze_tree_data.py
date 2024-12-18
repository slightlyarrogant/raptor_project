#!/usr/bin/env python3

import pickle
from pathlib import Path
from typing import Optional
import sys
from src.models.data_models import TreeData, NodeData

def print_node_details(node: NodeData, level: int = 0, max_text_length: int = 100):
    """Print details of a single node with proper indentation."""
    indent = "  " * level
    print(f"\n{indent}Node ID: {node.id}")
    print(f"{indent}Level: {node.level}")
    
    # Print title if exists
    if hasattr(node, 'title') and node.title:
        print(f"{indent}Title: {node.title}")
    
    # Print text preview
    if node.text:
        text_preview = node.text[:max_text_length] + "..." if len(node.text) > max_text_length else node.text
        print(f"{indent}Text Preview: {text_preview}")
    
    # Print metadata
    if node.metadata:
        print(f"{indent}Metadata:")
        for key, value in node.metadata.items():
            if isinstance(value, str):
                value = value[:max_text_length] + "..." if len(value) > max_text_length else value
            print(f"{indent}  {key}: {value}")
    
    # Print parent ID
    if node.parent_id:
        print(f"{indent}Parent ID: {node.parent_id}")
    
    # Print children IDs
    if node.children_ids:
        print(f"{indent}Children IDs: {node.children_ids}")
    
    print(f"{indent}{'-' * 80}")

def analyze_tree_data(tree_data: TreeData):
    """Analyze and print TreeData structure."""
    print("\n=== Tree Structure Analysis ===")
    print(f"Total nodes: {len(tree_data.nodes)}")
    print(f"Total edges: {len(tree_data.edges)}")
    
    # Find root node
    root_node = next((node for node in tree_data.nodes if node.level == 0 or not node.parent_id), None)
    if root_node:
        print(f"Root ID: {root_node.id}")
    else:
        print("No root node found!")
    
    # Count nodes per level
    level_counts = {}
    for node in tree_data.nodes:
        level_counts[node.level] = level_counts.get(node.level, 0) + 1
    
    print("\nNodes per level:")
    for level in sorted(level_counts.keys()):
        print(f"Level {level}: {level_counts[level]} nodes")
    
    # Create node lookup for easier traversal
    node_lookup = {node.id: node for node in tree_data.nodes}
    
    def print_tree_recursive(node_id: str, level: int = 0):
        """Print tree structure recursively."""
        node = node_lookup.get(node_id)
        if not node:
            return
            
        print_node_details(node, level)
        
        # Find children
        children = [edge.target for edge in tree_data.edges if edge.source == node_id]
        for child_id in children:
            print_tree_recursive(child_id, level + 1)
    
    print("\n=== Tree Hierarchy ===")
    if root_node:
        print_tree_recursive(root_node.id)
    else:
        print("Cannot print hierarchy without root node")

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_tree_data.py <path_to_pickle_file>")
        sys.exit(1)
    
    pickle_path = Path(sys.argv[1])
    if not pickle_path.exists():
        print(f"Error: File not found: {pickle_path}")
        sys.exit(1)
    
    try:
        with open(pickle_path, 'rb') as f:
            tree_data = pickle.load(f)
        
        if not isinstance(tree_data, TreeData):
            print(f"Error: File does not contain TreeData (found {type(tree_data)})")
            sys.exit(1)
            
        analyze_tree_data(tree_data)
        
    except Exception as e:
        print(f"Error analyzing tree data: {str(e)}")
        raise

if __name__ == "__main__":
    main()
