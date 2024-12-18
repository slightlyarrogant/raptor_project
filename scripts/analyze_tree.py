#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Dict, Any

def print_node_info(node: Dict[str, Any], level: int = 0):
    """Print detailed information about a node and its metadata."""
    indent = "  " * level
    node_id = node.get('id', 'NO_ID')
    metadata = node.get('metadata', {})
    if isinstance(metadata, list):
        metadata = metadata[0] if metadata else {}
    
    # Get important fields
    title = metadata.get('title', '')
    main_topic = metadata.get('main_topic', '')
    summary = metadata.get('summary', '')
    node_type = metadata.get('node_type', '')
    node_level = metadata.get('level', -1)
    
    # Print node info
    print(f"{indent}Node ID: {node_id}")
    print(f"{indent}Level: {node_level}")
    print(f"{indent}Type: {node_type}")
    if title:
        print(f"{indent}Title: {title}")
    if main_topic:
        print(f"{indent}Main Topic: {main_topic}")
    if summary:
        print(f"{indent}Summary: {summary[:100]}...")
    print(f"{indent}Current Name in Visualization: {node.get('name', 'NO_NAME')}")
    print(f"{indent}{'-' * 80}")
    
    # Process children recursively
    children = node.get('children', [])
    for child in children:
        print_node_info(child, level + 1)

def main():
    # Load the sunburst data
    data_path = Path("/home/bogdan/Desktop/CFI_Tools/raptor_project/data/test/visualizations/data/sunburst_data.json")
    with open(data_path) as f:
        data = json.load(f)
    
    print("=== Tree Structure Analysis ===")
    print(f"Total nodes in hierarchy: {len(data.get('processed', {}).get('ids', []))}")
    print("=" * 80)
    
    # Analyze the hierarchy
    hierarchy = data.get('hierarchy', {})
    if hierarchy:
        print_node_info(hierarchy)
    else:
        print("No hierarchy found in the data!")

if __name__ == "__main__":
    main()
