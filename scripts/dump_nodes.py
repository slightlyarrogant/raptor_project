import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from src.tree.tree_manager import TreeManager
from src.models.config import Config

def dump_all_nodes():
    """Dump all nodes from the tree into a JSON file."""
    # Initialize config and tree manager
    config = Config()
    tree_manager = TreeManager(config)
    
    # Get all nodes
    nodes = tree_manager.get_all_nodes()
    
    # Convert nodes to dictionary format
    nodes_data = []
    for node in nodes:
        node_dict = {
            'id': node.id,
            'text': node.text,
            'metadata': node.metadata,
            'parent_id': node.parent_id if hasattr(node, 'parent_id') else None,
            'children': [child.id for child in node.children] if hasattr(node, 'children') else []
        }
        nodes_data.append(node_dict)
    
    # Save to JSON file
    output_file = project_root / 'data' / 'all_nodes.json'
    with open(output_file, 'w') as f:
        json.dump(nodes_data, f, indent=2)
    
    print(f"Dumped {len(nodes_data)} nodes to {output_file}")

if __name__ == "__main__":
    dump_all_nodes()
