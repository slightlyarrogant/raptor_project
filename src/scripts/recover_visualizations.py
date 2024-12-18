#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

import logging
import json
from pathlib import Path
import click
from src.visualization.tree_viz import TreeVisualizer
from src.summarization.summary_manager import SummaryManager
from src.utils.config import DEFAULT_CONFIG
from src.storage.store_manager import StoreManager
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def extract_tree_from_store(store_path: Path, index_name: str) -> dict:
    """Extract tree data from file storage."""
    try:
        store_manager = StoreManager()
        tree_data = {
            'internal_nodes': {},
            'leaf_nodes': {}
        }
        
        # Load chunks
        chunks_dir = store_path / index_name / "chunks"
        if not chunks_dir.exists():
            raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
            
        # Load nodes from chunk files
        for chunk_file in chunks_dir.glob("*.json"):
            with open(chunk_file, 'r') as f:
                chunk_data = json.load(f)
                node_id = chunk_data.get('id')
                if not node_id:
                    continue
                    
                node_data = {
                    'id': node_id,
                    'summary': chunk_data.get('summary', ''),
                    'title': chunk_data.get('title', ''),
                    'children': chunk_data.get('children', []),
                    'metadata': chunk_data.get('metadata', {})
                }
                
                if chunk_data.get('children'):
                    tree_data['internal_nodes'][node_id] = node_data
                else:
                    tree_data['leaf_nodes'][node_id] = {
                        **node_data,
                        'text': chunk_data.get('text', '')
                    }
                    
        return tree_data
        
    except Exception as e:
        logger.error(f"Failed to extract tree data: {str(e)}")
        raise

def fix_incomplete_summaries(tree_data: dict, summary_manager: SummaryManager) -> dict:
    """Fix incomplete or problematic summaries in the tree."""
    try:
        # Process internal nodes
        for node_id, node_data in tree_data.get('internal_nodes', {}).items():
            summary = node_data.get('summary', '')
            if not summary or len(summary) < 50:
                # Get child texts
                child_texts = []
                for child_id in node_data['children']:
                    if child_id in tree_data['leaf_nodes']:
                        child_texts.append(tree_data['leaf_nodes'][child_id]['text'])
                    elif child_id in tree_data['internal_nodes']:
                        child_texts.append(tree_data['internal_nodes'][child_id]['summary'])
                        
                if child_texts:
                    new_summary = summary_manager.generate_summary(child_texts)
                    node_data['summary'] = new_summary
                    logger.info(f"Generated new summary for node {node_id}")
                    
        # Process leaf nodes
        for node_id, node_data in tree_data.get('leaf_nodes', {}).items():
            summary = node_data.get('summary', '')
            if not summary or len(summary) < 50:
                text = node_data.get('text', '')
                if text:
                    new_summary = summary_manager.generate_summary([text])
                    node_data['summary'] = new_summary
                    logger.info(f"Generated new summary for leaf node {node_id}")
                    
        return tree_data
        
    except Exception as e:
        logger.error(f"Failed to fix summaries: {str(e)}")
        raise

@click.command()
@click.option('--index-name', default='raptor-technicalbase', help='Name of the index to recover')
@click.option('--store-dir', default='cache', help='Directory containing stored data')
@click.option('--output-dir', default=None, help='Directory for output visualizations')
def recover_visualizations(index_name: str, store_dir: str, output_dir: str = None):
    """Recover and regenerate visualizations from stored tree data."""
    try:
        logger.info(f"Starting visualization recovery for index: {index_name}")
        
        # Setup paths
        store_path = Path(store_dir)
        if output_dir is None:
            output_dir = f"data/{index_name}/visualizations"
        viz_dir = Path(output_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract tree data from storage
        tree_data = extract_tree_from_store(store_path, index_name)
        
        # Initialize components
        summary_manager = SummaryManager(DEFAULT_CONFIG)
        
        # Fix any problematic summaries
        tree_data = fix_incomplete_summaries(tree_data, summary_manager)
        
        # Initialize visualizer
        visualizer = TreeVisualizer(viz_dir)
        
        # Generate tree visualization
        visualizer.create_tree_visualization(tree_data, 'tree_viz.html')
        logger.info("✓ Tree visualization created")
        
        # Generate network visualization
        visualizer.create_network(tree_data, 'network_viz.html')
        logger.info("✓ Network visualization created")
        
        # Generate health dashboard
        visualizer.create_tree_health_dashboard(tree_data, 'tree_health.html')
        logger.info("✓ Tree health dashboard created")
        
        # Generate document quality report
        visualizer.create_document_quality_report(tree_data, 'doc_quality.html')
        logger.info("✓ Document quality report created")
        
        # Generate cluster distribution
        visualizer.create_cluster_distribution(tree_data, 'cluster_dist.png')
        logger.info("✓ Cluster distribution created")
        
        # Generate 3D embedding visualization
        visualizer.create_embedding_visualization(tree_data, 'embedding_viz.html')
        logger.info("✓ 3D embedding visualization created")
        
        # Generate sunburst visualization
        visualizer.create_sunburst(tree_data, 'sunburst.html')
        logger.info("✓ Sunburst visualization created")
        
        # Generate D3.js network
        visualizer.create_network_d3(tree_data, 'network_d3.html')
        logger.info("✓ D3.js network visualization created")
        
        # Generate interactive explorer
        visualizer.create_interactive_tree_explorer(tree_data, 'tree_explorer.html')
        logger.info("✓ Interactive tree explorer created")
        
        logger.info(f"✓ All visualizations have been regenerated in: {viz_dir}")
        
    except Exception as e:
        logger.error(f"Recovery failed: {str(e)}")
        raise

if __name__ == "__main__":
    recover_visualizations()
