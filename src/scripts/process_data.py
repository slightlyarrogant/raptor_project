"""
RAPTOR Data Processing Script
Handles batch processing of documents for the RAPTOR system.
"""

# Standard library imports
import sys
import time
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Third-party imports
import logging
import psutil
import humanize
from tqdm import tqdm
import tiktoken
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# RAPTOR imports
from src.utils.text_utils import split_into_chunks
from src.utils.config import DEFAULT_CONFIG, RAPTOR_INDICES
from src.tree.tree_manager import TreeManager
from src.visualization.tree_viz import TreeVisualizer

# Constants
MAX_TOKENS_PER_CHUNK = 4000
SUPPORTED_FILE_TYPES = {'.txt', '.md', '.rst', '.org', '.wiki'}


class ProcessingMonitor:
    """Monitors and tracks document processing statistics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'chunked_docs': 0,  # Total number of chunks
            'total_tokens': 0,  # Total tokens processed
            'avg_chunk_size': 0,  # Average tokens per chunk
            'total_tree_elements': 0,
            'tree_health': {
                'leaf_nodes': 0,
                'branch_nodes': 0,
                'root_summaries': 0,
                'avg_cluster_size': 0,
                'max_depth': 0,
                'total_clusters': 0,
                'empty_clusters': 0,
                'balanced_score': 0
            },
            'file_types': {},
            'api_calls': {
                'embeddings': 0,
                'summarization': 0
            },
            'processing_time': 0  # Initialize processing time
        }
        
    def update_stats(self, **kwargs) -> None:
        """Update processing statistics."""
        for key, value in kwargs.items():
            if key == 'total_tokens':
                self.stats['total_tokens'] += value
            elif key == 'chunked_docs':
                self.stats['chunked_docs'] += value
            elif key == 'avg_chunk_size':
                # Recalculate average chunk size
                if self.stats['chunked_docs'] > 0:
                    self.stats['avg_chunk_size'] = self.stats['total_tokens'] / self.stats['chunked_docs']
            elif key in self.stats:
                if isinstance(self.stats[key], dict):
                    self.stats[key].update(value)
                else:
                    self.stats[key] += value
                    
        # Update processing time
        self.stats['processing_time'] = time.time() - self.start_time
        
    def get_memory_usage(self) -> str:
        """Get current memory usage of the process."""
        process = psutil.Process()
        memory_info = process.memory_info()
        return humanize.naturalsize(memory_info.rss)
        
    def save_stats(self) -> None:
        """Save processing statistics to file."""
        self.stats['processing_time'] = time.time() - self.start_time
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'processing_stats_{timestamp}.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
            
    def print_summary(self) -> None:
        """Print processing summary with detailed statistics."""
        print("\n=== Processing Summary ===")
        print(f"Total files: {self.stats['total_files']}")
        print(f"Successfully processed: {self.stats['processed_files']}")
        print(f"Failed: {self.stats['failed_files']}")
        
        print("\nTree Health Metrics:")
        print(f"- Leaf nodes: {self.stats['tree_health']['leaf_nodes']}")
        print(f"- Branch nodes: {self.stats['tree_health']['branch_nodes']}")
        print(f"- Root summaries: {self.stats['tree_health']['root_summaries']}")
        print(f"- Maximum tree depth: {self.stats['tree_health']['max_depth']}")
        print(f"- Average cluster size: {self.stats['tree_health']['avg_cluster_size']:.1f}")
        print(f"- Tree balance score: {self.stats['tree_health']['balanced_score']:.2f}")
        print(f"- Empty clusters: {self.stats['tree_health']['empty_clusters']}")
        
        print("\nNamespace Statistics:")
        print(f"- Chunks namespace: {self.stats['chunked_docs']} chunks expected")
        print(f"- Summaries namespace: {self.stats['tree_health']['total_clusters']} summaries expected")
        
        print("\nProcessing Details:")
        print(f"- Documents chunked: {self.stats['chunked_docs']}")
        print(f"- Average chunk size: {self.stats['avg_chunk_size']:.0f} tokens")
        print(f"- Total clusters: {self.stats['tree_health']['total_clusters']}")
        print(f"- Summaries generated: {self.stats.get('summaries_generated', 0)}")
        
        print("\nAPI Usage:")
        print(f"- Embedding calls: {self.stats['api_calls']['embeddings']}")
        print(f"- Summarization calls: {self.stats['api_calls']['summarization']}")
        
        print(f"\nProcessing time: {self.stats['processing_time']:.2f}s")
        print(f"Memory usage: {self.get_memory_usage()}")
        
        if self.stats.get('errors'):
            print("\nErrors encountered:")
            for error_type, count in self.stats['errors'].items():
                print(f"- {error_type}: {count}")
            
        # Add actual Pinecone counts if available
        if hasattr(self, 'pinecone_stats'):
            print("\nActual Pinecone Counts:")
            for namespace, count in self.pinecone_stats.items():
                print(f"- {namespace}: {count} vectors")


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def setup_index_directories(index_name: str) -> None:
    """Create necessary directories for index."""
    dirs = [
        f"data/{index_name}/raw",
        f"data/{index_name}/processed",
        f"data/{index_name}/failed",
        f"data/{index_name}/archive"
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    # Copy files from data/raw to index-specific raw directory if needed
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        index_raw_dir = Path(f"data/{index_name}/raw")
        for file in raw_dir.glob('*'):
            if not (index_raw_dir / file.name).exists():
                shutil.copy2(file, index_raw_dir)
                
    logger.info(f"Directory structure verified for index: data/{index_name}")

def move_file(index_name: str, filename: str, success: bool) -> None:
    """Move processed file to appropriate directory."""
    source_dir = f"data/{index_name}/raw"
    dest_dir = f"data/{index_name}/{'processed' if success else 'failed'}"
    
    try:
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        
        if os.path.exists(source_path):
            shutil.move(source_path, dest_path)
            logger.info(f"Moved {filename} to {'processed' if success else 'failed'} directory")
        else:
            logger.error(f"Source file not found: {source_path}")
            
    except Exception as e:
        logger.error(f"Error moving file {filename}: {str(e)}")

def select_raptor_index() -> Tuple[str, Dict]:
    """Allow user to select which index to process."""
    print("\nAvailable RAPTOR indices:")
    for idx, (name, config) in enumerate(RAPTOR_INDICES.items(), 1):
        print(f"{idx}. {name} - {config['description']}")
    
    while True:
        try:
            choice = int(input("\nSelect index number to process: "))
            if 1 <= choice <= len(RAPTOR_INDICES):
                index_name = list(RAPTOR_INDICES.keys())[choice - 1]
                return index_name, RAPTOR_INDICES[index_name]
        except ValueError:
            pass
        print("Invalid choice. Please try again.")

def print_processing_strategy() -> None:
    """Print suggested processing strategy."""
    print("\nSuggested Processing Strategy:")
    print("1. Start with a small batch (5-10 files) as a test run")
    print("2. Check the processed results in data/processed/")
    print("3. Review any failures in data/failed/")
    print("4. Adjust batch size based on performance")
    print("5. Monitor processing.log for detailed information")
    print(f"\nSupported file types: {', '.join(SUPPORTED_FILE_TYPES)}")
    print("\nTo begin processing:")
    print("1. Place your documents in data/raw/")
    print("2. Run this script")
    print("3. Monitor progress in the console and logs")

def process_documents(index_name: str, index_config: dict, batch_size: int = 5) -> None:
    """Process documents in batches."""
    try:
        # Setup directories
        setup_index_directories(index_name)
        logger.warning("Starting component initialization...")
        
        # 1. Initialize Pinecone
        try:
            pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            logger.warning(f"✓ Pinecone index initialized: {index_name}")
            logger.warning(f"Current index stats: {stats}")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
        
        # 2. Create configuration with more aggressive clustering
        # Get file list and process
        raw_files = list(Path(f"data/{index_name}/raw").glob('*'))
        total_files = len(raw_files)
        monitor.stats['total_files'] = total_files
        
        if not raw_files:
            logger.warning("No files found to process")
            return
            
        logger.warning(f"Found {total_files} files to process")


        config = {
            'embedding': {
                'model_name': "text-embedding-3-small",
                'dimension': 1536,
                'batch_size': batch_size,
                'api_key': os.getenv('OPENAI_API_KEY')
            },
            'storage': {
                'store': index,
                'api_key': os.getenv('PINECONE_API_KEY'),
                'index_name': index_name,
                'namespace': index_config['namespace'],
                'dimension': 1536,
                'metric': 'cosine'
            },
            'summarization': {
                'model_name': 'gpt-4o-mini',
                'max_tokens': 500,
                'temperature': 0.3,
                'api_key': os.getenv('OPENAI_API_KEY')
            },
            'clustering': {
                'umap': {
                    'n_neighbors': 15,
                    'n_components': 3,  # More dimensions for better separation
                    'metric': 'cosine'
                },
                'gmm': {
                    'max_clusters': min(200, total_files // 3),  # Much more aggressive
                    'threshold': 0.3,  # Lower threshold for more clusters
                    'n_init': 10  # More initialization attempts
                },
                'similarity_threshold': 0.2,  # Much lower for more merging
                'min_cluster_size': 5,    # Smaller minimum size
                'max_cluster_size': 25,   # Smaller maximum size
                'target_children': 3,     # Target children per parent
                'force_merge': True,      # Force merge remaining clusters
                'min_similarity': 0.05,   # Very low minimum similarity
                'decay_rate': 0.9,       # Slower threshold decay
                'merge_threshold': 0.15   # Low threshold for merging
            }
        }
        
        logger.warning("✓ Configuration prepared")
        
        # Initialize TreeManager (which initializes all other managers)
        tree_manager = TreeManager(config)
        logger.warning("✓ TreeManager initialized")
        

        
        # Process in batches
        total_batches = (total_files + batch_size - 1) // batch_size
        for batch_num in range(1, total_batches + 1):
            start_idx = (batch_num - 1) * batch_size
            end_idx = min(start_idx + batch_size, total_files)
            batch_files = raw_files[start_idx:end_idx]
            
            process_batch(batch_files, batch_num, total_batches, tree_manager, monitor, index_name)
            
        # Get final stats from PineconeManager
        monitor.pinecone_stats = tree_manager.pinecone_manager.get_namespace_stats()
        
        # Print summary
        monitor.print_summary()
        logger.info("Data processing completed")
        
        # Get tree structure from tree manager
        tree_structure = tree_manager.build_tree()  # Store the tree structure

        # Create visualization directory with correct path
        viz_dir = Path(f"analysis_outputs/{index_name}/tree_viz")
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Initialize visualizer with embeddings
        visualizer = TreeVisualizer(
            output_dir=viz_dir,
            embeddings=tree_manager.get_all_embeddings()
        )

        # Generate visualizations using the tree structure
        logger.info("\nGenerating Tree Visualizations:")
        logger.info("-" * 50)

        # Pass tree_structure to visualization methods with proper paths
        visualizations = [
            ('tree_structure.html', visualizer.visualize_tree_structure),
            ('cluster_distribution.png', visualizer.visualize_cluster_distribution),
            ('embeddings_3d.html', visualizer.visualize_embeddings),
            ('tree_sunburst.html', visualizer.visualize_tree_sunburst),
            ('cluster_network.html', visualizer.visualize_cluster_network)
        ]

        for viz_name, viz_func in visualizations:
            try:
                viz_func(tree_structure, viz_dir / viz_name)
                logger.info(f"✓ Generated {viz_name}")
            except Exception as viz_error:
                logger.warning(f"Failed to generate {viz_name}: {str(viz_error)}")
                continue

        logger.info(f"\nVisualizations saved to: {viz_dir}")
        
        # Print tree metrics
        print_tree_metrics(tree_structure)
        
    except Exception as e:
        logger.error(f"Error during data processing: {str(e)}")
        raise

def process_batch(
    batch_files: List[Path],
    batch_num: int,
    total_batches: int,
    tree_manager: TreeManager,
    monitor: ProcessingMonitor,
    index_name: str
) -> None:
    """Process a batch of files."""
    logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
    logger.info(f"Memory usage: {monitor.get_memory_usage()}")
    
    documents = []
    chunks = []
    chunk_metadata = []
    current_files = []
    failed_files = []
    total_chunks = 0
    total_tokens = 0
    
    try:
        # First, read and chunk all files in the batch
        for file_idx, file_path in enumerate(batch_files):
            try:
                content = read_and_chunk_file(file_path)
                if content:
                    # Track each chunk with its metadata
                    for chunk_idx, chunk in enumerate(content):
                        chunks.append(chunk)
                        chunk_metadata.append({
                            'file_name': file_path.name,
                            'file_index': file_idx,
                            'chunk_index': chunk_idx,
                            'total_chunks': len(content)
                        })
                        total_chunks += 1
                        total_tokens += count_tokens(chunk)
                    
                    documents.extend(content)
                    current_files.append(file_path.name)
                else:
                    failed_files.append(file_path.name)
                    
                logger.info(f"File {file_path.name} generated {len(content)} chunks")
                
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {str(e)}")
                failed_files.append(file_path.name)
        
        # Then process documents if any were successfully read
        if chunks:
            try:
                logger.info(f"Processing {len(chunks)} total chunks from {len(current_files)} files")
                logger.info(f"First chunk metadata: {chunk_metadata[0]}")
                result = tree_manager.process_documents(chunks, chunk_metadata)
                
                # Update monitor stats with chunk information
                monitor.update_stats(
                    processed_files=len(current_files),
                    chunked_docs=total_chunks,
                    total_tokens=total_tokens,
                    avg_chunk_size=total_tokens / total_chunks if total_chunks > 0 else 0,
                    total_tree_elements=len(result['tree_structure']),
                    tree_health={
                        'leaf_nodes': len([c for c in result['tree_structure'] if not c.get('children')]),
                        'branch_nodes': len([c for c in result['tree_structure'] if c.get('children')]),
                        'root_summaries': len([c for c in result['tree_structure'] if c.get('level', 0) == 0]),
                        'avg_cluster_size': len(chunks) / len(result['tree_structure']) if result['tree_structure'] else 0,
                        'max_depth': max(c.get('level', 0) for c in result['tree_structure']) + 1 if result['tree_structure'] else 0,
                        'total_clusters': len(result['tree_structure']),
                        'empty_clusters': 0,
                        'balanced_score': 1.0
                    },
                    api_calls={
                        'embeddings': len(chunks),
                        'summarization': len(result['tree_structure'])
                    }
                )
                
                # Move successfully processed files
                for filename in current_files:
                    move_file(index_name, filename, True)
                    
            except Exception as e:
                logger.error(f"Error processing documents: {str(e)}")
                failed_files.extend(current_files)
                
    except Exception as e:
        logger.error(f"Error processing batch {batch_num}: {str(e)}")
        failed_files.extend(current_files)
    
    finally:
        # Move failed files
        for filename in set(failed_files):
            move_file(index_name, filename, False)

def read_and_chunk_file(file_path: Path) -> List[str]:
    """Read file content and split into chunks if needed."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        token_count = count_tokens(content)
        logger.info(f"File {file_path.name}: {token_count} tokens")
        
        if token_count > MAX_TOKENS_PER_CHUNK:
            chunks = split_into_chunks(content)
            logger.info(f"Split {file_path.name} into {len(chunks)} chunks due to token count {token_count}")
            return chunks
        return [content]

def restore_files_to_raw(index_name: str) -> None:
    """Move all files back to raw directory."""
    try:
        # Get paths
        processed_dir = Path(f"data/{index_name}/processed")
        failed_dir = Path(f"data/{index_name}/failed")
        raw_dir = Path(f"data/{index_name}/raw")
        
        # Ensure raw directory exists
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Move files from processed
        if processed_dir.exists():
            for file_path in processed_dir.glob('*'):
                try:
                    shutil.move(str(file_path), str(raw_dir / file_path.name))
                    logger.info(f"Restored {file_path.name} from processed to raw")
                except Exception as e:
                    logger.error(f"Error restoring {file_path.name}: {str(e)}")
        
        # Move files from failed
        if failed_dir.exists():
            for file_path in failed_dir.glob('*'):
                try:
                    shutil.move(str(file_path), str(raw_dir / file_path.name))
                    logger.info(f"Restored {file_path.name} from failed to raw")
                except Exception as e:
                    logger.error(f"Error restoring {file_path.name}: {str(e)}")
                    
        logger.info("All files restored to raw directory")
        
    except Exception as e:
        logger.error(f"Error during file restoration: {str(e)}")

def print_tree_metrics(tree: Dict):
    """Print comprehensive tree quality metrics."""
    logger.info("\nTree Quality Metrics:")
    logger.info("=" * 50)
    
    # Structure metrics
    logger.info("\nStructure:")
    logger.info(f"- Total Nodes: {tree['metadata']['total_nodes']}")
    logger.info(f"- Leaf Nodes: {tree['metadata']['leaf_nodes']}")
    logger.info(f"- Maximum Depth: {tree['metadata']['max_depth']}")
    logger.info(f"- Average Branching Factor: {tree['metadata']['avg_branching']:.2f}")
    logger.info(f"- Tree Balance Score: {tree['metadata']['balance_score']:.2f}")
    
    # Content metrics
    logger.info("\nContent:")
    logger.info(f"- Total Documents: {tree['metadata']['total_documents']}")
    logger.info(f"- Average Cluster Size: {tree['metadata']['average_cluster_size']:.2f}")
    logger.info(f"- Document Distribution Score: {tree['metadata'].get('distribution_score', 0):.2f}")
    
    # Quality metrics
    logger.info("\nQuality:")
    logger.info(f"- Average Coherence: {tree['metadata'].get('avg_coherence', 0):.4f}")
    logger.info(f"- Minimum Coherence: {tree['metadata'].get('min_coherence', 0):.4f}")
    logger.info(f"- Silhouette Score: {tree['metadata'].get('silhouette_score', 0):.4f}")
    
    # Print warning if any metrics are below thresholds
    warnings = []
    if tree['metadata'].get('balance_score', 0) < 0.7:
        warnings.append("Low balance score - tree might be uneven")
    if tree['metadata'].get('avg_coherence', 0) < 0.6:
        warnings.append("Low coherence - clusters might not be well-formed")
    if tree['metadata'].get('silhouette_score', 0) < 0.3:
        warnings.append("Low silhouette score - clusters might not be well-separated")
        
    if warnings:
        logger.warning("\nWarnings:")
        for warning in warnings:
            logger.warning(f"- {warning}")

def main():
    """Main entry point for the script."""
    try:
        logger.info("Script initialized")
        print("\n=== RAPTOR Data Processing ===")
        
        # Test imports and functionality
        logger.info("Testing text splitting...")
        test_text = "This is a test."
        chunks = split_into_chunks(test_text)
        logger.info("Text splitting function works")
        
        # Show processing strategy
        print_processing_strategy()
        
        # Get user input and process
        if input("\nProceed with processing? (y/n): ").lower() == 'y':
            index_name, index_config = select_raptor_index()
            batch_size = int(input("Enter batch size (default 5): ") or "5")
            try:
                process_documents(index_name, index_config, batch_size)
            finally:
                # Ask if user wants to restore files
                if input("\nRestore files to raw directory? (y/n): ").lower() == 'y':
                    restore_files_to_raw(index_name)
        else:
            print("Processing cancelled")
            
    except Exception as e:
        logger.error(f"Error in main script: {str(e)}")
        raise

if __name__ == "__main__":
    main()
