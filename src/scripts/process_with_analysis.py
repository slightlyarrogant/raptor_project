import logging
from pathlib import Path
import click
from typing import Dict, List, Optional, Union, Any, Tuple
import os
import sys
import json
import time
import numpy as np
from collections import Counter
import shutil

from src.scripts.analyze_documents import DocumentAnalyzer
from src.storage.store_manager import StoreManager
from src.tree.tree_manager import TreeManager
from src.models.data_models import TreeData
from src.utils.config import (
    DEFAULT_CONFIG, PINECONE_API_KEY, PINECONE_REGION, PINECONE_CLOUD,
    PINECONE_INDEX_NAME, PINECONE_NAMESPACE, PINECONE_DIMENSION, PINECONE_METRIC
)
from src.visualization.tree_viz import TreeVisualizer
from src.prepare.data_loader import normalize_documents
from src.scripts.process_data import process_documents
from src.utils.openai_client import UnifiedAIClient
from src.embedding.embed_manager import EmbedManager
from src.storage.pinecone_manager import PineconeManager
from src.utils.env_check import check_environment
from src.clustering.cluster_manager import ClusterManager
from src.summarization.summary_manager import SummaryManager
from src.text.chunk_manager import ChunkManager

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_pinecone_config() -> Dict:
    """Load Pinecone configuration from environment variables."""
    return {
        'api_key': os.getenv('PINECONE_API_KEY'),
        'environment': os.getenv('PINECONE_REGION'),
        'cloud': os.getenv('PINECONE_CLOUD'),
        'dimension': int(os.getenv('PINECONE_DIMENSION', '1536')),
        'metric': os.getenv('PINECONE_METRIC', 'cosine')
    }

def initialize_components(index_name: str) -> Dict:
    """Initialize all required components."""
    try:
        components = {}
        
        # Initialize Pinecone first
        pinecone_config = {
            'api_key': os.getenv('PINECONE_API_KEY'),
            'environment': os.getenv('PINECONE_REGION'),
            'cloud': os.getenv('PINECONE_CLOUD'),
            'dimension': int(os.getenv('PINECONE_DIMENSION', '1536')),
            'metric': os.getenv('PINECONE_METRIC', 'cosine'),
            'index_name': index_name
        }
        pinecone_manager = PineconeManager(pinecone_config)
        components['pinecone_manager'] = pinecone_manager
        logger.info("✓ Pinecone initialized")
        
        # Initialize store manager with Pinecone
        store_config = {
            'index_name': index_name,
            'namespace': f"{index_name}_namespace",
            'pinecone': pinecone_config
        }
        store_manager = StoreManager(store_config)
        components['store_manager'] = store_manager
        logger.info("✓ StoreManager initialized")
        
        # Initialize AI client first since other components need it
        ai_client = UnifiedAIClient()
        components['ai_client'] = ai_client
        logger.info("✓ AI Client initialized")
        
        # Initialize embedding manager with AI client
        embed_config = {
            'model_name': 'text-embedding-3-small',
            'dimension': int(os.getenv('PINECONE_DIMENSION', '1536')),
            'batch_size': 100
        }
        embed_manager = EmbedManager(embed_config)
        embed_manager.unifiedai_client = ai_client
        components['embed_manager'] = embed_manager
        logger.info("✓ EmbedManager initialized")
        
        # Initialize summary manager with proper config and AI client
        summary_config = {'summarization': DEFAULT_CONFIG['summarization']}
        summary_manager = SummaryManager(summary_config)
        summary_manager.unifiedai_client = ai_client
        components['summary_manager'] = summary_manager
        logger.info("✓ SummaryManager initialized")
        
        # Initialize cluster manager
        cluster_config = {
            'dimension': int(os.getenv('PINECONE_DIMENSION', '1536')),
            'threshold': 0.15,
            'max_levels': 5,
            'min_cluster_size': 2,
            'target_children': 20
        }
        cluster_manager = ClusterManager(cluster_config)
        components['cluster_manager'] = cluster_manager
        logger.info("✓ ClusterManager initialized")
        
        # Initialize tree manager with all required managers
        tree_config = {
            'index_name': index_name,
            'max_cluster_size': 20,  # Move this to top level
            'min_cluster_size': 2,
            'balance_threshold': 0.3,
            'clustering': DEFAULT_CONFIG['clustering'],
            'embedding': DEFAULT_CONFIG['embedding'],
            'summarization': DEFAULT_CONFIG['summarization']
        }
        
        logger.info(f"Initializing TreeManager with config: {tree_config}")
        tree_manager = TreeManager(tree_config)
        tree_manager.unifiedai_client = ai_client
        tree_manager.embed_manager = components['embed_manager']
        tree_manager.cluster_manager = components['cluster_manager']
        tree_manager.store_manager = components['store_manager']
        tree_manager.summary_manager = summary_manager
        tree_manager.chunk_manager = ChunkManager(tree_config)
        tree_manager._initialize_managers()
        components['tree_manager'] = tree_manager
        logger.info("✓ TreeManager initialized")
        
        return components
        
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        raise

def validate_environment():
    """Validate environment before running pipeline."""
    try:
        logger.info("Validating environment...")
        
        # Check required API keys
        required_keys = {
            'OPENAI_API_KEY': 'OpenAI API key',
            'PINECONE_API_KEY': 'Pinecone API key',
            'PINECONE_REGION': 'Pinecone region',
            'PINECONE_CLOUD': 'Pinecone cloud'
        }
        
        missing_keys = []
        for key, desc in required_keys.items():
            if not os.getenv(key):
                missing_keys.append(desc)
        
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
            
        # Check and create required directories following instructions.md structure
        required_dirs = [
            os.path.join('data'),
            os.path.join('data', 'raw'),
            os.path.join('data', 'processed'),
            os.path.join('data', 'failed'),
            os.path.join('data', 'archive')
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Directory {dir_path} exists")
            
        # Test model access
        client = UnifiedAIClient()
        test_response = client.chat_completion(
            messages=[{"role": "user", "content": "Test connection"}],
            model='claude-3-5-haiku-20241022'
        )
        logger.info("✓ Model access verified")
        
        # Test embedding
        test_embedding = client.get_embeddings(["Test embedding"])
        logger.info("✓ Embedding generation verified")
        
        logger.info("✓ Environment validation complete")
        return True
        
    except Exception as e:
        logger.error(f"Environment validation failed: {str(e)}")
        return False

@click.command()
@click.option('--index-name', default='raptor-cfi', help='Name of the Pinecone index')
def run_pipeline(index_name: str = 'raptor-cfi'):
    """Run the complete Raptor analysis pipeline."""
    try:
        logger.info("\n=== Starting Raptor Pipeline ===")
        
        # Initialize components
        components = initialize_components(index_name)
        if not components:
            logger.error("Failed to initialize components")
            return
        
        # Setup index-specific directories
        index_dir = Path(f"data/{index_name}")
        raw_dir = index_dir / "raw"
        processed_dir = index_dir / "processed"
        failed_dir = index_dir / "failed"
        viz_dir = index_dir / "visualizations"
        doc_viz_dir = viz_dir / "document_analysis"
        tree_dir = index_dir / "tree"
        
        # Create all directories
        for dir_path in [raw_dir, processed_dir, failed_dir, viz_dir, doc_viz_dir, tree_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        
        # Check for documents in raw directory
        if not raw_dir.exists() or not any(raw_dir.iterdir()):
            logger.error(f"No documents found in raw directory: {raw_dir}")
            return
        
        logger.info(f"Processing documents from: {raw_dir}")
        documents = load_documents(str(raw_dir))
        if not documents:
            raise ValueError("No documents found to process")
            
        # Generate document analysis visualizations
        analyzer = DocumentAnalyzer(doc_viz_dir)
        analyzer.analyze_corpus(
            texts=[doc['text'] for doc in documents],
            metadata=[doc['metadata'] for doc in documents]
        )
        logger.info("✓ Document analysis visualizations created")
        
        # Process documents with chunking
        chunked_docs = []
        chunk_metadata = []
        for doc in documents:
            chunks = list(ChunkManager().chunk_document(doc['text'], doc['metadata']))
            for chunk_text, metadata in chunks:
                chunked_docs.append(chunk_text)
                chunk_metadata.append(metadata)
        
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        
        # Check cache for existing embeddings
        cached_embeddings = []
        texts_to_embed = []
        metadata_to_embed = []
        
        for i, (text, metadata) in enumerate(zip(chunked_docs, chunk_metadata)):
            cached_embedding = components['store_manager'].get_embedding(text)
            if cached_embedding is not None:
                cached_embeddings.append((i, cached_embedding))
                logger.debug(f"Found cached embedding for chunk {i}")
            else:
                texts_to_embed.append(text)
                metadata_to_embed.append(metadata)
        
        # Generate embeddings only for uncached texts
        if texts_to_embed:
            logger.info(f"Generating embeddings for {len(texts_to_embed)} uncached chunks...")
            new_embeddings = components['embed_manager'].embed_texts(texts_to_embed, metadata_to_embed)
            
            # Cache the new embeddings in batches
            batch_size = components['embed_manager'].config['embedding']['batch_size']
            for i in range(0, len(texts_to_embed), batch_size):
                batch_end = min(i + batch_size, len(texts_to_embed))
                batch_texts = texts_to_embed[i:batch_end]
                batch_embeddings = new_embeddings[i:batch_end]
                batch_metadata = metadata_to_embed[i:batch_end]
                
                # Store each embedding in the batch
                for text, embedding, metadata in zip(batch_texts, batch_embeddings, batch_metadata):
                    components['store_manager'].save_embedding(text, embedding, metadata)
                logger.info(f"✓ Cached batch {i//batch_size + 1} ({len(batch_texts)} embeddings)")
                
            logger.info("✓ All new embeddings cached")
        else:
            new_embeddings = []
        
        # Combine cached and new embeddings in correct order
        embeddings = [None] * len(chunked_docs)
        for i, emb in cached_embeddings:
            embeddings[i] = emb
            
        current_new_idx = 0
        for i in range(len(embeddings)):
            if embeddings[i] is None:
                embeddings[i] = new_embeddings[current_new_idx]
                current_new_idx += 1
        
        logger.info(f"Using {len(cached_embeddings)} cached and {len(texts_to_embed)} new embeddings")
        
        # Add texts and embeddings to components
        components['texts'] = chunked_docs
        components['embeddings'] = embeddings
        components['metadata'] = chunk_metadata
        
        # Execute processing with progress tracking
        tree_data, stats = execute_document_processing(components)
        
        # Create visualizations only if we have valid tree data
        if tree_data and stats:
            logger.info("\nCreating tree visualizations...")
            visualizer = TreeVisualizer(index_name=index_name)
            visualizer.create_visualizations(tree_data)
            logger.info("✓ Tree visualizations created")

        return True

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

def analyze_documents(docs_path: Path, analyzer: DocumentAnalyzer, index_name: str) -> dict:
    """Analyze documents and generate visualizations."""
    try:
        # Create visualization directory
        viz_dir = Path(f"analysis_outputs/{index_name}/document_analysis")
        viz_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created document analysis directory: {viz_dir}")
        
        # Set output directory for analyzer
        analyzer.output_dir = viz_dir
        
        texts = []
        metadata = []
        
        # Load and analyze documents
        for file_path in docs_path.glob('*'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    texts.append(text)
                    metadata.append({
                        'file_name': file_path.name,
                        'file_path': str(file_path),
                        'file_type': file_path.suffix,
                        'size': len(text),
                        'created_at': time.ctime(file_path.stat().st_ctime),
                        'modified_at': time.ctime(file_path.stat().st_mtime)
                    })
                    logger.info(f"Loaded: {file_path.name} ({len(text)} chars)")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                continue
        
        if not texts:
            raise ValueError(f"No valid documents found in {docs_path}")
            
        # Run analysis with enhanced metadata
        stats = analyzer.analyze_corpus(
            texts=[doc['text'] for doc in documents],
            metadata=[doc['metadata'] for doc in documents]
        )
        
        # Extract document types
        doc_types = Counter(meta['file_type'] for meta in metadata)
        stats['document_types'] = dict(doc_types)
        
        # Calculate metadata completeness
        required_fields = ['file_name', 'file_path', 'file_type', 'size', 'created_at', 'modified_at']
        completeness_scores = []
        for meta in metadata:
            score = sum(1 for field in required_fields if field in meta) / len(required_fields)
            completeness_scores.append({'score': score, 'file': meta['file_name']})
        stats['metadata_completeness'] = completeness_scores
        
        # Extract section patterns
        section_patterns = []
        for text in texts:
            sections = text.split('\n\n')  # Simple section detection
            section_patterns.append(len(sections))
        stats['section_breaks'] = section_patterns
        
        # Extract technical terms - Convert to Counter
        tech_terms = Counter()
        for text in texts:
            # Simple technical term detection (words with uppercase letters or underscores)
            words = text.split()
            tech_terms.update(word for word in words if '_' in word or (any(c.isupper() for c in word) and not word.isupper()))
        stats['technical_terms'] = tech_terms  # Store as Counter, not dict
        
        # Generate visualizations with enhanced data
        analyzer._generate_length_distribution(texts)
        analyzer._generate_word_cloud(texts)
        analyzer._generate_file_stats(metadata)
        analyzer._generate_content_analysis(texts)
        
        # Verify visualization files were created
        expected_files = [
            'length_distribution.png',
            'word_cloud.png',
            'file_types.png',
            'words_per_doc.png'
        ]
        
        for file_name in expected_files:
            file_path = viz_dir / file_name
            if file_path.exists():
                logger.info(f"Created visualization: {file_name}")
            else:
                logger.warning(f"Missing visualization: {file_name}")
        
        return {
            'stats': stats,
            'texts': texts,
            'metadata': metadata
        }
        
    except Exception as e:
        logger.error(f"Error in document analysis: {str(e)}")
        raise

def execute_document_processing(components: Dict, input_dir: str = None) -> Tuple[TreeData, Dict]:
    """Execute document processing pipeline."""
    failed_files = []
    processing_stats = {
        'total_files': 0,
        'processed_files': 0,
        'failed_files': 0,
        'errors': {}
    }
    
    try:
        # Get components
        tree_manager = components['tree_manager']
        index_name = tree_manager.config['index_name']
        
        # Get texts and embeddings from components
        texts = components.get('texts', [])
        embeddings = components.get('embeddings', [])
        metadata = components.get('metadata', [])
        
        processing_stats['total_files'] = len(texts)
        
        # Track which files are being processed
        current_batch = []
        for i, (text, meta) in enumerate(zip(texts, metadata)):
            try:
                current_batch.append(meta.get('filename', f'doc_{i}'))
                
                # Process documents with embeddings
                if i == len(texts) - 1 or len(current_batch) >= 10:  # Process in batches of 10
                    tree_data, stats = tree_manager.process_documents_with_embeddings(
                        texts=texts[i-len(current_batch)+1:i+1],
                        embeddings=np.array(embeddings[i-len(current_batch)+1:i+1]),
                        metadata=metadata[i-len(current_batch)+1:i+1]
                    )
                    processing_stats['processed_files'] += len(current_batch)
                    current_batch = []
                    
            except Exception as e:
                # Track failed files
                error_type = type(e).__name__
                error_msg = str(e)
                if error_type not in processing_stats['errors']:
                    processing_stats['errors'][error_type] = {'count': 0, 'files': [], 'message': error_msg}
                processing_stats['errors'][error_type]['count'] += 1
                processing_stats['errors'][error_type]['files'].extend(current_batch)
                failed_files.extend(current_batch)
                processing_stats['failed_files'] += len(current_batch)
                current_batch = []
                logger.error(f"Failed to process files {current_batch}: {error_msg}")
                continue
        
        # Initialize visualizer with index name
        visualizer = TreeVisualizer(index_name=index_name)
        
        # Create visualizations with proper namespacing
        visualizer.create_visualizations(tree_data)
        
        # Save tree data with namespace
        output_dir = Path('data') / index_name / 'tree'
        output_dir.mkdir(parents=True, exist_ok=True)
        tree_data_file = output_dir / 'tree_data.json'
        
        with open(tree_data_file, 'w') as f:
            json.dump(tree_data.to_dict(), f, indent=2)
        logger.info(f"\nTree data saved to {tree_data_file}")
        
        # Save failed files list for retry
        if failed_files:
            failed_dir = Path('data') / index_name / 'failed'
            failed_dir.mkdir(parents=True, exist_ok=True)
            failed_files_path = failed_dir / 'failed_files.json'
            with open(failed_files_path, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'failed_files': failed_files,
                    'error_details': processing_stats['errors']
                }, f, indent=2)
            logger.warning(f"\nFailed files saved to {failed_files_path}")
            logger.warning("You can retry these files later using the retry_failed_files.py script")
        
        # Log processing summary
        logger.info("\n=== Processing Summary ===")
        logger.info(f"Total files: {processing_stats['total_files']}")
        logger.info(f"Successfully processed: {processing_stats['processed_files']}")
        logger.info(f"Failed: {processing_stats['failed_files']}")
        if processing_stats['errors']:
            logger.info("\nErrors by type:")
            for error_type, details in processing_stats['errors'].items():
                logger.info(f"- {error_type}: {details['count']} files")
                logger.info(f"  Message: {details['message']}")
                logger.info(f"  Affected files: {', '.join(details['files'])}")
        
        return tree_data, processing_stats
        
    except Exception as e:
        logger.error(f"Failed to execute document processing: {str(e)}")
        raise ValueError("Failed to build tree: " + str(e))

def load_documents(docs_path: Union[str, Path]) -> List[Dict]:
    """Load documents with their metadata."""
    raw_dir = Path(docs_path)
    documents = []
    logger.info(f"Looking for documents in: {raw_dir}")
    
    # Define supported file extensions
    supported_extensions = ("*.txt", "*.md")
    
    # Iterate through all supported file types
    for pattern in supported_extensions:
        for file_path in raw_dir.glob(pattern):
            try:
                logger.info(f"Processing file: {file_path.name}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if text:  # Only add non-empty documents
                        # Use filename without extension as the document name
                        doc_name = file_path.stem
                        doc = {
                            'id': str(file_path),  # Use full path as unique ID
                            'name': doc_name,  # Use filename without extension as name
                            'text': text,
                            'path': str(file_path),  # Keep the path for reference
                            'metadata': {
                                'filename': file_path.name,
                                'path': str(file_path),
                                'size': file_path.stat().st_size,
                                'modified': file_path.stat().st_mtime,
                                'type': file_path.suffix[1:]  # Store file type without dot
                            }
                        }
                        documents.append(doc)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
    
    logger.info(f"Found {len(documents)} valid documents")
    if documents:
        # Log breakdown by file type
        type_counts = {}
        for doc in documents:
            doc_type = doc['metadata']['type']
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        for doc_type, count in type_counts.items():
            logger.info(f"- {count} {doc_type} files")
    
    return documents

def move_processed_files(documents: List[Dict], raw_dir: Path, processed_dir: Path, failed_dir: Path, logger: logging.Logger):
    """Move successfully processed files from raw to processed directory."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    
    successful_moves = 0
    failed_moves = 0
    
    for doc in documents:
        try:
            source_path = Path(doc['metadata']['path'])
            if not source_path.exists():
                logger.warning(f"Source file not found: {source_path}")
                continue
                
            # Create relative path to maintain directory structure
            rel_path = source_path.relative_to(raw_dir)
            target_path = processed_dir / rel_path
            
            # Create target directory if it doesn't exist
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            shutil.move(str(source_path), str(target_path))
            successful_moves += 1
            logger.info(f"✓ Moved to processed: {rel_path}")
            
        except Exception as e:
            failed_moves += 1
            # Move to failed directory if move fails
            try:
                failed_path = failed_dir / rel_path
                failed_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_path), str(failed_path))
                logger.error(f"Failed to process, moved to failed directory: {rel_path}")
            except Exception as move_error:
                logger.error(f"Failed to move file to either processed or failed directory: {str(move_error)}")
            logger.error(f"Error processing file {doc.get('metadata', {}).get('path', 'unknown')}: {str(e)}")
    
    logger.info(f"\nFile Movement Summary:")
    logger.info(f"✓ Successfully moved: {successful_moves} files")
    if failed_moves > 0:
        logger.warning(f"⚠️ Failed to process: {failed_moves} files")

def process_documents(config: Dict):
    """Process documents with enhanced logging and vector offloading."""
    logger.info("\n=== Starting Document Processing ===")
    start_time = time.time()
    
    try:
        # Initialize components
        client = UnifiedAIClient()
        embed_manager = EmbedManager(client)
        
        # Set up configuration
        config['embedding'] = {
            'manager': embed_manager,
            'model': 'text-embedding-3-small',  # Use correct model from config
            'batch_size': 100
        }
        config['storage'] = {
            'pinecone': {
                'api_key': PINECONE_API_KEY,
                'environment': PINECONE_REGION,
                'cloud': PINECONE_CLOUD,
                'index_name': PINECONE_INDEX_NAME,
                'dimension': PINECONE_DIMENSION,
                'metric': PINECONE_METRIC,
                'namespace': PINECONE_NAMESPACE
            }
        }
        
        # Initialize TreeManager with configuration
        tree_manager = TreeManager(config)
        
        return tree_manager
        
    except Exception as e:
        logger.error(f"Error in process_documents: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline()
