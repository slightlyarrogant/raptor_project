import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.storage.document_manifest import DocumentManifest
from src.storage.pinecone_manager import PineconeManager
from src.storage.vector_manager import VectorManager
from src.text.chunk_manager import ChunkManager
from src.tree.tree_manager import TreeManager
from src.tree.tree_updater import TreeUpdater
from src.visualization.tree_viz import TreeVisualizer
from src.monitoring.system_monitor import SystemMonitor
import numpy as np
import time

logger = logging.getLogger(__name__)

class IncrementalProcessor:
    def __init__(self, index_name: str, components: Dict):
        """Initialize the incremental processor."""
        self.index_name = index_name
        self.components = components
        self.manifest = DocumentManifest(index_name)
        self.index_dir = Path(f"data/{index_name}")
        
        # Directory structure
        self.raw_dir = self.index_dir / "raw"
        self.processed_dir = self.index_dir / "processed"
        self.failed_dir = self.index_dir / "failed"
        self.viz_dir = self.index_dir / "visualizations"
        
        # Ensure directories exist
        for dir_path in [self.raw_dir, self.processed_dir, self.failed_dir, self.viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize managers
        self.vector_manager = VectorManager(
            components['storage']['manager'],
            components['storage']['namespace'],
            cache_config={
                'vector_cache_size': 50000,
                'vector_cache_ttl': 7200,
                'metadata_cache_size': 100000,
                'metadata_cache_ttl': 14400,
                'query_cache_size': 5000
            }
        )
        
        # Initialize tree updater
        self.tree_updater = TreeUpdater(
            components['tree_manager'],
            self.vector_manager,  # Use VectorManager instead of PineconeManager
            components['storage']['namespace']
        )
        
        # Initialize system monitor
        self.monitor = SystemMonitor(self.index_dir)

    def process_incrementally(self) -> bool:
        """Process new or changed documents incrementally."""
        try:
            # Check if index exists and has vectors
            store_manager: PineconeManager = self.components['storage']['manager']
            index_stats = store_manager.index.describe_index_stats()
            has_existing_vectors = any(ns.vector_count > 0 for ns in index_stats.namespaces.values())
            
            # Get changed documents
            changed_docs = self.manifest.get_changed_documents(self.raw_dir)
            if not changed_docs:
                logger.info("No new or modified documents found.")
                return True
                
            logger.info(f"Found {len(changed_docs)} new or modified documents")
            
            # Process documents
            if has_existing_vectors:
                logger.info("Index contains existing vectors - performing incremental update")
                success = self._update_existing_index(changed_docs)
            else:
                logger.info("Index is empty - performing full processing")
                success = self._process_new_index(changed_docs)
                
            return success
            
        except Exception as e:
            logger.error(f"Incremental processing failed: {str(e)}")
            return False

    def _update_existing_index(self, changed_docs: List[Path]) -> bool:
        """Update existing index with new or modified documents."""
        try:
            # Process each document
            for doc_path in changed_docs:
                try:
                    # Register document in manifest
                    doc_entry = self.manifest.register_document(doc_path, {
                        "path": str(doc_path),
                        "status": "pending"
                    })
                    
                    # Process document
                    success = self._process_single_document(doc_path)
                    
                    # Update status and move file
                    if success:
                        self.manifest.update_document_status(doc_path, "processed")
                        self._move_to_processed(doc_path)
                    else:
                        self.manifest.update_document_status(doc_path, "failed")
                        self._move_to_failed(doc_path)
                        
                except Exception as e:
                    logger.error(f"Failed to process document {doc_path}: {str(e)}")
                    self.manifest.update_document_status(doc_path, "failed")
                    self._move_to_failed(doc_path)
                    
            return True
            
        except Exception as e:
            logger.error(f"Failed to update existing index: {str(e)}")
            return False

    def _process_new_index(self, documents: List[Path]) -> bool:
        """Process documents for a new or empty index."""
        try:
            # Initialize components
            chunker = ChunkManager()
            embed_manager = self.components['embed_manager']
            store_manager = self.components['storage']['manager']
            tree_manager = self.components['tree_manager']
            base_ns = self.components['storage']['namespace']
            
            # Process all documents
            all_chunks = []
            all_metadata = []
            doc_chunk_map = {}  # Track chunks per document
            
            # Create chunks for all documents
            for doc_path in documents:
                try:
                    with open(doc_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # Chunk document
                    chunks = list(chunker.chunk_document(text, {"path": str(doc_path)}))
                    chunk_texts = [chunk[0] for chunk in chunks]
                    chunk_metadata = [chunk[1] for chunk in chunks]
                    
                    # Store chunk information
                    start_idx = len(all_chunks)
                    all_chunks.extend(chunk_texts)
                    all_metadata.extend(chunk_metadata)
                    doc_chunk_map[str(doc_path)] = {
                        'start_idx': start_idx,
                        'num_chunks': len(chunks)
                    }
                    
                    logger.info(f"Created {len(chunks)} chunks from document: {doc_path.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to chunk document {doc_path}: {str(e)}")
                    return False
            
            # Generate embeddings for all chunks
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
            all_embeddings = embed_manager.embed_texts(all_chunks, all_metadata)
            
            # Build tree structure
            logger.info("Building document tree...")
            metadata = [{} for _ in range(len(all_chunks))]
            tree_data = tree_manager.build_tree(all_chunks, all_embeddings, metadata)
            if not tree_data:
                logger.error("Failed to build tree structure")
                return False
            
            # Prepare and store vectors
            chunks_ns = f"{base_ns}_chunks"
            summaries_ns = f"{base_ns}_summaries"
            
            # Process leaf nodes (chunks)
            chunk_vectors = []
            for doc_path, chunk_info in doc_chunk_map.items():
                start_idx = chunk_info['start_idx']
                num_chunks = chunk_info['num_chunks']
                chunk_ids = []
                
                for i in range(num_chunks):
                    idx = start_idx + i
                    vector_id = f"chunk_{Path(doc_path).stem}_{int(time.time())}_{i}"
                    chunk_ids.append(vector_id)
                    
                    metadata = all_metadata[idx]
                    metadata.update({
                        'text': all_chunks[idx],
                        'vector_id': vector_id,
                        'node_type': 'leaf',
                        'source_doc': doc_path,
                        'chunk_index': i,
                        'tree_node_id': tree_data.get('node_ids', {}).get(idx, '')
                    })
                    
                    chunk_vectors.append({
                        'id': vector_id,
                        'values': all_embeddings[idx].tolist() if isinstance(all_embeddings[idx], np.ndarray) else all_embeddings[idx],
                        'metadata': metadata
                    })
                
                # Update manifest for this document
                self.manifest.update_document_status(
                    Path(doc_path),
                    "processed",
                    chunk_ids=chunk_ids
                )
            
            # Store leaf nodes
            if chunk_vectors:
                store_manager.upsert_batch(chunk_vectors, chunks_ns)
                logger.info(f"✓ Upserted {len(chunk_vectors)} leaf nodes")
            
            # Process and store internal nodes
            summary_vectors = []
            for node_id, node_data in tree_data.get('internal_nodes', {}).items():
                if node_data.get('embedding') is not None:
                    vector_id = f"summary_{int(time.time())}_{node_id}"
                    metadata = {
                        'text': node_data.get('summary', ''),
                        'vector_id': vector_id,
                        'node_type': 'internal',
                        'tree_node_id': node_id,
                        'depth': node_data.get('depth', 0),
                        'num_children': len(node_data.get('children', [])),
                        'children_ids': node_data.get('children', [])
                    }
                    summary_vectors.append({
                        'id': vector_id,
                        'values': node_data['embedding'].tolist() if isinstance(node_data['embedding'], np.ndarray) else node_data['embedding'],
                        'metadata': metadata
                    })
            
            # Store internal nodes
            if summary_vectors:
                store_manager.upsert_batch(summary_vectors, summaries_ns)
                logger.info(f"✓ Upserted {len(summary_vectors)} internal nodes")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process document batch: {str(e)}")
            return False

    def _process_single_document(self, doc_path: Path) -> bool:
        """Process a single document for incremental updates."""
        start_time = time.time()
        try:
            # Load and chunk document
            with open(doc_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Create chunks using ChunkManager
            chunker = ChunkManager()
            chunks = list(chunker.chunk_document(text, {"path": str(doc_path)}))
            chunk_texts = [chunk[0] for chunk in chunks]
            chunk_metadata = [chunk[1] for chunk in chunks]
            
            logger.info(f"Created {len(chunks)} chunks from document: {doc_path.name}")
            
            # Generate embeddings
            embed_manager = self.components['embed_manager']
            embeddings = embed_manager.embed_texts(chunk_texts, chunk_metadata)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Check if document exists in manifest
            old_chunk_ids = []
            if self.manifest.document_exists(doc_path):
                old_chunk_ids = self.manifest.get_document_chunk_ids(doc_path)
            
            # Update tree structure
            tree_data = self.tree_updater.update_tree(
                chunk_texts,
                embeddings,
                removed_chunk_ids=old_chunk_ids if old_chunk_ids else None,
                modified_chunk_ids=None
            )
            
            if not tree_data:
                logger.error("Failed to update tree structure")
                return False
            
            # Get new chunk IDs from tree data
            new_chunk_ids = []
            for node_id in tree_data.get('leaf_nodes', {}):
                new_chunk_ids.append(f"chunk_{int(time.time())}_{node_id}")
            
            # Update manifest
            self.manifest.update_document_status(
                doc_path,
                "processed",
                chunk_ids=new_chunk_ids
            )
            
            # Update monitoring metrics
            chunks_created = len(chunks)
            processing_time = time.time() - start_time
            self.monitor.update_processing_metrics(
                docs_processed=1,
                chunks_created=chunks_created,
                processing_time=processing_time
            )
            
            # Update vector metrics
            vector_stats = self.vector_manager.get_stats()
            self.monitor.update_vector_metrics(vector_stats)
            
            # Update system metrics
            self.monitor.update_system_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process document {doc_path}: {str(e)}")
            self.monitor.update_processing_metrics(docs_failed=1)
            return False

    def _move_to_processed(self, doc_path: Path):
        """Move document to processed directory."""
        target_path = self.processed_dir / doc_path.relative_to(self.raw_dir)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.rename(target_path)

    def _move_to_failed(self, doc_path: Path):
        """Move document to failed directory."""
        target_path = self.failed_dir / doc_path.relative_to(self.raw_dir)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.rename(target_path)
