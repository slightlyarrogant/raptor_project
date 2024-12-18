#!/usr/bin/env python3

import logging
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set
import argparse
import os
from src.tree.tree_manager import TreeManager
from src.embeddings.openai_manager import OpenAIManager
from src.storage.store_manager import StoreManager
import time

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProcessRecovery:
    def __init__(self, index_name: str):
        self.index_name = index_name
        self.config = {
            'index_name': index_name,
            'batch_size': 1000,
            'pinecone': {
                'api_key': os.getenv('PINECONE_API_KEY'),
                'index_name': index_name
            }
        }
        
        # Initialize store manager
        self.store_manager = StoreManager(self.config)
        self.embed_manager = OpenAIManager()
        self.tree_manager = TreeManager({
            'embedding': {
                'manager': self.embed_manager,
                'model': 'text-embedding-3-small',
                'batch_size': 100
            },
            'clustering': {
                'method': 'kmeans',
                'min_cluster_size': 2,
                'max_clusters': 5,
                'dimension': 20,
                'threshold': 0.15,
                'max_levels': 5
            },
            'storage': {
                'manager': self.store_manager
            }
        })

    def analyze_state(self) -> Dict:
        """Analyze the current state of processing"""
        # Get all batch files
        data_path = self.store_manager.store['data_path']
        index_path = self.store_manager.store['index_path']
        
        # Load indices
        chunk_index = self.store_manager._load_or_create_index('chunks')
        embedding_index = self.store_manager._load_or_create_index('embeddings')
        node_index = self.store_manager._load_or_create_index('nodes')
        
        # Get batch files
        chunks_pattern = "chunks/batch_*.json"
        embeddings_pattern = "embeddings/batch_*.npy"
        nodes_pattern = "nodes_batch_*.json"
        
        chunks = set(int(f.stem.split('_')[1]) for f in Path(data_path).glob(chunks_pattern))
        embeddings = set(int(f.stem.split('_')[1]) for f in Path(data_path).glob(embeddings_pattern))
        nodes = set(int(f.stem.split('_')[2]) for f in Path(data_path).glob(nodes_pattern))
        
        # For embeddings, we want sequential batches
        if embeddings:
            max_emb_batch = max(embeddings)
            missing_embeddings = set(range(max_emb_batch + 1)) - embeddings
        else:
            missing_embeddings = set()
            
        # For nodes, we'll renumber them to be sequential
        if nodes:
            sorted_nodes = sorted(nodes)
            node_mapping = {old: new for new, old in enumerate(sorted_nodes)}
            self.node_batch_mapping = node_mapping
        else:
            self.node_batch_mapping = {}
            
        # For chunks, we'll match embedding batches
        missing_chunks = embeddings - chunks
        
        # Log findings
        logger.info(f"Found {len(chunks)} chunk batches")
        logger.info(f"Found {len(embeddings)} embedding batches")
        logger.info(f"Found {len(nodes)} node batches")
        logger.info(f"Missing chunk batches: {len(missing_chunks)}")
        logger.info(f"Missing embedding batches: {len(missing_embeddings)}")
        if self.node_batch_mapping:
            logger.info(f"Will renumber node batches from {min(nodes)}-{max(nodes)} to 0-{len(nodes)-1}")
        
        return {
            'missing_chunks': missing_chunks,
            'missing_embeddings': missing_embeddings,
            'node_mapping': self.node_batch_mapping,
            'chunk_index': chunk_index,
            'embedding_index': embedding_index,
            'node_index': node_index
        }

    def recover_embeddings(self, missing_batches: Set[int], chunk_index: Dict):
        """Generate missing embeddings"""
        if not missing_batches:
            logger.info("No missing embedding batches to recover")
            return
            
        logger.info(f"Recovering {len(missing_batches)} embedding batches")
        for batch_num in sorted(missing_batches):
            try:
                # Load chunk batch using new directory structure
                chunk_file = os.path.join(self.store_manager.store['data_path'], 'chunks', f"batch_{batch_num}.json")
                if not os.path.exists(chunk_file):
                    logger.error(f"Chunk batch {batch_num} not found")
                    continue
                    
                with open(chunk_file, 'r') as f:
                    chunks = json.load(f)
                
                # Generate embeddings
                texts = [chunk.get('text', '') for chunk in chunks]
                embeddings = self.embed_manager.get_embeddings(texts)
                
                # Save embeddings with metadata
                embedding_file = os.path.join(self.store_manager.store['data_path'], 'embeddings', f"batch_{batch_num}.npy")
                np.save(embedding_file, np.array(embeddings))
                
                # Save metadata separately
                metadata_batch = []
                for i, chunk in enumerate(chunks):
                    metadata_batch.append({
                        'text': chunk.get('text', ''),
                        'metadata': self.generate_metadata(chunk, batch_num, i),
                        'id': chunk.get('id'),
                        'batch_num': batch_num,
                        'index': i
                    })
                
                metadata_file = os.path.join(self.store_manager.store['data_path'], 'embeddings_metadata', f"batch_{batch_num}.json")
                os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
                with open(metadata_file, 'w') as f:
                    json.dump(metadata_batch, f, indent=2)
                
                logger.info(f"Recovered embeddings for batch {batch_num}")
                
            except Exception as e:
                logger.error(f"Failed to recover embeddings for batch {batch_num}: {str(e)}")

    def recover_chunks(self, missing_batches: Set[int], embedding_index: Dict):
        """Generate missing chunks from embeddings"""
        if not missing_batches:
            logger.info("No missing chunk batches to recover")
            return
            
        logger.info(f"Recovering {len(missing_batches)} chunk batches")
        data_path = self.store_manager.store['data_path']
        
        # Load the embedding index
        try:
            with open(os.path.join(self.store_manager.store['index_path'], 'embeddings_index.json'), 'r') as f:
                embedding_index = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load embedding index: {str(e)}")
            return
        
        for batch_num in sorted(missing_batches):
            try:
                # Load embedding batch
                embedding_file = os.path.join(data_path, 'embeddings', f"batch_{batch_num}.npy")
                metadata_file = os.path.join(data_path, 'embeddings_metadata', f"batch_{batch_num}.json")
                
                if not os.path.exists(embedding_file) or not os.path.exists(metadata_file):
                    logger.error(f"Embedding batch {batch_num} or its metadata not found")
                    continue
                    
                embeddings = np.load(embedding_file)
                with open(metadata_file, 'r') as f:
                    metadata_batch = json.load(f)
                
                # Create chunks with full metadata
                chunks = []
                for i, metadata in enumerate(metadata_batch):
                    chunk_id = metadata.get('id', f"chunk_{batch_num}_{i}")
                    chunk_metadata = metadata.get('metadata', {})
                    
                    # Ensure required metadata fields
                    if 'filename' not in chunk_metadata:
                        chunk_metadata['filename'] = f"unknown_file_{batch_num}"
                    if 'chunk_num' not in chunk_metadata:
                        chunk_metadata['chunk_num'] = i
                    
                    # Add embedding info to metadata
                    chunk_metadata.update({
                        'batch_num': batch_num,
                        'batch_index': i,
                        'embedding_size': len(embeddings[i]) if i < len(embeddings) else 0
                    })
                    
                    chunks.append({
                        'id': chunk_id,
                        'text': metadata.get('text', ''),
                        'metadata': chunk_metadata
                    })
                    
                    # If there's a summary, save it
                    if 'summary' in chunk_metadata:
                        self.store_manager.save_summary(
                            text_id=chunk_id,
                            summary=chunk_metadata['summary'],
                            metadata=chunk_metadata
                        )
                
                # Save chunks
                chunk_file = os.path.join(data_path, 'chunks', f"batch_{batch_num}.json")
                os.makedirs(os.path.dirname(chunk_file), exist_ok=True)
                with open(chunk_file, 'w') as f:
                    json.dump(chunks, f, indent=2)
                
                logger.info(f"Recovered {len(chunks)} chunks for batch {batch_num}")
                
            except Exception as e:
                logger.error(f"Failed to recover chunks for batch {batch_num}: {str(e)}")

    def generate_metadata(self, chunk: Dict, batch_num: int, index: int) -> Dict:
        """Generate metadata for a document chunk.
        
        Args:
            chunk: The document chunk dictionary.
            batch_num: The batch number for the chunk.
            index: The index of the chunk in the batch.
        
        Returns:
            A dictionary containing the generated metadata.
        """
        metadata = chunk.get('metadata', {})
        
        # Ensure required metadata fields
        if 'filename' not in metadata:
            metadata['filename'] = chunk.get('filename', f"unknown_file_{batch_num}")
        if 'chunk_num' not in metadata:
            metadata['chunk_num'] = index

        # Add additional metadata fields if necessary
        metadata['batch_num'] = batch_num
        metadata['text'] = chunk.get('text', '')
        metadata['creation_time'] = time.time()
        metadata['is_leaf'] = True  # Assuming chunks are leaf nodes
        metadata['node_type'] = 'leaf'  # Set node type to leaf
        
        return metadata

    def build_tree(self):
        """Build the document tree from recovered chunks"""
        logger.info("Building document tree from recovered chunks...")
        try:
            # Get all chunks
            chunks = []
            chunk_dir = os.path.join(self.store_manager.store['data_path'], 'chunks')
            for batch_file in sorted(Path(chunk_dir).glob("batch_*.json")):
                with open(batch_file, 'r') as f:
                    batch_chunks = json.load(f)
                chunks.extend(batch_chunks)
            
            if not chunks:
                logger.error("No chunks found to build tree")
                return False
            
            # Get corresponding embeddings
            embeddings = []
            embedding_dir = os.path.join(self.store_manager.store['data_path'], 'embeddings')
            for batch_file in sorted(Path(embedding_dir).glob("batch_*.npy")):
                batch_embeddings = np.load(batch_file)
                embeddings.extend(batch_embeddings)
            
            if len(chunks) != len(embeddings):
                logger.error(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")
                return False
            
            # Build tree
            tree = self.tree_manager.build_tree(
                texts=[chunk['text'] for chunk in chunks],
                embeddings=np.array(embeddings),
                metadata=[chunk['metadata'] for chunk in chunks]
            )
            
            logger.info("Tree building complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build tree: {str(e)}")
            return False

    def clean_corrupted_state(self):
        """Clean up any corrupted state"""
        logger.info("Cleaning up corrupted state...")
        
        data_path = self.store_manager.store['data_path']
        index_path = self.store_manager.store['index_path']
        
        # First analyze state to get node batch mapping
        state = self.analyze_state()
        
        # Renumber node batches if needed
        if self.node_batch_mapping:
            # First make copies of all files to avoid conflicts
            for old_num in self.node_batch_mapping:
                new_num = self.node_batch_mapping[old_num]
                old_file = Path(data_path) / f"nodes_batch_{old_num}.json"
                temp_file = Path(data_path) / f"nodes_batch_{old_num}.temp"
                if old_file.exists():
                    old_file.rename(temp_file)
            
            # Now rename temp files to new numbers
            for old_num in self.node_batch_mapping:
                new_num = self.node_batch_mapping[old_num]
                temp_file = Path(data_path) / f"nodes_batch_{old_num}.temp"
                new_file = Path(data_path) / f"nodes_batch_{new_num}.json"
                if temp_file.exists():
                    logger.info(f"Renaming node batch {old_num} to {new_num}")
                    temp_file.rename(new_file)
        
        # Update batch counters based on actual files
        max_chunk = max(int(f.stem.split('_')[1]) for f in Path(data_path).glob("chunks/batch_*.json")) if list(Path(data_path).glob("chunks/batch_*.json")) else -1
        max_emb = max(int(f.stem.split('_')[1]) for f in Path(data_path).glob("embeddings/batch_*.npy")) if list(Path(data_path).glob("embeddings/batch_*.npy")) else -1
        max_node = max(self.node_batch_mapping.values()) if self.node_batch_mapping else -1
        
        self.store_manager.batch_counters = {
            'chunks': max_chunk + 1,
            'embeddings': max_emb + 1,
            'nodes': max_node + 1
        }
        self.store_manager._save_index('batch_counters', self.store_manager.batch_counters)
        
        # Clean up any partial batches
        for pattern in ['chunks/batch_*.json', 'embeddings/batch_*.npy', 'nodes_batch_*.json']:
            for f in Path(data_path).glob(pattern):
                try:
                    if pattern == 'nodes_batch_*.json':
                        batch_num = int(f.stem.split('_')[2])
                        # Only remove if it's not a new numbered batch
                        if batch_num >= len(self.node_batch_mapping):
                            f.unlink()
                            logger.info(f"Removed unmapped node batch file: {f}")
                    else:
                        batch_num = int(f.stem.split('_')[1])
                        if batch_num < 0:  # Invalid batch number
                            f.unlink()
                            logger.info(f"Removed corrupted batch file: {f}")
                except (ValueError, IndexError):
                    f.unlink()
                    logger.info(f"Removed malformed batch file: {f}")
                    
        logger.info("State cleanup complete")

    def recover_process(self):
        """Main recovery process"""
        try:
            # Clean up any corrupted state first
            self.clean_corrupted_state()
            
            # Analyze current state
            logger.info("Analyzing current state...")
            state = self.analyze_state()
            
            # Recover missing chunks from embeddings
            self.recover_chunks(state['missing_chunks'], state['embedding_index'])
            
            # Recover embeddings
            self.recover_embeddings(state['missing_embeddings'], state['chunk_index'])
            
            # Build tree from recovered data
            if self.build_tree():
                logger.info("Tree building successful")
            else:
                logger.error("Failed to build tree from recovered data")
            
            # Final state check
            logger.info("Checking final state...")
            final_state = self.analyze_state()
            
            if not any([final_state['missing_chunks'], final_state['missing_embeddings']]):
                logger.info("Recovery complete - all batches are now present")
            else:
                logger.warning("Recovery completed with some batches still missing - may need another run")
                
        except Exception as e:
            logger.error(f"Recovery failed: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Recover processing state for an index')
    parser.add_argument('--index-name', type=str, required=True,
                       help='Name of the index (e.g., raptor-technicalbase)')
    args = parser.parse_args()
    
    recovery = ProcessRecovery(args.index_name)
    recovery.recover_process()

if __name__ == '__main__':
    main()
