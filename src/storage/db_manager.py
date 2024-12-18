import logging
import json
import numpy as np
from pathlib import Path
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import tempfile
import shutil

logger = logging.getLogger(__name__)

class DatabaseManager:
    """File-based cache manager that stores each piece of data in individual files for maximum resilience."""
    
    def __init__(self, data_dir: str, index_name: str):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir) / index_name
        self.temp_dir = self.data_dir / 'temp'
        
        # Create directory structure
        self.chunks_dir = self.data_dir / "chunks"
        self.embeddings_dir = self.data_dir / "embeddings"
        self.clusters_dir = self.data_dir / "clusters"
        self.summaries_dir = self.data_dir / "summaries"
        
        for dir in [self.chunks_dir, self.embeddings_dir, 
                   self.clusters_dir, self.summaries_dir, self.temp_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        # Create or load state file
        self.state_file = self.data_dir / "state.json"
        self._load_or_create_state()
        
        self.logger.info(f"Initialized file-based cache at {self.data_dir}")

    def _get_temp_file(self, target_file: Path) -> Path:
        """Get a temporary file path in the temp directory."""
        return self.temp_dir / f"{target_file.name}.tmp"

    def _atomic_write(self, data: Dict, target_file: Path) -> None:
        """Write data atomically using a temporary file."""
        temp_file = self._get_temp_file(target_file)
        try:
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            # Atomic rename
            temp_file.replace(target_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise e

    def _load_or_create_state(self):
        """Load or create the state tracking file"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    self.state = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load state file: {e}")
                self.state = self._create_initial_state()
        else:
            self.state = self._create_initial_state()
        self._save_state()

    def _create_initial_state(self) -> Dict:
        """Create initial state structure"""
        return {
            'started_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'stats': {
                'chunks_created': 0,
                'embeddings_created': 0,
                'clusters_created': 0,
                'summaries_created': 0
            },
            'processing_status': 'initializing'
        }

    def _save_state(self):
        """Save current state to file"""
        self.state['last_updated'] = datetime.now().isoformat()
        self._atomic_write(self.state, self.state_file)

    def save_chunk(self, doc_id: str, chunk_index: int, text: str, metadata: Dict):
        """Save a text chunk to an individual JSON file"""
        chunk_id = f"{doc_id}_chunk{chunk_index}"
        chunk_file = self.chunks_dir / f"{chunk_id}.json"
        
        data = {
            'text': text,
            'metadata': {
                **metadata,
                'doc_id': doc_id,
                'chunk_index': chunk_index,
                'created_at': datetime.now().isoformat()
            }
        }
        
        self._atomic_write(data, chunk_file)
        
        # Update state
        self.state['stats']['chunks_created'] += 1
        self._save_state()
        
        self.logger.debug(f"Saved chunk {chunk_id}")

    def save_embedding(self, doc_id: str, chunk_index: int, embedding: np.ndarray):
        """Save an embedding vector to a numpy file"""
        chunk_id = f"{doc_id}_chunk{chunk_index}"
        emb_file = self.embeddings_dir / f"{chunk_id}.npy"
        temp_file = self._get_temp_file(emb_file)
        
        try:
            np.save(temp_file, embedding)
            temp_file.replace(emb_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise e
        
        # Update state
        self.state['stats']['embeddings_created'] += 1
        self._save_state()
        
        self.logger.debug(f"Saved embedding for {chunk_id}")

    def save_cluster(self, level: int, cluster_id: str, data: Dict):
        """Save cluster information to JSON file"""
        cluster_dir = self.clusters_dir / f"level{level}"
        cluster_dir.mkdir(exist_ok=True)
        
        cluster_file = cluster_dir / f"{cluster_id}.json"
        
        # Add metadata
        data['metadata'] = {
            **data.get('metadata', {}),
            'created_at': datetime.now().isoformat(),
            'level': level
        }
        
        self._atomic_write(data, cluster_file)
        
        # Update state
        self.state['stats']['clusters_created'] += 1
        self._save_state()
        
        self.logger.debug(f"Saved cluster {cluster_id} at level {level}")

    def save_summary(self, cluster_id: str, summary: str):
        """Save a cluster summary to JSON file"""
        summary_file = self.summaries_dir / f"{cluster_id}.json"
        
        data = {
            'summary': summary,
            'metadata': {
                'created_at': datetime.now().isoformat()
            }
        }
        
        self._atomic_write(data, summary_file)
        
        # Update state
        self.state['stats']['summaries_created'] += 1
        self._save_state()
        
        self.logger.debug(f"Saved summary for cluster {cluster_id}")

    def cleanup_temp_files(self):
        """Clean up temporary files older than 1 hour."""
        try:
            for temp_file in self.temp_dir.glob('*.tmp'):
                if (datetime.now().timestamp() - temp_file.stat().st_mtime) > 3600:
                    temp_file.unlink()
        except Exception as e:
            self.logger.error(f"Error cleaning up temp files: {str(e)}")

    def __del__(self):
        """Cleanup on object destruction."""
        self.cleanup_temp_files()

    def save_node(self, node_id: str, parent_id: Optional[str], level: int, text: str, metadata: Dict, embedding_id: Optional[str] = None) -> bool:
        """Save a node to the database"""
        try:
            node_data = {
                'id': node_id,
                'parent_id': parent_id,
                'level': level,
                'text': text,
                'metadata': metadata,
                'embedding_id': embedding_id,
                'created_at': datetime.now().isoformat()
            }
            
            # Save to clusters directory using level-based subdirectories
            level_dir = self.clusters_dir / f"level_{level}"
            level_dir.mkdir(exist_ok=True)
            
            node_file = level_dir / f"{node_id}.json"
            temp_file = node_file.with_suffix('.tmp')
            
            # Write to temporary file first
            with open(temp_file, 'w') as f:
                json.dump(node_data, f, indent=2)
            
            # Atomic rename
            temp_file.rename(node_file)
            
            # Update state
            self.state['stats']['clusters_created'] += 1
            self._save_state()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save node {node_id}: {str(e)}")
            return False

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Retrieve a node from the database"""
        try:
            # Search in all level directories
            for level_dir in self.clusters_dir.glob("level_*"):
                node_file = level_dir / f"{node_id}.json"
                if node_file.exists():
                    with open(node_file) as f:
                        return json.load(f)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve node {node_id}: {str(e)}")
            return None

    def get_chunks(self, doc_id: Optional[str] = None) -> List[Dict]:
        """Get all chunks or chunks for a specific document"""
        chunks = []
        pattern = f"{doc_id}_chunk*.json" if doc_id else "*.json"
        
        for chunk_file in self.chunks_dir.glob(pattern):
            try:
                with open(chunk_file) as f:
                    chunks.append(json.load(f))
            except Exception as e:
                self.logger.error(f"Failed to load chunk {chunk_file}: {e}")
                
        return chunks

    def get_embedding(self, doc_id: str, chunk_index: int) -> Optional[np.ndarray]:
        """Get embedding for a specific chunk"""
        chunk_id = f"{doc_id}_chunk{chunk_index}"
        emb_file = self.embeddings_dir / f"{chunk_id}.npy"
        
        if emb_file.exists():
            try:
                return np.load(emb_file)
            except Exception as e:
                self.logger.error(f"Failed to load embedding {chunk_id}: {e}")
                return None
        return None

    def get_clusters(self, level: Optional[int] = None) -> List[Dict]:
        """Get all clusters or clusters at a specific level"""
        clusters = []
        if level is not None:
            cluster_dir = self.clusters_dir / f"level{level}"
            if not cluster_dir.exists():
                return []
            files = cluster_dir.glob("*.json")
        else:
            files = self.clusters_dir.glob("**/*.json")
            
        for cluster_file in files:
            try:
                with open(cluster_file) as f:
                    clusters.append(json.load(f))
            except Exception as e:
                self.logger.error(f"Failed to load cluster {cluster_file}: {e}")
                
        return clusters

    def get_summary(self, cluster_id: str) -> Optional[str]:
        """Get summary for a specific cluster"""
        summary_file = self.summaries_dir / f"{cluster_id}.json"
        
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    data = json.load(f)
                return data['summary']
            except Exception as e:
                self.logger.error(f"Failed to load summary {cluster_id}: {e}")
                return None
        return None

    def get_state(self) -> Dict:
        """Get current processing state"""
        return self.state.copy()

    def update_status(self, status: str):
        """Update processing status"""
        self.state['processing_status'] = status
        self._save_state()

    def clear_cache(self):
        """Clear all cached data"""
        import shutil
        for dir in [self.chunks_dir, self.embeddings_dir, 
                   self.clusters_dir, self.summaries_dir]:
            shutil.rmtree(dir)
            dir.mkdir()
        
        self.state = self._create_initial_state()
        self._save_state()
        self.logger.info("Cache cleared")
