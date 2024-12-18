import sqlite3
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, index_name: str):
        """Initialize cache manager with SQLite database."""
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.db_path = self.cache_dir / f"{index_name}_processing.db"
        self.logger = logging.getLogger(__name__)
        self.conn = None
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize SQLite database with required tables."""
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            cursor = self.conn.cursor()
            
            # Create tables
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding TEXT,
                    created_at INTEGER
                );
                
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    parent_id TEXT,
                    created_at INTEGER,
                    FOREIGN KEY(parent_id) REFERENCES nodes(node_id)
                );
                
                CREATE TABLE IF NOT EXISTS clusters (
                    cluster_id TEXT PRIMARY KEY,
                    nodes TEXT NOT NULL,
                    summary TEXT,
                    embedding TEXT,
                    created_at INTEGER
                );
                
                CREATE TABLE IF NOT EXISTS vectors (
                    vector_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at INTEGER
                );
            """)
            
            self.conn.commit()
            self.logger.info(f"Initialized cache database at {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cache database: {str(e)}")
            raise
            
    def get_all_nodes(self) -> List[Dict]:
        """Get all nodes with their embeddings and metadata."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT chunk_id, text, metadata, embedding 
                FROM chunks 
                WHERE embedding IS NOT NULL
            """)
            
            nodes = []
            for row in cursor.fetchall():
                chunk_id, text, metadata_json, embedding_json = row
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    node = {
                        'id': chunk_id,
                        'text': text,
                        'metadata': metadata,
                        'embedding': json.loads(embedding_json) if embedding_json else [],
                        'cluster_id': metadata.get('cluster_id')
                    }
                    nodes.append(node)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to decode JSON for chunk {chunk_id}: {str(e)}")
                    continue
            
            self.logger.info(f"Retrieved {len(nodes)} nodes from cache")
            for i, node in enumerate(nodes[:3]):
                self.logger.info(f"Node {i} cluster_id: {node.get('cluster_id', 'missing')}")
            
            return nodes
            
        except Exception as e:
            self.logger.error(f"Failed to get nodes from cache: {str(e)}")
            raise
            
    def store_chunk(self, chunk_id: str, text: str, metadata: Dict):
        """Store document chunk in cache."""
        try:
            # Verify metadata has required fields
            if 'filename' not in metadata:
                self.logger.warning(f"No filename in metadata for chunk {chunk_id}")
                metadata['filename'] = chunk_id.split('_')[0]
            
            # Ensure cluster_id is preserved if present
            if 'cluster_id' in metadata:
                self.logger.info(f"Preserving cluster_id: {metadata['cluster_id']} for chunk {chunk_id}")
            
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO chunks (chunk_id, text, metadata, created_at) VALUES (?, ?, ?, ?)",
                (chunk_id, text, json.dumps(metadata), int(time.time()))
            )
            self.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to store chunk {chunk_id}: {str(e)}")
            raise
            
    def store_embedding(self, chunk_id: str, embedding: List[float]):
        """Store embedding for a chunk."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE chunks SET embedding = ? WHERE chunk_id = ?",
                (json.dumps(embedding), chunk_id)
            )
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store embedding for chunk {chunk_id}: {str(e)}")
            raise
            
    def get_unprocessed_chunks(self, batch_size: int = 10) -> List[Dict]:
        """Get chunks that don't have embeddings yet."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT chunk_id, text, metadata FROM chunks WHERE embedding IS NULL LIMIT ?",
                (batch_size,)
            )
            return [
                {
                    'chunk_id': row[0],
                    'text': row[1],
                    'metadata': json.loads(row[2])
                }
                for row in cursor.fetchall()
            ]
        except Exception as e:
            self.logger.error(f"Failed to get unprocessed chunks: {str(e)}")
            raise
            
    def store_cluster(self, cluster_id: str, nodes: List[str], summary: Optional[str] = None):
        """Store cluster information."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO clusters (cluster_id, nodes, summary, created_at) VALUES (?, ?, ?, ?)",
                (cluster_id, json.dumps(nodes), summary, int(time.time()))
            )
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store cluster {cluster_id}: {str(e)}")
            raise
            
    def get_cluster_data(self) -> List[Dict]:
        """Get all cluster data for final tree building."""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT cluster_id, nodes, summary FROM clusters")
            return [
                {
                    'cluster_id': row[0],
                    'nodes': json.loads(row[1]),
                    'summary': row[2]
                }
                for row in cursor.fetchall()
            ]
        except Exception as e:
            self.logger.error(f"Failed to get cluster data: {str(e)}")
            raise
            
    def get_all_vectors(self) -> List[Dict]:
        """Retrieve all vectors from cache."""
        try:
            self.logger.info("Retrieving all vectors from cache")
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT v.vector_id, v.text, v.embedding, v.metadata, v.created_at
                FROM vectors v
                ORDER BY v.created_at ASC
            """)
            
            vectors = []
            for row in cursor.fetchall():
                vector = {
                    'vector_id': row[0],
                    'text': row[1],
                    'embedding': json.loads(row[2]),
                    'metadata': json.loads(row[3]),
                    'created_at': row[4]
                }
                vectors.append(vector)
                
            self.logger.info(f"Retrieved {len(vectors)} vectors from cache")
            return vectors
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve vectors from cache: {str(e)}")
            raise
            
    def clear_vectors(self) -> None:
        """Clear all vectors from cache after successful offload."""
        try:
            self.logger.info("Clearing vectors from cache")
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM vectors")
            self.conn.commit()
            self.logger.info("Cache cleared successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {str(e)}")
            raise
            
    def cleanup(self):
        """Clean up database connection."""
        if self.conn:
            self.conn.close()