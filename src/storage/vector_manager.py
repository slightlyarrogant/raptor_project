"""Vector lifecycle management and caching for efficient vector operations."""

import logging
from typing import Dict, List, Optional, Set, Tuple
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import numpy as np
from cachetools import TTLCache, LRUCache

from src.storage.pinecone_manager import PineconeManager

logger = logging.getLogger(__name__)

class VectorManager:
    def __init__(self, store_manager: PineconeManager, namespace: str, cache_config: Optional[Dict] = None):
        """
        Initialize vector manager with caching and cleanup capabilities.
        
        Args:
            store_manager: PineconeManager instance
            namespace: Base namespace for vectors
            cache_config: Optional cache configuration with settings:
                - vector_cache_size: Max number of vectors to cache (default: 10000)
                - vector_cache_ttl: Time to live for cached vectors in seconds (default: 3600)
                - metadata_cache_size: Max number of metadata entries (default: 50000)
                - metadata_cache_ttl: Time to live for metadata in seconds (default: 7200)
        """
        self.store_manager = store_manager
        self.base_namespace = namespace
        self.chunks_ns = f"{namespace}_chunks"
        self.summaries_ns = f"{namespace}_summaries"
        
        # Set up cache configuration
        self.cache_config = cache_config or {}
        self._setup_caches()
        
        # Initialize vector version tracking
        self.version_file = Path(f"data/{namespace}/vector_versions.json")
        self.version_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_versions()
        
        # Initialize statistics tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.query_times = []

    def _setup_caches(self):
        """Initialize vector and metadata caches."""
        # Vector cache: Store frequently accessed vectors
        self.vector_cache = TTLCache(
            maxsize=self.cache_config.get('vector_cache_size', 10000),
            ttl=self.cache_config.get('vector_cache_ttl', 3600)
        )
        
        # Metadata cache: Store frequently accessed metadata
        self.metadata_cache = TTLCache(
            maxsize=self.cache_config.get('metadata_cache_size', 50000),
            ttl=self.cache_config.get('metadata_cache_ttl', 7200)
        )
        
        # Query result cache: Store recent query results
        self.query_cache = LRUCache(
            maxsize=self.cache_config.get('query_cache_size', 1000)
        )

    def _load_versions(self):
        """Load vector version information from file."""
        try:
            if self.version_file.exists():
                with open(self.version_file, 'r') as f:
                    self.versions = json.load(f)
            else:
                self.versions = {
                    'chunks': {},
                    'summaries': {},
                    'last_cleanup': None
                }
                self._save_versions()
        except Exception as e:
            logger.error(f"Failed to load vector versions: {str(e)}")
            self.versions = {
                'chunks': {},
                'summaries': {},
                'last_cleanup': None
            }

    def _save_versions(self):
        """Save vector version information to file."""
        try:
            with open(self.version_file, 'w') as f:
                json.dump(self.versions, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save vector versions: {str(e)}")

    def get_vectors(self, vector_ids: List[str], namespace: Optional[str] = None) -> List[Dict]:
        """
        Get vectors by IDs with caching.
        
        Args:
            vector_ids: List of vector IDs to retrieve
            namespace: Optional namespace override
            
        Returns:
            List of vector dictionaries
        """
        ns = namespace or self.chunks_ns
        results = []
        missing_ids = []
        
        # Check cache first
        for vid in vector_ids:
            cache_key = f"{ns}:{vid}"
            if cache_key in self.vector_cache:
                self.cache_hits += 1
                results.append(self.vector_cache[cache_key])
            else:
                self.cache_misses += 1
                missing_ids.append(vid)
        
        # Fetch missing vectors
        if missing_ids:
            start_time = time.time()
            fetched = self.store_manager.get_vectors_by_ids(missing_ids, ns)
            query_time = time.time() - start_time
            self.query_times.append(query_time)
            for vector in fetched:
                if vector:
                    cache_key = f"{ns}:{vector['id']}"
                    self.vector_cache[cache_key] = vector
                    results.append(vector)
        
        return results

    def query_vectors(self, query: Dict, namespace: str, vector: Optional[List[float]] = None,
                     top_k: int = 10, include_metadata: bool = True) -> List[Dict]:
        """
        Query vectors with caching for identical queries.
        
        Args:
            query: Query filter
            namespace: Namespace to query
            vector: Optional query vector
            top_k: Number of results to return
            include_metadata: Whether to include metadata
            
        Returns:
            List of matching vectors
        """
        # Generate cache key
        cache_key = f"{namespace}:{json.dumps(query)}:{json.dumps(vector) if vector else ''}:{top_k}"
        
        # Check cache
        if cache_key in self.query_cache:
            self.cache_hits += 1
            return self.query_cache[cache_key]
        
        # Perform query
        start_time = time.time()
        results = self.store_manager.query_vectors(query, namespace, vector, top_k)
        query_time = time.time() - start_time
        self.query_times.append(query_time)
        
        # Cache results and vectors
        self.query_cache[cache_key] = results
        for vector in results:
            if vector:
                vcache_key = f"{namespace}:{vector['id']}"
                self.vector_cache[vcache_key] = vector
        
        return results

    def upsert_vectors(self, vectors: List[Dict], namespace: str, version: Optional[str] = None) -> bool:
        """
        Upsert vectors with version tracking.
        
        Args:
            vectors: List of vectors to upsert
            namespace: Target namespace
            version: Optional version identifier
            
        Returns:
            True if successful
        """
        try:
            # Generate version if not provided
            if not version:
                version = f"v_{int(time.time())}"
            
            # Track vector versions
            version_dict = self.versions['chunks'] if namespace == self.chunks_ns else self.versions['summaries']
            for vector in vectors:
                if vector and 'id' in vector:
                    version_dict[vector['id']] = {
                        'version': version,
                        'timestamp': int(time.time())
                    }
            
            # Perform upsert
            success = self.store_manager.upsert_batch(vectors, namespace)
            
            if success:
                # Update cache
                for vector in vectors:
                    if vector:
                        cache_key = f"{namespace}:{vector['id']}"
                        self.vector_cache[cache_key] = vector
                
                # Save version information
                self._save_versions()
                
                # Trigger cleanup if needed
                self._check_cleanup_needed()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {str(e)}")
            return False

    def delete_vectors(self, vector_ids: List[str], namespace: str) -> bool:
        """
        Delete vectors and update caches.
        
        Args:
            vector_ids: List of vector IDs to delete
            namespace: Namespace containing the vectors
            
        Returns:
            True if successful
        """
        try:
            # Remove from cache
            for vid in vector_ids:
                cache_key = f"{namespace}:{vid}"
                self.vector_cache.pop(cache_key, None)
            
            # Remove from versions
            version_dict = self.versions['chunks'] if namespace == self.chunks_ns else self.versions['summaries']
            for vid in vector_ids:
                version_dict.pop(vid, None)
            
            # Delete from store
            success = self.store_manager.delete_vectors(vector_ids, namespace)
            
            if success:
                # Save version information
                self._save_versions()
                
                # Clear query cache as results might be affected
                self.query_cache.clear()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete vectors: {str(e)}")
            return False

    def _check_cleanup_needed(self):
        """Check if vector cleanup is needed based on last cleanup time."""
        try:
            last_cleanup = self.versions.get('last_cleanup')
            if not last_cleanup or (int(time.time()) - last_cleanup) > 86400:  # 24 hours
                self._cleanup_old_vectors()
        except Exception as e:
            logger.error(f"Failed to check cleanup status: {str(e)}")

    def _cleanup_old_vectors(self):
        """Clean up old vector versions."""
        try:
            cutoff_time = int(time.time()) - (7 * 86400)  # 7 days
            
            # Find old vectors
            old_chunks = []
            old_summaries = []
            
            for vid, info in self.versions['chunks'].items():
                if info['timestamp'] < cutoff_time:
                    old_chunks.append(vid)
            
            for vid, info in self.versions['summaries'].items():
                if info['timestamp'] < cutoff_time:
                    old_summaries.append(vid)
            
            # Delete old vectors
            if old_chunks:
                self.delete_vectors(old_chunks, self.chunks_ns)
                logger.info(f"Cleaned up {len(old_chunks)} old chunk vectors")
            
            if old_summaries:
                self.delete_vectors(old_summaries, self.summaries_ns)
                logger.info(f"Cleaned up {len(old_summaries)} old summary vectors")
            
            # Update last cleanup time
            self.versions['last_cleanup'] = int(time.time())
            self._save_versions()
            
        except Exception as e:
            logger.error(f"Failed to clean up old vectors: {str(e)}")

    def get_stats(self) -> Dict:
        """Get vector operation statistics."""
        try:
            # Get total vectors
            total_chunks = len(self.vector_cache)
            total_summaries = len(self.metadata_cache)
            
            # Get vectors by age
            now = datetime.now()
            vectors_24h = 0
            vectors_7d = 0
            vectors_30d = 0
            
            for version_dict in [self.versions['chunks'], self.versions['summaries']]:
                for info in version_dict.values():
                    vector_age = now - datetime.fromtimestamp(info['timestamp'])
                    if vector_age <= timedelta(hours=24):
                        vectors_24h += 1
                    if vector_age <= timedelta(days=7):
                        vectors_7d += 1
                    if vector_age <= timedelta(days=30):
                        vectors_30d += 1
            
            # Calculate average query time
            avg_query_time = (
                sum(self.query_times) / len(self.query_times)
                if self.query_times else 0.0
            )
            
            return {
                'total_chunks': total_chunks,
                'total_summaries': total_summaries,
                'vectors_by_age': {
                    '24h': vectors_24h,
                    '7d': vectors_7d,
                    '30d': vectors_30d
                },
                'cache_stats': {
                    'hits': self.cache_hits,
                    'misses': self.cache_misses,
                    'hit_ratio': (
                        self.cache_hits / (self.cache_hits + self.cache_misses)
                        if (self.cache_hits + self.cache_misses) > 0 else 0.0
                    )
                },
                'avg_query_time': avg_query_time
            }
            
        except Exception as e:
            logger.error(f"Failed to get vector stats: {str(e)}")
            return {}

    def get_vector_stats(self) -> Dict:
        """Get statistics about vector usage and versions."""
        try:
            current_time = int(time.time())
            stats = {
                'total_chunks': len(self.versions['chunks']),
                'total_summaries': len(self.versions['summaries']),
                'vectors_by_age': {
                    '24h': 0,
                    '7d': 0,
                    '30d': 0,
                    'older': 0
                },
                'cache_stats': {
                    'vector_cache_size': len(self.vector_cache),
                    'metadata_cache_size': len(self.metadata_cache),
                    'query_cache_size': len(self.query_cache)
                }
            }
            
            # Calculate age distribution
            for version_dict in [self.versions['chunks'], self.versions['summaries']]:
                for info in version_dict.values():
                    age = current_time - info['timestamp']
                    if age < 86400:  # 24 hours
                        stats['vectors_by_age']['24h'] += 1
                    elif age < 7 * 86400:  # 7 days
                        stats['vectors_by_age']['7d'] += 1
                    elif age < 30 * 86400:  # 30 days
                        stats['vectors_by_age']['30d'] += 1
                    else:
                        stats['vectors_by_age']['older'] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get vector stats: {str(e)}")
            return {}
