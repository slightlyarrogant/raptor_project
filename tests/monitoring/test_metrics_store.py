"""Tests for the MetricsStore class."""

import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
import time

from src.monitoring.metrics_store import MetricsStore

class TestMetricsStore(unittest.TestCase):
    """Test suite for MetricsStore."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir)
        self.metrics_store = MetricsStore(self.base_dir)
        
        # Sample metrics for testing
        self.processing_metrics = {
            'total_documents': 100,
            'processed_documents': 95,
            'failed_documents': 5,
            'total_chunks': 500,
            'avg_chunks_per_doc': 5.0,
            'processing_time': 50.0,
            'avg_time_per_doc': 0.5
        }
        
        self.vector_metrics = {
            'total_vectors': 1000,
            'vectors_24h': 100,
            'vectors_7d': 500,
            'vectors_30d': 900,
            'cache_hits': 800,
            'cache_misses': 200,
            'cache_hit_ratio': 0.8,
            'avg_query_time': 0.05
        }
        
        self.system_metrics = {
            'cpu_percent': 45.5,
            'memory_percent': 60.0,
            'disk_usage_percent': 55.5,
            'network_bytes_sent': 1024,
            'network_bytes_recv': 2048
        }
    
    def test_store_processing_metrics(self):
        """Test storing and retrieving processing metrics."""
        # Store metrics
        self.metrics_store.store_processing_metrics(self.processing_metrics)
        
        # Retrieve latest metrics
        latest = self.metrics_store.get_latest_metrics('processing')
        
        # Check values
        self.assertEqual(latest['total_documents'], 100)
        self.assertEqual(latest['processed_documents'], 95)
        self.assertEqual(latest['failed_documents'], 5)
        self.assertEqual(latest['total_chunks'], 500)
        self.assertEqual(latest['avg_chunks_per_doc'], 5.0)
        self.assertEqual(latest['processing_time'], 50.0)
        self.assertEqual(latest['avg_time_per_doc'], 0.5)
    
    def test_store_vector_metrics(self):
        """Test storing and retrieving vector metrics."""
        # Store metrics
        self.metrics_store.store_vector_metrics(self.vector_metrics)
        
        # Retrieve latest metrics
        latest = self.metrics_store.get_latest_metrics('vector')
        
        # Check values
        self.assertEqual(latest['total_vectors'], 1000)
        self.assertEqual(latest['vectors_24h'], 100)
        self.assertEqual(latest['vectors_7d'], 500)
        self.assertEqual(latest['vectors_30d'], 900)
        self.assertEqual(latest['cache_hits'], 800)
        self.assertEqual(latest['cache_misses'], 200)
        self.assertEqual(latest['cache_hit_ratio'], 0.8)
        self.assertEqual(latest['avg_query_time'], 0.05)
    
    def test_store_system_metrics(self):
        """Test storing and retrieving system metrics."""
        # Store metrics
        self.metrics_store.store_system_metrics(self.system_metrics)
        
        # Retrieve latest metrics
        latest = self.metrics_store.get_latest_metrics('system')
        
        # Check values
        self.assertEqual(latest['cpu_percent'], 45.5)
        self.assertEqual(latest['memory_percent'], 60.0)
        self.assertEqual(latest['disk_usage_percent'], 55.5)
        self.assertEqual(latest['network_bytes_sent'], 1024)
        self.assertEqual(latest['network_bytes_recv'], 2048)
    
    def test_get_metrics_history(self):
        """Test retrieving historical metrics."""
        # Store multiple metrics with different timestamps
        for i in range(5):
            metrics = self.processing_metrics.copy()
            metrics['total_documents'] += i
            self.metrics_store.store_processing_metrics(metrics)
            time.sleep(0.001)  # Ensure unique timestamps
        
        # Get history with limit
        history = self.metrics_store.get_metrics_history('processing', limit=3)
        self.assertEqual(len(history), 3)
        
        # Check ordering (descending by timestamp)
        self.assertTrue(history[0]['total_documents'] > history[1]['total_documents'])
    
    def test_get_aggregated_metrics(self):
        """Test retrieving aggregated metrics."""
        # Store metrics for different times
        start_time = datetime.now() - timedelta(days=2)
        
        for i in range(48):  # 48 hours of data
            metrics = self.system_metrics.copy()
            metrics['cpu_percent'] += i
            
            # Manually insert with specific timestamp using microsecond precision
            timestamp = int((start_time + timedelta(hours=i)).timestamp() * 1000000)
            self.metrics_store._store_with_timestamp('system', metrics, timestamp)
            time.sleep(0.001)  # Ensure unique timestamps
        
        # Get hourly aggregation
        hourly = self.metrics_store.get_aggregated_metrics(
            'system',
            'hour',
            start_time=start_time
        )
        self.assertGreater(len(hourly), 0)
        
        # Get daily aggregation
        daily = self.metrics_store.get_aggregated_metrics(
            'system',
            'day',
            start_time=start_time
        )
        self.assertGreater(len(daily), 0)
    
    def test_cleanup_old_metrics(self):
        """Test cleaning up old metrics."""
        # Store old metrics
        old_time = datetime.now() - timedelta(days=40)
        with sqlite3.connect(str(self.metrics_store.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO processing_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                int(old_time.timestamp()),
                100, 95, 5, 500, 5.0, 50.0, 0.5
            ))
        
        # Store recent metrics
        self.metrics_store.store_processing_metrics(self.processing_metrics)
        
        # Clean up old metrics
        self.metrics_store.cleanup_old_metrics(retention_days=30)
        
        # Check that only recent metrics remain
        history = self.metrics_store.get_metrics_history('processing')
        for metric in history:
            self.assertGreater(
                metric['timestamp'],
                int((datetime.now() - timedelta(days=30)).timestamp())
            )
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
