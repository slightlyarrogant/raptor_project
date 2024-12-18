"""Tests for the SystemMonitor class."""

import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import json
import psutil

from src.monitoring.system_monitor import SystemMonitor

class TestSystemMonitor(unittest.TestCase):
    """Test suite for SystemMonitor."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir)
        self.monitor = SystemMonitor(self.base_dir)
        
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
    
    def test_update_processing_metrics(self):
        """Test updating processing metrics."""
        self.monitor.update_processing_metrics(self.processing_metrics)
        
        # Get latest metrics from store
        latest = self.monitor.metrics_store.get_latest_metrics('processing')
        
        # Check values
        self.assertEqual(latest['total_documents'], 100)
        self.assertEqual(latest['processed_documents'], 95)
        self.assertEqual(latest['failed_documents'], 5)
    
    def test_update_vector_metrics(self):
        """Test updating vector metrics."""
        self.monitor.update_vector_metrics(self.vector_metrics)
        
        # Get latest metrics from store
        latest = self.monitor.metrics_store.get_latest_metrics('vector')
        
        # Check values
        self.assertEqual(latest['total_vectors'], 1000)
        self.assertEqual(latest['cache_hits'], 800)
        self.assertEqual(latest['cache_hit_ratio'], 0.8)
    
    def test_update_system_metrics(self):
        """Test updating system metrics."""
        self.monitor.update_system_metrics()
        
        # Get latest metrics from store
        latest = self.monitor.metrics_store.get_latest_metrics('system')
        
        # Check that values are within reasonable ranges
        self.assertGreaterEqual(latest['cpu_percent'], 0)
        self.assertLessEqual(latest['cpu_percent'], 100)
        self.assertGreaterEqual(latest['memory_percent'], 0)
        self.assertLessEqual(latest['memory_percent'], 100)
        self.assertGreaterEqual(latest['disk_usage_percent'], 0)
        self.assertLessEqual(latest['disk_usage_percent'], 100)
    
    def test_generate_dashboard(self):
        """Test dashboard generation."""
        # Add some test data
        self.monitor.update_processing_metrics(self.processing_metrics)
        self.monitor.update_vector_metrics(self.vector_metrics)
        self.monitor.update_system_metrics()
        
        # Generate dashboard
        dashboard_path = self.base_dir / "test_dashboard.html"
        self.monitor.generate_dashboard(dashboard_path)
        
        # Check that dashboard file exists
        self.assertTrue(dashboard_path.exists())
        self.assertGreater(dashboard_path.stat().st_size, 0)
    
    def test_get_system_health(self):
        """Test system health status."""
        # Add test data
        self.monitor.update_processing_metrics(self.processing_metrics)
        self.monitor.update_vector_metrics(self.vector_metrics)
        self.monitor.update_system_metrics()
        
        # Get health status
        health = self.monitor.get_system_health()
        
        # Check structure and values
        self.assertIn('processing', health)
        self.assertIn('vector', health)
        self.assertIn('system', health)
        
        self.assertIn('status', health['processing'])
        self.assertIn('metrics', health['processing'])
        
        self.assertIn('status', health['vector'])
        self.assertIn('metrics', health['vector'])
        
        self.assertIn('status', health['system'])
        self.assertIn('metrics', health['system'])
    
    def test_network_io_calculation(self):
        """Test network I/O rate calculation."""
        # Update metrics multiple times
        self.monitor.update_system_metrics()
        initial = self.monitor.metrics_store.get_latest_metrics('system')
        
        # Wait a bit and update again
        import time
        time.sleep(1)
        self.monitor.update_system_metrics()
        updated = self.monitor.metrics_store.get_latest_metrics('system')
        
        # Check that rates are calculated
        self.assertIsInstance(updated['network_bytes_sent'], (int, float))
        self.assertIsInstance(updated['network_bytes_recv'], (int, float))
    
    def test_historical_data_visualization(self):
        """Test historical data visualization."""
        # Add multiple data points
        for i in range(5):
            metrics = self.processing_metrics.copy()
            metrics['total_documents'] += i
            self.monitor.update_processing_metrics(metrics)
            
            vector_metrics = self.vector_metrics.copy()
            vector_metrics['total_vectors'] += i * 100
            self.monitor.update_vector_metrics(vector_metrics)
            
            self.monitor.update_system_metrics()
        
        # Generate dashboard
        dashboard_path = self.base_dir / "historical_dashboard.html"
        self.monitor.generate_dashboard(dashboard_path)
        
        # Check dashboard file
        self.assertTrue(dashboard_path.exists())
        self.assertGreater(dashboard_path.stat().st_size, 0)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
