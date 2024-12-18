"""Load testing for monitoring system."""

import unittest
import tempfile
from pathlib import Path
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from src.monitoring.system_monitor import SystemMonitor

class TestMonitoringLoad(unittest.TestCase):
    """Load testing suite for monitoring system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir)
        self.monitor = SystemMonitor(self.base_dir)
    
    def generate_random_metrics(self):
        """Generate random metrics for testing."""
        return {
            'processing': {
                'total_documents': random.randint(1000, 10000),
                'processed_documents': random.randint(900, 9900),
                'failed_documents': random.randint(0, 100),
                'total_chunks': random.randint(5000, 50000),
                'avg_chunks_per_doc': random.uniform(3.0, 7.0),
                'processing_time': random.uniform(100.0, 1000.0),
                'avg_time_per_doc': random.uniform(0.1, 1.0)
            },
            'vector': {
                'total_vectors': random.randint(10000, 100000),
                'vectors_24h': random.randint(1000, 10000),
                'vectors_7d': random.randint(5000, 50000),
                'vectors_30d': random.randint(8000, 80000),
                'cache_hits': random.randint(8000, 90000),
                'cache_misses': random.randint(1000, 10000),
                'cache_hit_ratio': random.uniform(0.7, 0.95),
                'avg_query_time': random.uniform(0.01, 0.1)
            }
        }
    
    def test_concurrent_updates(self):
        """Test concurrent metric updates."""
        num_threads = 10
        updates_per_thread = 100
        
        def update_metrics():
            for _ in range(updates_per_thread):
                metrics = self.generate_random_metrics()
                self.monitor.update_processing_metrics(metrics['processing'])
                self.monitor.update_vector_metrics(metrics['vector'])
                self.monitor.update_system_metrics()
                time.sleep(0.001)  # Smaller delay to allow for unique timestamps
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(update_metrics) for _ in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Will raise any exceptions that occurred
        
        end_time = time.time()
        total_updates = num_threads * updates_per_thread
        time_taken = end_time - start_time
        updates_per_second = total_updates / time_taken
        
        print(f"\nLoad Test Results:")
        print(f"Total Updates: {total_updates}")
        print(f"Time Taken: {time_taken:.2f} seconds")
        print(f"Updates per Second: {updates_per_second:.2f}")
        
        # Verify data integrity
        history = self.monitor.metrics_store.get_metrics_history('processing')
        self.assertGreaterEqual(len(history), total_updates * 0.8)  # Allow for some overlap
    
    def test_dashboard_generation_performance(self):
        """Test dashboard generation with large dataset."""
        # Generate large dataset
        num_data_points = 1000
        start_time = datetime.now() - timedelta(hours=24)
        
        print("\nGenerating test data...")
        for i in range(num_data_points):
            metrics = self.generate_random_metrics()
            self.monitor.update_processing_metrics(metrics['processing'])
            self.monitor.update_vector_metrics(metrics['vector'])
            self.monitor.update_system_metrics()
        
        print("Generating dashboard...")
        dashboard_start = time.time()
        dashboard_path = self.base_dir / "load_test_dashboard.html"
        self.monitor.generate_dashboard(dashboard_path)
        dashboard_time = time.time() - dashboard_start
        
        print(f"Dashboard Generation Time: {dashboard_time:.2f} seconds")
        
        # Check dashboard size
        dashboard_size = dashboard_path.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"Dashboard Size: {dashboard_size:.2f} MB")
        
        # Performance assertions
        self.assertLess(dashboard_time, 10.0)  # Should generate within 10 seconds
        self.assertLess(dashboard_size, 10.0)  # Should be less than 10MB
    
    def test_metric_aggregation_performance(self):
        """Test metric aggregation performance with large dataset."""
        # Generate large dataset
        num_data_points = 1000
        start_time = datetime.now() - timedelta(days=7)
        
        print("\nGenerating test data for aggregation...")
        for i in range(num_data_points):
            metrics = self.generate_random_metrics()
            self.monitor.update_processing_metrics(metrics['processing'])
            time.sleep(0.01)  # Small delay to ensure unique timestamps
        
        print("Testing aggregation performance...")
        
        # Test different aggregation intervals
        intervals = ['hour', 'day', 'week', 'month']
        for interval in intervals:
            agg_start = time.time()
            aggregated = self.monitor.metrics_store.get_aggregated_metrics(
                'processing',
                interval,
                start_time=start_time
            )
            agg_time = time.time() - agg_start
            
            print(f"{interval.capitalize()} Aggregation Time: {agg_time:.2f} seconds")
            print(f"Number of {interval} groups: {len(aggregated)}")
            
            # Performance assertions
            self.assertLess(agg_time, 5.0)  # Should aggregate within 5 seconds
    
    def test_cleanup_performance(self):
        """Test cleanup performance with large dataset."""
        num_old_points = 1000
        num_recent_points = 1000
        retention_days = 7
        
        print("\nGenerating old test data...")
        old_start = datetime.now() - timedelta(days=retention_days + 1)
        for i in range(num_old_points):
            metrics = self.generate_random_metrics()
            self.monitor.metrics_store._store_with_timestamp(
                'processing',
                metrics['processing'],
                int(old_start.timestamp() * 1000000 + i)  # Use microsecond precision
            )
        
        print("Generating recent test data...")
        recent_start = datetime.now() - timedelta(days=1)
        for i in range(num_recent_points):
            metrics = self.generate_random_metrics()
            self.monitor.metrics_store._store_with_timestamp(
                'processing',
                metrics['processing'],
                int(recent_start.timestamp() * 1000000 + i)  # Use microsecond precision
            )
        
        print("Testing cleanup performance...")
        start_time = time.time()
        self.monitor.metrics_store.cleanup_old_metrics(retention_days)
        cleanup_time = time.time() - start_time
        print(f"Cleanup Time: {cleanup_time:.2f} seconds")
        
        # Verify cleanup
        history = self.monitor.metrics_store.get_metrics_history('processing')
        self.assertLess(len(history), num_old_points + num_recent_points)
        self.assertGreaterEqual(len(history), num_recent_points * 0.9)  # Allow for some overlap
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir)
