"""System monitoring and metrics collection."""

import logging
import time
import psutil
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import threading
from queue import Queue
import numpy as np
from dataclasses import dataclass
from src.utils.error_handler import error_handler, RaptorError

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System metrics data structure."""
    cpu_percent: float
    memory_percent: float
    disk_usage: Dict[str, float]
    processing_queue_size: int
    active_processes: int
    error_count: int
    timestamp: float

class MetricsCollector:
    """Collects system metrics."""
    
    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self.metrics_queue = Queue()
        self.stop_event = threading.Event()
        self.collector_thread = None
        
    @error_handler
    def start_collection(self):
        """Start metrics collection in a separate thread."""
        self.collector_thread = threading.Thread(target=self._collect_metrics)
        self.collector_thread.start()
        
    def stop_collection(self):
        """Stop metrics collection."""
        self.stop_event.set()
        if self.collector_thread:
            self.collector_thread.join()
            
    @error_handler
    def _collect_metrics(self):
        """Collect system metrics periodically."""
        while not self.stop_event.is_set():
            metrics = SystemMetrics(
                cpu_percent=psutil.cpu_percent(),
                memory_percent=psutil.virtual_memory().percent,
                disk_usage={
                    path.mountpoint: psutil.disk_usage(path.mountpoint).percent
                    for path in psutil.disk_partitions()
                },
                processing_queue_size=self._get_queue_size(),
                active_processes=len(psutil.Process().children()),
                error_count=self._get_error_count(),
                timestamp=time.time()
            )
            self.metrics_queue.put(metrics)
            time.sleep(self.collection_interval)
            
    def _get_queue_size(self) -> int:
        """Get current processing queue size."""
        # Implement queue size checking logic
        return 0
        
    def _get_error_count(self) -> int:
        """Get error count from logs."""
        # Implement error count checking logic
        return 0

class AlertManager:
    """Manages system alerts."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alert_thresholds = config.get('alert_thresholds', {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'error_rate': 0.1
        })
        
    @error_handler
    def check_alerts(self, metrics: SystemMetrics) -> List[str]:
        """Check metrics against thresholds and return alerts."""
        alerts = []
        
        # CPU usage alert
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append(f"High CPU usage: {metrics.cpu_percent}%")
            
        # Memory usage alert
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append(f"High memory usage: {metrics.memory_percent}%")
            
        # Disk usage alerts
        for mount, usage in metrics.disk_usage.items():
            if usage > self.alert_thresholds['disk_percent']:
                alerts.append(f"High disk usage on {mount}: {usage}%")
                
        return alerts

class SystemMonitor:
    """Main system monitoring class."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.metrics_collector = MetricsCollector(
            collection_interval=config.get('collection_interval', 60)
        )
        self.alert_manager = AlertManager(config)
        self.metrics_history: List[SystemMetrics] = []
        self.alert_history: List[Dict] = []
        
    @error_handler
    def start_monitoring(self):
        """Start system monitoring."""
        logger.info("Starting system monitoring...")
        self.metrics_collector.start_collection()
        
    def stop_monitoring(self):
        """Stop system monitoring."""
        logger.info("Stopping system monitoring...")
        self.metrics_collector.stop_collection()
        
    @error_handler
    def process_metrics(self):
        """Process collected metrics and generate alerts."""
        while not self.metrics_collector.metrics_queue.empty():
            metrics = self.metrics_collector.metrics_queue.get()
            self.metrics_history.append(metrics)
            
            # Check for alerts
            alerts = self.alert_manager.check_alerts(metrics)
            if alerts:
                alert_entry = {
                    'timestamp': metrics.timestamp,
                    'alerts': alerts
                }
                self.alert_history.append(alert_entry)
                for alert in alerts:
                    logger.warning(f"System Alert: {alert}")
                    
    @error_handler
    def generate_report(self, output_dir: Path) -> Path:
        """Generate monitoring report."""
        report = {
            'timestamp': time.time(),
            'metrics_summary': self._generate_metrics_summary(),
            'alerts_summary': self._generate_alerts_summary()
        }
        
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"monitoring_report_{int(time.time())}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report_path
        
    def _generate_metrics_summary(self) -> Dict:
        """Generate summary statistics for collected metrics."""
        if not self.metrics_history:
            return {}
            
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_percent for m in self.metrics_history]
        
        return {
            'cpu': {
                'mean': np.mean(cpu_values),
                'max': np.max(cpu_values),
                'min': np.min(cpu_values)
            },
            'memory': {
                'mean': np.mean(memory_values),
                'max': np.max(memory_values),
                'min': np.min(memory_values)
            },
            'total_errors': sum(m.error_count for m in self.metrics_history)
        }
        
    def _generate_alerts_summary(self) -> Dict:
        """Generate summary of alerts."""
        if not self.alert_history:
            return {}
            
        return {
            'total_alerts': len(self.alert_history),
            'recent_alerts': self.alert_history[-10:],  # Last 10 alerts
            'alert_types': self._count_alert_types()
        }
        
    def _count_alert_types(self) -> Dict[str, int]:
        """Count occurrences of each alert type."""
        alert_counts = {}
        for entry in self.alert_history:
            for alert in entry['alerts']:
                alert_type = alert.split(':')[0]
                alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
        return alert_counts
