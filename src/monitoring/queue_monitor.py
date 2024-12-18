"""Real-time monitoring for document processing queue."""

import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.processing.queue_manager import DocumentQueueManager
from src.utils.config import load_config

logger = logging.getLogger(__name__)

class QueueMonitor:
    """Monitor for document processing queue metrics."""
    
    def __init__(self, queue_manager: DocumentQueueManager):
        """Initialize monitor with queue manager."""
        self.queue_manager = queue_manager
        self.metrics_history = {
            'queue_size': deque(maxlen=100),
            'processing_rate': deque(maxlen=100),
            'error_rate': deque(maxlen=100),
            'timestamps': deque(maxlen=100)
        }
    
    def collect_metrics(self) -> Dict:
        """Collect current queue metrics."""
        metrics = {
            'queue_size': self.queue_manager.queue_size(),
            'failed_queue_size': self.queue_manager.failed_queue_size(),
            'processing_rate': self.queue_manager.get_processing_rate(),
            'error_rate': self.queue_manager.get_error_rate(),
            'timestamp': datetime.now()
        }
        
        # Update history
        self.metrics_history['queue_size'].append(metrics['queue_size'])
        self.metrics_history['processing_rate'].append(metrics['processing_rate'])
        self.metrics_history['error_rate'].append(metrics['error_rate'])
        self.metrics_history['timestamps'].append(metrics['timestamp'])
        
        return metrics

def create_metrics_plot(monitor: QueueMonitor):
    """Create plotly figure for metrics visualization."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Queue Size', 'Processing Rate', 'Error Rate')
    )
    
    timestamps = list(monitor.metrics_history['timestamps'])
    
    # Queue size plot
    fig.add_trace(
        go.Scatter(x=timestamps, y=list(monitor.metrics_history['queue_size']),
                  name='Queue Size'),
        row=1, col=1
    )
    
    # Processing rate plot
    fig.add_trace(
        go.Scatter(x=timestamps, y=list(monitor.metrics_history['processing_rate']),
                  name='Docs/Min'),
        row=2, col=1
    )
    
    # Error rate plot
    fig.add_trace(
        go.Scatter(x=timestamps, y=list(monitor.metrics_history['error_rate']),
                  name='Errors/Min'),
        row=3, col=1
    )
    
    fig.update_layout(height=900, showlegend=True)
    return fig

def run_dashboard():
    """Run Streamlit dashboard."""
    st.title('Document Processing Monitor')
    
    # Initialize queue manager and monitor
    config = load_config()
    queue_manager = DocumentQueueManager(config)
    monitor = QueueMonitor(queue_manager)
    
    # Display current metrics
    st.header('Current Status')
    metrics = monitor.collect_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('Queue Size', metrics['queue_size'])
    with col2:
        st.metric('Failed Queue', metrics['failed_queue_size'])
    with col3:
        st.metric('Processing Rate', f"{metrics['processing_rate']:.1f}/min")
    with col4:
        st.metric('Error Rate', f"{metrics['error_rate']:.2f}%")
    
    # Display metrics plot
    st.header('Historical Metrics')
    fig = create_metrics_plot(monitor)
    st.plotly_chart(fig, use_container_width=True)
    
    # Failed documents table
    st.header('Failed Documents')
    failed_docs = queue_manager.list_failed_documents()
    if failed_docs:
        st.table(failed_docs)
    else:
        st.info('No failed documents')
    
    # Auto-refresh
    time.sleep(5)
    st.experimental_rerun()

if __name__ == '__main__':
    run_dashboard()
