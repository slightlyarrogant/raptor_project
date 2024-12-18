"""Persistent storage for system metrics using SQLite."""

import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import json

logger = logging.getLogger(__name__)

class MetricsStore:
    """Manages persistent storage of system metrics."""
    
    def __init__(self, base_dir: Path):
        """Initialize metrics store."""
        self.base_dir = base_dir
        self.db_dir = base_dir / "data" / "metrics" / "db"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir / "metrics.db"
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        try:
            with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
                cursor = conn.cursor()
                
                # Processing metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS processing_metrics (
                        timestamp INTEGER NOT NULL,
                        total_documents INTEGER NOT NULL,
                        processed_documents INTEGER NOT NULL,
                        failed_documents INTEGER NOT NULL,
                        total_chunks INTEGER NOT NULL,
                        avg_chunks_per_doc REAL NOT NULL,
                        processing_time REAL NOT NULL,
                        avg_time_per_doc REAL NOT NULL,
                        PRIMARY KEY (timestamp)
                    )
                """)
                
                # Vector metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS vector_metrics (
                        timestamp INTEGER NOT NULL,
                        total_vectors INTEGER NOT NULL,
                        vectors_24h INTEGER NOT NULL,
                        vectors_7d INTEGER NOT NULL,
                        vectors_30d INTEGER NOT NULL,
                        cache_hits INTEGER NOT NULL,
                        cache_misses INTEGER NOT NULL,
                        cache_hit_ratio REAL NOT NULL,
                        avg_query_time REAL NOT NULL,
                        PRIMARY KEY (timestamp)
                    )
                """)
                
                # System metrics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        timestamp INTEGER NOT NULL,
                        cpu_percent REAL NOT NULL,
                        memory_percent REAL NOT NULL,
                        disk_usage_percent REAL NOT NULL,
                        network_bytes_sent INTEGER NOT NULL,
                        network_bytes_recv INTEGER NOT NULL,
                        PRIMARY KEY (timestamp)
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def store_processing_metrics(self, metrics: Dict):
        """Store processing metrics."""
        try:
            timestamp = int(datetime.now().timestamp() * 1000000)  # Use microsecond precision
            
            with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO processing_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    metrics['total_documents'],
                    metrics['processed_documents'],
                    metrics['failed_documents'],
                    metrics['total_chunks'],
                    metrics['avg_chunks_per_doc'],
                    metrics['processing_time'],
                    metrics['avg_time_per_doc']
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store processing metrics: {str(e)}")
    
    def store_vector_metrics(self, metrics: Dict):
        """Store vector metrics."""
        try:
            timestamp = int(datetime.now().timestamp() * 1000000)  # Use microsecond precision
            
            with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO vector_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    metrics['total_vectors'],
                    metrics['vectors_24h'],
                    metrics['vectors_7d'],
                    metrics['vectors_30d'],
                    metrics['cache_hits'],
                    metrics['cache_misses'],
                    metrics['cache_hit_ratio'],
                    metrics['avg_query_time']
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store vector metrics: {str(e)}")
    
    def store_system_metrics(self, metrics: Dict):
        """Store system metrics."""
        try:
            timestamp = int(datetime.now().timestamp() * 1000000)  # Use microsecond precision
            
            with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO system_metrics VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    metrics['cpu_percent'],
                    metrics['memory_percent'],
                    metrics['disk_usage_percent'],
                    metrics['network_bytes_sent'],
                    metrics['network_bytes_recv']
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store system metrics: {str(e)}")
    
    def _store_with_timestamp(self, metric_type: str, metrics: Dict, timestamp: int):
        """Store metrics with a specific timestamp (for testing)."""
        try:
            table_name = f"{metric_type}_metrics"
            
            # Build query based on metric type
            if metric_type == 'processing':
                fields = ['timestamp', 'total_documents', 'processed_documents', 'failed_documents',
                         'total_chunks', 'avg_chunks_per_doc', 'processing_time', 'avg_time_per_doc']
                values = [timestamp] + [metrics[k] for k in fields[1:]]
            elif metric_type == 'vector':
                fields = ['timestamp', 'total_vectors', 'vectors_24h', 'vectors_7d', 'vectors_30d',
                         'cache_hits', 'cache_misses', 'cache_hit_ratio', 'avg_query_time']
                values = [timestamp] + [metrics[k] for k in fields[1:]]
            elif metric_type == 'system':
                fields = ['timestamp', 'cpu_percent', 'memory_percent', 'disk_usage_percent',
                         'network_bytes_sent', 'network_bytes_recv']
                values = [timestamp] + [metrics[k] for k in fields[1:]]
            else:
                raise ValueError(f"Unsupported metric type: {metric_type}")
            
            placeholders = ','.join(['?' for _ in values])
            
            with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                    INSERT INTO {table_name} VALUES ({placeholders})
                """, values)
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store {metric_type} metrics with timestamp: {str(e)}")
    
    def get_metrics_history(
        self,
        metric_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get historical metrics data."""
        try:
            table_name = f"{metric_type}_metrics"
            query = f"SELECT * FROM {table_name}"
            params = []
            
            # Add time range conditions if specified
            if start_time or end_time:
                conditions = []
                if start_time:
                    conditions.append("timestamp >= ?")
                    params.append(int(start_time.timestamp() * 1000000))  # Use microsecond precision
                if end_time:
                    conditions.append("timestamp <= ?")
                    params.append(int(end_time.timestamp() * 1000000))  # Use microsecond precision
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            # Add ordering and limit
            query += " ORDER BY timestamp DESC"
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert microsecond timestamps to seconds for display
                result = []
                for row in rows:
                    row_dict = dict(row)
                    row_dict['timestamp'] = row_dict['timestamp'] // 1000000  # Convert back to seconds
                    result.append(row_dict)
                return result
                
        except Exception as e:
            logger.error(f"Failed to get metrics history: {str(e)}")
            return []
    
    def get_aggregated_metrics(
        self,
        metric_type: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """Get aggregated metrics over specified interval."""
        try:
            # Define time grouping based on interval
            if interval == 'hour':
                time_group = "strftime('%Y-%m-%d %H:00:00', datetime(timestamp / 1000000, 'unixepoch'))"  # Use microsecond precision
            elif interval == 'day':
                time_group = "strftime('%Y-%m-%d', datetime(timestamp / 1000000, 'unixepoch'))"  # Use microsecond precision
            elif interval == 'week':
                time_group = "strftime('%Y-%W', datetime(timestamp / 1000000, 'unixepoch'))"  # Use microsecond precision
            elif interval == 'month':
                time_group = "strftime('%Y-%m', datetime(timestamp / 1000000, 'unixepoch'))"  # Use microsecond precision
            else:
                raise ValueError(f"Unsupported interval: {interval}")
            
            table_name = f"{metric_type}_metrics"
            
            # Get column names for the table
            with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [row[1] for row in cursor.fetchall()]
            
            # Build aggregation query
            agg_columns = []
            for col in columns:
                if col == 'timestamp':
                    agg_columns.append(f"{time_group} as time_bucket")
                else:
                    agg_columns.append(f"avg({col}) as {col}")
            
            query = f"""
                SELECT {', '.join(agg_columns)}
                FROM {table_name}
            """
            
            params = []
            
            # Add time range conditions if specified
            if start_time or end_time:
                conditions = []
                if start_time:
                    conditions.append("timestamp >= ?")
                    params.append(int(start_time.timestamp() * 1000000))  # Use microsecond precision
                if end_time:
                    conditions.append("timestamp <= ?")
                    params.append(int(end_time.timestamp() * 1000000))  # Use microsecond precision
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            query += f" GROUP BY {time_group} ORDER BY time_bucket DESC"
            
            with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get aggregated metrics: {str(e)}")
            return []
    
    def cleanup_old_metrics(self, retention_days: int = 30):
        """Clean up metrics older than specified retention period."""
        try:
            cutoff_timestamp = int((datetime.now() - timedelta(days=retention_days)).timestamp() * 1000000)  # Use microsecond precision
            
            with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
                cursor = conn.cursor()
                
                # Clean up each metrics table
                for table in ['processing_metrics', 'vector_metrics', 'system_metrics']:
                    cursor.execute(f"""
                        DELETE FROM {table}
                        WHERE timestamp < ?
                    """, (cutoff_timestamp,))
                
                conn.commit()
                
            logger.info(f"Cleaned up metrics older than {retention_days} days")
            
        except Exception as e:
            logger.error(f"Failed to clean up old metrics: {str(e)}")
    
    def get_latest_metrics(self, metric_type: str) -> Optional[Dict]:
        """Get the most recent metrics of specified type."""
        try:
            table_name = f"{metric_type}_metrics"
            
            with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(f"""
                    SELECT * FROM {table_name}
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                
                # Convert microsecond timestamps to seconds for display
                if row:
                    row_dict = dict(row)
                    row_dict['timestamp'] = row_dict['timestamp'] // 1000000  # Convert back to seconds
                    return row_dict
                else:
                    return None
                
        except Exception as e:
            logger.error(f"Failed to get latest metrics: {str(e)}")
            return None
