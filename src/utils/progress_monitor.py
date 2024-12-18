import time
import logging
import functools
from typing import Any, Callable, Optional
from threading import Thread, Event
import signal
from datetime import datetime

logger = logging.getLogger(__name__)

class ProgressTimeout(Exception):
    """Raised when operation exceeds timeout without progress."""
    pass

class ProgressMonitor:
    def __init__(self, 
                 timeout: int = 300,  # 5 minutes default
                 progress_interval: int = 30,  # Check progress every 30 seconds
                 retry_count: int = 3,
                 retry_delay: int = 60):  # Wait 60 seconds between retries
        self.timeout = timeout
        self.progress_interval = progress_interval
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.stop_event = Event()
        self.last_progress = time.time()
        self.operation_name = None
        
    def __call__(self, operation_name: str = None):
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.operation_name = operation_name or func.__name__
                
                for attempt in range(self.retry_count):
                    try:
                        # Start progress monitoring thread
                        monitor_thread = Thread(target=self._monitor_progress)
                        monitor_thread.daemon = True
                        monitor_thread.start()
                        
                        # Reset progress timestamp
                        self.last_progress = time.time()
                        
                        # Execute operation
                        result = func(*args, **kwargs)
                        
                        # Stop monitoring
                        self.stop_event.set()
                        monitor_thread.join()
                        
                        return result
                        
                    except ProgressTimeout as e:
                        logger.warning(f"Operation {self.operation_name} timed out (attempt {attempt + 1}/{self.retry_count})")
                        if attempt < self.retry_count - 1:
                            logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                            time.sleep(self.retry_delay)
                            continue
                        raise
                        
                    except Exception as e:
                        logger.error(f"Operation {self.operation_name} failed: {str(e)}")
                        raise
                        
            return wrapper
        return decorator
        
    def _monitor_progress(self):
        """Monitor progress and raise timeout if stalled."""
        while not self.stop_event.is_set():
            time.sleep(self.progress_interval)
            
            time_since_progress = time.time() - self.last_progress
            if time_since_progress > self.timeout:
                logger.error(f"Operation {self.operation_name} stalled for {time_since_progress:.1f} seconds")
                raise ProgressTimeout(f"Operation {self.operation_name} exceeded timeout of {self.timeout} seconds without progress")
                
    def update_progress(self):
        """Update last progress timestamp."""
        self.last_progress = time.time()
        
progress_monitor = ProgressMonitor()  # Global instance for convenience 