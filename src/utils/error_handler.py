import logging
import logging.handlers
from typing import Optional, Dict, Any, Callable, TypeVar, List
from functools import wraps
import traceback
import time
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic function return type
T = TypeVar('T')

class RaptorError(Exception):
    """Base exception for all RAPTOR errors."""
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code or 'RAPTOR_ERROR'
        self.details = details or {}
        self.timestamp = time.time()
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format for logging."""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
            'type': self.__class__.__name__
        }

class ConfigurationError(RaptorError):
    """Raised when there's a configuration-related error."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, 'CONFIG_ERROR', details)

class ProcessingError(RaptorError):
    """Raised when there's an error during document processing."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, 'PROCESSING_ERROR', details)

class StorageError(RaptorError):
    """Raised when there's an error with storage operations."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, 'STORAGE_ERROR', details)

class TreeError(RaptorError):
    """Raised when there's an error with tree operations."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, 'TREE_ERROR', details)

class APIError(RaptorError):
    """Raised when there's an error with external API calls."""
    def __init__(self, message: str, service: str, details: Dict[str, Any] = None):
        details = details or {}
        details['service'] = service
        super().__init__(message, 'API_ERROR', details)

class ValidationError(RaptorError):
    """Raised when there's a validation error."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, 'VALIDATION_ERROR', details)

class RetryableError(RaptorError):
    """Base class for errors that can be retried."""
    def __init__(self, message: str, error_code: str, details: Dict[str, Any] = None):
        details = details or {}
        details['is_retryable'] = True
        super().__init__(message, error_code, details)

class RetryableAPIError(RetryableError, APIError):
    """Retryable API error."""
    def __init__(self, message: str, service: str, details: Dict[str, Any] = None):
        RetryableError.__init__(self, message, 'RETRYABLE_API_ERROR', details)
        self.details['service'] = service

def with_retries(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (RetryableError,)
) -> Callable:
    """Decorator for implementing retry logic."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            attempt = 0
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    attempt += 1
                    
                    if attempt < max_attempts:
                        wait_time = delay * (backoff_factor ** (attempt - 1))
                        logger.warning(
                            f"Retry attempt {attempt}/{max_attempts} for {func.__name__} "
                            f"after {wait_time:.2f}s due to: {str(e)}"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"Max retry attempts ({max_attempts}) reached for {func.__name__}"
                        )
            
            raise last_exception
        return wrapper
    return decorator

class ErrorTracker:
    """Tracks errors and maintains error statistics."""
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.error_stats_file = log_dir / 'error_stats.json'
        self.failed_items_file = log_dir / 'failed_items.json'
        self._load_stats()

    def _load_stats(self):
        """Load error statistics from file."""
        try:
            if self.error_stats_file.exists():
                with open(self.error_stats_file, 'r') as f:
                    self.stats = json.load(f)
            else:
                self.stats = {'error_counts': {}, 'last_update': None}
        except Exception as e:
            logger.error(f"Failed to load error stats: {e}")
            self.stats = {'error_counts': {}, 'last_update': None}

    def track_error(self, error: RaptorError, item_id: str = None):
        """Track an error occurrence."""
        error_type = error.__class__.__name__
        self.stats['error_counts'][error_type] = self.stats['error_counts'].get(error_type, 0) + 1
        self.stats['last_update'] = datetime.now().isoformat()

        if item_id:
            self._track_failed_item(item_id, error)

        self._save_stats()

    def _track_failed_item(self, item_id: str, error: RaptorError):
        """Track failed items for retry."""
        try:
            if self.failed_items_file.exists():
                with open(self.failed_items_file, 'r') as f:
                    failed_items = json.load(f)
            else:
                failed_items = {}

            failed_items[item_id] = {
                'error': error.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'retries': failed_items.get(item_id, {}).get('retries', 0) + 1
            }

            with open(self.failed_items_file, 'w') as f:
                json.dump(failed_items, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to track failed item: {e}")

    def _save_stats(self):
        """Save error statistics to file."""
        try:
            with open(self.error_stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save error stats: {e}")

def setup_error_logging(log_dir: Path = None):
    """Setup error logging configuration."""
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        error_log_path = log_dir / 'error.log'
        
        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            str(error_log_path),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.ERROR)
        
        # Create detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
            'Additional Context: %(error_info)s\n'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
    # Ensure we're capturing errors at the right level
    logger.setLevel(logging.ERROR)

# Initialize error tracker
error_tracker = None

def initialize_error_tracking(log_dir: Path):
    """Initialize the error tracker."""
    global error_tracker
    error_tracker = ErrorTracker(log_dir)
    setup_error_logging(log_dir)

def handle_error(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Standardized error handling function."""
    error_info = {
        'error_type': type(error).__name__,
        'message': str(error),
        'timestamp': time.time(),
        'traceback': traceback.format_exc(),
        'context': context or {}
    }
    
    if isinstance(error, RaptorError):
        error_info.update({
            'error_code': error.error_code,
            'details': error.details
        })
    
    # Log the error
    logger.error(
        f"Error occurred: {error_info['error_type']} - {error_info['message']}",
        extra={'error_info': error_info}
    )
    
    # Track error if tracker is initialized
    if error_tracker and isinstance(error, RaptorError):
        error_tracker.track_error(error, context.get('item_id') if context else None)
    
    return error_info

def error_handler(func):
    """Decorator for standardized error handling."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            context = {
                'function': func.__name__,
                'args': str(args),
                'kwargs': str(kwargs)
            }
            error_info = handle_error(e, context)
            
            # Re-raise as RaptorError if it's not already one
            if not isinstance(e, RaptorError):
                raise RaptorError(
                    f"Error in {func.__name__}: {str(e)}",
                    error_code='INTERNAL_ERROR',
                    details=error_info
                ) from e
            raise
    return wrapper