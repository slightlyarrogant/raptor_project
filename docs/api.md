# RAPTOR API Documentation

## Core Components

### TreeManager

The `TreeManager` class is the main entry point for document processing and tree management.

```python
from src.tree.tree_manager import TreeManager

tree_manager = TreeManager(config: Dict[str, Any])
```

#### Methods

##### process_documents
```python
def process_documents(
    self,
    documents: List[str],
    metadata: List[Dict[str, Any]]
) -> TreeData:
    """
    Process a list of documents and build a hierarchical tree structure.
    
    Args:
        documents: List of document texts
        metadata: List of metadata dictionaries for each document
        
    Returns:
        TreeData object containing the hierarchical structure
        
    Raises:
        ProcessingError: If document processing fails
        ConfigurationError: If configuration is invalid
    """
```

##### get_node_subtree
```python
def get_node_subtree(
    self,
    node_id: str,
    max_depth: int = None
) -> Dict[str, Any]:
    """
    Get the subtree starting from a specific node.
    
    Args:
        node_id: ID of the root node for the subtree
        max_depth: Maximum depth to traverse (None for unlimited)
        
    Returns:
        Dictionary containing the subtree structure
    """
```

### PineconeManager

The `PineconeManager` class handles vector storage and retrieval operations.

```python
from src.storage.pinecone_manager import PineconeManager

pinecone_manager = PineconeManager(config: Dict[str, Any])
```

#### Methods

##### store_vectors
```python
def store_vectors(
    self,
    vectors: np.ndarray,
    metadata: List[Dict[str, Any]]
) -> None:
    """
    Store vectors and their metadata in Pinecone.
    
    Args:
        vectors: Array of vectors to store (shape: [n_vectors, dimension])
        metadata: List of metadata dictionaries for each vector
        
    Raises:
        StorageError: If storage operation fails
    """
```

##### query_vectors
```python
def query_vectors(
    self,
    query_vector: np.ndarray,
    top_k: int = 10,
    filter: Dict = None
) -> Dict[str, Any]:
    """
    Query similar vectors from storage.
    
    Args:
        query_vector: Vector to query (shape: [dimension])
        top_k: Number of results to return
        filter: Optional metadata filter
        
    Returns:
        Dictionary containing matches and their scores
    """
```

### SystemMonitor

The `SystemMonitor` class provides system monitoring and alerting functionality.

```python
from src.monitoring.system_monitor import SystemMonitor

monitor = SystemMonitor(config: Dict[str, Any])
```

#### Methods

##### start_monitoring
```python
def start_monitoring(self) -> None:
    """
    Start system monitoring in a background thread.
    """
```

##### process_metrics
```python
def process_metrics(self) -> None:
    """
    Process collected metrics and generate alerts.
    """
```

##### generate_report
```python
def generate_report(
    self,
    output_dir: Path
) -> Path:
    """
    Generate monitoring report.
    
    Args:
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report file
    """
```

### ConfigValidator

The `ConfigValidator` class handles configuration validation and management.

```python
from src.utils.config_validator import ConfigValidator

validator = ConfigValidator()
```

#### Methods

##### validate_config
```python
def validate_config(
    self,
    config: Dict[str, Any],
    section: str = None
) -> Dict[str, Any]:
    """
    Validate configuration against schema.
    
    Args:
        config: Configuration dictionary to validate
        section: Optional section name to validate
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ConfigurationError: If validation fails
    """
```

##### load_config
```python
def load_config(
    self,
    config_path: Path
) -> Dict[str, Any]:
    """
    Load and validate configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ConfigurationError: If file not found or invalid
    """
```

## Data Structures

### TreeData
```python
class TreeData:
    """
    Hierarchical tree structure containing document clusters.
    
    Attributes:
        root: Root node of the tree
        nodes: Dictionary of all nodes by ID
        metadata: Tree metadata
    """
```

### SystemMetrics
```python
@dataclass
class SystemMetrics:
    """
    System metrics data structure.
    
    Attributes:
        cpu_percent: CPU usage percentage
        memory_percent: Memory usage percentage
        disk_usage: Dictionary of disk usage by mount point
        processing_queue_size: Number of items in processing queue
        active_processes: Number of active processing threads
        error_count: Number of errors in current period
        timestamp: Unix timestamp of metrics collection
    """
```

### ConfigSchema
```python
@dataclass
class ConfigSchema:
    """
    Configuration schema definition.
    
    Attributes:
        name: Field name
        type: Expected type
        required: Whether field is required
        default: Default value if not provided
        constraints: Dictionary of validation constraints
        description: Field description
    """
```

## Error Handling

### Base Exception
```python
class RaptorError(Exception):
    """
    Base exception for all RAPTOR errors.
    
    Attributes:
        message: Error message
        error_code: Error code string
        details: Dictionary of error details
        timestamp: Unix timestamp of error
    """
```

### Specific Exceptions
```python
class ConfigurationError(RaptorError):
    """Configuration-related errors."""

class ProcessingError(RaptorError):
    """Document processing errors."""

class StorageError(RaptorError):
    """Storage operation errors."""

class TreeError(RaptorError):
    """Tree operation errors."""
```

### Error Handler Decorator
```python
@error_handler
def your_function():
    """
    Decorator that provides standardized error handling.
    
    Features:
    - Catches all exceptions
    - Logs errors with context
    - Converts to RaptorError if needed
    - Preserves stack traces
    """
```

## Configuration Schema

### Pinecone Schema
```python
pinecone_schema = {
    'api_key': ConfigSchema(
        name='api_key',
        type=str,
        required=True,
        description="Pinecone API key"
    ),
    'dimension': ConfigSchema(
        name='dimension',
        type=int,
        required=True,
        constraints={'min': 1, 'max': 1536},
        description="Vector dimension"
    ),
    # ... other fields
}
```

### Embedding Schema
```python
embedding_schema = {
    'model': ConfigSchema(
        name='model',
        type=str,
        required=True,
        default='text-embedding-3-small',
        description="Embedding model name"
    ),
    # ... other fields
}
```

### Clustering Schema
```python
clustering_schema = {
    'method': ConfigSchema(
        name='method',
        type=str,
        required=True,
        default='kmeans',
        constraints={'allowed': ['kmeans', 'hdbscan', 'hierarchical']},
        description="Clustering algorithm"
    ),
    # ... other fields
}
```

### Monitoring Schema
```python
monitoring_schema = {
    'collection_interval': ConfigSchema(
        name='collection_interval',
        type=int,
        required=True,
        default=60,
        constraints={'min': 10, 'max': 3600},
        description="Metrics collection interval in seconds"
    ),
    # ... other fields
}
``` 