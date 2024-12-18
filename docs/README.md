# RAPTOR Documentation

## Overview
RAPTOR (Recursive Analysis and Processing Tool for Organized Retrieval) is a system designed for processing, analyzing, and organizing large collections of documents using hierarchical clustering and semantic analysis.

## Key Features
- Hierarchical document clustering
- Semantic analysis and embedding
- Tree-based document organization
- Real-time system monitoring
- Robust error handling
- Configurable processing pipeline

## System Requirements
- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- Storage space dependent on document collection size

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/raptor_project.git
cd raptor_project
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Configuration
RAPTOR uses a hierarchical configuration system with validation. Key configuration sections:

### Pinecone Configuration
```json
{
    "pinecone": {
        "api_key": "your-api-key",
        "dimension": 1536,
        "metric": "cosine",
        "cloud": "aws",
        "region": "us-west-1",
        "index_name": "your-index"
    }
}
```

### Embedding Configuration
```json
{
    "embedding": {
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "batch_size": 8
    }
}
```

### Clustering Configuration
```json
{
    "clustering": {
        "method": "kmeans",
        "min_cluster_size": 2,
        "max_clusters": 10,
        "dimension": 20,
        "threshold": 0.15,
        "max_levels": 3
    }
}
```

### Monitoring Configuration
```json
{
    "monitoring": {
        "collection_interval": 60,
        "alert_thresholds": {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "error_rate": 0.1
        }
    }
}
```

## Usage

### Basic Usage
```python
from src.tree.tree_manager import TreeManager
from pathlib import Path

# Initialize TreeManager with configuration
config = {
    "embedding": {...},
    "clustering": {...},
    "pinecone": {...}
}
tree_manager = TreeManager(config)

# Process documents
docs = ["Document 1 content", "Document 2 content"]
metadata = [
    {"filename": "doc1.txt", "index": 0},
    {"filename": "doc2.txt", "index": 1}
]
tree_data = tree_manager.process_documents(docs, metadata)
```

### Monitoring
```python
from src.monitoring.system_monitor import SystemMonitor

# Initialize monitoring
monitor_config = {
    "collection_interval": 60,
    "alert_thresholds": {...}
}
monitor = SystemMonitor(monitor_config)

# Start monitoring
monitor.start_monitoring()

# Process metrics and check alerts
monitor.process_metrics()

# Generate report
report_path = monitor.generate_report(Path("reports"))
```

## Error Handling
RAPTOR provides a comprehensive error handling system:

```python
from src.utils.error_handler import error_handler, RaptorError

@error_handler
def your_function():
    # Your code here
    pass

try:
    result = your_function()
except RaptorError as e:
    print(f"Error: {e.message}")
    print(f"Error Code: {e.error_code}")
    print(f"Details: {e.details}")
```

## Directory Structure
```
raptor_project/
├── src/
│   ├── analysis/       # Document analysis components
│   ├── clustering/     # Clustering algorithms
│   ├── embedding/      # Embedding generation
│   ├── monitoring/     # System monitoring
│   ├── prepare/       # Data preparation
│   ├── scripts/       # Utility scripts
│   ├── storage/       # Data storage components
│   ├── summarization/ # Text summarization
│   ├── tree/          # Tree management
│   ├── utils/         # Utility functions
│   └── visualization/ # Visualization components
├── data/              # Data storage
├── docs/              # Documentation
├── tests/            # Test suite
└── requirements.txt  # Project dependencies
```

## Testing
Run the test suite:
```bash
python -m pytest tests/
```

## Monitoring and Maintenance
- Monitor system metrics through the monitoring dashboard
- Check error logs in `data/{index_name}/logs/error.log`
- Review monitoring reports in `data/{index_name}/reports/`
- Regularly backup data in `data/{index_name}/backup/`

## Best Practices
1. Always validate configuration before deployment
2. Monitor system resources during large processing jobs
3. Implement proper error handling using the provided decorators
4. Regularly check and rotate logs
5. Keep embeddings and clustering configurations consistent
6. Backup data before major operations

## Troubleshooting
Common issues and solutions:

1. **Memory Issues**
   - Reduce batch size in embedding configuration
   - Increase system swap space
   - Monitor memory usage through dashboard

2. **Processing Errors**
   - Check error logs for specific error messages
   - Verify input data format and encoding
   - Ensure all required fields are present in metadata

3. **Clustering Issues**
   - Adjust clustering parameters (min_cluster_size, threshold)
   - Check vector dimensions match configuration
   - Verify embedding quality and consistency

4. **Storage Issues**
   - Verify Pinecone API key and permissions
   - Check index exists and has correct dimensions
   - Monitor storage usage and cleanup old data

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License
[Your License Here]

## Contact
[Your Contact Information] 