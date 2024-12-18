# RAPTOR Troubleshooting Guide

## Common Issues and Solutions

### 1. Document Processing Issues

#### Large Document Processing Fails
**Symptoms:**
- Memory errors during processing
- Process killed by system
- Timeout errors

**Solutions:**
1. Reduce batch size:
```python
config['embedding']['batch_size'] = 4  # Reduce from default 8
```

2. Enable chunking:
```python
config['processing']['enable_chunking'] = True
config['processing']['chunk_size'] = 500  # Tokens per chunk
```

3. Monitor memory usage:
```bash
docker stats raptor
```

#### Embedding Generation Fails
**Symptoms:**
- OpenAI API errors
- Timeout during embedding generation
- Inconsistent vector dimensions

**Solutions:**
1. Check API key and rate limits:
```bash
# Verify API key
echo $OPENAI_API_KEY

# Check rate limits
python -m src.scripts.check_api_limits
```

2. Implement retry logic:
```python
config['embedding']['max_retries'] = 3
config['embedding']['retry_delay'] = 5  # seconds
```

3. Verify input text:
```python
from src.utils.text_validator import validate_text

# Add validation step
validate_text(document_text)
```

### 2. Clustering Issues

#### Unbalanced Clusters
**Symptoms:**
- One very large cluster
- Many single-document clusters
- Poor tree structure

**Solutions:**
1. Adjust clustering parameters:
```python
config['clustering'].update({
    'min_cluster_size': 3,
    'max_cluster_size': 50,
    'threshold': 0.15
})
```

2. Try different algorithms:
```python
# Switch to HDBSCAN
config['clustering']['method'] = 'hdbscan'
config['clustering']['min_samples'] = 2
```

3. Analyze cluster distribution:
```bash
python -m src.scripts.analyze_clusters
```

#### Poor Cluster Quality
**Symptoms:**
- Unrelated documents clustered together
- Similar documents in different clusters
- Inconsistent hierarchy

**Solutions:**
1. Adjust similarity threshold:
```python
config['clustering']['threshold'] = 0.2  # Increase for stricter clustering
```

2. Implement cluster validation:
```python
config['clustering'].update({
    'validate_clusters': True,
    'min_coherence': 0.3,
    'max_dispersion': 0.7
})
```

3. Analyze cluster quality:
```bash
python -m src.scripts.evaluate_clusters
```

### 3. Storage Issues

#### Pinecone Connection Problems
**Symptoms:**
- Connection timeouts
- Authentication errors
- Index not found

**Solutions:**
1. Verify Pinecone configuration:
```bash
# Check environment variables
env | grep PINECONE

# Test connection
python -m src.scripts.test_pinecone
```

2. Initialize index properly:
```python
from src.storage.pinecone_manager import PineconeManager

# Ensure index exists
manager = PineconeManager(config)
manager.initialize_index()
```

3. Monitor Pinecone status:
```bash
python -m src.scripts.check_pinecone_status
```

#### Vector Storage Issues
**Symptoms:**
- Failed vector uploads
- Missing vectors
- Incorrect metadata

**Solutions:**
1. Validate vectors before storage:
```python
from src.utils.vector_validator import validate_vectors

# Add validation
validate_vectors(vectors, expected_dim=1536)
```

2. Implement batch processing:
```python
config['storage'].update({
    'batch_size': 100,
    'max_retries': 3,
    'validate_upload': True
})
```

3. Monitor storage operations:
```bash
python -m src.scripts.monitor_storage
```

### 4. System Performance Issues

#### High Memory Usage
**Symptoms:**
- System slowdown
- OOM errors
- Swap usage

**Solutions:**
1. Monitor memory usage:
```bash
# Check container memory
docker stats

# Check system memory
free -h
```

2. Optimize memory settings:
```python
config['system'].update({
    'max_workers': 2,  # Reduce parallel processing
    'batch_size': 4,   # Reduce batch size
    'clear_cache': True
})
```

3. Implement memory limits:
```bash
# Set container memory limit
docker update --memory 8G raptor
```

#### Slow Processing Speed
**Symptoms:**
- Long processing times
- Queue buildup
- Timeouts

**Solutions:**
1. Profile performance:
```bash
python -m src.scripts.profile_performance
```

2. Optimize processing:
```python
config['processing'].update({
    'enable_caching': True,
    'parallel_processing': True,
    'optimize_embeddings': True
})
```

3. Monitor processing times:
```bash
python -m src.scripts.monitor_performance
```

### 5. Monitoring and Logging Issues

#### Missing or Incomplete Logs
**Symptoms:**
- Missing error information
- Incomplete tracking
- No performance data

**Solutions:**
1. Configure logging properly:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/raptor.log'),
        logging.StreamHandler()
    ]
)
```

2. Enable detailed logging:
```python
config['logging'].update({
    'log_level': 'DEBUG',
    'enable_performance_logging': True,
    'log_retention_days': 14
})
```

3. Monitor log files:
```bash
# Check log size
du -h data/logs/

# Monitor logs in real-time
tail -f data/logs/raptor.log
```

#### Alert System Issues
**Symptoms:**
- Missing alerts
- False positives
- Delayed notifications

**Solutions:**
1. Configure alert thresholds:
```python
config['monitoring']['alert_thresholds'].update({
    'cpu_percent': 80.0,
    'memory_percent': 85.0,
    'error_rate': 0.1,
    'processing_delay': 300  # seconds
})
```

2. Implement alert validation:
```python
config['monitoring'].update({
    'validate_alerts': True,
    'min_alert_interval': 300,  # seconds
    'alert_deduplication': True
})
```

3. Test alert system:
```bash
python -m src.scripts.test_alerts
```

### 6. Data Quality Issues

#### Incorrect or Missing Metadata
**Symptoms:**
- Missing document information
- Incorrect relationships
- Poor search results

**Solutions:**
1. Validate metadata:
```python
from src.utils.metadata_validator import validate_metadata

# Add validation
validate_metadata(metadata, required_fields=['filename', 'type', 'source'])
```

2. Implement metadata enrichment:
```python
config['processing'].update({
    'enrich_metadata': True,
    'extract_keywords': True,
    'detect_language': True
})
```

3. Audit metadata:
```bash
python -m src.scripts.audit_metadata
```

#### Data Consistency Issues
**Symptoms:**
- Inconsistent document versions
- Missing relationships
- Broken tree structure

**Solutions:**
1. Enable consistency checks:
```python
config['storage'].update({
    'enable_consistency_check': True,
    'validate_references': True,
    'auto_repair': True
})
```

2. Implement versioning:
```python
config['processing'].update({
    'enable_versioning': True,
    'track_changes': True,
    'version_retention': 30  # days
})
```

3. Run consistency check:
```bash
python -m src.scripts.check_consistency
```

### Emergency Procedures

#### System Recovery
1. Stop processing:
```bash
docker-compose stop raptor
```

2. Backup data:
```bash
./scripts/backup.sh
```

3. Clear problematic data:
```bash
python -m src.scripts.clear_problematic_data
```

4. Restore from backup:
```bash
python -m src.scripts.restore_backup
```

#### Performance Recovery
1. Clear caches:
```bash
python -m src.scripts.clear_caches
```

2. Reset connections:
```bash
python -m src.scripts.reset_connections
```

3. Restart services:
```bash
docker-compose restart
```

### Diagnostic Tools

#### System Diagnostics
```bash
# Run full system diagnostic
python -m src.scripts.run_diagnostics

# Generate diagnostic report
python -m src.scripts.generate_report

# Check system health
python -m src.scripts.health_check
```

#### Performance Analysis
```bash
# Profile system performance
python -m src.scripts.profile_system

# Analyze bottlenecks
python -m src.scripts.analyze_bottlenecks

# Generate performance report
python -m src.scripts.performance_report
```

#### Data Validation
```bash
# Validate all data
python -m src.scripts.validate_data

# Check data integrity
python -m src.scripts.check_integrity

# Generate validation report
python -m src.scripts.validation_report
``` 