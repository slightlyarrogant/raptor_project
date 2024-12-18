import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import os
from dataclasses import dataclass
from src.utils.error_handler import ConfigurationError, error_handler

logger = logging.getLogger(__name__)

@dataclass
class ConfigSchema:
    """Configuration schema definition."""
    name: str
    type: type
    required: bool = True
    default: Any = None
    constraints: Optional[Dict[str, Any]] = None
    description: str = ""

class ConfigValidator:
    """Validates configuration settings against defined schemas."""
    
    def __init__(self):
        self.schemas = {
            'pinecone': {
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
                'metric': ConfigSchema(
                    name='metric',
                    type=str,
                    required=True,
                    constraints={'allowed': ['cosine', 'euclidean', 'dotproduct']},
                    description="Distance metric"
                ),
                'cloud': ConfigSchema(
                    name='cloud',
                    type=str,
                    required=True,
                    description="Cloud provider"
                ),
                'region': ConfigSchema(
                    name='region',
                    type=str,
                    required=True,
                    description="Cloud region"
                ),
                'index_name': ConfigSchema(
                    name='index_name',
                    type=str,
                    required=True,
                    description="Index name"
                )
            },
            'embedding': {
                'model': ConfigSchema(
                    name='model',
                    type=str,
                    required=True,
                    default='text-embedding-3-small',
                    description="Embedding model name"
                ),
                'dimensions': ConfigSchema(
                    name='dimensions',
                    type=int,
                    required=True,
                    default=1536,
                    constraints={'min': 1, 'max': 1536},
                    description="Embedding dimensions"
                ),
                'batch_size': ConfigSchema(
                    name='batch_size',
                    type=int,
                    required=True,
                    default=8,
                    constraints={'min': 1, 'max': 100},
                    description="Batch size for embedding generation"
                )
            },
            'clustering': {
                'method': ConfigSchema(
                    name='method',
                    type=str,
                    required=True,
                    default='kmeans',
                    constraints={'allowed': ['kmeans', 'hdbscan', 'hierarchical']},
                    description="Clustering algorithm"
                ),
                'min_cluster_size': ConfigSchema(
                    name='min_cluster_size',
                    type=int,
                    required=True,
                    default=2,
                    constraints={'min': 2},
                    description="Minimum cluster size"
                ),
                'max_clusters': ConfigSchema(
                    name='max_clusters',
                    type=int,
                    required=True,
                    default=10,
                    constraints={'min': 2},
                    description="Maximum number of clusters"
                ),
                'dimension': ConfigSchema(
                    name='dimension',
                    type=int,
                    required=True,
                    default=20,
                    constraints={'min': 1},
                    description="Clustering dimension"
                ),
                'threshold': ConfigSchema(
                    name='threshold',
                    type=float,
                    required=True,
                    default=0.15,
                    constraints={'min': 0.0, 'max': 1.0},
                    description="Clustering threshold"
                ),
                'max_levels': ConfigSchema(
                    name='max_levels',
                    type=int,
                    required=True,
                    default=3,
                    constraints={'min': 1, 'max': 10},
                    description="Maximum tree levels"
                )
            },
            'monitoring': {
                'collection_interval': ConfigSchema(
                    name='collection_interval',
                    type=int,
                    required=True,
                    default=60,
                    constraints={'min': 10, 'max': 3600},
                    description="Metrics collection interval in seconds"
                ),
                'alert_thresholds': ConfigSchema(
                    name='alert_thresholds',
                    type=dict,
                    required=True,
                    default={
                        'cpu_percent': 80.0,
                        'memory_percent': 85.0,
                        'disk_percent': 90.0,
                        'error_rate': 0.1
                    },
                    description="Alert thresholds"
                )
            }
        }
    
    @error_handler
    def validate_config(self, config: Dict[str, Any], section: str = None) -> Dict[str, Any]:
        """Validate configuration against schema."""
        if section and section not in self.schemas:
            raise ConfigurationError(f"Unknown configuration section: {section}")
            
        sections = [section] if section else self.schemas.keys()
        validated_config = {}
        
        for sec in sections:
            if sec not in config:
                if any(schema.required for schema in self.schemas[sec].values()):
                    raise ConfigurationError(f"Missing required section: {sec}")
                continue
                
            validated_config[sec] = self._validate_section(config[sec], sec)
            
        return validated_config if section else validated_config[section]
    
    def _validate_section(self, config: Dict[str, Any], section: str) -> Dict[str, Any]:
        """Validate a specific configuration section."""
        validated = {}
        schema = self.schemas[section]
        
        # Check for required fields
        for field_name, field_schema in schema.items():
            if field_name not in config:
                if field_schema.required:
                    if field_schema.default is not None:
                        validated[field_name] = field_schema.default
                        logger.warning(
                            f"Using default value for required field {field_name} "
                            f"in section {section}: {field_schema.default}"
                        )
                    else:
                        raise ConfigurationError(
                            f"Missing required field {field_name} in section {section}"
                        )
                continue
                
            value = config[field_name]
            validated[field_name] = self._validate_field(value, field_schema, section)
            
        return validated
    
    def _validate_field(self, value: Any, schema: ConfigSchema, section: str) -> Any:
        """Validate a single configuration field."""
        # Type checking
        if not isinstance(value, schema.type):
            try:
                value = schema.type(value)
            except (ValueError, TypeError):
                raise ConfigurationError(
                    f"Invalid type for {schema.name} in section {section}. "
                    f"Expected {schema.type.__name__}, got {type(value).__name__}"
                )
                
        # Constraint validation
        if schema.constraints:
            if 'min' in schema.constraints and value < schema.constraints['min']:
                raise ConfigurationError(
                    f"Value for {schema.name} in section {section} "
                    f"must be >= {schema.constraints['min']}"
                )
                
            if 'max' in schema.constraints and value > schema.constraints['max']:
                raise ConfigurationError(
                    f"Value for {schema.name} in section {section} "
                    f"must be <= {schema.constraints['max']}"
                )
                
            if 'allowed' in schema.constraints and value not in schema.constraints['allowed']:
                raise ConfigurationError(
                    f"Invalid value for {schema.name} in section {section}. "
                    f"Must be one of: {', '.join(schema.constraints['allowed'])}"
                )
                
        return value
    
    @error_handler
    def load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load and validate configuration from file."""
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path) as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {str(e)}")
            
        return self.validate_config(config)
    
    @error_handler
    def save_config(self, config: Dict[str, Any], config_path: Path) -> None:
        """Save validated configuration to file."""
        # Validate before saving
        validated_config = self.validate_config(config)
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(validated_config, f, indent=2)
            
        logger.info(f"Configuration saved to {config_path}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        default_config = {}
        for section, schema in self.schemas.items():
            default_config[section] = {
                name: field.default if field.default is not None else None
                for name, field in schema.items()
            }
        return default_config 