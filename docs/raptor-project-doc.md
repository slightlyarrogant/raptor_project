# RAPTOR (Recursive Analysis and Processing Tool for Organized Retrieval)

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [API Specification](#api-specification)
5. [Data Models](#data-models)
6. [Implementation Guidelines](#implementation-guidelines)
7. [Deployment](#deployment)

## Overview

RAPTOR is a document processing and analysis system that organizes documents into hierarchical structures using semantic analysis and clustering. The system provides a web interface for document upload, processing configuration, and visualization of document relationships.

### Key Features
- Document processing and semantic analysis
- Hierarchical clustering and organization
- Multiple vector store support
- OpenAI embedding integration
- Real-time processing status
- Interactive visualization
- REST API access

## Architecture

### Intermediate Solution Architecture
```mermaid
graph TB
    subgraph Frontend
        UI[Web UI]
        FH[File Handler]
        TV[Tree Visualizer]
    end

    subgraph Backend
        API[FastAPI Service]
        Auth[Authentication]
        PP[File Preprocessor]
        PE[Document Processor]
        TC[Tree Constructor]
    end

    subgraph Storage
        FS[File Storage]
        VDB[(Vector Store)]
        Cache[(Redis Cache)]
    end

    %% Frontend Interactions
    UI --> FH
    UI --> TV
    
    %% Frontend to Backend
    FH --> API
    TV --> API
    
    %% Backend Flow
    API --> Auth
    API --> PP
    PP --> PE
    PE --> TC
    
    %% Storage Interactions
    PP --> FS
    PE --> VDB
    TC --> VDB
    TC --> Cache
```

### Components

1. **Frontend Layer**
   - Web Interface (React/Vue)
   - File Upload Component
   - Configuration Interface
   - Tree Visualization
   - Status Monitoring

2. **Backend Layer**
   - FastAPI Application
   - Authentication Service
   - Document Processor
   - Tree Manager
   - Embedding Service

3. **Storage Layer**
   - Vector Store (Pinecone/Qdrant/FAISS)
   - File Storage
   - Redis Cache

## Embedding Models

### Supported Models

| Model Name | Dimensions | Max Input Tokens | Description | Cost per 1K tokens |
|------------|------------|------------------|-------------|-------------------|
| text-embedding-3-small | 1536 | 8191 | Newest model, best price/performance ratio | $0.00002 |
| text-embedding-3-large | 3072 | 8191 | Newest model, highest performance | $0.00013 |
| text-embedding-ada-002 | 1536 | 8191 | Legacy model | $0.00010 |

### Vector Store Support

#### Pinecone
```json
{
    "type": "pinecone",
    "settings": {
        "api_key": "string",
        "environment": "string",
        "index_name": "string",
        "metric": "cosine"
    }
}
```

#### Qdrant
```json
{
    "type": "qdrant",
    "settings": {
        "api_key": "string",
        "url": "string",
        "collection_name": "string",
        "distance": "Cosine"
    }
}
```

#### FAISS
```json
{
    "type": "faiss",
    "settings": {
        "index_type": "IndexFlatIP",
        "storage_path": "string"
    }
}
```

## API Specification

### Authentication

#### POST /auth/login
Login to the system
```json
Request:
{
    "username": "string",
    "password": "string"
}

Response:
{
    "access_token": "string",
    "token_type": "bearer"
}
```

#### POST /auth/refresh
Refresh access token
```json
Request:
{
    "refresh_token": "string"
}

Response:
{
    "access_token": "string",
    "token_type": "bearer"
}
```

### Configuration

#### GET /config/embedding-models
Get available embedding models
```json
Response:
{
    "models": [
        {
            "id": "text-embedding-3-small",
            "dimensions": 1536,
            "max_tokens": 8191,
            "cost_per_1k_tokens": 0.00002,
            "recommended": true
        }
    ]
}
```

#### POST /config/system
Configure system settings
```json
Request:
{
    "embedding": {
        "model": "text-embedding-3-small",
        "batch_size": 8,
        "retry_attempts": 3
    },
    "vector_store": {
        "type": "pinecone",
        "settings": {
            "api_key": "string",
            "environment": "string",
            "index_name": "string"
        }
    }
}
```

### Document Management

#### POST /documents/upload
Upload documents
```json
Request:
- Multipart form data
- Files under "documents" key
- Optional metadata JSON

Response:
{
    "job_id": "string",
    "uploaded_files": [
        {
            "filename": "string",
            "size": integer,
            "mime_type": "string",
            "status": "pending"
        }
    ]
}
```

#### GET /documents/{doc_id}
Get document details
```json
Response:
{
    "id": "string",
    "filename": "string",
    "status": "string",
    "metadata": {
        "size": integer,
        "upload_date": "datetime",
        "last_processed": "datetime",
        "mime_type": "string"
    }
}
```

### Processing

#### POST /processing/start
Start processing job
```json
Request:
{
    "doc_ids": ["string"],
    "settings": {
        "chunk_size": "dynamic",
        "overlap": integer,
        "batch_size": integer
    }
}
```

#### GET /processing/status/{job_id}
Get job status
```json
Response:
{
    "job_id": "string",
    "status": "string",
    "progress": {
        "percentage": float,
        "current_file": "string",
        "processed_files": integer,
        "total_files": integer
    }
}
```

### Tree Management

#### GET /tree
Get tree structure
```json
Response:
{
    "tree_id": "string",
    "root": {
        "id": "string",
        "type": "root",
        "children": []
    },
    "metadata": {
        "total_nodes": integer,
        "depth": integer,
        "last_updated": "datetime"
    }
}
```

#### POST /tree/search
Search within tree
```json
Request:
{
    "query": "string",
    "filters": {
        "document_types": ["string"],
        "date_range": {
            "start": "datetime",
            "end": "datetime"
        }
    }
}
```

## Data Models

### Document
```python
class Document(BaseModel):
    id: str
    filename: str
    content: str
    mime_type: str
    size: int
    upload_date: datetime
    metadata: Dict[str, Any]
```

### Processing Job
```python
class ProcessingJob(BaseModel):
    id: str
    status: str
    doc_ids: List[str]
    settings: Dict[str, Any]
    progress: float
    created_at: datetime
    updated_at: datetime
```

### Tree Node
```python
class TreeNode(BaseModel):
    id: str
    type: str
    content: str
    metadata: Dict[str, Any]
    children: List[str]
    parent: Optional[str]
```

## Implementation Guidelines

### Project Structure
```
raptor/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   └── public/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   └── services/
│   └── tests/
└── deployment/
    ├── docker/
    └── config/
```

### Technology Stack
- Frontend: React/Vue.js
- Backend: FastAPI
- Database: PostgreSQL
- Vector Store: Pinecone/Qdrant/FAISS
- Cache: Redis
- Authentication: JWT

## Deployment

### Requirements
- Python 3.8+
- Node.js 14+
- PostgreSQL 12+
- Redis 6+
- 8GB RAM minimum
- 50GB storage minimum

### Development Setup
```bash
# Backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend
npm install
npm run dev

# Database
createdb raptor
alembic upgrade head
```

### Production Deployment
```bash
# Build frontend
cd frontend
npm run build

# Start services
docker-compose up -d
```

### Environment Variables
```env
# API Keys
OPENAI_API_KEY=your-openai-key
VECTOR_STORE_API_KEY=your-store-key

# Database
DATABASE_URL=postgresql://user:pass@localhost/raptor

# Redis
REDIS_URL=redis://localhost

# JWT
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
```

### Monitoring
- System Health: `/health`
- Metrics: `/metrics` (Prometheus format)
- Logs: `logs/raptor.log`

## Security Considerations

1. **Authentication**
   - JWT-based authentication
   - Token refresh mechanism
   - Rate limiting

2. **API Security**
   - Input validation
   - Request sanitization
   - CORS configuration

3. **Data Security**
   - Encryption at rest
   - Secure file handling
   - API key management

4. **Monitoring**
   - Error tracking
   - Access logging
   - Performance monitoring

