# RAPTOR Deployment Guide

## Production Deployment

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 16GB RAM minimum
- 100GB storage minimum
- Linux environment (recommended)

### Environment Setup

1. Create production environment file:
```bash
cp .env.example .env.prod
```

Edit `.env.prod` with production values:
```env
# API Keys
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key

# Pinecone Configuration
PINECONE_DIMENSION=1536
PINECONE_METRIC=cosine
PINECONE_CLOUD=aws
PINECONE_REGION=us-west-1
PINECONE_INDEX_NAME=your-prod-index

# System Configuration
LOG_LEVEL=INFO
COLLECTION_INTERVAL=60
MAX_BATCH_SIZE=8
MAX_WORKERS=4
ENABLE_MONITORING=true
```

2. Create production configuration:
```bash
cp config.example.json config.prod.json
```

Edit `config.prod.json`:
```json
{
    "embedding": {
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "batch_size": 8
    },
    "clustering": {
        "method": "hdbscan",
        "min_cluster_size": 3,
        "max_clusters": 10,
        "dimension": 20,
        "threshold": 0.15,
        "max_levels": 3
    },
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

### Docker Setup

1. Create Dockerfile:
```dockerfile
FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/logs data/reports data/backup

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run with gunicorn
CMD ["gunicorn", "src.api:app", "--bind", "0.0.0.0:8000", "--workers", "4"]
```

2. Create docker-compose.yml:
```yaml
version: '3.8'

services:
  raptor:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./config.prod.json:/app/config.json
    env_file:
      - .env.prod
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  monitoring:
    build: .
    command: python -m src.monitoring.run_dashboard
    volumes:
      - ./data:/app/data
    env_file:
      - .env.prod
    ports:
      - "8050:8050"
    depends_on:
      - raptor

  scheduler:
    build: .
    command: python -m src.scripts.scheduled_processor
    volumes:
      - ./data:/app/data
    env_file:
      - .env.prod
    depends_on:
      - raptor
```

### Production Deployment Steps

1. Build and start services:
```bash
docker-compose -f docker-compose.yml up -d
```

2. Initialize Pinecone index:
```bash
docker-compose exec raptor python -m src.scripts.init_pinecone
```

3. Verify deployment:
```bash
docker-compose ps
curl http://localhost:8000/health
```

### Monitoring Setup

1. Configure logging:
```bash
mkdir -p data/logs
touch data/logs/error.log
touch data/logs/access.log
```

2. Setup log rotation:
```bash
cat > /etc/logrotate.d/raptor << EOF
/app/data/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        kill -USR1 $(cat /app/gunicorn.pid 2>/dev/null) 2>/dev/null || true
    endscript
}
EOF
```

3. Configure monitoring dashboard:
```bash
mkdir -p data/monitoring
cp config/monitoring.example.json data/monitoring/config.json
```

### Backup Configuration

1. Create backup script:
```bash
mkdir -p scripts
cat > scripts/backup.sh << EOF
#!/bin/bash
BACKUP_DIR="/app/data/backup"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup data
tar -czf $BACKUP_DIR/data_$TIMESTAMP.tar.gz /app/data

# Backup Pinecone vectors (if needed)
python -m src.scripts.backup_vectors

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
EOF
chmod +x scripts/backup.sh
```

2. Setup daily backup cron:
```bash
echo "0 0 * * * /app/scripts/backup.sh" | crontab -
```

### Security Considerations

1. Configure firewall:
```bash
ufw allow 8000/tcp  # API
ufw allow 8050/tcp  # Monitoring dashboard
ufw enable
```

2. Setup SSL/TLS:
```bash
# Install certbot
apt-get install certbot python3-certbot-nginx

# Get certificate
certbot --nginx -d your-domain.com
```

3. Configure secure headers in nginx:
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Content-Security-Policy "default-src 'self';" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Performance Tuning

1. Configure gunicorn:
```python
# gunicorn.conf.py
import multiprocessing

workers = multiprocessing.cpu_count() * 2 + 1
threads = 2
worker_class = 'gevent'
worker_connections = 1000
timeout = 300
keepalive = 2
```

2. Configure system limits:
```bash
# /etc/security/limits.conf
*          soft    nofile      65535
*          hard    nofile      65535
```

3. Optimize Python settings:
```bash
export PYTHONOPTIMIZE=2
export PYTHONHASHSEED=random
```

### Maintenance Procedures

1. Update application:
```bash
# Pull latest changes
git pull origin main

# Build new image
docker-compose build

# Update containers
docker-compose up -d
```

2. Backup data:
```bash
# Manual backup
./scripts/backup.sh

# Verify backup
ls -l data/backup/
```

3. Monitor logs:
```bash
# View application logs
docker-compose logs -f raptor

# View monitoring logs
docker-compose logs -f monitoring
```

4. Health checks:
```bash
# Check API health
curl http://localhost:8000/health

# Check monitoring
curl http://localhost:8050/health

# Check system resources
docker stats
```

### Troubleshooting

1. Container issues:
```bash
# Check container status
docker-compose ps

# View container logs
docker-compose logs -f [service_name]

# Restart service
docker-compose restart [service_name]
```

2. Performance issues:
```bash
# Check resource usage
docker stats

# View slow queries
docker-compose exec raptor python -m src.scripts.analyze_performance

# Clear cache if needed
docker-compose exec raptor python -m src.scripts.clear_cache
```

3. Storage issues:
```bash
# Check disk usage
df -h

# Clean old data
docker-compose exec raptor python -m src.scripts.cleanup_old_data

# Optimize storage
docker-compose exec raptor python -m src.scripts.optimize_storage
```

### Scaling Considerations

1. Horizontal scaling:
- Use Docker Swarm or Kubernetes for container orchestration
- Configure load balancer for multiple instances
- Set up distributed caching with Redis

2. Vertical scaling:
- Increase container resource limits
- Optimize batch processing parameters
- Tune database connection pools

3. Storage scaling:
- Implement data archival strategy
- Use object storage for large files
- Configure data retention policies 