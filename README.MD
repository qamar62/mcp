# MCP Docker Infrastructure

## Overview
This is a complete Model Context Protocol (MCP) infrastructure built with Docker Compose, designed to work with Ollama for AI model interactions.

┌─────────────────────────────────────────────────────────────┐
│                    MCP LXC Container                        │
│                                                             │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │               Docker Compose Stack                      │  │
│  │                                                         │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │     MCP      │  │     MCP      │  │     MCP      │  │  │
│  │  │ Orchestrator │  │ FileSystem   │  │  Database    │  │  │
│  │  │  Container   │  │  Container   │  │  Container   │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │  │
│  │                                                         │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │     MCP      │  │     MCP      │  │    Redis     │  │  │
│  │  │   Web API    │  │  Monitoring  │  │   (Cache)    │  │  │
│  │  │  Container   │  │  Container   │  │  Container   │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                             │
│  External Network Connection to Ollama LXC                 │
└─────────────────────────────────────────────────────────────┘

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Ollama running on `192.168.1.132:11434`
- Available models: mistral:7b, qwen3:8b, phi4-mini-reasoning:3.8b, deepseek-r1:8b

### Deployment
```bash
# Clone or copy this directory to your server
cd /path/to/mcp-infrastructure

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Access Points
- **Main API**: http://localhost:8080
- **Monitoring**: http://localhost:3000  
- **Nginx Proxy**: http://localhost:80
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

## Architecture

### Services

#### 1. MCP Orchestrator (Port 8080)
- **Purpose**: Main coordinator service that manages MCP servers and Ollama integration
- **File**: `orchestrator/main.py`
- **Features**:
  - Query routing to Ollama models
  - MCP server coordination
  - Context gathering from multiple agents
  - Redis caching

#### 2. MCP Filesystem Server (Port 8081)
- **Purpose**: File system operations through MCP protocol
- **File**: `servers/filesystem/filesystem_server.py`
- **Features**:
  - Read/write file operations
  - Directory listing
  - Secure path validation
  - File metadata access

#### 3. MCP Database Server (Port 8082)
- **Purpose**: Database operations through MCP protocol
- **File**: `servers/database/database_server.py`
- **Features**:
  - SQL query execution
  - PostgreSQL and SQLite support
  - Connection pooling
  - Query result formatting

#### 4. MCP Web API Server (Port 8083)
- **Purpose**: External web API access through MCP protocol
- **File**: `servers/webapi/webapi_server.py`
- **Features**:
  - HTTP request handling
  - Web scraping capabilities
  - RSS feed parsing
  - Rate limiting

#### 5. Monitoring Server (Port 3000)
- **Purpose**: System monitoring and health checks
- **File**: `monitoring/monitor.py`
- **Features**:
  - Service health monitoring
  - System metrics collection
  - Alert management
  - Performance tracking

#### 6. Supporting Services
- **Redis**: Caching and session management
- **PostgreSQL**: Primary database storage
- **Nginx**: Reverse proxy and load balancing

## Project Structure

```
mcp-infrastructure/
├── docker-compose.yml          # Main orchestration file
├── mcp-docker.sh              # Setup script
├── README.md                  # This file
│
├── orchestrator/              # Main coordinator service
│   ├── Dockerfile
│   ├── main.py               # FastAPI orchestrator
│   └── requirements.txt
│
├── servers/                   # MCP server implementations
│   ├── filesystem/
│   │   ├── Dockerfile
│   │   ├── filesystem_server.py
│   │   └── requirements.txt
│   ├── database/
│   │   ├── Dockerfile
│   │   ├── database_server.py
│   │   └── requirements.txt
│   └── webapi/
│       ├── Dockerfile
│       ├── webapi_server.py
│       └── requirements.txt
│
├── monitoring/               # Monitoring and metrics
│   ├── Dockerfile
│   ├── monitor.py
│   ├── requirements.txt
│   └── config/
│
├── nginx/                    # Reverse proxy
│   └── nginx.conf
│
├── init-scripts/            # Database initialization
│   └── init.sql
│
├── config/                  # Configuration files
├── logs/                    # Application logs
└── data/                    # Persistent data
    ├── sqlite/             # SQLite databases
    └── shared/             # Shared file storage
```

## Configuration

### Environment Variables

#### Orchestrator
- `OLLAMA_URL`: Ollama API endpoint (default: http://192.168.1.132:11434)
- `REDIS_URL`: Redis connection string
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)

#### Filesystem Server
- `ALLOWED_PATHS`: Comma-separated allowed directories
- `MAX_FILE_SIZE`: Maximum file size for operations

#### Database Server
- `POSTGRES_URL`: PostgreSQL connection string
- `SQLITE_PATH`: SQLite database directory

#### Web API Server
- `API_RATE_LIMIT`: Requests per minute limit
- `CACHE_TTL`: Cache time-to-live in seconds

### Ollama Models
The system is configured to work with your available models:
- `mistral:7b` (default)
- `qwen3:8b`
- `phi4-mini-reasoning:3.8b`
- `deepseek-r1:8b`

## API Usage

### Query Ollama with MCP Context
```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What files are in the data directory?",
    "model": "mistral:7b",
    "agents": ["filesystem"],
    "include_context": true
  }'
```

### Execute MCP Tools
```bash
curl -X POST http://localhost:8080/tool \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "filesystem",
    "tool": "list_directory",
    "arguments": {"directory_path": "/data"}
  }'
```

### Health Checks
```bash
# Overall system health
curl http://localhost:8080/health

# Individual service health
curl http://localhost:8081/health  # Filesystem
curl http://localhost:8082/health  # Database
curl http://localhost:8083/health  # Web API
curl http://localhost:3000/health  # Monitoring
```

## Monitoring

### System Metrics
- CPU and memory usage
- Disk space utilization
- Service response times
- Error rates and alerts

### Service Status
- Health check results
- Connection status to Ollama
- MCP server availability
- Database connectivity

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   - Verify Ollama is running on 192.168.1.132:11434
   - Check network connectivity
   - Ensure firewall allows connections

2. **Service Won't Start**
   - Check Docker logs: `docker-compose logs [service-name]`
   - Verify port availability
   - Check configuration files

3. **Database Connection Issues**
   - Ensure PostgreSQL is running
   - Check database credentials
   - Verify network connectivity

### Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f mcp-orchestrator
docker-compose logs -f mcp-filesystem
```

## Development

### Adding New MCP Servers
1. Create new directory under `servers/`
2. Implement FastAPI server with MCP protocol
3. Add Dockerfile and requirements.txt
4. Update docker-compose.yml
5. Register in orchestrator's MCP_SERVERS

### Custom Configuration
- Modify environment variables in docker-compose.yml
- Update nginx.conf for custom routing
- Add initialization scripts to init-scripts/

## Security Notes

- File system access is restricted to configured paths
- Database connections use environment variables
- Nginx provides reverse proxy protection
- Health checks ensure service availability

## Performance Tuning

- Adjust Redis memory limits
- Configure PostgreSQL connection pooling
- Set appropriate resource limits in docker-compose.yml
- Monitor and adjust based on usage patterns