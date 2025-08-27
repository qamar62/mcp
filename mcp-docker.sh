#!/bin/bash
# setup-mcp-docker.sh - Setup MCP infrastructure with Docker Compose

set -e

echo "🚀 Setting up MCP Infrastructure with Docker Compose..."

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p /opt/mcp-infrastructure/{orchestrator,servers/{filesystem,database,webapi},monitoring,nginx,config,logs,data/{sqlite,shared},init-scripts}

cd /opt/mcp-infrastructure

# Create requirements.txt files
echo "📦 Creating requirements files..."

# Orchestrator requirements
cat > orchestrator/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
aiohttp==3.9.0
redis==5.0.1
sqlalchemy==2.0.23
asyncpg==0.29.0
pydantic==2.5.0
python-multipart==0.0.6
pyyaml==6.0.1
jinja2==3.1.2
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
# MCP SDK - we'll handle this differently since pip install mcp failed
# We'll use a local implementation or alternative
EOF

# Filesystem server requirements
cat > servers/filesystem/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
aiofiles==23.2.0
watchdog==3.0.0
python-magic==0.4.27
pillow==10.1.0
EOF

# Database server requirements  
cat > servers/database/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
asyncpg==0.29.0
sqlite3
pandas==2.1.4
aiosqlite==0.19.0
EOF

# Web API server requirements
cat > servers/webapi/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
httpx==0.25.2
aiohttp==3.9.0
beautifulsoup4==4.12.2
feedparser==6.0.10
EOF

# Create orchestrator main.py
echo "🎯 Creating orchestrator application..."
cat > orchestrator/main.py << 'EOF'
#!/usr/bin/env python3
"""
MCP Orchestrator - Docker version
Main FastAPI application that coordinates MCP servers
"""
import asyncio
import os
from typing import Dict, List, Optional
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import aiohttp
import redis.asyncio as redis
import yaml

# Setup logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MCP Orchestrator",
    description="Model Context Protocol Orchestrator for Docker",
    version="1.0.0"
)

# Configuration
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://192.168.1.100:11434')
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')

# MCP Server endpoints
MCP_SERVERS = {
    'filesystem': 'http://mcp-filesystem:8081',
    'database': 'http://mcp-database:8082', 
    'webapi': 'http://mcp-webapi:8083'
}

# Global variables
redis_client = None
mcp_sessions = {}

class QueryRequest(BaseModel):
    question: str
    agents: Optional[List[str]] = None
    model: Optional[str] = "llama2"
    include_context: bool = True

class ToolRequest(BaseModel):
    agent: str
    tool: str
    arguments: dict

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    global redis_client
    
    try:
        # Connect to Redis
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("Connected to Redis")
        
        # Initialize MCP server connections
        await initialize_mcp_servers()
        
        logger.info("MCP Orchestrator started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start orchestrator: {e}")
        raise

async def initialize_mcp_servers():
    """Initialize connections to MCP servers"""
    global mcp_sessions
    
    for server_name, server_url in MCP_SERVERS.items():
        try:
            # Test connection to MCP server
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{server_url}/health") as response:
                    if response.status == 200:
                        mcp_sessions[server_name] = server_url
                        logger.info(f"Connected to MCP server: {server_name}")
                    else:
                        logger.warning(f"MCP server {server_name} not ready")
        except Exception as e:
            logger.warning(f"Failed to connect to {server_name}: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mcp_servers": list(mcp_sessions.keys()),
        "ollama_url": OLLAMA_URL
    }

@app.get("/status")
async def get_status():
    """Get detailed status"""
    return {
        "orchestrator": "running",
        "mcp_servers": len(mcp_sessions),
        "connected_servers": list(mcp_sessions.keys()),
        "ollama_url": OLLAMA_URL,
        "redis_connected": redis_client is not None,
        "uptime": "TODO: implement",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/query")
async def enhanced_query(request: QueryRequest):
    """Query Ollama with MCP context"""
    try:
        context = ""
        
        if request.include_context:
            context = await gather_mcp_context(request.agents or list(mcp_sessions.keys()))
        
        # Create enhanced prompt
        prompt = f"""
Context from MCP servers:
{context}

Question: {request.question}

Please answer the question using the context provided above.
"""
        
        # Query Ollama
        response = await query_ollama(prompt, request.model)
        
        return {
            "question": request.question,
            "response": response,
            "agents_used": request.agents or list(mcp_sessions.keys()),
            "context_included": request.include_context,
            "context_length": len(context),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def gather_mcp_context(agent_names: List[str]) -> str:
    """Gather context from MCP servers"""
    context = ""
    
    for agent_name in agent_names:
        if agent_name not in mcp_sessions:
            continue
            
        try:
            server_url = mcp_sessions[agent_name]
            
            async with aiohttp.ClientSession() as session:
                # Get resources from MCP server
                async with session.get(f"{server_url}/resources") as response:
                    if response.status == 200:
                        resources = await response.json()
                        context += f"\n=== {agent_name.upper()} SERVER ===\n"
                        context += f"Resources available: {len(resources.get('resources', []))}\n"
                        
                        # Add sample resource content
                        for resource in resources.get('resources', [])[:3]:
                            context += f"- {resource.get('name', 'Unknown')}\n"
                
                # Get tools from MCP server
                async with session.get(f"{server_url}/tools") as response:
                    if response.status == 200:
                        tools = await response.json()
                        tool_names = [t.get('name') for t in tools.get('tools', [])]
                        if tool_names:
                            context += f"Available tools: {', '.join(tool_names)}\n"
                            
        except Exception as e:
            logger.warning(f"Error gathering context from {agent_name}: {e}")
            continue
    
    return context

async def query_ollama(prompt: str, model: str = "llama2") -> str:
    """Send query to Ollama"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "No response from Ollama")
                else:
                    return f"Error: Ollama returned status {response.status}"
                    
    except Exception as e:
        logger.error(f"Error querying Ollama: {e}")
        return f"Error connecting to Ollama: {str(e)}"

@app.post("/tool")
async def execute_tool(request: ToolRequest):
    """Execute tool on MCP server"""
    if request.agent not in mcp_sessions:
        raise HTTPException(status_code=404, detail=f"Agent {request.agent} not found")
    
    try:
        server_url = mcp_sessions[request.agent]
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{server_url}/execute/{request.tool}",
                json=request.arguments
            ) as response:
                result = await response.json()
                
                return {
                    "agent": request.agent,
                    "tool": request.tool,
                    "arguments": request.arguments,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
                
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List available MCP agents"""
    return {
        "agents": list(mcp_sessions.keys()),
        "count": len(mcp_sessions),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
EOF

# Create basic filesystem server
echo "📁 Creating filesystem MCP server..."
cat > servers/filesystem/filesystem_server.py << 'EOF'
#!/usr/bin/env python3
"""
MCP Filesystem Server - Docker version
Provides file system access through HTTP API
"""
import os
import json
from pathlib import Path
from typing import List, Dict
import aiofiles
import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Filesystem Server", version="1.0.0")

# Configuration
ALLOWED_PATHS = os.getenv('ALLOWED_PATHS', '/data,/shared').split(',')
MAX_FILE_SIZE = os.getenv('MAX_FILE_SIZE', '10MB')

class FileInfo(BaseModel):
    name: str
    path: str
    size: int
    modified: str
    is_directory: bool

@app.get("/health")
async def health_check():
    return {"status": "healthy", "server": "filesystem"}

@app.get("/resources")
async def list_resources():
    """List available file resources"""
    resources = []
    
    for allowed_path in ALLOWED_PATHS:
        if os.path.exists(allowed_path):
            for root, dirs, files in os.walk(allowed_path):
                for file in files[:10]:  # Limit results
                    file_path = os.path.join(root, file)
                    try:
                        stat = os.stat(file_path)
                        resources.append({
                            "uri": f"file://{file_path}",
                            "name": file,
                            "mimeType": "text/plain",
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
                    except:
                        continue
    
    return {"resources": resources}

@app.get("/tools")
async def list_tools():
    """List available tools"""
    return {
        "tools": [
            {
                "name": "read_file",
                "description": "Read file contents",
                "parameters": {"file_path": "string"}
            },
            {
                "name": "write_file", 
                "description": "Write file contents",
                "parameters": {"file_path": "string", "content": "string"}
            },
            {
                "name": "list_directory",
                "description": "List directory contents",
                "parameters": {"directory_path": "string"}
            }
        ]
    }

@app.post("/execute/{tool_name}")
async def execute_tool(tool_name: str, arguments: dict):
    """Execute filesystem tool"""
    try:
        if tool_name == "read_file":
            return await read_file(arguments.get("file_path"))
        elif tool_name == "write_file":
            return await write_file(arguments.get("file_path"), arguments.get("content"))
        elif tool_name == "list_directory":
            return await list_directory(arguments.get("directory_path"))
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tool: {tool_name}")
    
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def read_file(file_path: str):
    """Read file contents"""
    if not is_path_allowed(file_path):
        raise HTTPException(status_code=403, detail="Path not allowed")
    
    try:
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
        
        return {
            "result": content,
            "file_path": file_path,
            "size": len(content)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

async def write_file(file_path: str, content: str):
    """Write file contents"""
    if not is_path_allowed(file_path):
        raise HTTPException(status_code=403, detail="Path not allowed")
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(content)
        
        return {
            "result": f"File written successfully: {file_path}",
            "file_path": file_path,
            "size": len(content)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing file: {str(e)}")

async def list_directory(directory_path: str):
    """List directory contents"""
    if not is_path_allowed(directory_path):
        raise HTTPException(status_code=403, detail="Path not allowed")
    
    try:
        items = []
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            stat = os.stat(item_path)
            
            items.append({
                "name": item,
                "path": item_path,
                "is_directory": os.path.isdir(item_path),
                "size": stat.st_size if not os.path.isdir(item_path) else 0,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        return {
            "result": items,
            "directory": directory_path,
            "count": len(items)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing directory: {str(e)}")

def is_path_allowed(path: str) -> bool:
    """Check if path is within allowed directories"""
    abs_path = os.path.abspath(path)
    
    for allowed in ALLOWED_PATHS:
        if abs_path.startswith(os.path.abspath(allowed)):
            return True
    
    return False

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
EOF

# Make filesystem server executable
chmod +x servers/filesystem/filesystem_server.py

# Create basic database server (placeholder)
echo "🗄️ Creating database MCP server..."
cat > servers/database/database_server.py << 'EOF'
#!/usr/bin/env python3
"""
MCP Database Server - Docker version
Provides database access through HTTP API
"""
import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Database Server", version="1.0.0")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "server": "database"}

@app.get("/resources")
async def list_resources():
    """List available database resources"""
    return {
        "resources": [
            {"uri": "db://sqlite/main", "name": "SQLite Database", "type": "database"},
            {"uri": "db://postgres/mcpdata", "name": "PostgreSQL Database", "type": "database"}
        ]
    }

@app.get("/tools")
async def list_tools():
    """List available database tools"""
    return {
        "tools": [
            {
                "name": "execute_query",
                "description": "Execute SQL query",
                "parameters": {"query": "string", "database": "string"}
            }
        ]
    }

@app.post("/execute/{tool_name}")
async def execute_tool(tool_name: str, arguments: dict):
    """Execute database tool"""
    # Placeholder implementation
    return {
        "result": f"Database tool {tool_name} executed with args: {arguments}",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
EOF

# Create web API server (placeholder)
echo "🌐 Creating web API MCP server..."
cat > servers/webapi/webapi_server.py << 'EOF'
#!/usr/bin/env python3
"""
MCP Web API Server - Docker version  
Provides external API access through HTTP API
"""
import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Web API Server", version="1.0.0")

@app.get("/health")