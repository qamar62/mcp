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
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://192.168.1.132:11434')
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
    model: Optional[str] = "mistral:7b"
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

async def query_ollama(prompt: str, model: str = "mistral:7b") -> str:
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
