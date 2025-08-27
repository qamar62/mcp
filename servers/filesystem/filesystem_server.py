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