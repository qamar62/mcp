import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent
import os

app = Server("filesystem-server")

@app.list_resources()
async def list_resources():
    """List available files in current directory"""
    files = []
    for file in os.listdir('.'):
        if os.path.isfile(file):
            files.append(Resource(
                uri=f"file://{os.path.abspath(file)}",
                name=file,
                mimeType="text/plain"
            ))
    return files

@app.read_resource()
async def read_resource(uri: str):
    """Read file contents"""
    if uri.startswith("file://"):
        file_path = uri[7:]  # Remove 'file://' prefix
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            return TextContent(type="text", text=content)
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")

@app.list_tools()
async def list_tools():
    """List available tools"""
    return [
        Tool(
            name="create_file",
            description="Create a new file with specified content",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["filename", "content"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    """Execute tool calls"""
    if name == "create_file":
        filename = arguments.get("filename")
        content = arguments.get("content")
        
        with open(filename, 'w') as f:
            f.write(content)
        
        return TextContent(
            type="text",
            text=f"File '{filename}' created successfully"
        )

if __name__ == "__main__":
    asyncio.run(stdio_server(app))