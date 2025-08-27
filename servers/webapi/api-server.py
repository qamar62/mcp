# api_server.py
import requests
from mcp.server import Server

app = Server("api-server")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="fetch_weather",
            description="Get weather information",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        )
    ]

# Add API call implementation...