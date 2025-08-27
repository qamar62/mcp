# database_server.py
import sqlite3
from mcp.server import Server

app = Server("database-server")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="query_database",
            description="Execute SQL query on local database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "database": {"type": "string"}
                },
                "required": ["query", "database"]
            }
        )
    ]

# Add database query implementation...