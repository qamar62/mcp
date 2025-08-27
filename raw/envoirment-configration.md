{
  "ollama": {
    "url": "http://localhost:11434",
    "default_model": "llama2"
  },
  "mcp_servers": [
    {
      "name": "filesystem",
      "command": ["python3", "filesystem_server.py"],
      "description": "Local file system access"
    },
    {
      "name": "database",
      "command": ["python3", "database_server.py"],
      "description": "Database query agent"
    }
  ]
}