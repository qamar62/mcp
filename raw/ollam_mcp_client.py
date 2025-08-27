import asyncio
import json
import subprocess
import requests
from mcp.client import ClientSession
from mcp.client.stdio import stdio_client

class OllamaMCPClient:
    def __init__(self, ollama_url="http://localhost:11434"):
        self.ollama_url = ollama_url
        self.session = None
    
    async def connect_to_mcp_server(self, server_command):
        """Connect to MCP server"""
        self.session = await stdio_client(server_command)
        await self.session.__aenter__()
        
        # Initialize the session
        init_result = await self.session.initialize()
        print(f"Connected to MCP server: {init_result}")
    
    async def get_available_resources(self):
        """Get list of available resources from MCP server"""
        if not self.session:
            raise Exception("Not connected to MCP server")
        
        resources = await self.session.list_resources()
        return resources.resources
    
    async def read_resource(self, uri):
        """Read resource content"""
        if not self.session:
            raise Exception("Not connected to MCP server")
        
        result = await self.session.read_resource(uri)
        return result.contents[0].text if result.contents else ""
    
    def query_ollama(self, prompt, model="llama2"):
        """Send query to Ollama"""
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json().get("response", "")
    
    async def enhanced_query(self, question, model="llama2"):
        """Query Ollama with MCP context"""
        # Get available resources
        resources = await self.get_available_resources()
        
        # Build context from resources
        context = "Available resources:\n"
        for resource in resources[:5]:  # Limit to first 5 resources
            try:
                content = await self.read_resource(resource.uri)
                context += f"\n--- {resource.name} ---\n"
                context += content[:500] + "...\n"  # Limit content length
            except:
                continue
        
        # Create enhanced prompt
        enhanced_prompt = f"""
Context from available resources:
{context}

Question: {question}

Please answer the question using the context provided above.
"""
        
        return self.query_ollama(enhanced_prompt, model)

# Usage example
async def main():
    client = OllamaMCPClient()
    
    # Start MCP server (adjust path as needed)
    server_command = ["python3", "filesystem_server.py"]
    
    await client.connect_to_mcp_server(server_command)
    
    # Query with MCP context
    response = await client.enhanced_query(
        "What files are available and what do they contain?",
        model="llama2"  # or whatever model you have installed
    )
    
    print("Ollama Response with MCP Context:")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())