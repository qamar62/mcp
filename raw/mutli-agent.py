class MultiAgentOrchestrator:
    def __init__(self):
        self.agents = {}
    
    async def add_agent(self, name, server_command):
        """Add a new MCP agent"""
        session = await stdio_client(server_command)
        await session.__aenter__()
        await session.initialize()
        self.agents[name] = session
    
    async def route_query(self, query, agent_name=None):
        """Route query to appropriate agent or all agents"""
        if agent_name and agent_name in self.agents:
            # Query specific agent
            return await self._query_agent(self.agents[agent_name], query)
        else:
            # Query all agents and combine results
            results = {}
            for name, session in self.agents.items():
                try:
                    results[name] = await self._query_agent(session, query)
                except:
                    continue
            return results
    
    async def _query_agent(self, session, query):
        """Query individual agent"""
        # Implementation depends on your specific needs
        resources = await session.list_resources()
        return {"resources": len(resources.resources)}

# Usage
async def multi_agent_example():
    orchestrator = MultiAgentOrchestrator()
    
    # Add different agents
    await orchestrator.add_agent("filesystem", ["python3", "filesystem_server.py"])
    await orchestrator.add_agent("database", ["python3", "database_server.py"])
    await orchestrator.add_agent("api", ["python3", "api_server.py"])
    
    # Route queries
    results = await orchestrator.route_query("What data is available?")
    print(results)