-- Initialize MCP database schema
CREATE TABLE IF NOT EXISTS mcp_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(50) DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS mcp_resources (
    id SERIAL PRIMARY KEY,
    uri VARCHAR(500) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mcp_tools (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    agent_name VARCHAR(100) NOT NULL,
    description TEXT,
    parameters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mcp_queries (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    model_used VARCHAR(100),
    agents_used TEXT[],
    response TEXT,
    context_length INTEGER,
    response_time FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_mcp_sessions_agent ON mcp_sessions(agent_name);
CREATE INDEX IF NOT EXISTS idx_mcp_resources_agent ON mcp_resources(agent_name);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_agent ON mcp_tools(agent_name);
CREATE INDEX IF NOT EXISTS idx_mcp_queries_created ON mcp_queries(created_at);
