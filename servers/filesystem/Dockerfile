# servers/filesystem/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    file \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY . .

# Create directories
RUN mkdir -p /data /shared

# Expose port for health checks
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8081/health')" || exit 1

# Run the MCP filesystem server
CMD ["python", "filesystem_server.py"]