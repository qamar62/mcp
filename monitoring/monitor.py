#!/usr/bin/env python3
"""
MCP Monitoring Server - Docker version
Provides monitoring and metrics for MCP infrastructure
"""
import os
import asyncio
import logging
from datetime import datetime
from typing import Dict, List
import psutil
import aiohttp
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Monitoring Server", version="1.0.0")

# Configuration
MONITOR_INTERVAL = int(os.getenv('MONITOR_INTERVAL', '30'))
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379')
ALERT_WEBHOOK_URL = os.getenv('ALERT_WEBHOOK_URL', '')

# MCP Services to monitor
MCP_SERVICES = {
    'orchestrator': 'http://mcp-orchestrator:8080',
    'filesystem': 'http://mcp-filesystem:8081',
    'database': 'http://mcp-database:8082',
    'webapi': 'http://mcp-webapi:8083'
}

# Global variables
redis_client = None
monitoring_data = {}

class ServiceStatus(BaseModel):
    name: str
    status: str
    response_time: float
    last_check: str

@app.on_event("startup")
async def startup_event():
    """Initialize monitoring on startup"""
    global redis_client
    
    try:
        # Connect to Redis
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        await redis_client.ping()
        logger.info("Monitoring connected to Redis")
        
        # Start background monitoring
        asyncio.create_task(monitor_services())
        
        logger.info("MCP Monitoring started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "server": "monitoring",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status")
async def get_status():
    """Get monitoring status"""
    return {
        "monitoring": "running",
        "services_monitored": len(MCP_SERVICES),
        "monitor_interval": MONITOR_INTERVAL,
        "redis_connected": redis_client is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/services")
async def get_services_status():
    """Get status of all monitored services"""
    return {
        "services": monitoring_data,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_total_gb": round(disk.total / (1024**3), 2)
            },
            "services": monitoring_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def monitor_services():
    """Background task to monitor MCP services"""
    global monitoring_data
    
    while True:
        try:
            for service_name, service_url in MCP_SERVICES.items():
                status = await check_service_health(service_name, service_url)
                monitoring_data[service_name] = status
                
                # Store in Redis for persistence
                if redis_client:
                    await redis_client.hset(
                        "mcp:monitoring:services", 
                        service_name, 
                        f"{status['status']}:{status['response_time']}"
                    )
            
            logger.info(f"Monitoring check completed for {len(MCP_SERVICES)} services")
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
        
        await asyncio.sleep(MONITOR_INTERVAL)

async def check_service_health(service_name: str, service_url: str) -> Dict:
    """Check health of a specific service"""
    start_time = datetime.now()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{service_url}/health",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                
                if response.status == 200:
                    status = "healthy"
                else:
                    status = f"unhealthy (HTTP {response.status})"
                
                return {
                    "status": status,
                    "response_time": round(response_time, 3),
                    "last_check": end_time.isoformat(),
                    "url": service_url
                }
                
    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "response_time": 10.0,
            "last_check": datetime.now().isoformat(),
            "url": service_url
        }
    except Exception as e:
        return {
            "status": f"error: {str(e)}",
            "response_time": 0.0,
            "last_check": datetime.now().isoformat(),
            "url": service_url
        }

@app.get("/alerts")
async def get_alerts():
    """Get current alerts"""
    alerts = []
    
    for service_name, data in monitoring_data.items():
        if data.get('status') != 'healthy':
            alerts.append({
                "service": service_name,
                "status": data.get('status'),
                "response_time": data.get('response_time'),
                "last_check": data.get('last_check'),
                "severity": "critical" if "error" in data.get('status', '') else "warning"
            })
    
    return {
        "alerts": alerts,
        "count": len(alerts),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
