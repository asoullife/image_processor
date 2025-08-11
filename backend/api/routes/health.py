"""Health check API routes."""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging
import asyncio
from datetime import datetime

import sys
from pathlib import Path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from api.dependencies import get_database, get_config
from database.connection import DatabaseManager
from config.config_loader import AppConfig

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint.
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Adobe Stock Image Processor API",
        "version": "1.0.0"
    }

@router.get("/detailed")
async def detailed_health_check(
    db: DatabaseManager = Depends(get_database),
    config: AppConfig = Depends(get_config)
) -> Dict[str, Any]:
    """Detailed health check with database and configuration status.
    
    Args:
        db: Database manager
        config: Application configuration
        
    Returns:
        Detailed health status information
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Adobe Stock Image Processor API",
        "version": "1.0.0",
        "components": {}
    }
    
    # Check database connection
    try:
        async with db.get_session() as session:
            from sqlalchemy import text
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
        
        health_status["components"]["database"] = {
            "status": "healthy",
            "message": "Database connection successful"
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}"
        }
        health_status["status"] = "unhealthy"
    
    # Check configuration
    try:
        # Verify essential configuration is loaded
        assert config.database is not None
        assert config.processing is not None
        
        health_status["components"]["configuration"] = {
            "status": "healthy",
            "message": "Configuration loaded successfully"
        }
    except Exception as e:
        logger.error(f"Configuration health check failed: {e}")
        health_status["components"]["configuration"] = {
            "status": "unhealthy",
            "message": f"Configuration error: {str(e)}"
        }
        health_status["status"] = "unhealthy"
    
    # Check system resources
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_status["components"]["system"] = {
            "status": "healthy",
            "cpu_usage": f"{cpu_percent}%",
            "memory_usage": f"{memory.percent}%",
            "disk_usage": f"{disk.percent}%",
            "available_memory": f"{memory.available / (1024**3):.2f} GB"
        }
        
        # Mark as warning if resources are high
        if cpu_percent > 80 or memory.percent > 80 or disk.percent > 90:
            health_status["components"]["system"]["status"] = "warning"
            if health_status["status"] == "healthy":
                health_status["status"] = "warning"
                
    except ImportError:
        health_status["components"]["system"] = {
            "status": "unknown",
            "message": "psutil not available for system monitoring"
        }
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        health_status["components"]["system"] = {
            "status": "error",
            "message": f"System check failed: {str(e)}"
        }
    
    return health_status

@router.get("/database")
async def database_health_check(
    db: DatabaseManager = Depends(get_database)
) -> Dict[str, Any]:
    """Database-specific health check.
    
    Args:
        db: Database manager
        
    Returns:
        Database health status
    """
    try:
        start_time = datetime.utcnow()
        
        async with db.get_session() as session:
            from sqlalchemy import text
            
            # Test basic connection
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
            
            # Test table existence
            result = await session.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('projects', 'processing_sessions', 'image_results')
            """))
            table_count = result.scalar()
            
            # Test database version
            result = await session.execute(text("SELECT version()"))
            db_version = result.scalar()
            
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "status": "healthy",
            "response_time_seconds": response_time,
            "database_version": db_version,
            "tables_found": table_count,
            "expected_tables": 3,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.get("/ready")
async def readiness_check(
    db: DatabaseManager = Depends(get_database)
) -> Dict[str, Any]:
    """Kubernetes-style readiness check.
    
    Args:
        db: Database manager
        
    Returns:
        Readiness status
    """
    try:
        # Quick database connectivity check
        async with db.get_session() as session:
            from sqlalchemy import text
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")

@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Kubernetes-style liveness check.
    
    Returns:
        Liveness status
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat()
    }