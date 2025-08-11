"""FastAPI dependency injection system."""

from fastapi import Depends, HTTPException, Request
from typing import AsyncGenerator, Optional
import logging

import sys
from pathlib import Path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from database.connection import DatabaseManager, get_db_session
from config.config_loader import AppConfig, get_config as load_config
from core.services import ProjectService, SessionService, AnalysisService
from core.checkpoint_manager import CheckpointManager
from core.recovery_service import RecoveryService

logger = logging.getLogger(__name__)

# Global instances
_db_manager: Optional[DatabaseManager] = None
_config: Optional[AppConfig] = None
_checkpoint_manager: Optional[CheckpointManager] = None
_recovery_service: Optional[RecoveryService] = None

async def get_database() -> AsyncGenerator[DatabaseManager, None]:
    """Get database manager dependency.
    
    Yields:
        DatabaseManager instance
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()
    
    try:
        yield _db_manager
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database connection error")

def get_config(request: Request = None) -> AppConfig:
    """Get application configuration dependency.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Application configuration
    """
    global _config
    
    # Try to get config from app state first
    if request and hasattr(request.app.state, 'config'):
        return request.app.state.config
    
    # Load config if not cached
    if _config is None:
        try:
            _config = load_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise HTTPException(status_code=500, detail="Configuration error")
    
    return _config

async def get_project_service(
    db: DatabaseManager = Depends(get_database),
    config: AppConfig = Depends(get_config)
) -> ProjectService:
    """Get project service dependency.
    
    Args:
        db: Database manager
        config: Application configuration
        
    Returns:
        Project service instance
    """
    return ProjectService(db, config)

async def get_session_service(
    db: DatabaseManager = Depends(get_database),
    config: AppConfig = Depends(get_config)
) -> SessionService:
    """Get session service dependency.
    
    Args:
        db: Database manager
        config: Application configuration
        
    Returns:
        Session service instance
    """
    return SessionService(db, config)

async def get_analysis_service(
    config: AppConfig = Depends(get_config)
) -> AnalysisService:
    """Get analysis service dependency.
    
    Args:
        config: Application configuration
        
    Returns:
        Analysis service instance
    """
    return AnalysisService(config)

async def get_checkpoint_manager(
    db: DatabaseManager = Depends(get_database),
    config: AppConfig = Depends(get_config)
) -> CheckpointManager:
    """Get checkpoint manager dependency.
    
    Args:
        db: Database manager
        config: Application configuration
        
    Returns:
        Checkpoint manager instance
    """
    global _checkpoint_manager
    
    if _checkpoint_manager is None:
        checkpoint_interval = getattr(config.processing, 'checkpoint_interval', 10)
        _checkpoint_manager = CheckpointManager(db, checkpoint_interval)
    
    return _checkpoint_manager

async def get_recovery_service(
    db: DatabaseManager = Depends(get_database),
    checkpoint_manager: CheckpointManager = Depends(get_checkpoint_manager)
) -> RecoveryService:
    """Get recovery service dependency.
    
    Args:
        db: Database manager
        checkpoint_manager: Checkpoint manager
        
    Returns:
        Recovery service instance
    """
    global _recovery_service
    
    if _recovery_service is None:
        _recovery_service = RecoveryService(db, checkpoint_manager)
    
    return _recovery_service