"""Application startup sequence with crash detection and recovery."""

import logging
import asyncio
from typing import List, Dict, Any
from datetime import datetime

import sys
from pathlib import Path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from database.connection import DatabaseManager
from core.checkpoint_manager import CheckpointManager
from core.recovery_service import RecoveryService
from config.config_loader import get_config

logger = logging.getLogger(__name__)

async def startup_sequence():
    """Execute application startup sequence with crash detection.
    
    This function:
    1. Initializes database connections
    2. Detects crashed sessions from previous runs
    3. Prepares recovery options for crashed sessions
    4. Logs recovery information for user awareness
    """
    try:
        logger.info("Starting Adobe Stock Image Processor startup sequence")
        
        # Initialize configuration
        config = get_config()
        
        # Initialize database
        db_manager = DatabaseManager()
        await db_manager.initialize()
        logger.info("Database connection initialized")
        
        # Initialize checkpoint and recovery services
        checkpoint_interval = getattr(config.processing, 'checkpoint_interval', 10)
        checkpoint_manager = CheckpointManager(db_manager, checkpoint_interval)
        recovery_service = RecoveryService(db_manager, checkpoint_manager)
        
        # Detect crashed sessions
        logger.info("Detecting crashed sessions from previous runs...")
        crashed_sessions = await recovery_service.detect_crashes_on_startup()
        
        if crashed_sessions:
            logger.warning(f"Found {len(crashed_sessions)} crashed sessions")
            
            # Log details about each crashed session
            for session_info in crashed_sessions:
                session_id = session_info["session_id"]
                crash_type = session_info["crash_type"]
                processed_count = session_info["processed_count"]
                total_images = session_info["total_images"]
                can_recover = session_info["can_recover"]
                
                logger.warning(
                    f"Crashed session {session_id}: "
                    f"type={crash_type}, "
                    f"progress={processed_count}/{total_images}, "
                    f"recoverable={can_recover}"
                )
                
                if can_recover:
                    # Prepare recovery options for user
                    recovery_options = await recovery_service.prepare_recovery_options(
                        session_id
                    )
                    
                    if recovery_options:
                        logger.info(
                            f"Recovery options prepared for session {session_id}. "
                            f"Available options: {[opt['option'] for opt in recovery_options.get('available_options', [])]}"
                        )
                else:
                    logger.error(
                        f"Session {session_id} cannot be recovered due to corrupted checkpoint data"
                    )
            
            # Create summary for frontend
            recovery_summary = await _create_recovery_summary(crashed_sessions)
            logger.info(f"Recovery summary: {recovery_summary}")
            
        else:
            logger.info("No crashed sessions detected - clean startup")
        
        # Cleanup old checkpoints to prevent database bloat
        await _cleanup_old_checkpoints(checkpoint_manager)
        
        # Verify database integrity
        await _verify_database_integrity(db_manager)
        
        logger.info("Startup sequence completed successfully")
        
    except Exception as e:
        logger.error(f"Startup sequence failed: {e}")
        raise

async def _create_recovery_summary(crashed_sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a summary of crashed sessions for frontend display.
    
    Args:
        crashed_sessions: List of crashed session information
        
    Returns:
        Recovery summary dictionary
    """
    try:
        total_crashed = len(crashed_sessions)
        recoverable_count = sum(1 for s in crashed_sessions if s["can_recover"])
        
        # Group by crash type
        crash_types = {}
        for session in crashed_sessions:
            crash_type = session["crash_type"]
            crash_types[crash_type] = crash_types.get(crash_type, 0) + 1
        
        # Calculate total lost progress
        total_processed = sum(s["processed_count"] for s in crashed_sessions)
        total_images = sum(s["total_images"] for s in crashed_sessions)
        
        return {
            "total_crashed_sessions": total_crashed,
            "recoverable_sessions": recoverable_count,
            "non_recoverable_sessions": total_crashed - recoverable_count,
            "crash_types": crash_types,
            "total_processed_images": total_processed,
            "total_images_in_crashed_sessions": total_images,
            "recovery_rate": (recoverable_count / total_crashed * 100) if total_crashed > 0 else 0,
            "detected_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to create recovery summary: {e}")
        return {}

async def _cleanup_old_checkpoints(checkpoint_manager: CheckpointManager):
    """Clean up old checkpoints to prevent database bloat.
    
    Args:
        checkpoint_manager: Checkpoint manager instance
    """
    try:
        logger.info("Cleaning up old checkpoints...")
        
        # This would require getting all sessions and cleaning up their checkpoints
        # For now, we'll implement a basic cleanup strategy
        
        # Get all sessions from database
        from database.models import ProcessingSession
        from sqlalchemy import select
        
        async with checkpoint_manager.db_manager.get_session() as session:
            stmt = select(ProcessingSession)
            result = await session.execute(stmt)
            all_sessions = result.scalars().all()
            
            cleanup_count = 0
            for processing_session in all_sessions:
                # Keep only 5 most recent checkpoints per session
                success = await checkpoint_manager.cleanup_old_checkpoints(
                    processing_session.id, keep_count=5
                )
                if success:
                    cleanup_count += 1
            
            logger.info(f"Cleaned up checkpoints for {cleanup_count} sessions")
        
    except Exception as e:
        logger.warning(f"Checkpoint cleanup failed: {e}")

async def _verify_database_integrity(db_manager: DatabaseManager):
    """Verify basic database integrity.
    
    Args:
        db_manager: Database manager instance
    """
    try:
        logger.info("Verifying database integrity...")
        
        async with db_manager.get_session() as session:
            # Check if all required tables exist
            from database.models import (
                Project, ProcessingSession, ImageResult, 
                Checkpoint, SimilarityGroup, ProcessingLog
            )
            
            # Simple existence check by counting records
            from sqlalchemy import func, select
            
            tables_to_check = [
                ("projects", Project),
                ("processing_sessions", ProcessingSession),
                ("image_results", ImageResult),
                ("checkpoints", Checkpoint),
                ("similarity_groups", SimilarityGroup),
                ("processing_logs", ProcessingLog)
            ]
            
            for table_name, model_class in tables_to_check:
                try:
                    stmt = select(func.count(model_class.id))
                    result = await session.execute(stmt)
                    count = result.scalar()
                    logger.debug(f"Table {table_name}: {count} records")
                except Exception as e:
                    logger.error(f"Table {table_name} integrity check failed: {e}")
                    raise
            
            logger.info("Database integrity verification completed")
        
    except Exception as e:
        logger.error(f"Database integrity verification failed: {e}")
        raise

async def detect_and_prepare_recovery():
    """Standalone function to detect crashes and prepare recovery options.
    
    This can be called from the main application or CLI tools.
    
    Returns:
        List of crashed sessions with recovery options
    """
    try:
        # Initialize services
        config = get_config()
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        checkpoint_interval = getattr(config.processing, 'checkpoint_interval', 10)
        checkpoint_manager = CheckpointManager(db_manager, checkpoint_interval)
        recovery_service = RecoveryService(db_manager, checkpoint_manager)
        
        # Detect crashes
        crashed_sessions = await recovery_service.detect_crashes_on_startup()
        
        # Prepare recovery options for each
        recovery_prepared_sessions = []
        for session_info in crashed_sessions:
            if session_info["can_recover"]:
                session_id = session_info["session_id"]
                recovery_options = await recovery_service.prepare_recovery_options(session_id)
                
                if recovery_options:
                    session_info["recovery_options"] = recovery_options
                    recovery_prepared_sessions.append(session_info)
        
        return recovery_prepared_sessions
        
    except Exception as e:
        logger.error(f"Failed to detect and prepare recovery: {e}")
        return []

if __name__ == "__main__":
    # Allow running startup sequence standalone
    asyncio.run(startup_sequence())