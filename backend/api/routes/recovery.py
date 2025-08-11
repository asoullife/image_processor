"""Recovery and resume API routes."""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from uuid import UUID
import logging

import sys
from pathlib import Path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from api.dependencies import get_recovery_service, get_checkpoint_manager
from core.recovery_service import RecoveryService
from core.checkpoint_manager import CheckpointManager, CheckpointType, ResumeOption
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

class ResumeRequest(BaseModel):
    """Request model for resume operations."""
    resume_option: str  # "continue", "restart_batch", "fresh_start"
    user_confirmed: bool = False

class CheckpointRequest(BaseModel):
    """Request model for manual checkpoint creation."""
    checkpoint_type: str = "manual"  # "image", "batch", "milestone", "manual"
    force: bool = False

@router.get("/crashed-sessions")
async def get_crashed_sessions(
    recovery_service: RecoveryService = Depends(get_recovery_service)
) -> List[Dict[str, Any]]:
    """Get list of crashed sessions that can be recovered.
    
    Args:
        recovery_service: Recovery service instance
        
    Returns:
        List of crashed session information
    """
    try:
        crashed_sessions = await recovery_service.detect_crashes_on_startup()
        return crashed_sessions
        
    except Exception as e:
        logger.error(f"Failed to get crashed sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to detect crashed sessions")

@router.get("/sessions/{session_id}/recovery-options")
async def get_recovery_options(
    session_id: UUID,
    recovery_service: RecoveryService = Depends(get_recovery_service)
) -> Dict[str, Any]:
    """Get recovery options for a specific session.
    
    Args:
        session_id: Session UUID
        recovery_service: Recovery service instance
        
    Returns:
        Recovery options and session information
    """
    try:
        recovery_options = await recovery_service.prepare_recovery_options(session_id)
        
        if not recovery_options:
            raise HTTPException(
                status_code=404, 
                detail="Session not found or not recoverable"
            )
        
        return recovery_options
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recovery options for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recovery options")

@router.post("/sessions/{session_id}/recover")
async def execute_recovery(
    session_id: UUID,
    request: ResumeRequest,
    recovery_service: RecoveryService = Depends(get_recovery_service)
) -> Dict[str, Any]:
    """Execute recovery for a crashed session.
    
    Args:
        session_id: Session UUID
        request: Recovery request with selected option
        recovery_service: Recovery service instance
        
    Returns:
        Recovery execution results
    """
    try:
        success, message, start_index = await recovery_service.execute_recovery(
            session_id=session_id,
            recovery_option=request.resume_option,
            user_confirmed=request.user_confirmed
        )
        
        if not success:
            raise HTTPException(status_code=400, detail=message)
        
        # Verify data integrity after recovery
        integrity_results = await recovery_service.verify_data_integrity_after_recovery(
            session_id, start_index
        )
        
        return {
            "success": True,
            "message": message,
            "start_index": start_index,
            "integrity_check": integrity_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute recovery for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Recovery execution failed")

@router.get("/sessions/{session_id}/checkpoints")
async def get_session_checkpoints(
    session_id: UUID,
    checkpoint_manager: CheckpointManager = Depends(get_checkpoint_manager)
) -> List[Dict[str, Any]]:
    """Get all checkpoints for a session.
    
    Args:
        session_id: Session UUID
        checkpoint_manager: Checkpoint manager instance
        
    Returns:
        List of checkpoint data
    """
    try:
        checkpoints = await checkpoint_manager.get_all_checkpoints(session_id)
        
        return [
            {
                "checkpoint_id": cp.checkpoint_id,
                "checkpoint_type": cp.checkpoint_type.value,
                "processed_count": cp.session_state.processed_count,
                "approved_count": cp.session_state.approved_count,
                "rejected_count": cp.session_state.rejected_count,
                "created_at": cp.created_at.isoformat(),
                "can_resume": cp.can_resume,
                "recovery_options": [opt.value for opt in cp.recovery_options],
                "integrity_hash": cp.integrity_hash
            }
            for cp in checkpoints
        ]
        
    except Exception as e:
        logger.error(f"Failed to get checkpoints for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session checkpoints")

@router.get("/sessions/{session_id}/latest-checkpoint")
async def get_latest_checkpoint(
    session_id: UUID,
    checkpoint_manager: CheckpointManager = Depends(get_checkpoint_manager)
) -> Dict[str, Any]:
    """Get the latest checkpoint for a session.
    
    Args:
        session_id: Session UUID
        checkpoint_manager: Checkpoint manager instance
        
    Returns:
        Latest checkpoint data
    """
    try:
        checkpoint = await checkpoint_manager.get_latest_checkpoint(session_id)
        
        if not checkpoint:
            raise HTTPException(status_code=404, detail="No checkpoint found for session")
        
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_type": checkpoint.checkpoint_type.value,
            "session_state": {
                "processed_count": checkpoint.session_state.processed_count,
                "approved_count": checkpoint.session_state.approved_count,
                "rejected_count": checkpoint.session_state.rejected_count,
                "current_batch": checkpoint.session_state.current_batch,
                "current_image_index": checkpoint.session_state.current_image_index,
                "last_processing_rate": checkpoint.session_state.last_processing_rate,
                "memory_usage_mb": checkpoint.session_state.memory_usage_mb,
                "gpu_memory_usage_mb": checkpoint.session_state.gpu_memory_usage_mb,
                "error_count": checkpoint.session_state.error_count,
                "last_error": checkpoint.session_state.last_error
            },
            "created_at": checkpoint.created_at.isoformat(),
            "can_resume": checkpoint.can_resume,
            "recovery_options": [opt.value for opt in checkpoint.recovery_options],
            "integrity_hash": checkpoint.integrity_hash
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get latest checkpoint for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get latest checkpoint")

@router.post("/sessions/{session_id}/checkpoint")
async def create_manual_checkpoint(
    session_id: UUID,
    request: CheckpointRequest,
    checkpoint_manager: CheckpointManager = Depends(get_checkpoint_manager)
) -> Dict[str, Any]:
    """Create a manual checkpoint for a session.
    
    Args:
        session_id: Session UUID
        request: Checkpoint creation request
        checkpoint_manager: Checkpoint manager instance
        
    Returns:
        Created checkpoint information
    """
    try:
        # Convert string to CheckpointType enum
        checkpoint_type_map = {
            "image": CheckpointType.IMAGE,
            "batch": CheckpointType.BATCH,
            "milestone": CheckpointType.MILESTONE,
            "manual": CheckpointType.MANUAL,
            "emergency": CheckpointType.EMERGENCY
        }
        
        checkpoint_type = checkpoint_type_map.get(request.checkpoint_type, CheckpointType.MANUAL)
        
        checkpoint_data = await checkpoint_manager.create_session_checkpoint(
            session_id=session_id,
            checkpoint_type=checkpoint_type,
            force=request.force
        )
        
        if not checkpoint_data:
            raise HTTPException(status_code=400, detail="Failed to create checkpoint")
        
        return {
            "success": True,
            "checkpoint_id": checkpoint_data.checkpoint_id,
            "checkpoint_type": checkpoint_data.checkpoint_type.value,
            "processed_count": checkpoint_data.session_state.processed_count,
            "created_at": checkpoint_data.created_at.isoformat(),
            "message": "Checkpoint created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create manual checkpoint for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create checkpoint")

@router.post("/sessions/{session_id}/resume")
async def resume_session(
    session_id: UUID,
    request: ResumeRequest,
    checkpoint_manager: CheckpointManager = Depends(get_checkpoint_manager)
) -> Dict[str, Any]:
    """Resume a session from checkpoint.
    
    Args:
        session_id: Session UUID
        request: Resume request with selected option
        checkpoint_manager: Checkpoint manager instance
        
    Returns:
        Resume execution results
    """
    try:
        # Get latest checkpoint
        latest_checkpoint = await checkpoint_manager.get_latest_checkpoint(session_id)
        if not latest_checkpoint:
            raise HTTPException(status_code=404, detail="No checkpoint found for session")
        
        # Verify checkpoint integrity
        integrity_ok = await checkpoint_manager.verify_checkpoint_integrity(latest_checkpoint)
        if not integrity_ok:
            raise HTTPException(status_code=400, detail="Checkpoint data is corrupted")
        
        # Convert resume option string to enum
        resume_option_map = {
            "continue": ResumeOption.CONTINUE,
            "restart_batch": ResumeOption.RESTART_BATCH,
            "fresh_start": ResumeOption.FRESH_START
        }
        
        resume_option = resume_option_map.get(request.resume_option)
        if not resume_option:
            raise HTTPException(status_code=400, detail=f"Invalid resume option: {request.resume_option}")
        
        # Execute resume
        success, start_index = await checkpoint_manager.execute_resume(
            session_id, resume_option, latest_checkpoint
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to resume session")
        
        return {
            "success": True,
            "message": f"Session resumed successfully from image {start_index + 1}",
            "start_index": start_index,
            "resume_option": request.resume_option
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Session resume failed")

@router.delete("/sessions/{session_id}/checkpoints")
async def cleanup_old_checkpoints(
    session_id: UUID,
    keep_count: int = Query(5, ge=1, le=20),
    checkpoint_manager: CheckpointManager = Depends(get_checkpoint_manager)
) -> Dict[str, str]:
    """Clean up old checkpoints for a session.
    
    Args:
        session_id: Session UUID
        keep_count: Number of recent checkpoints to keep
        checkpoint_manager: Checkpoint manager instance
        
    Returns:
        Cleanup confirmation
    """
    try:
        success = await checkpoint_manager.cleanup_old_checkpoints(session_id, keep_count)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to cleanup checkpoints")
        
        return {"message": f"Old checkpoints cleaned up, kept {keep_count} most recent"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cleanup checkpoints for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup checkpoints")

@router.post("/sessions/{session_id}/verify-integrity")
async def verify_session_integrity(
    session_id: UUID,
    recovery_service: RecoveryService = Depends(get_recovery_service)
) -> Dict[str, Any]:
    """Verify data integrity for a session.
    
    Args:
        session_id: Session UUID
        recovery_service: Recovery service instance
        
    Returns:
        Integrity verification results
    """
    try:
        # Use start_index = 0 for general integrity check
        integrity_results = await recovery_service.verify_data_integrity_after_recovery(
            session_id, 0
        )
        
        return integrity_results
        
    except Exception as e:
        logger.error(f"Failed to verify integrity for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Integrity verification failed")

@router.get("/recovery-statistics")
async def get_recovery_statistics(
    recovery_service: RecoveryService = Depends(get_recovery_service)
) -> Dict[str, Any]:
    """Get recovery operation statistics.
    
    Args:
        recovery_service: Recovery service instance
        
    Returns:
        Recovery statistics
    """
    try:
        stats = await recovery_service.get_recovery_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get recovery statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recovery statistics")

@router.post("/sessions/{session_id}/emergency-checkpoint")
async def create_emergency_checkpoint(
    session_id: UUID,
    error_message: str = Query(..., description="Error message that triggered emergency checkpoint"),
    checkpoint_manager: CheckpointManager = Depends(get_checkpoint_manager)
) -> Dict[str, Any]:
    """Create an emergency checkpoint when an error occurs.
    
    Args:
        session_id: Session UUID
        error_message: Error message that triggered the checkpoint
        checkpoint_manager: Checkpoint manager instance
        
    Returns:
        Emergency checkpoint information
    """
    try:
        checkpoint_data = await checkpoint_manager.create_emergency_checkpoint(
            session_id, error_message
        )
        
        if not checkpoint_data:
            raise HTTPException(status_code=400, detail="Failed to create emergency checkpoint")
        
        return {
            "success": True,
            "checkpoint_id": checkpoint_data.checkpoint_id,
            "checkpoint_type": checkpoint_data.checkpoint_type.value,
            "processed_count": checkpoint_data.session_state.processed_count,
            "error_message": error_message,
            "created_at": checkpoint_data.created_at.isoformat(),
            "message": "Emergency checkpoint created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create emergency checkpoint for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create emergency checkpoint")