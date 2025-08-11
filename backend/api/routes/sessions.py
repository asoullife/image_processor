"""Processing session API routes."""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from uuid import UUID
import logging

import sys
from pathlib import Path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from api.dependencies import get_session_service
from core.services import SessionService
from api.schemas import SessionResponse, SessionCreate

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", response_model=List[SessionResponse])
async def list_sessions(
    project_id: Optional[UUID] = Query(None),
    status: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    service: SessionService = Depends(get_session_service)
) -> List[SessionResponse]:
    """List processing sessions with optional filtering.
    
    Args:
        project_id: Filter by project ID
        status: Filter by session status
        skip: Number of sessions to skip
        limit: Maximum number of sessions to return
        service: Session service
        
    Returns:
        List of processing sessions
    """
    try:
        sessions = await service.list_sessions(
            project_id=project_id,
            status_filter=status,
            skip=skip,
            limit=limit
        )
        
        return [SessionResponse.from_orm(session) for session in sessions]
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to list sessions")

@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: UUID,
    service: SessionService = Depends(get_session_service)
) -> SessionResponse:
    """Get session by ID.
    
    Args:
        session_id: Session UUID
        service: Session service
        
    Returns:
        Session information
    """
    try:
        session = await service.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionResponse.from_orm(session)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session")

@router.get("/{session_id}/status")
async def get_session_status(
    session_id: UUID,
    service: SessionService = Depends(get_session_service)
) -> Dict[str, Any]:
    """Get detailed session status and progress.
    
    Args:
        session_id: Session UUID
        service: Session service
        
    Returns:
        Session status and progress information
    """
    try:
        status = await service.get_session_status(session_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session status {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session status")

@router.get("/{session_id}/results")
async def get_session_results(
    session_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    decision_filter: Optional[str] = Query(None),
    service: SessionService = Depends(get_session_service)
) -> Dict[str, Any]:
    """Get session processing results.
    
    Args:
        session_id: Session UUID
        skip: Number of results to skip
        limit: Maximum number of results to return
        decision_filter: Filter by decision (approved/rejected)
        service: Session service
        
    Returns:
        Session results with pagination
    """
    try:
        results = await service.get_session_results(
            session_id=session_id,
            skip=skip,
            limit=limit,
            decision_filter=decision_filter
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to get session results {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session results")

@router.post("/{session_id}/resume")
async def resume_session(
    session_id: UUID,
    service: SessionService = Depends(get_session_service)
) -> Dict[str, str]:
    """Resume a paused or interrupted session.
    
    Args:
        session_id: Session UUID
        service: Session service
        
    Returns:
        Resume confirmation
    """
    try:
        success = await service.resume_session(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found or cannot be resumed")
        
        return {"message": "Session resumed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to resume session")

@router.post("/{session_id}/pause")
async def pause_session(
    session_id: UUID,
    service: SessionService = Depends(get_session_service)
) -> Dict[str, str]:
    """Pause a running session.
    
    Args:
        session_id: Session UUID
        service: Session service
        
    Returns:
        Pause confirmation
    """
    try:
        success = await service.pause_session(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found or not running")
        
        return {"message": "Session paused successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to pause session")

@router.delete("/{session_id}")
async def delete_session(
    session_id: UUID,
    service: SessionService = Depends(get_session_service)
) -> Dict[str, str]:
    """Delete a session and its results.
    
    Args:
        session_id: Session UUID
        service: Session service
        
    Returns:
        Deletion confirmation
    """
    try:
        success = await service.delete_session(session_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": "Session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")

@router.get("/concurrent")
async def get_concurrent_sessions(
    service: SessionService = Depends(get_session_service)
) -> List[Dict[str, Any]]:
    """Get all currently running sessions across all projects.
    
    Args:
        service: Session service
        
    Returns:
        List of concurrent sessions
    """
    try:
        sessions = await service.get_concurrent_sessions()
        
        return [
            {
                "id": str(session.id),
                "project_id": str(session.project_id),
                "project_name": session.project.name if session.project else "Unknown",
                "status": session.status,
                "total_images": session.total_images,
                "processed_images": session.processed_images,
                "approved_images": session.approved_images,
                "rejected_images": session.rejected_images,
                "start_time": session.start_time,
                "created_at": session.created_at,
                "updated_at": session.updated_at
            }
            for session in sessions
        ]
        
    except Exception as e:
        logger.error(f"Failed to get concurrent sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get concurrent sessions")

@router.get("/history")
async def get_session_history(
    project_id: Optional[UUID] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    service: SessionService = Depends(get_session_service)
) -> List[Dict[str, Any]]:
    """Get session history with optional project filtering.
    
    Args:
        project_id: Optional project ID to filter by
        limit: Maximum number of sessions to return
        service: Session service
        
    Returns:
        List of historical sessions
    """
    try:
        sessions = await service.get_session_history(project_id, limit)
        
        return [
            {
                "id": str(session.id),
                "project_id": str(session.project_id),
                "project_name": session.project.name if session.project else "Unknown",
                "status": session.status,
                "total_images": session.total_images,
                "processed_images": session.processed_images,
                "approved_images": session.approved_images,
                "rejected_images": session.rejected_images,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "created_at": session.created_at,
                "updated_at": session.updated_at
            }
            for session in sessions
        ]
        
    except Exception as e:
        logger.error(f"Failed to get session history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session history")

@router.put("/{session_id}/progress")
async def update_session_progress(
    session_id: UUID,
    progress_data: Dict[str, Any],
    service: SessionService = Depends(get_session_service)
) -> Dict[str, str]:
    """Update session progress counters.
    
    Args:
        session_id: Session UUID
        progress_data: Progress update data
        service: Session service
        
    Returns:
        Update confirmation
    """
    try:
        success = await service.update_session_progress(
            session_id=session_id,
            processed_count=progress_data.get("processed_count", 0),
            approved_count=progress_data.get("approved_count", 0),
            rejected_count=progress_data.get("rejected_count", 0),
            current_image=progress_data.get("current_image")
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"message": "Progress updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update session progress {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update session progress")