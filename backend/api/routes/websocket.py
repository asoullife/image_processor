"""WebSocket API routes for Socket.IO integration."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional
import logging

from ...websocket.socketio_manager import socketio_manager, ProgressData, ErrorData, CompletionData
from ..dependencies import get_database
from ...database.models import ProcessingSession

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/sessions/{session_id}/clients")
async def get_connected_clients(session_id: str):
    """Get number of connected clients for a session."""
    try:
        client_count = socketio_manager.get_connected_clients_count(session_id)
        return {
            "session_id": session_id,
            "connected_clients": client_count,
            "has_active_connections": client_count > 0
        }
    except Exception as e:
        logger.error(f"Error getting connected clients: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/broadcast/progress")
async def broadcast_progress(session_id: str, progress_data: ProgressData):
    """Manually broadcast progress update to session clients."""
    try:
        await socketio_manager.broadcast_progress(session_id, progress_data)
        return {
            "status": "success",
            "message": f"Progress broadcasted to session {session_id}",
            "clients_notified": socketio_manager.get_connected_clients_count(session_id)
        }
    except Exception as e:
        logger.error(f"Error broadcasting progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/broadcast/error")
async def broadcast_error(session_id: str, error_data: ErrorData):
    """Manually broadcast error to session clients."""
    try:
        await socketio_manager.broadcast_error(session_id, error_data)
        return {
            "status": "success",
            "message": f"Error broadcasted to session {session_id}",
            "clients_notified": socketio_manager.get_connected_clients_count(session_id)
        }
    except Exception as e:
        logger.error(f"Error broadcasting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/broadcast/completion")
async def broadcast_completion(session_id: str, completion_data: CompletionData):
    """Manually broadcast completion to session clients."""
    try:
        await socketio_manager.broadcast_completion(session_id, completion_data)
        return {
            "status": "success",
            "message": f"Completion broadcasted to session {session_id}",
            "clients_notified": socketio_manager.get_connected_clients_count(session_id)
        }
    except Exception as e:
        logger.error(f"Error broadcasting completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sessions/{session_id}/broadcast/stage")
async def broadcast_stage_change(session_id: str, stage: str, message: str = ""):
    """Manually broadcast stage change to session clients."""
    try:
        await socketio_manager.broadcast_stage_change(session_id, stage, message)
        return {
            "status": "success",
            "message": f"Stage change broadcasted to session {session_id}",
            "stage": stage,
            "clients_notified": socketio_manager.get_connected_clients_count(session_id)
        }
    except Exception as e:
        logger.error(f"Error broadcasting stage change: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sessions/{session_id}/cleanup")
async def cleanup_session(session_id: str):
    """Clean up session data and disconnect all clients."""
    try:
        await socketio_manager.cleanup_session(session_id)
        return {
            "status": "success",
            "message": f"Session {session_id} cleaned up successfully"
        }
    except Exception as e:
        logger.error(f"Error cleaning up session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/progress")
async def get_cached_progress(session_id: str):
    """Get cached progress data for a session."""
    try:
        if session_id in socketio_manager.progress_cache:
            progress_data = socketio_manager.progress_cache[session_id]
            return {
                "status": "success",
                "session_id": session_id,
                "progress": progress_data.dict(),
                "has_cached_data": True
            }
        else:
            return {
                "status": "success",
                "session_id": session_id,
                "progress": None,
                "has_cached_data": False,
                "message": "No cached progress data found"
            }
    except Exception as e:
        logger.error(f"Error getting cached progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def websocket_health():
    """Check WebSocket service health."""
    try:
        total_sessions = len(socketio_manager.progress_cache)
        total_clients = sum(len(clients) for clients in socketio_manager.connected_clients.values())
        
        return {
            "status": "healthy",
            "service": "websocket",
            "active_sessions": total_sessions,
            "connected_clients": total_clients,
            "redis_adapter": socketio_manager.progress_cache is not None
        }
    except Exception as e:
        logger.error(f"Error checking WebSocket health: {e}")
        raise HTTPException(status_code=500, detail=str(e))