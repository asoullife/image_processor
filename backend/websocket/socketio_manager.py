"""Socket.IO manager for real-time communication."""

import socketio
import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

class ProgressData(BaseModel):
    """Progress data structure for real-time updates."""
    session_id: str
    current_image: int
    total_images: int
    percentage: float
    current_filename: str
    approved_count: int
    rejected_count: int
    processing_speed: float  # images per second
    estimated_completion: Optional[datetime] = None
    current_stage: str  # "scanning", "processing", "reviewing", "complete"
    error_message: Optional[str] = None
    
    # Performance metrics
    memory_usage_mb: float = 0.0
    gpu_usage_percent: float = 0.0
    cpu_usage_percent: float = 0.0
    batch_processing_time: float = 0.0
    avg_image_processing_time: float = 0.0
    
    # Milestone tracking
    milestone_reached: Optional[str] = None  # "25%", "50%", "75%", "1000_images", etc.
    elapsed_time: float = 0.0  # Total elapsed time in seconds
    
    # Additional progress details
    current_batch: int = 0
    total_batches: int = 0
    images_per_second: float = 0.0
    eta_seconds: Optional[float] = None

class ErrorData(BaseModel):
    """Error data structure for real-time updates."""
    session_id: str
    error_type: str
    error_message: str
    timestamp: datetime
    recoverable: bool

class CompletionData(BaseModel):
    """Completion data structure for real-time updates."""
    session_id: str
    total_processed: int
    total_approved: int
    total_rejected: int
    processing_time: float
    completion_time: datetime
    output_folder: str
    
    # Performance summary
    avg_processing_speed: float  # images per second
    peak_memory_usage: float  # MB
    avg_gpu_usage: float  # percentage
    total_batches: int
    approval_rate: float  # percentage
    
    # Final statistics
    quality_issues_count: int = 0
    defect_issues_count: int = 0
    similarity_issues_count: int = 0
    compliance_issues_count: int = 0

# Create Socket.IO server with ASGI support
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    logger=True,
    engineio_logger=False,  # Reduce noise in logs
    ping_timeout=60,
    ping_interval=25
)

class MilestoneData(BaseModel):
    """Milestone notification data structure."""
    session_id: str
    milestone_type: str  # "percentage", "count", "time"
    milestone_value: str  # "25%", "1000_images", "1_hour"
    current_progress: int
    total_progress: int
    message: str
    timestamp: datetime
    performance_snapshot: Dict[str, Any]

class SocketIOManager:
    """Manage real-time progress updates via Socket.IO."""

    def __init__(self):
        self.progress_cache: Dict[str, ProgressData] = {}
        self.connected_clients: Dict[str, List[str]] = {}  # session_id -> [client_ids]
        self.client_sessions: Dict[str, str] = {}  # client_id -> session_id
        self.milestone_tracker: Dict[str, set] = {}  # session_id -> set of reached milestones

    async def join_session_room(self, sid: str, session_id: str):
        """Add client to session room."""
        try:
            await sio.enter_room(sid, f"session_{session_id}")
            
            # Track client-session mapping
            self.client_sessions[sid] = session_id
            if session_id not in self.connected_clients:
                self.connected_clients[session_id] = []
            if sid not in self.connected_clients[session_id]:
                self.connected_clients[session_id].append(sid)

            logger.info(f"Client {sid} joined session {session_id}")

            # Send cached progress if available
            if session_id in self.progress_cache:
                await sio.emit(
                    'progress_update',
                    self.progress_cache[session_id].dict(),
                    room=sid
                )
                logger.debug(f"Sent cached progress to client {sid}")

            # Send connection confirmation
            await sio.emit(
                'session_joined',
                {
                    'session_id': session_id,
                    'status': 'joined',
                    'timestamp': datetime.now().isoformat()
                },
                room=sid
            )

        except Exception as e:
            logger.error(f"Error joining session room: {e}")
            await sio.emit('error', {'message': f'Failed to join session: {str(e)}'}, room=sid)

    async def leave_session_room(self, sid: str, session_id: str):
        """Remove client from session room."""
        try:
            await sio.leave_room(sid, f"session_{session_id}")
            
            # Update tracking
            if sid in self.client_sessions:
                del self.client_sessions[sid]
            if session_id in self.connected_clients and sid in self.connected_clients[session_id]:
                self.connected_clients[session_id].remove(sid)
                if not self.connected_clients[session_id]:
                    del self.connected_clients[session_id]

            logger.info(f"Client {sid} left session {session_id}")

        except Exception as e:
            logger.error(f"Error leaving session room: {e}")

    async def broadcast_progress(self, session_id: str, progress_data: ProgressData):
        """Broadcast progress to all clients in session room."""
        try:
            # Cache progress data
            self.progress_cache[session_id] = progress_data

            # Check for milestones
            milestone = self._check_milestones(session_id, progress_data)
            if milestone:
                await self.broadcast_milestone(session_id, milestone)

            # Broadcast to all clients in session room
            await sio.emit(
                'progress_update',
                progress_data.model_dump(),
                room=f"session_{session_id}"
            )

            logger.debug(f"Broadcasted progress for session {session_id}: {progress_data.current_image}/{progress_data.total_images}")

        except Exception as e:
            logger.error(f"Error broadcasting progress: {e}")

    def _check_milestones(self, session_id: str, progress_data: ProgressData) -> Optional[MilestoneData]:
        """Check if any milestones have been reached."""
        if session_id not in self.milestone_tracker:
            self.milestone_tracker[session_id] = set()
        
        reached_milestones = self.milestone_tracker[session_id]
        
        # Percentage milestones
        percentage_milestones = [25, 50, 75, 90]
        for milestone in percentage_milestones:
            milestone_key = f"{milestone}%"
            if (progress_data.percentage >= milestone and 
                milestone_key not in reached_milestones):
                reached_milestones.add(milestone_key)
                
                return MilestoneData(
                    session_id=session_id,
                    milestone_type="percentage",
                    milestone_value=milestone_key,
                    current_progress=progress_data.current_image,
                    total_progress=progress_data.total_images,
                    message=f"Processing {milestone}% complete! ({progress_data.current_image:,}/{progress_data.total_images:,} images)",
                    timestamp=datetime.now(),
                    performance_snapshot={
                        "processing_speed": progress_data.processing_speed,
                        "memory_usage": progress_data.memory_usage_mb,
                        "gpu_usage": progress_data.gpu_usage_percent,
                        "approved_rate": (progress_data.approved_count / progress_data.current_image * 100) if progress_data.current_image > 0 else 0
                    }
                )
        
        # Count milestones
        count_milestones = [100, 500, 1000, 5000, 10000, 25000]
        for milestone in count_milestones:
            milestone_key = f"{milestone}_images"
            if (progress_data.current_image >= milestone and 
                milestone_key not in reached_milestones):
                reached_milestones.add(milestone_key)
                
                return MilestoneData(
                    session_id=session_id,
                    milestone_type="count",
                    milestone_value=milestone_key,
                    current_progress=progress_data.current_image,
                    total_progress=progress_data.total_images,
                    message=f"Milestone reached: {milestone:,} images processed! Speed: {progress_data.processing_speed:.1f} img/sec",
                    timestamp=datetime.now(),
                    performance_snapshot={
                        "processing_speed": progress_data.processing_speed,
                        "memory_usage": progress_data.memory_usage_mb,
                        "gpu_usage": progress_data.gpu_usage_percent,
                        "approved_count": progress_data.approved_count,
                        "rejected_count": progress_data.rejected_count
                    }
                )
        
        # Time milestones (every hour)
        if progress_data.elapsed_time > 0:
            hours_elapsed = int(progress_data.elapsed_time / 3600)
            if hours_elapsed > 0:
                milestone_key = f"{hours_elapsed}_hour{'s' if hours_elapsed > 1 else ''}"
                if milestone_key not in reached_milestones:
                    reached_milestones.add(milestone_key)
                    
                    return MilestoneData(
                        session_id=session_id,
                        milestone_type="time",
                        milestone_value=milestone_key,
                        current_progress=progress_data.current_image,
                        total_progress=progress_data.total_images,
                        message=f"Processing for {hours_elapsed} hour{'s' if hours_elapsed > 1 else ''}! Progress: {progress_data.percentage:.1f}%",
                        timestamp=datetime.now(),
                        performance_snapshot={
                            "elapsed_time": progress_data.elapsed_time,
                            "processing_speed": progress_data.processing_speed,
                            "eta_seconds": progress_data.eta_seconds
                        }
                    )
        
        return None

    async def broadcast_milestone(self, session_id: str, milestone_data: MilestoneData):
        """Broadcast milestone notification to session room."""
        try:
            await sio.emit(
                'milestone_reached',
                milestone_data.model_dump(),
                room=f"session_{session_id}"
            )

            logger.info(f"Milestone reached for session {session_id}: {milestone_data.milestone_value}")

        except Exception as e:
            logger.error(f"Error broadcasting milestone: {e}")

    async def broadcast_error(self, session_id: str, error_data: ErrorData):
        """Broadcast error to session room."""
        try:
            await sio.emit(
                'processing_error',
                error_data.model_dump(),
                room=f"session_{session_id}"
            )

            logger.warning(f"Broadcasted error for session {session_id}: {error_data.error_message}")

        except Exception as e:
            logger.error(f"Error broadcasting error: {e}")

    async def broadcast_completion(self, session_id: str, completion_data: CompletionData):
        """Broadcast completion to session room."""
        try:
            await sio.emit(
                'processing_complete',
                completion_data.model_dump(),
                room=f"session_{session_id}"
            )

            # Clean up cached progress
            if session_id in self.progress_cache:
                del self.progress_cache[session_id]

            logger.info(f"Broadcasted completion for session {session_id}")

        except Exception as e:
            logger.error(f"Error broadcasting completion: {e}")

    async def broadcast_stage_change(self, session_id: str, stage: str, message: str = ""):
        """Broadcast processing stage change."""
        try:
            await sio.emit(
                'stage_change',
                {
                    'session_id': session_id,
                    'stage': stage,
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                },
                room=f"session_{session_id}"
            )

            logger.info(f"Broadcasted stage change for session {session_id}: {stage}")

        except Exception as e:
            logger.error(f"Error broadcasting stage change: {e}")

    def get_connected_clients_count(self, session_id: str) -> int:
        """Get number of connected clients for a session."""
        return len(self.connected_clients.get(session_id, []))

    async def cleanup_session(self, session_id: str):
        """Clean up session data and disconnect clients."""
        try:
            # Remove from cache
            if session_id in self.progress_cache:
                del self.progress_cache[session_id]
            
            # Remove milestone tracking
            if session_id in self.milestone_tracker:
                del self.milestone_tracker[session_id]

            # Disconnect all clients in session
            if session_id in self.connected_clients:
                for client_id in self.connected_clients[session_id].copy():
                    await self.leave_session_room(client_id, session_id)

            logger.info(f"Cleaned up session {session_id}")

        except Exception as e:
            logger.error(f"Error cleaning up session: {e}")

# Global Socket.IO manager instance
socketio_manager = SocketIOManager()

# Socket.IO Event Handlers
@sio.event
async def connect(sid, environ):
    """Handle client connection."""
    try:
        logger.info(f"Client {sid} connected")
        await sio.emit(
            'connected', 
            {
                'status': 'Connected to server',
                'client_id': sid,
                'timestamp': datetime.now().isoformat()
            }, 
            room=sid
        )
    except Exception as e:
        logger.error(f"Error handling connection: {e}")

@sio.event
async def disconnect(sid):
    """Handle client disconnection."""
    try:
        # Clean up client tracking
        if sid in socketio_manager.client_sessions:
            session_id = socketio_manager.client_sessions[sid]
            await socketio_manager.leave_session_room(sid, session_id)

        logger.info(f"Client {sid} disconnected")
    except Exception as e:
        logger.error(f"Error handling disconnection: {e}")

@sio.event
async def join_session(sid, data):
    """Handle client joining session room."""
    try:
        session_id = data.get('session_id')
        if not session_id:
            await sio.emit('error', {'message': 'session_id is required'}, room=sid)
            return

        await socketio_manager.join_session_room(sid, session_id)
        logger.info(f"Client {sid} joined session {session_id}")

    except Exception as e:
        logger.error(f"Error joining session: {e}")
        await sio.emit('error', {'message': f'Failed to join session: {str(e)}'}, room=sid)

@sio.event
async def leave_session(sid, data):
    """Handle client leaving session room."""
    try:
        session_id = data.get('session_id')
        if not session_id:
            await sio.emit('error', {'message': 'session_id is required'}, room=sid)
            return

        await socketio_manager.leave_session_room(sid, session_id)
        logger.info(f"Client {sid} left session {session_id}")

    except Exception as e:
        logger.error(f"Error leaving session: {e}")
        await sio.emit('error', {'message': f'Failed to leave session: {str(e)}'}, room=sid)

@sio.event
async def get_session_status(sid, data):
    """Handle request for current session status."""
    try:
        session_id = data.get('session_id')
        if not session_id:
            await sio.emit('error', {'message': 'session_id is required'}, room=sid)
            return

        # Send cached progress if available
        if session_id in socketio_manager.progress_cache:
            progress_data = socketio_manager.progress_cache[session_id]
            await sio.emit('progress_update', progress_data.dict(), room=sid)
        else:
            await sio.emit(
                'session_status',
                {
                    'session_id': session_id,
                    'status': 'no_active_processing',
                    'message': 'No active processing for this session'
                },
                room=sid
            )

    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        await sio.emit('error', {'message': f'Failed to get session status: {str(e)}'}, room=sid)

@sio.event
async def pause_processing(sid, data):
    """Handle pause processing request."""
    try:
        session_id = data.get('session_id')
        if not session_id:
            await sio.emit('error', {'message': 'session_id is required'}, room=sid)
            return

        # TODO: Implement actual pause logic in processing engine
        # For now, just acknowledge the request
        await sio.emit(
            'processing_paused',
            {
                'session_id': session_id,
                'status': 'paused',
                'timestamp': datetime.now().isoformat()
            },
            room=f"session_{session_id}"
        )

        logger.info(f"Processing paused for session {session_id}")

    except Exception as e:
        logger.error(f"Error pausing processing: {e}")
        await sio.emit('error', {'message': f'Failed to pause processing: {str(e)}'}, room=sid)

@sio.event
async def resume_processing(sid, data):
    """Handle resume processing request."""
    try:
        session_id = data.get('session_id')
        if not session_id:
            await sio.emit('error', {'message': 'session_id is required'}, room=sid)
            return

        # TODO: Implement actual resume logic in processing engine
        # For now, just acknowledge the request
        await sio.emit(
            'processing_resumed',
            {
                'session_id': session_id,
                'status': 'resumed',
                'timestamp': datetime.now().isoformat()
            },
            room=f"session_{session_id}"
        )

        logger.info(f"Processing resumed for session {session_id}")

    except Exception as e:
        logger.error(f"Error resuming processing: {e}")
        await sio.emit('error', {'message': f'Failed to resume processing: {str(e)}'}, room=sid)

@sio.event
async def ping(sid, data):
    """Handle ping for connection testing."""
    try:
        await sio.emit('pong', {'timestamp': datetime.now().isoformat()}, room=sid)
    except Exception as e:
        logger.error(f"Error handling ping: {e}")