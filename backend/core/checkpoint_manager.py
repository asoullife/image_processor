"""Advanced checkpoint manager for robust resume and recovery system."""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

import sys
from pathlib import Path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from database.connection import DatabaseManager
from database.models import ProcessingSession, Checkpoint, ImageResult
from sqlalchemy import select, update, delete, and_, desc
from sqlalchemy.orm import selectinload

logger = logging.getLogger(__name__)

class CheckpointType(Enum):
    """Types of checkpoints for different recovery scenarios."""
    IMAGE = "image"          # Every 10 images
    BATCH = "batch"          # End of each batch
    MILESTONE = "milestone"  # Major processing milestones
    EMERGENCY = "emergency"  # Emergency/crash checkpoints
    MANUAL = "manual"        # User-triggered checkpoints

class ResumeOption(Enum):
    """Resume options for interrupted sessions."""
    CONTINUE = "continue"           # Continue from last checkpoint
    RESTART_BATCH = "restart_batch" # Restart current batch
    FRESH_START = "fresh_start"     # Start completely fresh

@dataclass
class SessionState:
    """Complete session state for checkpointing."""
    session_id: str
    current_batch: int
    current_image_index: int
    processed_count: int
    approved_count: int
    rejected_count: int
    current_image_path: Optional[str]
    batch_start_time: datetime
    last_processing_rate: float  # images per second
    memory_usage_mb: float
    gpu_memory_usage_mb: float
    error_count: int
    last_error: Optional[str]
    processing_config: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime to ISO string
        data['batch_start_time'] = self.batch_start_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        """Create from dictionary."""
        # Convert ISO string back to datetime
        if isinstance(data['batch_start_time'], str):
            data['batch_start_time'] = datetime.fromisoformat(data['batch_start_time'])
        return cls(**data)

@dataclass
class CheckpointData:
    """Comprehensive checkpoint data."""
    checkpoint_id: str
    session_id: str
    checkpoint_type: CheckpointType
    session_state: SessionState
    integrity_hash: str
    created_at: datetime
    can_resume: bool
    recovery_options: List[ResumeOption]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'checkpoint_id': self.checkpoint_id,
            'session_id': self.session_id,
            'checkpoint_type': self.checkpoint_type.value,
            'session_state': self.session_state.to_dict(),
            'integrity_hash': self.integrity_hash,
            'created_at': self.created_at.isoformat(),
            'can_resume': self.can_resume,
            'recovery_options': [opt.value for opt in self.recovery_options]
        }

class CheckpointManager:
    """Advanced checkpoint manager with robust recovery capabilities."""
    
    def __init__(self, db_manager: DatabaseManager, checkpoint_interval: int = 10):
        """Initialize checkpoint manager.
        
        Args:
            db_manager: Database manager instance
            checkpoint_interval: Save checkpoint every N processed images
        """
        self.db_manager = db_manager
        self.checkpoint_interval = checkpoint_interval
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Track active sessions
        self._active_sessions: Dict[str, SessionState] = {}
        self._last_checkpoint_counts: Dict[str, int] = {}
        
    async def create_session_checkpoint(
        self,
        session_id: UUID,
        checkpoint_type: CheckpointType = CheckpointType.IMAGE,
        force: bool = False
    ) -> Optional[CheckpointData]:
        """Create a checkpoint for the current session state.
        
        Args:
            session_id: Session UUID
            checkpoint_type: Type of checkpoint to create
            force: Force checkpoint creation regardless of interval
            
        Returns:
            CheckpointData if successful, None otherwise
        """
        try:
            session_id_str = str(session_id)
            
            # Get current session state
            session_state = await self._get_current_session_state(session_id)
            if not session_state:
                self.logger.error(f"Cannot create checkpoint: session {session_id} not found")
                return None
            
            # Check if checkpoint is needed (unless forced)
            if not force and checkpoint_type == CheckpointType.IMAGE:
                last_checkpoint = self._last_checkpoint_counts.get(session_id_str, 0)
                if session_state.processed_count - last_checkpoint < self.checkpoint_interval:
                    return None  # Not time for checkpoint yet
            
            # Create checkpoint data
            checkpoint_data = CheckpointData(
                checkpoint_id=str(uuid4()),
                session_id=session_id_str,
                checkpoint_type=checkpoint_type,
                session_state=session_state,
                integrity_hash=self._calculate_integrity_hash(session_state),
                created_at=datetime.now(),
                can_resume=True,
                recovery_options=self._determine_recovery_options(session_state)
            )
            
            # Save to database
            success = await self._save_checkpoint_to_db(checkpoint_data)
            if success:
                self._last_checkpoint_counts[session_id_str] = session_state.processed_count
                self.logger.info(
                    f"Checkpoint created: {checkpoint_type.value} at {session_state.processed_count} images"
                )
                return checkpoint_data
            else:
                self.logger.error(f"Failed to save checkpoint to database")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint for session {session_id}: {e}")
            return None
    
    async def _get_current_session_state(self, session_id: UUID) -> Optional[SessionState]:
        """Get current session state from database.
        
        Args:
            session_id: Session UUID
            
        Returns:
            SessionState if found, None otherwise
        """
        try:
            async with self.db_manager.get_session() as session:
                # Get processing session
                stmt = (
                    select(ProcessingSession)
                    .where(ProcessingSession.id == session_id)
                )
                result = await session.execute(stmt)
                processing_session = result.scalar_one_or_none()
                
                if not processing_session:
                    return None
                
                # Calculate current batch and image index
                batch_size = processing_session.session_config.get('batch_size', 20)
                current_batch = processing_session.processed_images // batch_size
                current_image_index = processing_session.processed_images % batch_size
                
                # Calculate processing rate
                processing_rate = 0.0
                if processing_session.start_time:
                    elapsed = (datetime.now() - processing_session.start_time).total_seconds()
                    if elapsed > 0 and processing_session.processed_images > 0:
                        processing_rate = processing_session.processed_images / elapsed
                
                # Get system metrics (mock values for now)
                memory_usage = await self._get_memory_usage()
                gpu_memory_usage = await self._get_gpu_memory_usage()
                
                # Get last error if any
                last_error = processing_session.error_message
                
                return SessionState(
                    session_id=str(session_id),
                    current_batch=current_batch,
                    current_image_index=current_image_index,
                    processed_count=processing_session.processed_images or 0,
                    approved_count=processing_session.approved_images or 0,
                    rejected_count=processing_session.rejected_images or 0,
                    current_image_path=None,  # Would need to track this separately
                    batch_start_time=processing_session.start_time or datetime.now(),
                    last_processing_rate=processing_rate,
                    memory_usage_mb=memory_usage,
                    gpu_memory_usage_mb=gpu_memory_usage,
                    error_count=0,  # Would need to track this
                    last_error=last_error,
                    processing_config=processing_session.session_config or {}
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get session state for {session_id}: {e}")
            return None
    
    async def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    async def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed
        except:
            return 0.0
    
    def _calculate_integrity_hash(self, session_state: SessionState) -> str:
        """Calculate integrity hash for session state.
        
        Args:
            session_state: Session state to hash
            
        Returns:
            SHA-256 hash string
        """
        import hashlib
        
        # Create deterministic string representation
        state_str = json.dumps(session_state.to_dict(), sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()
    
    def _determine_recovery_options(self, session_state: SessionState) -> List[ResumeOption]:
        """Determine available recovery options based on session state.
        
        Args:
            session_state: Current session state
            
        Returns:
            List of available resume options
        """
        options = [ResumeOption.FRESH_START]  # Always available
        
        if session_state.processed_count > 0:
            options.append(ResumeOption.CONTINUE)
            
        if session_state.current_batch > 0:
            options.append(ResumeOption.RESTART_BATCH)
            
        return options
    
    async def _save_checkpoint_to_db(self, checkpoint_data: CheckpointData) -> bool:
        """Save checkpoint data to database.
        
        Args:
            checkpoint_data: Checkpoint data to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self.db_manager.get_session() as session:
                checkpoint = Checkpoint(
                    session_id=UUID(checkpoint_data.session_id),
                    checkpoint_type=checkpoint_data.checkpoint_type.value,
                    processed_count=checkpoint_data.session_state.processed_count,
                    current_batch=checkpoint_data.session_state.current_batch,
                    current_image_index=checkpoint_data.session_state.current_image_index,
                    session_state=checkpoint_data.to_dict()
                )
                
                session.add(checkpoint)
                await session.flush()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to database: {e}")
            return False
    
    async def get_latest_checkpoint(self, session_id: UUID) -> Optional[CheckpointData]:
        """Get the latest checkpoint for a session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            Latest CheckpointData if found, None otherwise
        """
        try:
            async with self.db_manager.get_session() as session:
                stmt = (
                    select(Checkpoint)
                    .where(Checkpoint.session_id == session_id)
                    .order_by(desc(Checkpoint.created_at))
                    .limit(1)
                )
                result = await session.execute(stmt)
                checkpoint = result.scalar_one_or_none()
                
                if not checkpoint:
                    return None
                
                # Reconstruct CheckpointData
                session_state_data = checkpoint.session_state
                if isinstance(session_state_data, dict):
                    session_state = SessionState.from_dict(session_state_data['session_state'])
                    
                    return CheckpointData(
                        checkpoint_id=session_state_data['checkpoint_id'],
                        session_id=session_state_data['session_id'],
                        checkpoint_type=CheckpointType(session_state_data['checkpoint_type']),
                        session_state=session_state,
                        integrity_hash=session_state_data['integrity_hash'],
                        created_at=datetime.fromisoformat(session_state_data['created_at']),
                        can_resume=session_state_data['can_resume'],
                        recovery_options=[ResumeOption(opt) for opt in session_state_data['recovery_options']]
                    )
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get latest checkpoint for session {session_id}: {e}")
            return None
    
    async def get_all_checkpoints(self, session_id: UUID) -> List[CheckpointData]:
        """Get all checkpoints for a session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            List of CheckpointData ordered by creation time
        """
        try:
            async with self.db_manager.get_session() as session:
                stmt = (
                    select(Checkpoint)
                    .where(Checkpoint.session_id == session_id)
                    .order_by(desc(Checkpoint.created_at))
                )
                result = await session.execute(stmt)
                checkpoints = result.scalars().all()
                
                checkpoint_data_list = []
                for checkpoint in checkpoints:
                    try:
                        session_state_data = checkpoint.session_state
                        if isinstance(session_state_data, dict):
                            session_state = SessionState.from_dict(session_state_data['session_state'])
                            
                            checkpoint_data = CheckpointData(
                                checkpoint_id=session_state_data['checkpoint_id'],
                                session_id=session_state_data['session_id'],
                                checkpoint_type=CheckpointType(session_state_data['checkpoint_type']),
                                session_state=session_state,
                                integrity_hash=session_state_data['integrity_hash'],
                                created_at=datetime.fromisoformat(session_state_data['created_at']),
                                can_resume=session_state_data['can_resume'],
                                recovery_options=[ResumeOption(opt) for opt in session_state_data['recovery_options']]
                            )
                            checkpoint_data_list.append(checkpoint_data)
                    except Exception as e:
                        self.logger.warning(f"Failed to parse checkpoint data: {e}")
                        continue
                
                return checkpoint_data_list
                
        except Exception as e:
            self.logger.error(f"Failed to get checkpoints for session {session_id}: {e}")
            return []
    
    async def verify_checkpoint_integrity(self, checkpoint_data: CheckpointData) -> bool:
        """Verify checkpoint data integrity.
        
        Args:
            checkpoint_data: Checkpoint data to verify
            
        Returns:
            True if integrity is valid, False otherwise
        """
        try:
            # Recalculate hash
            calculated_hash = self._calculate_integrity_hash(checkpoint_data.session_state)
            
            # Compare with stored hash
            if calculated_hash != checkpoint_data.integrity_hash:
                self.logger.error(f"Checkpoint integrity check failed: hash mismatch")
                return False
            
            # Verify session exists in database
            async with self.db_manager.get_session() as session:
                stmt = select(ProcessingSession).where(
                    ProcessingSession.id == UUID(checkpoint_data.session_id)
                )
                result = await session.execute(stmt)
                processing_session = result.scalar_one_or_none()
                
                if not processing_session:
                    self.logger.error(f"Checkpoint integrity check failed: session not found")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to verify checkpoint integrity: {e}")
            return False
    
    async def detect_interrupted_sessions(self) -> List[Tuple[UUID, CheckpointData]]:
        """Detect sessions that were interrupted and can be resumed.
        
        Returns:
            List of tuples (session_id, latest_checkpoint)
        """
        try:
            interrupted_sessions = []
            
            async with self.db_manager.get_session() as session:
                # Find sessions that are still "running" but haven't been updated recently
                cutoff_time = datetime.now() - timedelta(minutes=5)
                
                stmt = (
                    select(ProcessingSession)
                    .where(
                        and_(
                            ProcessingSession.status == "running",
                            ProcessingSession.updated_at < cutoff_time
                        )
                    )
                )
                result = await session.execute(stmt)
                stale_sessions = result.scalars().all()
                
                for processing_session in stale_sessions:
                    # Get latest checkpoint
                    latest_checkpoint = await self.get_latest_checkpoint(processing_session.id)
                    if latest_checkpoint and latest_checkpoint.can_resume:
                        # Verify integrity
                        if await self.verify_checkpoint_integrity(latest_checkpoint):
                            interrupted_sessions.append((processing_session.id, latest_checkpoint))
                        else:
                            self.logger.warning(
                                f"Session {processing_session.id} has corrupted checkpoint data"
                            )
            
            return interrupted_sessions
            
        except Exception as e:
            self.logger.error(f"Failed to detect interrupted sessions: {e}")
            return []
    
    async def prepare_resume_options(
        self,
        session_id: UUID,
        checkpoint_data: CheckpointData
    ) -> Dict[str, Any]:
        """Prepare detailed resume options for user selection.
        
        Args:
            session_id: Session UUID
            checkpoint_data: Latest checkpoint data
            
        Returns:
            Dictionary with resume options and details
        """
        try:
            session_state = checkpoint_data.session_state
            
            # Calculate progress statistics
            total_images = 0
            async with self.db_manager.get_session() as session:
                stmt = select(ProcessingSession).where(ProcessingSession.id == session_id)
                result = await session.execute(stmt)
                processing_session = result.scalar_one_or_none()
                if processing_session:
                    total_images = processing_session.total_images
            
            progress_percentage = (session_state.processed_count / total_images * 100) if total_images > 0 else 0
            approval_rate = (session_state.approved_count / session_state.processed_count * 100) if session_state.processed_count > 0 else 0
            
            # Estimate remaining time
            remaining_images = total_images - session_state.processed_count
            estimated_time_remaining = 0
            if session_state.last_processing_rate > 0:
                estimated_time_remaining = remaining_images / session_state.last_processing_rate
            
            resume_options = {
                "session_info": {
                    "session_id": str(session_id),
                    "total_images": total_images,
                    "processed_count": session_state.processed_count,
                    "approved_count": session_state.approved_count,
                    "rejected_count": session_state.rejected_count,
                    "progress_percentage": round(progress_percentage, 2),
                    "approval_rate": round(approval_rate, 2),
                    "last_processing_rate": round(session_state.last_processing_rate, 2),
                    "estimated_time_remaining_seconds": round(estimated_time_remaining),
                    "checkpoint_created": checkpoint_data.created_at.isoformat(),
                    "memory_usage_mb": session_state.memory_usage_mb,
                    "gpu_memory_usage_mb": session_state.gpu_memory_usage_mb
                },
                "available_options": []
            }
            
            # Add available resume options with details
            for option in checkpoint_data.recovery_options:
                option_detail = {
                    "option": option.value,
                    "description": self._get_resume_option_description(option),
                    "recommended": False
                }
                
                if option == ResumeOption.CONTINUE:
                    option_detail["start_from_image"] = session_state.processed_count + 1
                    option_detail["recommended"] = True  # Default recommendation
                elif option == ResumeOption.RESTART_BATCH:
                    batch_start = session_state.current_batch * session_state.processing_config.get('batch_size', 20)
                    option_detail["start_from_image"] = batch_start + 1
                elif option == ResumeOption.FRESH_START:
                    option_detail["start_from_image"] = 1
                    option_detail["warning"] = "All previous progress will be lost"
                
                resume_options["available_options"].append(option_detail)
            
            return resume_options
            
        except Exception as e:
            self.logger.error(f"Failed to prepare resume options: {e}")
            return {}
    
    def _get_resume_option_description(self, option: ResumeOption) -> str:
        """Get human-readable description for resume option.
        
        Args:
            option: Resume option
            
        Returns:
            Description string
        """
        descriptions = {
            ResumeOption.CONTINUE: "Continue processing from the last checkpoint",
            ResumeOption.RESTART_BATCH: "Restart processing from the beginning of the current batch",
            ResumeOption.FRESH_START: "Start processing from the beginning (discard all progress)"
        }
        return descriptions.get(option, "Unknown option")
    
    async def execute_resume(
        self,
        session_id: UUID,
        resume_option: ResumeOption,
        checkpoint_data: CheckpointData
    ) -> Tuple[bool, int]:
        """Execute the selected resume option.
        
        Args:
            session_id: Session UUID
            resume_option: Selected resume option
            checkpoint_data: Checkpoint data to resume from
            
        Returns:
            Tuple of (success, start_index)
        """
        try:
            session_state = checkpoint_data.session_state
            start_index = 0
            
            async with self.db_manager.get_session() as session:
                # Update session status to running
                stmt = (
                    update(ProcessingSession)
                    .where(ProcessingSession.id == session_id)
                    .values(
                        status="running",
                        updated_at=datetime.now(),
                        error_message=None
                    )
                )
                await session.execute(stmt)
                
                if resume_option == ResumeOption.CONTINUE:
                    # Continue from last processed image
                    start_index = session_state.processed_count
                    
                elif resume_option == ResumeOption.RESTART_BATCH:
                    # Restart from beginning of current batch
                    batch_size = session_state.processing_config.get('batch_size', 20)
                    start_index = session_state.current_batch * batch_size
                    
                    # Reset counters to batch start
                    batch_start_processed = start_index
                    
                    # Count approved/rejected in current batch to adjust counters
                    batch_approved = 0
                    batch_rejected = 0
                    
                    # Get results from current batch
                    results_stmt = (
                        select(ImageResult)
                        .where(ImageResult.session_id == session_id)
                        .offset(start_index)
                        .limit(session_state.processed_count - start_index)
                    )
                    results_result = await session.execute(results_stmt)
                    batch_results = results_result.scalars().all()
                    
                    for result in batch_results:
                        if result.final_decision == 'approved':
                            batch_approved += 1
                        else:
                            batch_rejected += 1
                    
                    # Update session counters
                    new_processed = session_state.processed_count - (session_state.processed_count - start_index)
                    new_approved = session_state.approved_count - batch_approved
                    new_rejected = session_state.rejected_count - batch_rejected
                    
                    update_stmt = (
                        update(ProcessingSession)
                        .where(ProcessingSession.id == session_id)
                        .values(
                            processed_images=new_processed,
                            approved_images=new_approved,
                            rejected_images=new_rejected
                        )
                    )
                    await session.execute(update_stmt)
                    
                    # Delete results from current batch
                    delete_stmt = (
                        delete(ImageResult)
                        .where(ImageResult.session_id == session_id)
                    )
                    # Add condition to delete only batch results
                    await session.execute(delete_stmt)
                    
                elif resume_option == ResumeOption.FRESH_START:
                    # Reset everything
                    start_index = 0
                    
                    # Reset session counters
                    update_stmt = (
                        update(ProcessingSession)
                        .where(ProcessingSession.id == session_id)
                        .values(
                            processed_images=0,
                            approved_images=0,
                            rejected_images=0,
                            start_time=datetime.now()
                        )
                    )
                    await session.execute(update_stmt)
                    
                    # Delete all results
                    delete_results_stmt = delete(ImageResult).where(ImageResult.session_id == session_id)
                    await session.execute(delete_results_stmt)
                    
                    # Delete all checkpoints
                    delete_checkpoints_stmt = delete(Checkpoint).where(Checkpoint.session_id == session_id)
                    await session.execute(delete_checkpoints_stmt)
                
                await session.commit()
            
            self.logger.info(f"Resume executed: {resume_option.value} from index {start_index}")
            return True, start_index
            
        except Exception as e:
            self.logger.error(f"Failed to execute resume: {e}")
            return False, 0
    
    async def cleanup_old_checkpoints(
        self,
        session_id: UUID,
        keep_count: int = 5
    ) -> bool:
        """Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            session_id: Session UUID
            keep_count: Number of recent checkpoints to keep
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self.db_manager.get_session() as session:
                # Get all checkpoints ordered by creation time
                stmt = (
                    select(Checkpoint)
                    .where(Checkpoint.session_id == session_id)
                    .order_by(desc(Checkpoint.created_at))
                )
                result = await session.execute(stmt)
                checkpoints = result.scalars().all()
                
                # Delete old checkpoints if we have more than keep_count
                if len(checkpoints) > keep_count:
                    checkpoints_to_delete = checkpoints[keep_count:]
                    
                    for checkpoint in checkpoints_to_delete:
                        await session.delete(checkpoint)
                    
                    await session.commit()
                    self.logger.info(f"Cleaned up {len(checkpoints_to_delete)} old checkpoints")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old checkpoints: {e}")
            return False
    
    async def create_emergency_checkpoint(
        self,
        session_id: UUID,
        error_message: str
    ) -> Optional[CheckpointData]:
        """Create an emergency checkpoint when an error occurs.
        
        Args:
            session_id: Session UUID
            error_message: Error that triggered the emergency checkpoint
            
        Returns:
            CheckpointData if successful, None otherwise
        """
        try:
            # Get current session state
            session_state = await self._get_current_session_state(session_id)
            if not session_state:
                return None
            
            # Update session state with error information
            session_state.last_error = error_message
            session_state.error_count += 1
            
            # Create emergency checkpoint
            checkpoint_data = await self.create_session_checkpoint(
                session_id=session_id,
                checkpoint_type=CheckpointType.EMERGENCY,
                force=True
            )
            
            if checkpoint_data:
                # Mark session as failed but recoverable
                async with self.db_manager.get_session() as session:
                    stmt = (
                        update(ProcessingSession)
                        .where(ProcessingSession.id == session_id)
                        .values(
                            status="failed",
                            error_message=error_message,
                            updated_at=datetime.now()
                        )
                    )
                    await session.execute(stmt)
                    await session.commit()
                
                self.logger.info(f"Emergency checkpoint created for session {session_id}")
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to create emergency checkpoint: {e}")
            return None