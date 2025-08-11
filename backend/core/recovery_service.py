"""Recovery service for handling session interruptions and resume operations."""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
from datetime import datetime, timedelta
from enum import Enum

import sys
from pathlib import Path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from database.connection import DatabaseManager
from database.models import ProcessingSession
from core.checkpoint_manager import CheckpointManager, CheckpointData, ResumeOption
from sqlalchemy import select, and_

logger = logging.getLogger(__name__)

class CrashType(Enum):
    """Types of crashes that can be detected."""
    POWER_FAILURE = "power_failure"
    SYSTEM_CRASH = "system_crash"
    APPLICATION_CRASH = "application_crash"
    NETWORK_FAILURE = "network_failure"
    OUT_OF_MEMORY = "out_of_memory"
    UNKNOWN = "unknown"

class RecoveryService:
    """Service for handling crash detection and recovery operations."""
    
    def __init__(self, db_manager: DatabaseManager, checkpoint_manager: CheckpointManager):
        """Initialize recovery service.
        
        Args:
            db_manager: Database manager instance
            checkpoint_manager: Checkpoint manager instance
        """
        self.db_manager = db_manager
        self.checkpoint_manager = checkpoint_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration
        self.crash_detection_timeout = timedelta(minutes=5)  # Consider crashed after 5 minutes
        self.max_recovery_attempts = 3
        
    async def detect_crashes_on_startup(self) -> List[Dict[str, Any]]:
        """Detect crashed sessions on application startup.
        
        Returns:
            List of crashed session information
        """
        try:
            crashed_sessions = []
            
            # Find sessions that were running but haven't been updated recently
            cutoff_time = datetime.now() - self.crash_detection_timeout
            
            async with self.db_manager.get_session() as session:
                stmt = (
                    select(ProcessingSession)
                    .where(
                        and_(
                            ProcessingSession.status.in_(["running", "created"]),
                            ProcessingSession.updated_at < cutoff_time
                        )
                    )
                )
                result = await session.execute(stmt)
                stale_sessions = result.scalars().all()
                
                for processing_session in stale_sessions:
                    # Analyze crash type
                    crash_type = await self._analyze_crash_type(processing_session)
                    
                    # Get latest checkpoint
                    latest_checkpoint = await self.checkpoint_manager.get_latest_checkpoint(
                        processing_session.id
                    )
                    
                    if latest_checkpoint:
                        # Verify checkpoint integrity
                        integrity_ok = await self.checkpoint_manager.verify_checkpoint_integrity(
                            latest_checkpoint
                        )
                        
                        crashed_session_info = {
                            "session_id": str(processing_session.id),
                            "project_id": str(processing_session.project_id),
                            "crash_type": crash_type.value,
                            "last_update": processing_session.updated_at.isoformat(),
                            "processed_count": processing_session.processed_images or 0,
                            "total_images": processing_session.total_images,
                            "has_valid_checkpoint": integrity_ok,
                            "checkpoint_data": latest_checkpoint if integrity_ok else None,
                            "can_recover": integrity_ok and latest_checkpoint.can_resume,
                            "error_message": processing_session.error_message
                        }
                        
                        crashed_sessions.append(crashed_session_info)
                        
                        self.logger.info(
                            f"Detected crashed session: {processing_session.id} "
                            f"({crash_type.value}, recoverable: {integrity_ok})"
                        )
            
            return crashed_sessions
            
        except Exception as e:
            self.logger.error(f"Failed to detect crashes on startup: {e}")
            return []
    
    async def _analyze_crash_type(self, processing_session: ProcessingSession) -> CrashType:
        """Analyze the type of crash based on session data.
        
        Args:
            processing_session: Processing session to analyze
            
        Returns:
            Detected crash type
        """
        try:
            # Check error message for clues
            if processing_session.error_message:
                error_msg = processing_session.error_message.lower()
                
                if "memory" in error_msg or "oom" in error_msg:
                    return CrashType.OUT_OF_MEMORY
                elif "network" in error_msg or "connection" in error_msg:
                    return CrashType.NETWORK_FAILURE
                elif "crash" in error_msg or "exception" in error_msg:
                    return CrashType.APPLICATION_CRASH
            
            # Check how long ago the session was last updated
            time_since_update = datetime.now() - processing_session.updated_at
            
            if time_since_update > timedelta(hours=1):
                # Likely a power failure or system crash
                return CrashType.POWER_FAILURE
            elif time_since_update > timedelta(minutes=10):
                # Likely a system crash
                return CrashType.SYSTEM_CRASH
            else:
                # Likely an application crash
                return CrashType.APPLICATION_CRASH
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze crash type: {e}")
            return CrashType.UNKNOWN
    
    async def prepare_recovery_options(
        self,
        session_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Prepare recovery options for a crashed session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            Recovery options dictionary or None if not recoverable
        """
        try:
            # Get latest checkpoint
            latest_checkpoint = await self.checkpoint_manager.get_latest_checkpoint(session_id)
            if not latest_checkpoint:
                self.logger.warning(f"No checkpoint found for session {session_id}")
                return None
            
            # Verify checkpoint integrity
            integrity_ok = await self.checkpoint_manager.verify_checkpoint_integrity(latest_checkpoint)
            if not integrity_ok:
                self.logger.error(f"Checkpoint integrity check failed for session {session_id}")
                return None
            
            # Get detailed resume options
            resume_options = await self.checkpoint_manager.prepare_resume_options(
                session_id, latest_checkpoint
            )
            
            # Add recovery-specific information
            recovery_info = {
                "recovery_available": True,
                "checkpoint_integrity": "valid",
                "data_loss_risk": self._assess_data_loss_risk(latest_checkpoint),
                "recommended_action": self._get_recommended_recovery_action(latest_checkpoint),
                "recovery_confidence": self._calculate_recovery_confidence(latest_checkpoint)
            }
            
            # Merge with resume options
            recovery_options = {**resume_options, "recovery_info": recovery_info}
            
            return recovery_options
            
        except Exception as e:
            self.logger.error(f"Failed to prepare recovery options for session {session_id}: {e}")
            return None
    
    def _assess_data_loss_risk(self, checkpoint_data: CheckpointData) -> str:
        """Assess the risk of data loss during recovery.
        
        Args:
            checkpoint_data: Checkpoint data to assess
            
        Returns:
            Risk level: "low", "medium", "high"
        """
        session_state = checkpoint_data.session_state
        
        # Calculate time since last checkpoint
        time_since_checkpoint = datetime.now() - checkpoint_data.created_at
        
        # Calculate potential lost work
        if session_state.last_processing_rate > 0:
            potentially_lost_images = int(
                time_since_checkpoint.total_seconds() * session_state.last_processing_rate
            )
        else:
            potentially_lost_images = 0
        
        if potentially_lost_images == 0:
            return "low"
        elif potentially_lost_images < 10:
            return "low"
        elif potentially_lost_images < 50:
            return "medium"
        else:
            return "high"
    
    def _get_recommended_recovery_action(self, checkpoint_data: CheckpointData) -> str:
        """Get recommended recovery action based on checkpoint data.
        
        Args:
            checkpoint_data: Checkpoint data to analyze
            
        Returns:
            Recommended action description
        """
        session_state = checkpoint_data.session_state
        
        # If very little progress, recommend fresh start
        if session_state.processed_count < 10:
            return "Fresh start recommended - minimal progress to lose"
        
        # If recent checkpoint, recommend continue
        time_since_checkpoint = datetime.now() - checkpoint_data.created_at
        if time_since_checkpoint < timedelta(minutes=2):
            return "Continue from checkpoint - minimal data loss risk"
        
        # If errors occurred, recommend restart batch
        if session_state.error_count > 0:
            return "Restart current batch - errors detected in recent processing"
        
        # Default recommendation
        return "Continue from checkpoint - best balance of progress and safety"
    
    def _calculate_recovery_confidence(self, checkpoint_data: CheckpointData) -> float:
        """Calculate confidence score for recovery success.
        
        Args:
            checkpoint_data: Checkpoint data to analyze
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 1.0
        session_state = checkpoint_data.session_state
        
        # Reduce confidence based on time since checkpoint
        time_since_checkpoint = datetime.now() - checkpoint_data.created_at
        if time_since_checkpoint > timedelta(minutes=5):
            confidence -= 0.2
        if time_since_checkpoint > timedelta(minutes=15):
            confidence -= 0.3
        
        # Reduce confidence based on error count
        if session_state.error_count > 0:
            confidence -= 0.1 * session_state.error_count
        
        # Reduce confidence if memory usage was high
        if session_state.memory_usage_mb > 8000:  # 8GB
            confidence -= 0.1
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    async def execute_recovery(
        self,
        session_id: UUID,
        recovery_option: str,
        user_confirmed: bool = False
    ) -> Tuple[bool, str, int]:
        """Execute recovery for a crashed session.
        
        Args:
            session_id: Session UUID
            recovery_option: Selected recovery option ("continue", "restart_batch", "fresh_start")
            user_confirmed: Whether user has confirmed the recovery action
            
        Returns:
            Tuple of (success, message, start_index)
        """
        try:
            if not user_confirmed:
                return False, "User confirmation required for recovery", 0
            
            # Get latest checkpoint
            latest_checkpoint = await self.checkpoint_manager.get_latest_checkpoint(session_id)
            if not latest_checkpoint:
                return False, "No valid checkpoint found for recovery", 0
            
            # Verify checkpoint integrity
            integrity_ok = await self.checkpoint_manager.verify_checkpoint_integrity(latest_checkpoint)
            if not integrity_ok:
                return False, "Checkpoint data is corrupted and cannot be used for recovery", 0
            
            # Convert recovery option to ResumeOption
            resume_option_map = {
                "continue": ResumeOption.CONTINUE,
                "restart_batch": ResumeOption.RESTART_BATCH,
                "fresh_start": ResumeOption.FRESH_START
            }
            
            resume_option = resume_option_map.get(recovery_option)
            if not resume_option:
                return False, f"Invalid recovery option: {recovery_option}", 0
            
            # Execute the resume
            success, start_index = await self.checkpoint_manager.execute_resume(
                session_id, resume_option, latest_checkpoint
            )
            
            if success:
                # Log recovery action
                self.logger.info(
                    f"Recovery executed successfully: session {session_id}, "
                    f"option {recovery_option}, start_index {start_index}"
                )
                
                # Create a new checkpoint to mark successful recovery
                await self.checkpoint_manager.create_session_checkpoint(
                    session_id=session_id,
                    checkpoint_type=self.checkpoint_manager.CheckpointType.MANUAL,
                    force=True
                )
                
                return True, f"Recovery successful - resuming from image {start_index + 1}", start_index
            else:
                return False, "Recovery failed - unable to restore session state", 0
            
        except Exception as e:
            self.logger.error(f"Failed to execute recovery for session {session_id}: {e}")
            return False, f"Recovery failed due to error: {str(e)}", 0
    
    async def verify_data_integrity_after_recovery(
        self,
        session_id: UUID,
        start_index: int
    ) -> Dict[str, Any]:
        """Verify data integrity after recovery operation.
        
        Args:
            session_id: Session UUID
            start_index: Index where processing will resume
            
        Returns:
            Integrity verification results
        """
        try:
            verification_results = {
                "integrity_check_passed": True,
                "issues_found": [],
                "recommendations": [],
                "safe_to_continue": True
            }
            
            async with self.db_manager.get_session() as session:
                # Get processing session
                stmt = select(ProcessingSession).where(ProcessingSession.id == session_id)
                result = await session.execute(stmt)
                processing_session = result.scalar_one_or_none()
                
                if not processing_session:
                    verification_results["integrity_check_passed"] = False
                    verification_results["issues_found"].append("Processing session not found")
                    verification_results["safe_to_continue"] = False
                    return verification_results
                
                # Check if counters are consistent
                expected_processed = processing_session.processed_images or 0
                expected_approved = processing_session.approved_images or 0
                expected_rejected = processing_session.rejected_images or 0
                
                # Count actual results in database
                from sqlalchemy import func
                from database.models import ImageResult
                
                count_stmt = (
                    select(
                        func.count(ImageResult.id).label('total'),
                        func.sum(
                            func.case(
                                (ImageResult.final_decision == 'approved', 1),
                                else_=0
                            )
                        ).label('approved'),
                        func.sum(
                            func.case(
                                (ImageResult.final_decision == 'rejected', 1),
                                else_=0
                            )
                        ).label('rejected')
                    )
                    .where(ImageResult.session_id == session_id)
                )
                count_result = await session.execute(count_stmt)
                counts = count_result.first()
                
                actual_total = counts.total or 0
                actual_approved = counts.approved or 0
                actual_rejected = counts.rejected or 0
                
                # Check for discrepancies
                if actual_total != expected_processed:
                    verification_results["issues_found"].append(
                        f"Processed count mismatch: expected {expected_processed}, found {actual_total}"
                    )
                
                if actual_approved != expected_approved:
                    verification_results["issues_found"].append(
                        f"Approved count mismatch: expected {expected_approved}, found {actual_approved}"
                    )
                
                if actual_rejected != expected_rejected:
                    verification_results["issues_found"].append(
                        f"Rejected count mismatch: expected {expected_rejected}, found {actual_rejected}"
                    )
                
                # Check for duplicate results
                duplicate_stmt = (
                    select(ImageResult.image_path, func.count(ImageResult.id))
                    .where(ImageResult.session_id == session_id)
                    .group_by(ImageResult.image_path)
                    .having(func.count(ImageResult.id) > 1)
                )
                duplicate_result = await session.execute(duplicate_stmt)
                duplicates = duplicate_result.fetchall()
                
                if duplicates:
                    verification_results["issues_found"].append(
                        f"Found {len(duplicates)} duplicate image results"
                    )
                    verification_results["recommendations"].append(
                        "Consider cleaning up duplicate results before continuing"
                    )
                
                # Check if start_index is reasonable
                if start_index > actual_total:
                    verification_results["issues_found"].append(
                        f"Start index {start_index} is beyond actual processed count {actual_total}"
                    )
                    verification_results["recommendations"].append(
                        f"Adjust start index to {actual_total} or consider fresh start"
                    )
                
                # Determine if it's safe to continue
                if len(verification_results["issues_found"]) > 0:
                    verification_results["integrity_check_passed"] = False
                    
                    # Critical issues that prevent continuation
                    critical_issues = [
                        issue for issue in verification_results["issues_found"]
                        if "not found" in issue or "beyond actual" in issue
                    ]
                    
                    if critical_issues:
                        verification_results["safe_to_continue"] = False
                    else:
                        verification_results["recommendations"].append(
                            "Minor inconsistencies detected but recovery can continue"
                        )
                
                return verification_results
            
        except Exception as e:
            self.logger.error(f"Failed to verify data integrity: {e}")
            return {
                "integrity_check_passed": False,
                "issues_found": [f"Integrity check failed: {str(e)}"],
                "recommendations": ["Manual intervention required"],
                "safe_to_continue": False
            }
    
    async def cleanup_failed_recovery_attempts(self, session_id: UUID) -> bool:
        """Clean up data from failed recovery attempts.
        
        Args:
            session_id: Session UUID
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            async with self.db_manager.get_session() as session:
                # Mark session as failed
                from sqlalchemy import update
                stmt = (
                    update(ProcessingSession)
                    .where(ProcessingSession.id == session_id)
                    .values(
                        status="failed",
                        error_message="Recovery failed - session marked for cleanup",
                        updated_at=datetime.now()
                    )
                )
                await session.execute(stmt)
                await session.commit()
            
            self.logger.info(f"Cleaned up failed recovery attempts for session {session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup failed recovery attempts: {e}")
            return False
    
    async def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about recovery operations.
        
        Returns:
            Recovery statistics dictionary
        """
        try:
            stats = {
                "total_crashed_sessions": 0,
                "recoverable_sessions": 0,
                "successful_recoveries": 0,
                "failed_recoveries": 0,
                "data_integrity_issues": 0,
                "most_common_crash_type": "unknown"
            }
            
            # This would be implemented with proper tracking in a production system
            # For now, return basic structure
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get recovery statistics: {e}")
            return {}