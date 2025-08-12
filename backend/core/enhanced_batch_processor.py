"""Enhanced batch processor with integrated checkpoint and recovery functionality."""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator
from uuid import UUID
from datetime import datetime
import time

import sys
from pathlib import Path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from database.connection import DatabaseManager
from core.checkpoint_manager import CheckpointManager, CheckpointType, SessionState
from core.recovery_service import RecoveryService
from core.services import SessionService
from realtime.socketio_manager import socketio_manager
from analyzers.analyzer_factory import AnalyzerFactory
from config.config_loader import AppConfig

logger = logging.getLogger(__name__)

class EnhancedBatchProcessor:
    """Enhanced batch processor with automatic checkpointing and recovery."""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        config: AppConfig,
        checkpoint_manager: CheckpointManager,
        recovery_service: RecoveryService,
        session_service: SessionService
    ):
        """Initialize enhanced batch processor.
        
        Args:
            db_manager: Database manager
            config: Application configuration
            checkpoint_manager: Checkpoint manager
            recovery_service: Recovery service
            session_service: Session service
        """
        self.db_manager = db_manager
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.recovery_service = recovery_service
        self.session_service = session_service
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Processing configuration
        self.batch_size = getattr(config.processing, 'batch_size', 20)
        self.checkpoint_interval = getattr(config.processing, 'checkpoint_interval', 10)
        self.max_workers = getattr(config.processing, 'max_workers', 4)
        
        # Initialize analyzer factory
        self.analyzer_factory = AnalyzerFactory(config)
        
        # Processing state
        self._processing_sessions: Dict[str, bool] = {}  # session_id -> is_running
        self._session_stats: Dict[str, Dict[str, Any]] = {}
        
    async def start_processing_with_recovery(
        self,
        session_id: UUID,
        image_paths: List[str],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Start processing with automatic recovery detection.
        
        Args:
            session_id: Session UUID
            image_paths: List of image paths to process
            progress_callback: Optional progress callback function
            
        Returns:
            Processing results summary
        """
        session_id_str = str(session_id)
        
        try:
            # Check if session was previously interrupted
            recovery_options = await self.recovery_service.prepare_recovery_options(session_id)
            
            if recovery_options and recovery_options.get("recovery_available"):
                self.logger.info(f"Recovery available for session {session_id}")
                
                # For automatic recovery, use continue option
                success, message, start_index = await self.recovery_service.execute_recovery(
                    session_id=session_id,
                    recovery_option="continue",
                    user_confirmed=True
                )
                
                if success:
                    self.logger.info(f"Session {session_id} recovered, resuming from index {start_index}")
                    # Continue processing from recovered index
                    remaining_paths = image_paths[start_index:] if start_index < len(image_paths) else []
                    if remaining_paths:
                        return await self._process_batch_with_checkpoints(
                            session_id, remaining_paths, start_index, progress_callback
                        )
                    else:
                        return {"status": "completed", "message": "Session already completed"}
                else:
                    self.logger.warning(f"Recovery failed for session {session_id}: {message}")
                    # Fall through to normal processing
            
            # Normal processing from start
            return await self._process_batch_with_checkpoints(
                session_id, image_paths, 0, progress_callback
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start processing for session {session_id}: {e}")
            
            # Create emergency checkpoint
            await self.checkpoint_manager.create_emergency_checkpoint(
                session_id, f"Processing failed: {str(e)}"
            )
            
            raise
    
    async def _process_batch_with_checkpoints(
        self,
        session_id: UUID,
        image_paths: List[str],
        start_index: int = 0,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Process images with automatic checkpointing.
        
        Args:
            session_id: Session UUID
            image_paths: List of image paths to process
            start_index: Index to start processing from
            progress_callback: Optional progress callback function
            
        Returns:
            Processing results summary
        """
        session_id_str = str(session_id)
        self._processing_sessions[session_id_str] = True
        
        # Initialize session stats
        self._session_stats[session_id_str] = {
            "start_time": datetime.now(),
            "processed_count": start_index,
            "approved_count": 0,
            "rejected_count": 0,
            "error_count": 0,
            "last_checkpoint": start_index,
            "processing_rate": 0.0
        }
        
        try:
            # Update session status
            await self.session_service.update_session_progress(
                session_id=session_id,
                processed_count=start_index,
                approved_count=0,
                rejected_count=0
            )
            
            # Process images in batches
            total_images = len(image_paths)
            current_index = start_index
            
            while current_index < total_images and self._processing_sessions.get(session_id_str, False):
                # Calculate batch end
                batch_end = min(current_index + self.batch_size, total_images)
                batch_paths = image_paths[current_index:batch_end]
                
                # Process current batch
                batch_results = await self._process_image_batch(
                    session_id, batch_paths, current_index
                )
                
                # Update counters
                stats = self._session_stats[session_id_str]
                stats["processed_count"] = batch_end
                
                for result in batch_results:
                    if result.get("final_decision") == "approved":
                        stats["approved_count"] += 1
                    else:
                        stats["rejected_count"] += 1
                
                # Calculate processing rate
                elapsed_time = (datetime.now() - stats["start_time"]).total_seconds()
                if elapsed_time > 0:
                    stats["processing_rate"] = (stats["processed_count"] - start_index) / elapsed_time
                
                # Update session progress
                await self.session_service.update_session_progress(
                    session_id=session_id,
                    processed_count=stats["processed_count"],
                    approved_count=stats["approved_count"],
                    rejected_count=stats["rejected_count"]
                )
                
                # Send real-time progress update
                if progress_callback:
                    await progress_callback({
                        "session_id": session_id_str,
                        "processed": stats["processed_count"],
                        "total": total_images,
                        "approved": stats["approved_count"],
                        "rejected": stats["rejected_count"],
                        "progress_percentage": (stats["processed_count"] / total_images) * 100,
                        "processing_rate": stats["processing_rate"],
                        "current_image": batch_paths[-1] if batch_paths else None
                    })
                
                # Create checkpoint if needed
                if (stats["processed_count"] - stats["last_checkpoint"]) >= self.checkpoint_interval:
                    checkpoint_data = await self.checkpoint_manager.create_session_checkpoint(
                        session_id=session_id,
                        checkpoint_type=CheckpointType.IMAGE,
                        force=False
                    )
                    
                    if checkpoint_data:
                        stats["last_checkpoint"] = stats["processed_count"]
                        self.logger.info(
                            f"Checkpoint created at {stats['processed_count']}/{total_images} images"
                        )
                
                # Move to next batch
                current_index = batch_end
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
            
            # Create final checkpoint
            await self.checkpoint_manager.create_session_checkpoint(
                session_id=session_id,
                checkpoint_type=CheckpointType.MILESTONE,
                force=True
            )
            
            # Mark session as completed
            final_stats = self._session_stats[session_id_str]
            
            return {
                "status": "completed",
                "total_processed": final_stats["processed_count"],
                "approved": final_stats["approved_count"],
                "rejected": final_stats["rejected_count"],
                "processing_time_seconds": (datetime.now() - final_stats["start_time"]).total_seconds(),
                "average_processing_rate": final_stats["processing_rate"]
            }
            
        except Exception as e:
            self.logger.error(f"Batch processing failed for session {session_id}: {e}")
            
            # Create emergency checkpoint
            await self.checkpoint_manager.create_emergency_checkpoint(
                session_id, f"Batch processing error: {str(e)}"
            )
            
            raise
        finally:
            # Cleanup
            self._processing_sessions.pop(session_id_str, None)
            self._session_stats.pop(session_id_str, None)
    
    async def _process_image_batch(
        self,
        session_id: UUID,
        image_paths: List[str],
        batch_start_index: int
    ) -> List[Dict[str, Any]]:
        """Process a batch of images.
        
        Args:
            session_id: Session UUID
            image_paths: List of image paths in this batch
            batch_start_index: Starting index of this batch
            
        Returns:
            List of processing results
        """
        batch_results = []
        
        # Process images concurrently within the batch
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single_image(image_path: str, index: int) -> Dict[str, Any]:
            async with semaphore:
                try:
                    start_time = time.time()
                    
                    # Perform image analysis
                    analysis_result = await self._analyze_image(image_path)
                    
                    processing_time = time.time() - start_time
                    
                    # Create result record
                    result = {
                        "session_id": str(session_id),
                        "image_path": image_path,
                        "filename": Path(image_path).name,
                        "batch_index": index,
                        "quality_scores": analysis_result.get("quality_result", {}),
                        "defect_results": analysis_result.get("defect_result", {}),
                        "compliance_results": analysis_result.get("compliance_result", {}),
                        "final_decision": analysis_result.get("final_decision", "rejected"),
                        "rejection_reasons": analysis_result.get("rejection_reasons", []),
                        "processing_time": processing_time,
                        "created_at": datetime.now().isoformat()
                    }
                    
                    # Save to database
                    await self._save_image_result(session_id, result)
                    
                    return result
                    
                except Exception as e:
                    self.logger.error(f"Failed to process image {image_path}: {e}")
                    
                    # Return error result
                    return {
                        "session_id": str(session_id),
                        "image_path": image_path,
                        "filename": Path(image_path).name,
                        "batch_index": index,
                        "final_decision": "rejected",
                        "rejection_reasons": [f"Processing error: {str(e)}"],
                        "processing_time": 0.0,
                        "error": str(e),
                        "created_at": datetime.now().isoformat()
                    }
        
        # Process all images in the batch concurrently
        tasks = [
            process_single_image(image_path, batch_start_index + i)
            for i, image_path in enumerate(image_paths)
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_results = []
        for result in batch_results:
            if isinstance(result, Exception):
                self.logger.error(f"Batch processing exception: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze a single image using the analyzer factory.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Analysis results
        """
        try:
            # Get analyzers
            quality_analyzer = self.analyzer_factory.get_quality_analyzer()
            defect_detector = self.analyzer_factory.get_defect_detector()
            compliance_checker = self.analyzer_factory.get_compliance_checker()
            
            # Perform analyses
            quality_result = quality_analyzer.analyze(image_path)
            defect_result = defect_detector.detect_defects(image_path)
            compliance_result = compliance_checker.check_compliance(image_path, {})
            
            # Determine final decision
            final_decision = "approved"
            rejection_reasons = []
            
            if not quality_result.passed:
                final_decision = "rejected"
                rejection_reasons.append("คุณภาพภาพต่ำ")
            
            if not defect_result.passed:
                final_decision = "rejected"
                rejection_reasons.append("พบความผิดปกติในภาพ")
            
            if not compliance_result.overall_compliance:
                final_decision = "rejected"
                rejection_reasons.append("ไม่ผ่านการตรวจสอบลิขสิทธิ์")
            
            return {
                "quality_result": {
                    "sharpness_score": quality_result.sharpness_score,
                    "noise_level": quality_result.noise_level,
                    "exposure_score": quality_result.exposure_score,
                    "color_balance_score": quality_result.color_balance_score,
                    "overall_score": quality_result.overall_score,
                    "passed": quality_result.passed
                },
                "defect_result": {
                    "defect_count": defect_result.defect_count,
                    "anomaly_score": defect_result.anomaly_score,
                    "confidence_scores": defect_result.confidence_scores,
                    "passed": defect_result.passed
                },
                "compliance_result": {
                    "logo_detections": len(compliance_result.logo_detections),
                    "privacy_violations": len(compliance_result.privacy_violations),
                    "overall_compliance": compliance_result.overall_compliance
                },
                "final_decision": final_decision,
                "rejection_reasons": rejection_reasons
            }
            
        except Exception as e:
            self.logger.error(f"Image analysis failed for {image_path}: {e}")
            return {
                "final_decision": "rejected",
                "rejection_reasons": [f"Analysis error: {str(e)}"],
                "error": str(e)
            }
    
    async def _save_image_result(self, session_id: UUID, result: Dict[str, Any]) -> bool:
        """Save image processing result to database.
        
        Args:
            session_id: Session UUID
            result: Processing result to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self.db_manager.get_session() as session:
                from database.models import ImageResult
                
                image_result = ImageResult(
                    session_id=session_id,
                    image_path=result["image_path"],
                    filename=result["filename"],
                    quality_scores=result.get("quality_scores"),
                    defect_results=result.get("defect_results"),
                    compliance_results=result.get("compliance_results"),
                    final_decision=result["final_decision"],
                    rejection_reasons=result.get("rejection_reasons", []),
                    processing_time=result.get("processing_time", 0.0)
                )
                
                session.add(image_result)
                await session.flush()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save image result: {e}")
            return False
    
    async def pause_processing(self, session_id: UUID) -> bool:
        """Pause processing for a session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            True if paused successfully
        """
        session_id_str = str(session_id)
        
        if session_id_str in self._processing_sessions:
            self._processing_sessions[session_id_str] = False
            
            # Create checkpoint before pausing
            await self.checkpoint_manager.create_session_checkpoint(
                session_id=session_id,
                checkpoint_type=CheckpointType.MANUAL,
                force=True
            )
            
            self.logger.info(f"Processing paused for session {session_id}")
            return True
        
        return False
    
    async def resume_processing(
        self,
        session_id: UUID,
        image_paths: List[str],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Resume processing for a paused session.
        
        Args:
            session_id: Session UUID
            image_paths: Complete list of image paths
            progress_callback: Optional progress callback function
            
        Returns:
            Processing results summary
        """
        try:
            # Get latest checkpoint to determine resume point
            latest_checkpoint = await self.checkpoint_manager.get_latest_checkpoint(session_id)
            
            if latest_checkpoint:
                start_index = latest_checkpoint.session_state.processed_count
                self.logger.info(f"Resuming session {session_id} from index {start_index}")
            else:
                start_index = 0
                self.logger.warning(f"No checkpoint found for session {session_id}, starting from beginning")
            
            # Resume processing
            return await self._process_batch_with_checkpoints(
                session_id, image_paths, start_index, progress_callback
            )
            
        except Exception as e:
            self.logger.error(f"Failed to resume processing for session {session_id}: {e}")
            raise
    
    def get_processing_status(self, session_id: UUID) -> Dict[str, Any]:
        """Get current processing status for a session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            Processing status information
        """
        session_id_str = str(session_id)
        
        if session_id_str in self._session_stats:
            stats = self._session_stats[session_id_str]
            is_running = self._processing_sessions.get(session_id_str, False)
            
            return {
                "session_id": session_id_str,
                "is_running": is_running,
                "processed_count": stats["processed_count"],
                "approved_count": stats["approved_count"],
                "rejected_count": stats["rejected_count"],
                "processing_rate": stats["processing_rate"],
                "start_time": stats["start_time"].isoformat(),
                "last_checkpoint": stats["last_checkpoint"]
            }
        
        return {
            "session_id": session_id_str,
            "is_running": False,
            "status": "not_found"
        }