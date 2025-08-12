"""Integration utilities for Socket.IO with processing engine."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from ..realtime.socketio_manager import socketio_manager, ProgressData, ErrorData, CompletionData

logger = logging.getLogger(__name__)

class ProcessingProgressBroadcaster:
    """Utility class to broadcast processing progress via Socket.IO."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.last_update_time = datetime.now()
        self.processed_count = 0
        self.total_count = 0
        self.approved_count = 0
        self.rejected_count = 0
        self.current_filename = ""
        self.current_stage = "initializing"
        self.processing_speeds = []  # Track recent speeds for averaging

    async def initialize(self, total_images: int, stage: str = "scanning"):
        """Initialize processing session."""
        self.total_count = total_images
        self.current_stage = stage
        
        await socketio_manager.broadcast_stage_change(
            self.session_id,
            stage,
            f"Found {total_images} images to process"
        )

    async def update_progress(
        self,
        current_image: int,
        filename: str,
        approved: bool = None,
        stage: str = None
    ):
        """Update processing progress."""
        self.processed_count = current_image
        self.current_filename = filename
        
        if stage:
            self.current_stage = stage

        if approved is not None:
            if approved:
                self.approved_count += 1
            else:
                self.rejected_count += 1

        # Calculate processing speed
        now = datetime.now()
        time_diff = (now - self.last_update_time).total_seconds()
        if time_diff > 0:
            speed = 1 / time_diff  # images per second
            self.processing_speeds.append(speed)
            # Keep only last 10 speeds for averaging
            if len(self.processing_speeds) > 10:
                self.processing_speeds.pop(0)

        self.last_update_time = now

        # Calculate average speed and ETA
        avg_speed = sum(self.processing_speeds) / len(self.processing_speeds) if self.processing_speeds else 0
        remaining_images = self.total_count - self.processed_count
        eta = None
        if avg_speed > 0 and remaining_images > 0:
            eta_seconds = remaining_images / avg_speed
            eta = now + timedelta(seconds=eta_seconds)

        # Create progress data
        progress_data = ProgressData(
            session_id=self.session_id,
            current_image=current_image,
            total_images=self.total_count,
            percentage=(current_image / self.total_count * 100) if self.total_count > 0 else 0,
            current_filename=filename,
            approved_count=self.approved_count,
            rejected_count=self.rejected_count,
            processing_speed=avg_speed,
            estimated_completion=eta.isoformat() if eta else None,
            current_stage=self.current_stage
        )

        # Broadcast progress
        await socketio_manager.broadcast_progress(self.session_id, progress_data)

    async def update_stage(self, stage: str, message: str = ""):
        """Update processing stage."""
        self.current_stage = stage
        await socketio_manager.broadcast_stage_change(self.session_id, stage, message)

    async def broadcast_error(
        self,
        error_type: str,
        error_message: str,
        recoverable: bool = True
    ):
        """Broadcast processing error."""
        error_data = ErrorData(
            session_id=self.session_id,
            error_type=error_type,
            error_message=error_message,
            timestamp=datetime.now(),
            recoverable=recoverable
        )

        await socketio_manager.broadcast_error(self.session_id, error_data)

    async def broadcast_completion(self, output_folder: str):
        """Broadcast processing completion."""
        end_time = datetime.now()
        processing_time = (end_time - self.start_time).total_seconds()

        completion_data = CompletionData(
            session_id=self.session_id,
            total_processed=self.processed_count,
            total_approved=self.approved_count,
            total_rejected=self.rejected_count,
            processing_time=processing_time,
            completion_time=end_time,
            output_folder=output_folder
        )

        await socketio_manager.broadcast_completion(self.session_id, completion_data)

class SocketIOProcessingIntegration:
    """Integration class for connecting processing engine with Socket.IO."""

    @staticmethod
    async def create_broadcaster(session_id: str) -> ProcessingProgressBroadcaster:
        """Create a new progress broadcaster for a session."""
        return ProcessingProgressBroadcaster(session_id)

    @staticmethod
    async def notify_session_start(session_id: str, total_images: int):
        """Notify clients that processing has started."""
        await socketio_manager.broadcast_stage_change(
            session_id,
            "starting",
            f"Starting processing of {total_images} images"
        )

    @staticmethod
    async def notify_session_pause(session_id: str):
        """Notify clients that processing has been paused."""
        await socketio_manager.broadcast_stage_change(
            session_id,
            "paused",
            "Processing has been paused"
        )

    @staticmethod
    async def notify_session_resume(session_id: str):
        """Notify clients that processing has been resumed."""
        await socketio_manager.broadcast_stage_change(
            session_id,
            "resumed",
            "Processing has been resumed"
        )

    @staticmethod
    async def notify_batch_complete(session_id: str, batch_number: int, total_batches: int):
        """Notify clients that a batch has been completed."""
        await socketio_manager.broadcast_stage_change(
            session_id,
            "batch_complete",
            f"Completed batch {batch_number} of {total_batches}"
        )

    @staticmethod
    async def notify_checkpoint_saved(session_id: str, checkpoint_number: int):
        """Notify clients that a checkpoint has been saved."""
        await socketio_manager.broadcast_stage_change(
            session_id,
            "checkpoint_saved",
            f"Checkpoint {checkpoint_number} saved"
        )

    @staticmethod
    async def cleanup_session(session_id: str):
        """Clean up session data."""
        await socketio_manager.cleanup_session(session_id)

# Example usage function for testing
async def example_processing_with_socketio(session_id: str, image_paths: list):
    """Example function showing how to integrate Socket.IO with processing."""
    
    # Create broadcaster
    broadcaster = await SocketIOProcessingIntegration.create_broadcaster(session_id)
    
    try:
        # Initialize
        await broadcaster.initialize(len(image_paths), "scanning")
        
        # Simulate processing
        for i, image_path in enumerate(image_paths, 1):
            # Update stage periodically
            if i % 100 == 0:
                await broadcaster.update_stage("processing", f"Processing batch {i//100 + 1}")
            
            # Simulate image processing
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Simulate approval/rejection (random for example)
            import random
            approved = random.choice([True, False])
            
            # Update progress
            await broadcaster.update_progress(
                current_image=i,
                filename=image_path.split('/')[-1],
                approved=approved,
                stage="processing"
            )
            
            # Simulate error occasionally
            if i % 500 == 0:
                await broadcaster.broadcast_error(
                    "processing_warning",
                    f"Minor issue processing image {i}, continuing...",
                    recoverable=True
                )
        
        # Complete processing
        await broadcaster.broadcast_completion("/path/to/output")
        
    except Exception as e:
        # Handle errors
        await broadcaster.broadcast_error(
            "processing_error",
            f"Processing failed: {str(e)}",
            recoverable=False
        )
        raise
    
    finally:
        # Cleanup
        await SocketIOProcessingIntegration.cleanup_session(session_id)