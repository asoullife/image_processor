#!/usr/bin/env python3
"""Test script for real-time monitoring and notifications system."""

import asyncio
import time
import logging
from datetime import datetime
from typing import List
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.performance_monitor import performance_monitor, PerformanceMonitor
from utils.console_notifier import console_notifier, ConsoleNotificationConfig
from realtime.socketio_manager import socketio_manager, ProgressData
from core.base import ProcessingResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockImageProcessor:
    """Mock image processor for testing monitoring system."""
    
    def __init__(self, session_id: str, total_images: int = 100):
        self.session_id = session_id
        self.total_images = total_images
        self.processed_count = 0
        self.approved_count = 0
        self.rejected_count = 0
        self.start_time = datetime.now()
    
    def process_image(self, image_path: str) -> ProcessingResult:
        """Mock image processing with random results."""
        import random
        
        # Simulate processing time
        time.sleep(random.uniform(0.1, 0.3))
        
        # Random decision
        decision = "approved" if random.random() > 0.3 else "rejected"
        
        result = ProcessingResult(
            image_path=image_path,
            filename=os.path.basename(image_path),
            final_decision=decision,
            rejection_reasons=["Low quality"] if decision == "rejected" else [],
            processing_time=random.uniform(0.1, 0.3),
            timestamp=datetime.now()
        )
        
        return result
    
    async def run_processing(self):
        """Run mock processing with real-time monitoring."""
        logger.info(f"Starting mock processing for session {self.session_id}")
        
        # Start monitoring
        performance_monitor.start_session(self.session_id)
        console_notifier.start_session(self.session_id, self.total_images)
        
        try:
            # Process images in batches
            batch_size = 10
            for batch_start in range(0, self.total_images, batch_size):
                batch_end = min(batch_start + batch_size, self.total_images)
                batch_paths = [f"test_image_{i:04d}.jpg" for i in range(batch_start, batch_end)]
                
                logger.info(f"Processing batch {batch_start//batch_size + 1}: {len(batch_paths)} images")
                
                batch_start_time = time.time()
                batch_results = []
                
                # Process batch
                for image_path in batch_paths:
                    result = self.process_image(image_path)
                    batch_results.append(result)
                    
                    self.processed_count += 1
                    if result.final_decision == "approved":
                        self.approved_count += 1
                    else:
                        self.rejected_count += 1
                
                batch_time = time.time() - batch_start_time
                
                # Update monitoring
                performance_monitor.update_session_progress(
                    self.session_id,
                    self.processed_count,
                    batch_time / len(batch_paths),
                    batch_time
                )
                
                # Get performance metrics
                perf_metrics = performance_monitor.get_current_performance(self.session_id)
                
                # Update console notifier
                current_filename = batch_paths[-1] if batch_paths else ""
                console_notifier.update_progress(
                    self.processed_count,
                    self.approved_count,
                    self.rejected_count,
                    perf_metrics.get('images_per_second', 0.0),
                    current_filename,
                    "processing",
                    perf_metrics
                )
                
                # Create progress data for Socket.IO
                elapsed_time = (datetime.now() - self.start_time).total_seconds()
                processing_speed = self.processed_count / elapsed_time if elapsed_time > 0 else 0.0
                
                # Calculate ETA
                eta_seconds = None
                if processing_speed > 0:
                    remaining = self.total_images - self.processed_count
                    eta_seconds = remaining / processing_speed
                
                progress_data = ProgressData(
                    session_id=self.session_id,
                    current_image=self.processed_count,
                    total_images=self.total_images,
                    percentage=(self.processed_count / self.total_images) * 100,
                    current_filename=current_filename,
                    approved_count=self.approved_count,
                    rejected_count=self.rejected_count,
                    processing_speed=processing_speed,
                    estimated_completion=datetime.now() + timedelta(seconds=eta_seconds) if eta_seconds else None,
                    current_stage="processing",
                    memory_usage_mb=perf_metrics.get('current_memory_mb', 0.0),
                    gpu_usage_percent=perf_metrics.get('current_gpu_percent', 0.0),
                    cpu_usage_percent=perf_metrics.get('current_cpu_percent', 0.0),
                    batch_processing_time=batch_time,
                    avg_image_processing_time=perf_metrics.get('avg_processing_time', 0.0),
                    elapsed_time=elapsed_time,
                    current_batch=(batch_start // batch_size) + 1,
                    total_batches=(self.total_images + batch_size - 1) // batch_size,
                    images_per_second=processing_speed,
                    eta_seconds=eta_seconds
                )
                
                # Broadcast progress (would normally be async)
                logger.info(f"Progress: {progress_data.current_image}/{progress_data.total_images} "
                           f"({progress_data.percentage:.1f}%) - Speed: {progress_data.processing_speed:.1f} img/s")
                
                # Simulate milestone notifications
                if self.processed_count in [25, 50, 75]:
                    console_notifier.notify_milestone(
                        "count",
                        f"{self.processed_count}_images",
                        f"Processed {self.processed_count} images! Current speed: {processing_speed:.1f} img/s"
                    )
                
                # Small delay between batches
                await asyncio.sleep(0.5)
            
            # Processing completed
            total_time = (datetime.now() - self.start_time).total_seconds()
            final_metrics = performance_monitor.end_session(self.session_id)
            
            console_notifier.notify_completion(
                self.processed_count,
                self.approved_count,
                self.rejected_count,
                total_time,
                final_metrics
            )
            
            logger.info(f"Mock processing completed: {self.processed_count} images in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            console_notifier.notify_error(str(e), recoverable=False)
        
        finally:
            console_notifier.stop_session()

async def test_console_notifications():
    """Test console notification system."""
    print("\n" + "="*80)
    print("TESTING CONSOLE NOTIFICATIONS")
    print("="*80)
    
    session_id = "test_console_session"
    processor = MockImageProcessor(session_id, total_images=50)
    
    await processor.run_processing()
    
    print("\nConsole notification test completed!")

async def test_performance_monitoring():
    """Test performance monitoring system."""
    print("\n" + "="*80)
    print("TESTING PERFORMANCE MONITORING")
    print("="*80)
    
    session_id = "test_perf_session"
    
    # Start monitoring
    metrics = performance_monitor.start_session(session_id)
    print(f"Started monitoring session: {session_id}")
    
    # Simulate processing
    for i in range(20):
        await asyncio.sleep(0.1)
        performance_monitor.update_session_progress(session_id, i + 1, 0.1, 1.0)
        
        if (i + 1) % 5 == 0:
            current_metrics = performance_monitor.get_current_performance(session_id)
            print(f"Progress {i + 1}/20: {current_metrics}")
    
    # End monitoring
    final_metrics = performance_monitor.end_session(session_id)
    print(f"Final metrics: {final_metrics}")
    
    print("Performance monitoring test completed!")

async def test_socketio_integration():
    """Test Socket.IO integration (without actual server)."""
    print("\n" + "="*80)
    print("TESTING SOCKET.IO INTEGRATION")
    print("="*80)
    
    session_id = "test_socketio_session"
    
    # Create mock progress data
    from datetime import timedelta
    
    progress_data = ProgressData(
        session_id=session_id,
        current_image=150,
        total_images=1000,
        percentage=15.0,
        current_filename="test_image_0150.jpg",
        approved_count=120,
        rejected_count=30,
        processing_speed=5.2,
        estimated_completion=datetime.now() + timedelta(minutes=15),
        current_stage="processing",
        memory_usage_mb=512.0,
        gpu_usage_percent=75.0,
        cpu_usage_percent=45.0,
        batch_processing_time=2.1,
        avg_image_processing_time=0.19,
        elapsed_time=300.0,
        current_batch=15,
        total_batches=100,
        images_per_second=5.2,
        eta_seconds=900.0
    )
    
    print(f"Mock progress data created: {progress_data}")
    
    # Test milestone checking
    milestone = socketio_manager._check_milestones(session_id, progress_data)
    if milestone:
        print(f"Milestone detected: {milestone}")
    else:
        print("No milestone detected")
    
    print("Socket.IO integration test completed!")

async def main():
    """Run all monitoring tests."""
    print("🚀 Starting Real-time Monitoring Tests")
    print("="*80)
    
    try:
        # Test individual components
        await test_performance_monitoring()
        await test_socketio_integration()
        
        # Test integrated console notifications
        await test_console_notifications()
        
        print("\n✅ All monitoring tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Import required modules
    from datetime import timedelta
    
    # Run tests
    asyncio.run(main())