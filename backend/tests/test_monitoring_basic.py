#!/usr/bin/env python3
"""Basic test for monitoring system components without external dependencies."""

import sys
import os
import time
from datetime import datetime, timedelta

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_socketio_manager():
    """Test Socket.IO manager components."""
    print("Testing Socket.IO Manager...")
    
    try:
        from websocket.socketio_manager import ProgressData, MilestoneData, SocketIOManager
        
        # Test ProgressData creation
        progress_data = ProgressData(
            session_id="test_session",
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
        
        print(f"✅ ProgressData created: {progress_data.session_id} - {progress_data.percentage:.1f}%")
        
        # Test SocketIOManager
        manager = SocketIOManager()
        
        # Test milestone checking
        milestone = manager._check_milestones("test_session", progress_data)
        if milestone:
            print(f"✅ Milestone detected: {milestone.milestone_value}")
        else:
            print("✅ No milestone detected (expected for first run)")
        
        print("✅ Socket.IO Manager test passed")
        return True
        
    except Exception as e:
        print(f"❌ Socket.IO Manager test failed: {e}")
        return False

def test_console_notifier_basic():
    """Test console notifier basic functionality."""
    print("\nTesting Console Notifier...")
    
    try:
        from utils.console_notifier import ConsoleNotifier, ConsoleNotificationConfig
        
        # Create notifier with basic config
        config = ConsoleNotificationConfig(
            show_progress_bar=True,
            show_statistics=True,
            show_performance_metrics=False,  # Disable to avoid psutil dependency
            update_interval=1.0
        )
        
        notifier = ConsoleNotifier(config)
        
        # Test session start
        notifier.start_session("test_session", 100)
        print("✅ Session started")
        
        # Test progress updates
        for i in range(1, 6):
            notifier.update_progress(
                current=i * 20,
                approved=i * 15,
                rejected=i * 5,
                speed=2.5,
                filename=f"test_image_{i:04d}.jpg",
                stage="processing"
            )
            time.sleep(0.5)  # Brief pause to see updates
        
        # Test milestone notification
        notifier.notify_milestone("percentage", "50%", "Halfway through processing!")
        
        # Test completion
        notifier.notify_completion(100, 75, 25, 40.0)
        
        # Stop session
        notifier.stop_session()
        
        print("✅ Console Notifier test passed")
        return True
        
    except Exception as e:
        print(f"❌ Console Notifier test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_processor_integration():
    """Test batch processor integration points."""
    print("\nTesting Batch Processor Integration...")
    
    try:
        from core.batch_processor import BatchConfig, BatchResult
        
        # Test BatchConfig
        config = BatchConfig(
            batch_size=50,
            max_workers=2,
            memory_threshold_mb=512,
            enable_memory_monitoring=False  # Disable to avoid psutil dependency
        )
        
        print(f"✅ BatchConfig created: batch_size={config.batch_size}")
        
        # Test BatchResult
        result = BatchResult(
            batch_id=1,
            processed_count=50,
            success_count=40,
            error_count=10,
            processing_time=25.5,
            memory_usage_mb=256.0,
            results=[],
            errors=[]
        )
        
        print(f"✅ BatchResult created: {result.success_count}/{result.processed_count} successful")
        
        print("✅ Batch Processor Integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Batch Processor Integration test failed: {e}")
        return False

def test_data_structures():
    """Test all data structures used in monitoring."""
    print("\nTesting Data Structures...")
    
    try:
        # Test progress data serialization
        from websocket.socketio_manager import ProgressData, CompletionData, ErrorData
        
        progress = ProgressData(
            session_id="test",
            current_image=50,
            total_images=100,
            percentage=50.0,
            current_filename="test.jpg",
            approved_count=40,
            rejected_count=10,
            processing_speed=2.5,
            current_stage="processing"
        )
        
        # Test model_dump conversion (for JSON serialization)
        progress_dict = progress.model_dump()
        print(f"✅ ProgressData serialization: {len(progress_dict)} fields")
        
        # Test completion data
        completion = CompletionData(
            session_id="test",
            total_processed=100,
            total_approved=80,
            total_rejected=20,
            processing_time=120.0,
            completion_time=datetime.now(),
            output_folder="/output",
            avg_processing_speed=2.5,
            peak_memory_usage=512.0,
            avg_gpu_usage=75.0,
            total_batches=10,
            approval_rate=80.0
        )
        
        completion_dict = completion.model_dump()
        print(f"✅ CompletionData serialization: {len(completion_dict)} fields")
        
        # Test error data
        error = ErrorData(
            session_id="test",
            error_type="processing_error",
            error_message="Test error message",
            timestamp=datetime.now(),
            recoverable=True
        )
        
        error_dict = error.model_dump()
        print(f"✅ ErrorData serialization: {len(error_dict)} fields")
        
        print("✅ Data Structures test passed")
        return True
        
    except Exception as e:
        print(f"❌ Data Structures test failed: {e}")
        return False

def main():
    """Run all basic monitoring tests."""
    print("🚀 Starting Basic Monitoring Tests")
    print("="*60)
    
    tests = [
        test_socketio_manager,
        test_data_structures,
        test_batch_processor_integration,
        test_console_notifier_basic,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "="*60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All basic monitoring tests passed!")
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)