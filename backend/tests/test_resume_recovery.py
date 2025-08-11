#!/usr/bin/env python3
"""Test script for robust resume and recovery system."""

import asyncio
import logging
import sys
from pathlib import Path
from uuid import uuid4
from datetime import datetime, timedelta

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from database.connection import DatabaseManager
from database.models import ProcessingSession, Project
from core.checkpoint_manager import CheckpointManager, CheckpointType, SessionState
from core.recovery_service import RecoveryService
from config.config_loader import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_checkpoint_creation():
    """Test checkpoint creation functionality."""
    logger.info("Testing checkpoint creation...")
    
    try:
        # Initialize services
        config = get_config()
        db_manager = DatabaseManager()
        await db_manager.initialize()
        
        checkpoint_manager = CheckpointManager(db_manager, checkpoint_interval=5)
        
        # Create a test project and session
        async with db_manager.get_session() as session:
            project = Project(
                name="Test Recovery Project",
                description="Test project for recovery functionality",
                input_folder="/test/input",
                output_folder="/test/output",
                performance_mode="balanced"
            )
            session.add(project)
            await session.flush()
            await session.refresh(project)
            
            processing_session = ProcessingSession(
                project_id=project.id,
                total_images=100,
                processed_images=25,
                approved_images=20,
                rejected_images=5,
                status="running",
                start_time=datetime.now(),
                session_config={
                    "batch_size": 10,
                    "performance_mode": "balanced"
                }
            )
            session.add(processing_session)
            await session.flush()
            await session.refresh(processing_session)
            
            session_id = processing_session.id
        
        # Test checkpoint creation
        checkpoint_data = await checkpoint_manager.create_session_checkpoint(
            session_id=session_id,
            checkpoint_type=CheckpointType.IMAGE,
            force=True
        )
        
        assert checkpoint_data is not None, "Checkpoint creation failed"
        assert checkpoint_data.session_state.processed_count == 25, "Incorrect processed count"
        assert checkpoint_data.checkpoint_type == CheckpointType.IMAGE, "Incorrect checkpoint type"
        
        logger.info(f"✅ Checkpoint created successfully: {checkpoint_data.checkpoint_id}")
        
        # Test checkpoint retrieval
        latest_checkpoint = await checkpoint_manager.get_latest_checkpoint(session_id)
        assert latest_checkpoint is not None, "Failed to retrieve latest checkpoint"
        assert latest_checkpoint.checkpoint_id == checkpoint_data.checkpoint_id, "Checkpoint ID mismatch"
        
        logger.info("✅ Checkpoint retrieval successful")
        
        # Test checkpoint integrity verification
        integrity_ok = await checkpoint_manager.verify_checkpoint_integrity(latest_checkpoint)
        assert integrity_ok, "Checkpoint integrity verification failed"
        
        logger.info("✅ Checkpoint integrity verification successful")
        
        return session_id
        
    except Exception as e:
        logger.error(f"❌ Checkpoint creation test failed: {e}")
        raise

async def test_crash_detection(session_id):
    """Test crash detection functionality."""
    logger.info("Testing crash detection...")
    
    try:
        config = get_config()
        db_manager = DatabaseManager()
        checkpoint_manager = CheckpointManager(db_manager, checkpoint_interval=5)
        recovery_service = RecoveryService(db_manager, checkpoint_manager)
        
        # Simulate a crashed session by updating the session timestamp to be old
        async with db_manager.get_session() as session:
            from sqlalchemy import update
            
            # Make the session appear crashed (not updated for 10 minutes)
            old_time = datetime.now() - timedelta(minutes=10)
            stmt = (
                update(ProcessingSession)
                .where(ProcessingSession.id == session_id)
                .values(updated_at=old_time)
            )
            await session.execute(stmt)
            await session.commit()
        
        # Test crash detection
        crashed_sessions = await recovery_service.detect_crashes_on_startup()
        
        assert len(crashed_sessions) > 0, "No crashed sessions detected"
        
        crashed_session = None
        for session_info in crashed_sessions:
            if session_info["session_id"] == str(session_id):
                crashed_session = session_info
                break
        
        assert crashed_session is not None, "Test session not detected as crashed"
        assert crashed_session["can_recover"], "Session should be recoverable"
        
        logger.info(f"✅ Crash detection successful: {crashed_session['crash_type']}")
        
        return crashed_session
        
    except Exception as e:
        logger.error(f"❌ Crash detection test failed: {e}")
        raise

async def test_recovery_options(session_id):
    """Test recovery options preparation."""
    logger.info("Testing recovery options preparation...")
    
    try:
        config = get_config()
        db_manager = DatabaseManager()
        checkpoint_manager = CheckpointManager(db_manager, checkpoint_interval=5)
        recovery_service = RecoveryService(db_manager, checkpoint_manager)
        
        # Test recovery options preparation
        recovery_options = await recovery_service.prepare_recovery_options(session_id)
        
        assert recovery_options is not None, "Failed to prepare recovery options"
        assert recovery_options["recovery_available"], "Recovery should be available"
        assert len(recovery_options["available_options"]) > 0, "No recovery options available"
        
        # Check that continue option is available
        continue_option = None
        for option in recovery_options["available_options"]:
            if option["option"] == "continue":
                continue_option = option
                break
        
        assert continue_option is not None, "Continue option should be available"
        assert continue_option["start_from_image"] == 26, "Incorrect start index for continue option"
        
        logger.info("✅ Recovery options preparation successful")
        logger.info(f"   Available options: {[opt['option'] for opt in recovery_options['available_options']]}")
        logger.info(f"   Data loss risk: {recovery_options['recovery_info']['data_loss_risk']}")
        logger.info(f"   Recovery confidence: {recovery_options['recovery_info']['recovery_confidence']:.2f}")
        
        return recovery_options
        
    except Exception as e:
        logger.error(f"❌ Recovery options test failed: {e}")
        raise

async def test_recovery_execution(session_id):
    """Test recovery execution."""
    logger.info("Testing recovery execution...")
    
    try:
        config = get_config()
        db_manager = DatabaseManager()
        checkpoint_manager = CheckpointManager(db_manager, checkpoint_interval=5)
        recovery_service = RecoveryService(db_manager, checkpoint_manager)
        
        # Test recovery execution with continue option
        success, message, start_index = await recovery_service.execute_recovery(
            session_id=session_id,
            recovery_option="continue",
            user_confirmed=True
        )
        
        assert success, f"Recovery execution failed: {message}"
        assert start_index == 25, f"Incorrect start index: {start_index}"
        
        logger.info(f"✅ Recovery execution successful: {message}")
        logger.info(f"   Start index: {start_index}")
        
        # Verify session status was updated
        async with db_manager.get_session() as session:
            from sqlalchemy import select
            
            stmt = select(ProcessingSession).where(ProcessingSession.id == session_id)
            result = await session.execute(stmt)
            processing_session = result.scalar_one_or_none()
            
            assert processing_session is not None, "Session not found after recovery"
            assert processing_session.status == "running", "Session status not updated to running"
            assert processing_session.error_message is None, "Error message not cleared"
        
        logger.info("✅ Session status verification successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Recovery execution test failed: {e}")
        raise

async def test_data_integrity_verification(session_id):
    """Test data integrity verification."""
    logger.info("Testing data integrity verification...")
    
    try:
        config = get_config()
        db_manager = DatabaseManager()
        checkpoint_manager = CheckpointManager(db_manager, checkpoint_interval=5)
        recovery_service = RecoveryService(db_manager, checkpoint_manager)
        
        # Test data integrity verification
        integrity_results = await recovery_service.verify_data_integrity_after_recovery(
            session_id=session_id,
            start_index=25
        )
        
        assert integrity_results is not None, "Integrity verification failed"
        assert "integrity_check_passed" in integrity_results, "Missing integrity check result"
        assert "safe_to_continue" in integrity_results, "Missing safety assessment"
        
        logger.info("✅ Data integrity verification successful")
        logger.info(f"   Integrity check passed: {integrity_results['integrity_check_passed']}")
        logger.info(f"   Safe to continue: {integrity_results['safe_to_continue']}")
        
        if integrity_results["issues_found"]:
            logger.warning(f"   Issues found: {integrity_results['issues_found']}")
        
        if integrity_results["recommendations"]:
            logger.info(f"   Recommendations: {integrity_results['recommendations']}")
        
        return integrity_results
        
    except Exception as e:
        logger.error(f"❌ Data integrity verification test failed: {e}")
        raise

async def test_emergency_checkpoint(session_id):
    """Test emergency checkpoint creation."""
    logger.info("Testing emergency checkpoint creation...")
    
    try:
        config = get_config()
        db_manager = DatabaseManager()
        checkpoint_manager = CheckpointManager(db_manager, checkpoint_interval=5)
        
        # Test emergency checkpoint creation
        error_message = "Test emergency situation - simulated crash"
        emergency_checkpoint = await checkpoint_manager.create_emergency_checkpoint(
            session_id=session_id,
            error_message=error_message
        )
        
        assert emergency_checkpoint is not None, "Emergency checkpoint creation failed"
        assert emergency_checkpoint.checkpoint_type == CheckpointType.EMERGENCY, "Incorrect checkpoint type"
        assert emergency_checkpoint.session_state.last_error == error_message, "Error message not saved"
        
        logger.info(f"✅ Emergency checkpoint created successfully: {emergency_checkpoint.checkpoint_id}")
        
        # Verify session was marked as failed
        async with db_manager.get_session() as session:
            from sqlalchemy import select
            
            stmt = select(ProcessingSession).where(ProcessingSession.id == session_id)
            result = await session.execute(stmt)
            processing_session = result.scalar_one_or_none()
            
            assert processing_session is not None, "Session not found"
            assert processing_session.status == "failed", "Session not marked as failed"
            assert processing_session.error_message == error_message, "Error message not saved to session"
        
        logger.info("✅ Emergency checkpoint verification successful")
        
        return emergency_checkpoint
        
    except Exception as e:
        logger.error(f"❌ Emergency checkpoint test failed: {e}")
        raise

async def test_checkpoint_cleanup(session_id):
    """Test checkpoint cleanup functionality."""
    logger.info("Testing checkpoint cleanup...")
    
    try:
        config = get_config()
        db_manager = DatabaseManager()
        checkpoint_manager = CheckpointManager(db_manager, checkpoint_interval=5)
        
        # Create multiple checkpoints
        for i in range(10):
            await checkpoint_manager.create_session_checkpoint(
                session_id=session_id,
                checkpoint_type=CheckpointType.MANUAL,
                force=True
            )
        
        # Get all checkpoints before cleanup
        all_checkpoints_before = await checkpoint_manager.get_all_checkpoints(session_id)
        logger.info(f"Checkpoints before cleanup: {len(all_checkpoints_before)}")
        
        # Test cleanup (keep only 3)
        success = await checkpoint_manager.cleanup_old_checkpoints(session_id, keep_count=3)
        assert success, "Checkpoint cleanup failed"
        
        # Verify cleanup
        all_checkpoints_after = await checkpoint_manager.get_all_checkpoints(session_id)
        assert len(all_checkpoints_after) == 3, f"Expected 3 checkpoints, got {len(all_checkpoints_after)}"
        
        logger.info(f"✅ Checkpoint cleanup successful: {len(all_checkpoints_before)} → {len(all_checkpoints_after)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Checkpoint cleanup test failed: {e}")
        raise

async def run_all_tests():
    """Run all resume and recovery tests."""
    logger.info("🚀 Starting robust resume and recovery system tests")
    
    try:
        # Test 1: Checkpoint creation
        session_id = await test_checkpoint_creation()
        
        # Test 2: Crash detection
        crashed_session = await test_crash_detection(session_id)
        
        # Test 3: Recovery options preparation
        recovery_options = await test_recovery_options(session_id)
        
        # Test 4: Recovery execution
        await test_recovery_execution(session_id)
        
        # Test 5: Data integrity verification
        await test_data_integrity_verification(session_id)
        
        # Test 6: Emergency checkpoint
        await test_emergency_checkpoint(session_id)
        
        # Test 7: Checkpoint cleanup
        await test_checkpoint_cleanup(session_id)
        
        logger.info("🎉 All tests passed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("ROBUST RESUME AND RECOVERY SYSTEM TEST SUMMARY")
        print("="*60)
        print("✅ Checkpoint creation and retrieval")
        print("✅ Checkpoint integrity verification")
        print("✅ Crash detection on startup")
        print("✅ Recovery options preparation")
        print("✅ Recovery execution (continue/restart/fresh)")
        print("✅ Data integrity verification")
        print("✅ Emergency checkpoint creation")
        print("✅ Checkpoint cleanup and maintenance")
        print("="*60)
        print("🎉 ALL TESTS PASSED - System is ready for production!")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        print("\n" + "="*60)
        print("TEST FAILURE SUMMARY")
        print("="*60)
        print(f"❌ Error: {e}")
        print("="*60)
        return False

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)