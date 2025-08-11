"""Unit tests for progress tracking functionality."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
import tempfile
import os
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from backend.core.progress_tracker import SQLiteProgressTracker
from backend.core.base import ProcessingResult, QualityResult, DefectResult, ComplianceResult


class TestSQLiteProgressTracker(unittest.TestCase):
    """Test cases for SQLiteProgressTracker class."""
    
    def setUp(self):
        """Set up test progress tracker."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        self.tracker = SQLiteProgressTracker(self.db_path, checkpoint_interval=5)
        
        # Test data
        self.test_input_folder = "/test/input"
        self.test_output_folder = "/test/output"
        self.test_total_images = 20
        self.test_config = {"batch_size": 10, "quality_threshold": 0.8}
    
    def tearDown(self):
        """Clean up test database."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_create_session(self):
        """Test creating a new session."""
        session_id = self.tracker.create_session(
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images,
            config=self.test_config
        )
        
        self.assertIsNotNone(session_id)
        self.assertTrue(session_id.startswith("session_"))
        
        # Verify session was created in database
        session_info = self.tracker.db_manager.get_session_info(session_id)
        self.assertIsNotNone(session_info)
        self.assertEqual(session_info['input_folder'], self.test_input_folder)
        self.assertEqual(session_info['output_folder'], self.test_output_folder)
        self.assertEqual(session_info['total_images'], self.test_total_images)
        self.assertEqual(session_info['config_snapshot'], self.test_config)
    
    def test_create_session_with_custom_id(self):
        """Test creating session with custom ID."""
        custom_id = "my_custom_session"
        session_id = self.tracker.create_session(
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images,
            session_id=custom_id
        )
        
        self.assertEqual(session_id, custom_id)
    
    def test_create_session_failure(self):
        """Test session creation failure handling."""
        # Create first session
        session_id = self.tracker.create_session(
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images,
            session_id="duplicate_test"
        )
        
        # Try to create duplicate session
        with self.assertRaises(RuntimeError):
            self.tracker.create_session(
                input_folder=self.test_input_folder,
                output_folder=self.test_output_folder,
                total_images=self.test_total_images,
                session_id="duplicate_test"
            )
    
    def test_save_checkpoint_basic(self):
        """Test basic checkpoint saving."""
        session_id = self.tracker.create_session(
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        
        # Create test results
        results = []
        for i in range(3):
            result = ProcessingResult(
                image_path=f"/test/image{i}.jpg",
                filename=f"image{i}.jpg",
                final_decision="approved" if i < 2 else "rejected",
                processing_time=1.0,
                timestamp=datetime.now()
            )
            results.append(result)
        
        # Save checkpoint
        success = self.tracker.save_checkpoint(
            session_id=session_id,
            processed_count=3,
            total_count=self.test_total_images,
            results=results
        )
        
        self.assertTrue(success)
        
        # Verify session progress was updated
        session_info = self.tracker.db_manager.get_session_info(session_id)
        self.assertEqual(session_info['processed_images'], 3)
        self.assertEqual(session_info['approved_images'], 2)
        self.assertEqual(session_info['rejected_images'], 1)
        
        # Verify results were saved
        saved_results = self.tracker.db_manager.get_session_results(session_id)
        self.assertEqual(len(saved_results), 3)
    
    def test_save_checkpoint_with_interval(self):
        """Test checkpoint saving with interval trigger."""
        session_id = self.tracker.create_session(
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        
        # Save checkpoint that should trigger checkpoint record (interval = 5)
        results = []
        for i in range(5):
            result = ProcessingResult(
                image_path=f"/test/image{i}.jpg",
                filename=f"image{i}.jpg",
                final_decision="approved",
                processing_time=1.0,
                timestamp=datetime.now()
            )
            results.append(result)
        
        success = self.tracker.save_checkpoint(
            session_id=session_id,
            processed_count=5,
            total_count=self.test_total_images,
            results=results
        )
        
        self.assertTrue(success)
        
        # Verify checkpoint record was created
        with self.tracker.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM checkpoints WHERE session_id = ?', (session_id,))
            checkpoint_count = cursor.fetchone()[0]
            self.assertEqual(checkpoint_count, 1)
    
    def test_load_checkpoint_no_session(self):
        """Test loading checkpoint for non-existent session."""
        checkpoint_data = self.tracker.load_checkpoint("nonexistent_session")
        self.assertIsNone(checkpoint_data)
    
    def test_load_checkpoint_no_checkpoints(self):
        """Test loading checkpoint for session without checkpoints."""
        session_id = self.tracker.create_session(
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        
        checkpoint_data = self.tracker.load_checkpoint(session_id)
        
        self.assertIsNotNone(checkpoint_data)
        self.assertFalse(checkpoint_data['has_checkpoint'])
        self.assertEqual(checkpoint_data['processed_count'], 0)
        self.assertTrue(checkpoint_data['can_resume'])
    
    def test_load_checkpoint_with_data(self):
        """Test loading checkpoint with existing data."""
        session_id = self.tracker.create_session(
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        
        # Save some progress first
        results = []
        for i in range(7):  # More than checkpoint interval
            result = ProcessingResult(
                image_path=f"/test/image{i}.jpg",
                filename=f"image{i}.jpg",
                final_decision="approved",
                processing_time=1.0,
                timestamp=datetime.now()
            )
            results.append(result)
        
        self.tracker.save_checkpoint(
            session_id=session_id,
            processed_count=7,
            total_count=self.test_total_images,
            results=results
        )
        
        # Load checkpoint
        checkpoint_data = self.tracker.load_checkpoint(session_id)
        
        self.assertIsNotNone(checkpoint_data)
        self.assertTrue(checkpoint_data['has_checkpoint'])
        self.assertEqual(checkpoint_data['processed_count'], 7)
        self.assertEqual(checkpoint_data['last_checkpoint_count'], 7)  # Checkpoint at processed count
        self.assertEqual(checkpoint_data['resume_from_index'], 7)  # Resume from after checkpoint
    
    def test_get_session_status(self):
        """Test getting session status."""
        session_id = self.tracker.create_session(
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        
        # Check initial status
        status = self.tracker.get_session_status(session_id)
        self.assertEqual(status, 'running')
        
        # Complete session
        self.tracker.complete_session(session_id, 'completed')
        
        # Check updated status
        status = self.tracker.get_session_status(session_id)
        self.assertEqual(status, 'completed')
        
        # Check non-existent session
        status = self.tracker.get_session_status("nonexistent")
        self.assertIsNone(status)
    
    def test_complete_session(self):
        """Test completing a session."""
        session_id = self.tracker.create_session(
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        
        success = self.tracker.complete_session(session_id, 'completed')
        self.assertTrue(success)
        
        # Verify session was completed
        session_info = self.tracker.db_manager.get_session_info(session_id)
        self.assertEqual(session_info['status'], 'completed')
        self.assertIsNotNone(session_info['end_time'])
    
    def test_complete_session_with_error(self):
        """Test completing session with error."""
        session_id = self.tracker.create_session(
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        
        error_message = "Test error occurred"
        success = self.tracker.complete_session(session_id, 'failed', error_message)
        self.assertTrue(success)
        
        # Verify error was saved
        session_info = self.tracker.db_manager.get_session_info(session_id)
        self.assertEqual(session_info['status'], 'failed')
        self.assertEqual(session_info['error_message'], error_message)
    
    def test_get_progress_summary(self):
        """Test getting progress summary."""
        session_id = self.tracker.create_session(
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        
        # Add some progress
        results = []
        for i in range(8):
            result = ProcessingResult(
                image_path=f"/test/image{i}.jpg",
                filename=f"image{i}.jpg",
                final_decision="approved" if i < 5 else "rejected",
                processing_time=1.0,
                timestamp=datetime.now()
            )
            results.append(result)
        
        self.tracker.save_checkpoint(
            session_id=session_id,
            processed_count=8,
            total_count=self.test_total_images,
            results=results
        )
        
        # Get progress summary
        summary = self.tracker.get_progress_summary(session_id)
        
        self.assertIsNotNone(summary)
        self.assertEqual(summary['session_id'], session_id)
        self.assertEqual(summary['total_images'], self.test_total_images)
        self.assertEqual(summary['processed_images'], 8)
        self.assertEqual(summary['approved_images'], 5)
        self.assertEqual(summary['rejected_images'], 3)
        self.assertEqual(summary['progress_percentage'], 40.0)  # 8/20 * 100
        self.assertEqual(summary['approval_rate'], 62.5)  # 5/8 * 100
        self.assertIn('elapsed_time_seconds', summary)
        self.assertIn('estimated_remaining_seconds', summary)
    
    def test_list_sessions(self):
        """Test listing sessions."""
        # Create multiple sessions
        session_ids = []
        for i in range(3):
            session_id = self.tracker.create_session(
                input_folder=f"/test/input{i}",
                output_folder=f"/test/output{i}",
                total_images=10,
                session_id=f"test_session_{i}"
            )
            session_ids.append(session_id)
        
        # Complete one session
        self.tracker.complete_session(session_ids[0], 'completed')
        
        # List all sessions
        all_sessions = self.tracker.list_sessions()
        self.assertEqual(len(all_sessions), 3)
        
        # List only running sessions
        running_sessions = self.tracker.list_sessions(status='running')
        self.assertEqual(len(running_sessions), 2)
        
        # Verify enhanced information is included
        for session in all_sessions:
            self.assertIn('progress_percentage', session)
            self.assertIn('approval_rate', session)
            self.assertIn('elapsed_time_seconds', session)
    
    def test_get_resumable_sessions(self):
        """Test getting resumable sessions."""
        # Create sessions with different statuses
        running_session = self.tracker.create_session(
            input_folder="/test/input1",
            output_folder="/test/output1",
            total_images=10,
            session_id="running_session"
        )
        
        completed_session = self.tracker.create_session(
            input_folder="/test/input2",
            output_folder="/test/output2",
            total_images=10,
            session_id="completed_session"
        )
        self.tracker.complete_session(completed_session, 'completed')
        
        # Get resumable sessions
        resumable = self.tracker.get_resumable_sessions()
        
        self.assertEqual(len(resumable), 1)
        self.assertEqual(resumable[0]['session_id'], running_session)
    
    def test_cleanup_session_data(self):
        """Test cleaning up session data."""
        session_id = self.tracker.create_session(
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        
        # Add some data
        results = [ProcessingResult(
            image_path="/test/image.jpg",
            filename="image.jpg",
            final_decision="approved",
            processing_time=1.0,
            timestamp=datetime.now()
        )]
        
        self.tracker.save_checkpoint(
            session_id=session_id,
            processed_count=1,
            total_count=self.test_total_images,
            results=results
        )
        
        # Cleanup session
        success = self.tracker.cleanup_session_data(session_id)
        self.assertTrue(success)
        
        # Verify data was deleted
        session_info = self.tracker.db_manager.get_session_info(session_id)
        self.assertIsNone(session_info)
        
        saved_results = self.tracker.db_manager.get_session_results(session_id)
        self.assertEqual(len(saved_results), 0)
    
    def test_get_session_results_summary(self):
        """Test getting session results summary."""
        session_id = self.tracker.create_session(
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        
        # Add results with different outcomes
        results = []
        for i in range(10):
            quality_result = QualityResult(
                sharpness_score=80.0 + i,
                noise_level=0.05,
                exposure_score=85.0,
                color_balance_score=88.0,
                resolution=(1920, 1080),
                file_size=2048000,
                overall_score=80.0 + i,
                passed=True
            )
            
            result = ProcessingResult(
                image_path=f"/test/image{i}.jpg",
                filename=f"image{i}.jpg",
                quality_result=quality_result,
                final_decision="approved" if i < 7 else "rejected",
                rejection_reasons=["low_quality"] if i >= 7 else [],
                processing_time=1.0 + (i * 0.1),
                timestamp=datetime.now()
            )
            results.append(result)
        
        self.tracker.save_checkpoint(
            session_id=session_id,
            processed_count=10,
            total_count=self.test_total_images,
            results=results
        )
        
        # Get results summary
        summary = self.tracker.get_session_results_summary(session_id)
        
        self.assertIsNotNone(summary)
        self.assertEqual(summary['session_id'], session_id)
        self.assertEqual(summary['total_processed'], 10)
        self.assertEqual(summary['approved'], 7)
        self.assertEqual(summary['rejected'], 3)
        self.assertAlmostEqual(summary['avg_processing_time'], 1.45, places=2)  # Average of 1.0 to 1.9
        self.assertAlmostEqual(summary['avg_quality_score'], 84.5, places=1)  # Average of 80-89
        
        # Check rejection breakdown
        self.assertIn('rejection_breakdown', summary)
        self.assertEqual(summary['rejection_breakdown']['low_quality'], 3)
    
    def test_force_checkpoint(self):
        """Test forcing a checkpoint save."""
        session_id = self.tracker.create_session(
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        
        # Add some progress (less than checkpoint interval)
        results = [ProcessingResult(
            image_path="/test/image.jpg",
            filename="image.jpg",
            final_decision="approved",
            processing_time=1.0,
            timestamp=datetime.now()
        )]
        
        self.tracker.save_checkpoint(
            session_id=session_id,
            processed_count=1,
            total_count=self.test_total_images,
            results=results
        )
        
        # Force checkpoint
        success = self.tracker.force_checkpoint(session_id)
        self.assertTrue(success)
        
        # Verify checkpoint was created
        with self.tracker.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM checkpoints WHERE session_id = ?', (session_id,))
            checkpoint_count = cursor.fetchone()[0]
            self.assertEqual(checkpoint_count, 1)
            
            # Verify it's marked as forced
            cursor.execute('SELECT checkpoint_data FROM checkpoints WHERE session_id = ?', (session_id,))
            checkpoint_data = json.loads(cursor.fetchone()[0])
            self.assertTrue(checkpoint_data.get('forced', False))
    
    def test_thread_safety(self):
        """Test thread safety of progress tracker."""
        import threading
        import time
        
        session_id = self.tracker.create_session(
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=100
        )
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(5):
                    result = ProcessingResult(
                        image_path=f"/test/worker{worker_id}_image{i}.jpg",
                        filename=f"worker{worker_id}_image{i}.jpg",
                        final_decision="approved",
                        processing_time=0.1,
                        timestamp=datetime.now()
                    )
                    
                    success = self.tracker.save_checkpoint(
                        session_id=session_id,
                        processed_count=(worker_id * 5) + i + 1,
                        total_count=100,
                        results=[result]
                    )
                    results.append(success)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertTrue(all(results), "Some checkpoint operations failed")
        
        # Verify final state
        summary = self.tracker.get_progress_summary(session_id)
        self.assertEqual(summary['processed_images'], 15)  # 3 workers * 5 results each


if __name__ == '__main__':
    unittest.main()