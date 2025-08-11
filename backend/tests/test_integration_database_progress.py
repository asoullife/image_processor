"""Integration tests for database and progress tracking working together."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
import tempfile
import os
from datetime import datetime

from backend.core.database import DatabaseManager
from backend.core.progress_tracker import SQLiteProgressTracker
from backend.core.base import ProcessingResult, QualityResult


class TestDatabaseProgressIntegration(unittest.TestCase):
    """Integration tests for database and progress tracking."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Create both components using same database
        self.db_manager = DatabaseManager(self.db_path)
        self.progress_tracker = SQLiteProgressTracker(self.db_path, checkpoint_interval=3)
    
    def tearDown(self):
        """Clean up test database."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_complete_workflow_with_resume(self):
        """Test complete workflow including interruption and resume."""
        # Step 1: Start processing session
        session_id = self.progress_tracker.create_session(
            input_folder="/test/input",
            output_folder="/test/output",
            total_images=10,
            config={"batch_size": 5}
        )
        
        # Step 2: Process first batch (3 images - triggers checkpoint)
        batch1_results = []
        for i in range(3):
            quality_result = QualityResult(
                sharpness_score=85.0 + i,
                noise_level=0.05,
                exposure_score=90.0,
                color_balance_score=88.0,
                resolution=(1920, 1080),
                file_size=2048000,
                overall_score=85.0 + i,
                passed=True
            )
            
            result = ProcessingResult(
                image_path=f"/test/batch1_image{i}.jpg",
                filename=f"batch1_image{i}.jpg",
                quality_result=quality_result,
                final_decision="approved",
                processing_time=1.0,
                timestamp=datetime.now()
            )
            batch1_results.append(result)
        
        # Save first checkpoint
        success = self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=3,
            total_count=10,
            results=batch1_results
        )
        self.assertTrue(success)
        
        # Step 3: Simulate interruption - check we can resume
        checkpoint_data = self.progress_tracker.load_checkpoint(session_id)
        self.assertIsNotNone(checkpoint_data)
        self.assertTrue(checkpoint_data['can_resume'])
        self.assertEqual(checkpoint_data['processed_count'], 3)
        self.assertTrue(checkpoint_data['has_checkpoint'])
        
        # Step 4: Resume processing with second batch
        batch2_results = []
        for i in range(3, 7):  # Process 4 more images
            result = ProcessingResult(
                image_path=f"/test/batch2_image{i}.jpg",
                filename=f"batch2_image{i}.jpg",
                final_decision="approved" if i < 6 else "rejected",
                rejection_reasons=["low_quality"] if i >= 6 else [],
                processing_time=1.2,
                timestamp=datetime.now()
            )
            batch2_results.append(result)
        
        # Save second checkpoint (should trigger at 6 images)
        success = self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=7,
            total_count=10,
            results=batch2_results
        )
        self.assertTrue(success)
        
        # Step 5: Complete processing
        success = self.progress_tracker.complete_session(session_id, 'completed')
        self.assertTrue(success)
        
        # Step 6: Verify final state
        progress_summary = self.progress_tracker.get_progress_summary(session_id)
        self.assertEqual(progress_summary['status'], 'completed')
        self.assertEqual(progress_summary['processed_images'], 7)
        self.assertEqual(progress_summary['approved_images'], 6)
        self.assertEqual(progress_summary['rejected_images'], 1)
        self.assertEqual(progress_summary['progress_percentage'], 70.0)  # 7/10 * 100
        
        # Step 7: Verify all results were saved
        all_results = self.db_manager.get_session_results(session_id)
        self.assertEqual(len(all_results), 7)
        
        # Step 8: Verify checkpoint records exist
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM checkpoints WHERE session_id = ?', (session_id,))
            checkpoint_count = cursor.fetchone()[0]
            self.assertEqual(checkpoint_count, 2)  # Two checkpoints at 3 and 6 images
        
        # Step 9: Get results summary
        results_summary = self.progress_tracker.get_session_results_summary(session_id)
        self.assertIsNotNone(results_summary)
        self.assertEqual(results_summary['total_processed'], 7)
        self.assertEqual(results_summary['approved'], 6)
        self.assertEqual(results_summary['rejected'], 1)
        self.assertIn('rejection_breakdown', results_summary)
    
    def test_multiple_sessions_isolation(self):
        """Test that multiple sessions are properly isolated."""
        # Create two sessions
        session1_id = self.progress_tracker.create_session(
            input_folder="/test/input1",
            output_folder="/test/output1",
            total_images=5,
            session_id="session_1"
        )
        
        session2_id = self.progress_tracker.create_session(
            input_folder="/test/input2",
            output_folder="/test/output2",
            total_images=8,
            session_id="session_2"
        )
        
        # Add results to both sessions
        for session_id, prefix in [(session1_id, "s1"), (session2_id, "s2")]:
            results = []
            for i in range(2):
                result = ProcessingResult(
                    image_path=f"/test/{prefix}_image{i}.jpg",
                    filename=f"{prefix}_image{i}.jpg",
                    final_decision="approved",
                    processing_time=1.0,
                    timestamp=datetime.now()
                )
                results.append(result)
            
            self.progress_tracker.save_checkpoint(
                session_id=session_id,
                processed_count=2,
                total_count=5 if session_id == session1_id else 8,
                results=results
            )
        
        # Verify sessions are isolated
        session1_results = self.db_manager.get_session_results(session1_id)
        session2_results = self.db_manager.get_session_results(session2_id)
        
        self.assertEqual(len(session1_results), 2)
        self.assertEqual(len(session2_results), 2)
        
        # Verify results belong to correct sessions
        for result in session1_results:
            self.assertTrue(result['filename'].startswith('s1_'))
        
        for result in session2_results:
            self.assertTrue(result['filename'].startswith('s2_'))
        
        # Verify progress summaries are separate
        summary1 = self.progress_tracker.get_progress_summary(session1_id)
        summary2 = self.progress_tracker.get_progress_summary(session2_id)
        
        self.assertEqual(summary1['total_images'], 5)
        self.assertEqual(summary2['total_images'], 8)
        self.assertEqual(summary1['processed_images'], 2)
        self.assertEqual(summary2['processed_images'], 2)
    
    def test_error_recovery_scenarios(self):
        """Test error recovery and data integrity."""
        session_id = self.progress_tracker.create_session(
            input_folder="/test/input",
            output_folder="/test/output",
            total_images=5
        )
        
        # Simulate processing with some errors
        results = []
        for i in range(3):
            result = ProcessingResult(
                image_path=f"/test/image{i}.jpg",
                filename=f"image{i}.jpg",
                final_decision="approved" if i < 2 else "rejected",
                error_message="Processing error occurred" if i == 2 else None,
                processing_time=1.0,
                timestamp=datetime.now()
            )
            results.append(result)
        
        # Save checkpoint with mixed results
        success = self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=3,
            total_count=5,
            results=results
        )
        self.assertTrue(success)
        
        # Simulate system failure and recovery
        checkpoint_data = self.progress_tracker.load_checkpoint(session_id)
        self.assertIsNotNone(checkpoint_data)
        self.assertTrue(checkpoint_data['can_resume'])
        
        # Verify error was recorded
        saved_results = self.db_manager.get_session_results(session_id)
        error_results = [r for r in saved_results if r['error_message'] is not None]
        self.assertEqual(len(error_results), 1)
        self.assertEqual(error_results[0]['error_message'], "Processing error occurred")
        
        # Complete session with error status
        self.progress_tracker.complete_session(session_id, 'failed', 'System encountered errors')
        
        # Verify final status
        final_status = self.progress_tracker.get_session_status(session_id)
        self.assertEqual(final_status, 'failed')
        
        session_info = self.db_manager.get_session_info(session_id)
        self.assertEqual(session_info['error_message'], 'System encountered errors')


if __name__ == '__main__':
    unittest.main()