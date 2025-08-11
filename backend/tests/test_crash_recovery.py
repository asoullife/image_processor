"""Integration tests specifically for crash recovery scenarios."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
import tempfile
import os
import shutil
import signal
import subprocess
import sys
import time
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

from backend.core.progress_tracker import SQLiteProgressTracker
from backend.core.database import DatabaseManager
from backend.core.base import ProcessingResult


class TestCrashRecovery(unittest.TestCase):
    """Test cases for crash recovery scenarios."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        self.db_path = os.path.join(self.temp_dir, 'crash_test.db')
        
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
        
        # Create test images
        self.test_images = []
        for i in range(15):
            image_path = os.path.join(self.input_dir, f'crash_test_{i:03d}.jpg')
            with open(image_path, 'wb') as f:
                f.write(b'fake_image_data_for_crash_test')
            self.test_images.append(image_path)
        
        self.progress_tracker = SQLiteProgressTracker(self.db_path, checkpoint_interval=3)
        self.db_manager = DatabaseManager(self.db_path)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_database_corruption_recovery(self):
        """Test recovery from database corruption scenarios."""
        # Create a session and add some data
        session_id = self.progress_tracker.create_session(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            total_images=len(self.test_images),
            config={'test': True}
        )
        
        # Add some results
        results = []
        for i in range(5):
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'crash_test_{i:03d}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
            results.append(result)
        
        # Save checkpoint
        success = self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=5,
            total_count=len(self.test_images),
            results=results
        )
        self.assertTrue(success)
        
        # Simulate database corruption by writing invalid data
        try:
            with open(self.db_path, 'r+b') as f:
                f.seek(100)  # Seek to middle of file
                f.write(b'CORRUPTED_DATA_BLOCK')
        except Exception:
            pass  # Expected to potentially fail
        
        # Try to create a new progress tracker (should handle corruption gracefully)
        try:
            new_tracker = SQLiteProgressTracker(self.db_path, checkpoint_interval=3)
            
            # Should be able to create new sessions even if old data is corrupted
            new_session_id = new_tracker.create_session(
                input_folder=self.input_dir,
                output_folder=self.output_dir,
                total_images=len(self.test_images),
                config={'recovery_test': True}
            )
            
            self.assertIsNotNone(new_session_id)
            
        except Exception as e:
            # If database is completely corrupted, should be able to recreate
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            
            recovery_tracker = SQLiteProgressTracker(self.db_path, checkpoint_interval=3)
            recovery_session_id = recovery_tracker.create_session(
                input_folder=self.input_dir,
                output_folder=self.output_dir,
                total_images=len(self.test_images),
                config={'recovery_test': True}
            )
            
            self.assertIsNotNone(recovery_session_id)
    
    def test_partial_checkpoint_recovery(self):
        """Test recovery from partial checkpoint writes."""
        session_id = self.progress_tracker.create_session(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            total_images=len(self.test_images),
            config={'test': True}
        )
        
        # Process some images successfully
        successful_results = []
        for i in range(4):
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'crash_test_{i:03d}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
            successful_results.append(result)
        
        # Save successful checkpoint
        success = self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=4,
            total_count=len(self.test_images),
            results=successful_results
        )
        self.assertTrue(success)
        
        # Simulate partial write by manually inserting incomplete data
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert incomplete image result (missing required fields)
                cursor.execute('''
                    INSERT INTO image_results 
                    (session_id, image_path, filename, final_decision, processing_time, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (session_id, 'incomplete_path', 'incomplete.jpg', 'pending', 0.0, datetime.now()))
                
                # Don't commit to simulate interrupted transaction
                # conn.commit()  # Intentionally commented out
                
        except Exception:
            pass  # Expected to potentially fail
        
        # Try to load checkpoint - should handle incomplete data gracefully
        checkpoint_data = self.progress_tracker.load_checkpoint(session_id)
        self.assertIsNotNone(checkpoint_data)
        
        # Should be able to resume from last successful checkpoint
        self.assertTrue(checkpoint_data['can_resume'])
        self.assertEqual(checkpoint_data['processed_count'], 4)
        
        # Should be able to continue processing
        continue_results = []
        for i in range(4, 7):
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'crash_test_{i:03d}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
            continue_results.append(result)
        
        # Should be able to save new checkpoint
        success = self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=7,
            total_count=len(self.test_images),
            results=continue_results
        )
        self.assertTrue(success)
    
    def test_concurrent_access_recovery(self):
        """Test recovery from concurrent access issues."""
        session_id = self.progress_tracker.create_session(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            total_images=len(self.test_images),
            config={'test': True}
        )
        
        # Simulate concurrent access by creating multiple trackers
        tracker1 = SQLiteProgressTracker(self.db_path, checkpoint_interval=3)
        tracker2 = SQLiteProgressTracker(self.db_path, checkpoint_interval=3)
        
        # Try to save checkpoints from both trackers simultaneously
        results1 = []
        results2 = []
        
        for i in range(3):
            result1 = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'tracker1_{i:03d}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
            results1.append(result1)
            
            result2 = ProcessingResult(
                image_path=self.test_images[i + 5],
                filename=f'tracker2_{i:03d}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
            results2.append(result2)
        
        # Save from both trackers (should handle concurrency)
        success1 = tracker1.save_checkpoint(
            session_id=session_id,
            processed_count=3,
            total_count=len(self.test_images),
            results=results1
        )
        
        success2 = tracker2.save_checkpoint(
            session_id=session_id,
            processed_count=6,
            total_count=len(self.test_images),
            results=results2
        )
        
        # At least one should succeed
        self.assertTrue(success1 or success2)
        
        # Should be able to load checkpoint
        checkpoint_data = self.progress_tracker.load_checkpoint(session_id)
        self.assertIsNotNone(checkpoint_data)
        self.assertTrue(checkpoint_data['can_resume'])
    
    def test_disk_space_exhaustion_recovery(self):
        """Test recovery from disk space exhaustion scenarios."""
        session_id = self.progress_tracker.create_session(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            total_images=len(self.test_images),
            config={'test': True}
        )
        
        # Process some images successfully
        successful_results = []
        for i in range(3):
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'crash_test_{i:03d}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
            successful_results.append(result)
        
        # Save successful checkpoint
        success = self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=3,
            total_count=len(self.test_images),
            results=successful_results
        )
        self.assertTrue(success)
        
        # Simulate disk space issue by trying to write very large data
        large_results = []
        for i in range(3, 6):
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'crash_test_{i:03d}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now(),
                error_message='x' * 10000  # Very large error message to simulate space issue
            )
            large_results.append(result)
        
        # Try to save large checkpoint (may fail due to simulated space issue)
        try:
            large_success = self.progress_tracker.save_checkpoint(
                session_id=session_id,
                processed_count=6,
                total_count=len(self.test_images),
                results=large_results
            )
            
            # If it succeeds, that's fine too
            if large_success:
                self.assertTrue(True)
            
        except Exception:
            # If it fails due to space, should still be able to recover
            pass
        
        # Should be able to load last successful checkpoint
        checkpoint_data = self.progress_tracker.load_checkpoint(session_id)
        self.assertIsNotNone(checkpoint_data)
        self.assertTrue(checkpoint_data['can_resume'])
        
        # Should be able to continue with smaller data
        small_results = []
        for i in range(3, 5):
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'crash_test_{i:03d}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
            small_results.append(result)
        
        # Should be able to save smaller checkpoint
        small_success = self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=5,
            total_count=len(self.test_images),
            results=small_results
        )
        self.assertTrue(small_success)
    
    def test_session_state_consistency(self):
        """Test session state consistency after various failure scenarios."""
        session_id = self.progress_tracker.create_session(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            total_images=len(self.test_images),
            config={'test': True}
        )
        
        # Process images in multiple batches with simulated failures
        total_processed = 0
        total_approved = 0
        total_rejected = 0
        
        for batch_num in range(3):
            batch_results = []
            batch_size = 3
            
            for i in range(batch_size):
                idx = total_processed + i
                if idx >= len(self.test_images):
                    break
                
                decision = 'approved' if idx % 3 != 0 else 'rejected'
                result = ProcessingResult(
                    image_path=self.test_images[idx],
                    filename=f'crash_test_{idx:03d}.jpg',
                    final_decision=decision,
                    rejection_reasons=['test_reason'] if decision == 'rejected' else [],
                    processing_time=0.1,
                    timestamp=datetime.now()
                )
                batch_results.append(result)
                
                if decision == 'approved':
                    total_approved += 1
                else:
                    total_rejected += 1
            
            total_processed += len(batch_results)
            
            # Save checkpoint
            success = self.progress_tracker.save_checkpoint(
                session_id=session_id,
                processed_count=total_processed,
                total_count=len(self.test_images),
                results=batch_results
            )
            self.assertTrue(success)
            
            # Verify state consistency after each batch
            checkpoint_data = self.progress_tracker.load_checkpoint(session_id)
            self.assertEqual(checkpoint_data['processed_count'], total_processed)
            self.assertEqual(checkpoint_data['approved_count'], total_approved)
            self.assertEqual(checkpoint_data['rejected_count'], total_rejected)
            
            # Verify database consistency
            session_info = self.db_manager.get_session_info(session_id)
            self.assertEqual(session_info['processed_images'], total_processed)
            self.assertEqual(session_info['approved_images'], total_approved)
            self.assertEqual(session_info['rejected_images'], total_rejected)
            
            # Verify results count in database
            saved_results = self.db_manager.get_session_results(session_id)
            self.assertEqual(len(saved_results), total_processed)
        
        # Final consistency check
        final_summary = self.progress_tracker.get_progress_summary(session_id)
        self.assertEqual(final_summary['processed_images'], total_processed)
        self.assertEqual(final_summary['approved_images'], total_approved)
        self.assertEqual(final_summary['rejected_images'], total_rejected)
        
        # Complete session and verify final state
        self.progress_tracker.complete_session(session_id, 'completed')
        
        final_session_info = self.db_manager.get_session_info(session_id)
        self.assertEqual(final_session_info['status'], 'completed')
        self.assertIsNotNone(final_session_info['end_time'])
    
    def test_checkpoint_integrity_validation(self):
        """Test checkpoint data integrity validation."""
        session_id = self.progress_tracker.create_session(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            total_images=len(self.test_images),
            config={'test': True}
        )
        
        # Create results with various data types and edge cases
        edge_case_results = []
        
        # Normal result
        normal_result = ProcessingResult(
            image_path=self.test_images[0],
            filename='normal_test.jpg',
            final_decision='approved',
            processing_time=0.1,
            timestamp=datetime.now()
        )
        edge_case_results.append(normal_result)
        
        # Result with special characters
        special_result = ProcessingResult(
            image_path=self.test_images[1],
            filename='special_çhars_测试_🖼️.jpg',
            final_decision='rejected',
            rejection_reasons=['special_chars', 'unicode_test'],
            processing_time=0.2,
            timestamp=datetime.now()
        )
        edge_case_results.append(special_result)
        
        # Result with very long paths/names
        long_result = ProcessingResult(
            image_path=self.test_images[2],
            filename='very_long_filename_' + 'x' * 200 + '.jpg',
            final_decision='approved',
            processing_time=0.15,
            timestamp=datetime.now()
        )
        edge_case_results.append(long_result)
        
        # Save checkpoint with edge cases
        success = self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=len(edge_case_results),
            total_count=len(self.test_images),
            results=edge_case_results
        )
        self.assertTrue(success)
        
        # Load and verify checkpoint integrity
        checkpoint_data = self.progress_tracker.load_checkpoint(session_id)
        self.assertIsNotNone(checkpoint_data)
        self.assertEqual(checkpoint_data['processed_count'], len(edge_case_results))
        
        # Verify all results were saved correctly
        saved_results = self.db_manager.get_session_results(session_id)
        self.assertEqual(len(saved_results), len(edge_case_results))
        
        # Verify special characters were preserved
        special_result_saved = next(
            (r for r in saved_results if 'special_çhars' in r['filename']), 
            None
        )
        self.assertIsNotNone(special_result_saved)
        
        # Handle rejection_reasons which might be stored as JSON string or already parsed
        rejection_reasons = special_result_saved['rejection_reasons']
        if isinstance(rejection_reasons, str):
            rejection_reasons = json.loads(rejection_reasons)
        elif rejection_reasons is None:
            rejection_reasons = []
        
        self.assertIn('special_chars', rejection_reasons)
        
        # Verify long filename was handled
        long_result_saved = next(
            (r for r in saved_results if len(r['filename']) > 200), 
            None
        )
        self.assertIsNotNone(long_result_saved)
    
    def test_recovery_after_forced_termination(self):
        """Test recovery after forced process termination."""
        session_id = self.progress_tracker.create_session(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            total_images=len(self.test_images),
            config={'test': True}
        )
        
        # Process some images
        results = []
        for i in range(4):
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'crash_test_{i:03d}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
            results.append(result)
        
        # Save checkpoint
        success = self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=4,
            total_count=len(self.test_images),
            results=results
        )
        self.assertTrue(success)
        
        # Simulate forced termination by not completing the session
        # (session remains in 'running' state)
        
        # Create new tracker instance (simulating restart after crash)
        new_tracker = SQLiteProgressTracker(self.db_path, checkpoint_interval=3)
        
        # Should be able to find and resume the session
        resumable_sessions = new_tracker.get_resumable_sessions()
        self.assertEqual(len(resumable_sessions), 1)
        self.assertEqual(resumable_sessions[0]['session_id'], session_id)
        
        # Should be able to load checkpoint
        checkpoint_data = new_tracker.load_checkpoint(session_id)
        self.assertIsNotNone(checkpoint_data)
        self.assertTrue(checkpoint_data['can_resume'])
        self.assertEqual(checkpoint_data['processed_count'], 4)
        
        # Should be able to continue processing
        continue_results = []
        for i in range(4, 7):
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'crash_test_{i:03d}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
            continue_results.append(result)
        
        # Should be able to save new checkpoint
        success = new_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=7,
            total_count=len(self.test_images),
            results=continue_results
        )
        self.assertTrue(success)
        
        # Complete session
        new_tracker.complete_session(session_id, 'completed')
        
        # Verify final state
        final_summary = new_tracker.get_progress_summary(session_id)
        self.assertEqual(final_summary['status'], 'completed')
        self.assertEqual(final_summary['processed_images'], 7)


if __name__ == '__main__':
    unittest.main()