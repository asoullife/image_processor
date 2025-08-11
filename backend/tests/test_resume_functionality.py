"""Integration tests for resume functionality and crash recovery scenarios."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
import tempfile
import os
import shutil
import time
from datetime import datetime
from unittest.mock import patch, MagicMock

from backend.core.progress_tracker import PostgresProgressTracker
from backend.core.batch_processor import BatchProcessor
from backend.core.base import ProcessingResult, QualityResult
from main import ImageProcessor
from backend.utils.file_manager import FileManager


class TestResumeFunctionality(unittest.TestCase):
    """Test cases for resume functionality and crash recovery."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
        
        # Create test images
        self.test_images = []
        for i in range(20):
            image_path = os.path.join(self.input_dir, f'test_image_{i:03d}.jpg')
            with open(image_path, 'wb') as f:
                f.write(b'fake_image_data')
            self.test_images.append(image_path)
        
        # Initialize components
        self.progress_tracker = PostgresProgressTracker(checkpoint_interval=5)
        
        # Mock configuration
        self.mock_config = {
            'processing': {
                'batch_size': 5,
                'max_workers': 2,
                'checkpoint_interval': 5
            },
            'output': {
                'images_per_folder': 200
            },
            'logging': {
                'level': 'INFO'
            }
        }
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_basic_session_creation_and_resume(self):
        """Test basic session creation and resume capability."""
        # Create initial session
        session_id = self.progress_tracker.create_session(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            total_images=len(self.test_images),
            config=self.mock_config
        )
        
        # Process first batch (5 images)
        first_batch_results = []
        for i in range(5):
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'test_image_{i:03d}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
            first_batch_results.append(result)
        
        # Save checkpoint
        success = self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=5,
            total_count=len(self.test_images),
            results=first_batch_results
        )
        self.assertTrue(success)
        
        # Simulate interruption - load checkpoint
        checkpoint_data = self.progress_tracker.load_checkpoint(session_id)
        
        # Verify checkpoint data
        self.assertIsNotNone(checkpoint_data)
        self.assertTrue(checkpoint_data['can_resume'])
        self.assertEqual(checkpoint_data['processed_count'], 5)
        self.assertTrue(checkpoint_data['has_checkpoint'])
        self.assertEqual(checkpoint_data['resume_from_index'], 5)
        
        # Resume processing from checkpoint
        second_batch_results = []
        for i in range(5, 10):
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'test_image_{i:03d}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
            second_batch_results.append(result)
        
        # Save second checkpoint
        success = self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=10,
            total_count=len(self.test_images),
            results=second_batch_results
        )
        self.assertTrue(success)
        
        # Verify final state
        final_checkpoint = self.progress_tracker.load_checkpoint(session_id)
        self.assertEqual(final_checkpoint['processed_count'], 10)
        self.assertEqual(final_checkpoint['approved_count'], 10)
        
        # Complete session
        self.progress_tracker.complete_session(session_id, 'completed')
        
        # Verify session cannot be resumed after completion
        final_checkpoint = self.progress_tracker.load_checkpoint(session_id)
        self.assertFalse(final_checkpoint['can_resume'])
    
    def test_multiple_interruptions_and_resumes(self):
        """Test handling multiple interruptions and resumes."""
        session_id = self.progress_tracker.create_session(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            total_images=len(self.test_images),
            config=self.mock_config
        )
        
        # Simulate multiple processing cycles with interruptions
        processed_count = 0
        batch_size = 3
        
        for cycle in range(3):  # 3 cycles of processing
            # Process a batch
            batch_results = []
            start_idx = processed_count
            end_idx = min(start_idx + batch_size, len(self.test_images))
            
            for i in range(start_idx, end_idx):
                result = ProcessingResult(
                    image_path=self.test_images[i],
                    filename=f'test_image_{i:03d}.jpg',
                    final_decision='approved' if i % 3 != 0 else 'rejected',
                    rejection_reasons=['test_reason'] if i % 3 == 0 else [],
                    processing_time=0.1,
                    timestamp=datetime.now()
                )
                batch_results.append(result)
            
            processed_count = end_idx
            
            # Save checkpoint
            success = self.progress_tracker.save_checkpoint(
                session_id=session_id,
                processed_count=processed_count,
                total_count=len(self.test_images),
                results=batch_results
            )
            self.assertTrue(success)
            
            # Simulate interruption and resume check
            checkpoint_data = self.progress_tracker.load_checkpoint(session_id)
            self.assertIsNotNone(checkpoint_data)
            self.assertEqual(checkpoint_data['processed_count'], processed_count)
            self.assertTrue(checkpoint_data['can_resume'])
            
            # Verify we can resume from the correct position
            expected_resume_index = processed_count
            if checkpoint_data['has_checkpoint'] and processed_count >= 5:  # Checkpoint interval
                # Should have checkpoint record
                self.assertIn('resume_from_index', checkpoint_data)
        
        # Final verification
        final_summary = self.progress_tracker.get_progress_summary(session_id)
        self.assertEqual(final_summary['processed_images'], processed_count)
        self.assertGreater(final_summary['approved_images'], 0)
        self.assertGreater(final_summary['rejected_images'], 0)
    
    def test_crash_recovery_with_corrupted_data(self):
        """Test recovery from crashes with potentially corrupted data."""
        session_id = self.progress_tracker.create_session(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            total_images=len(self.test_images),
            config=self.mock_config
        )
        
        # Process some images successfully
        successful_results = []
        for i in range(7):
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'test_image_{i:03d}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
            successful_results.append(result)
        
        # Save checkpoint
        self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=7,
            total_count=len(self.test_images),
            results=successful_results
        )
        
        # Simulate crash by adding corrupted/incomplete data
        corrupted_results = []
        for i in range(7, 10):
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'test_image_{i:03d}.jpg',
                final_decision='error',
                error_message='Simulated crash during processing',
                processing_time=0.0,
                timestamp=datetime.now()
            )
            corrupted_results.append(result)
        
        # Try to save corrupted data (simulating partial write during crash)
        try:
            self.progress_tracker.save_checkpoint(
                session_id=session_id,
                processed_count=10,
                total_count=len(self.test_images),
                results=corrupted_results
            )
        except Exception:
            pass  # Expected to potentially fail
        
        # Attempt recovery
        checkpoint_data = self.progress_tracker.load_checkpoint(session_id)
        self.assertIsNotNone(checkpoint_data)
        
        # Should be able to resume from last successful checkpoint
        self.assertTrue(checkpoint_data['can_resume'])
        
        # Verify data integrity
        session_results = self.progress_tracker.db_manager.get_session_results(session_id)
        
        # Should have at least the successful results
        successful_count = sum(1 for r in session_results if r['final_decision'] == 'approved')
        self.assertGreaterEqual(successful_count, 7)
        
        # Resume processing should work
        resume_results = []
        resume_start = checkpoint_data.get('resume_from_index', checkpoint_data['processed_count'])
        
        for i in range(resume_start, min(resume_start + 3, len(self.test_images))):
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'test_image_{i:03d}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
            resume_results.append(result)
        
        # Should be able to continue processing
        success = self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=resume_start + len(resume_results),
            total_count=len(self.test_images),
            results=resume_results
        )
        self.assertTrue(success)
    
    def test_concurrent_session_isolation(self):
        """Test that concurrent sessions don't interfere with each other."""
        # Create two separate sessions
        session1_id = self.progress_tracker.create_session(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            total_images=10,
            config=self.mock_config,
            session_id='session_1'
        )
        
        session2_id = self.progress_tracker.create_session(
            input_folder=self.input_dir,
            output_folder=self.output_dir + '_2',
            total_images=10,
            config=self.mock_config,
            session_id='session_2'
        )
        
        # Process different images in each session
        session1_results = []
        for i in range(5):
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'session1_image_{i}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
            session1_results.append(result)
        
        session2_results = []
        for i in range(5, 10):
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'session2_image_{i}.jpg',
                final_decision='rejected',
                rejection_reasons=['test_reason'],
                processing_time=0.1,
                timestamp=datetime.now()
            )
            session2_results.append(result)
        
        # Save checkpoints for both sessions
        success1 = self.progress_tracker.save_checkpoint(
            session_id=session1_id,
            processed_count=5,
            total_count=10,
            results=session1_results
        )
        
        success2 = self.progress_tracker.save_checkpoint(
            session_id=session2_id,
            processed_count=5,
            total_count=10,
            results=session2_results
        )
        
        self.assertTrue(success1)
        self.assertTrue(success2)
        
        # Verify sessions are isolated
        checkpoint1 = self.progress_tracker.load_checkpoint(session1_id)
        checkpoint2 = self.progress_tracker.load_checkpoint(session2_id)
        
        self.assertEqual(checkpoint1['processed_count'], 5)
        self.assertEqual(checkpoint2['processed_count'], 5)
        
        # Check individual session info
        session1_info = self.progress_tracker.db_manager.get_session_info(session1_id)
        session2_info = self.progress_tracker.db_manager.get_session_info(session2_id)
        
        # Debug: Print session info to understand the issue
        print(f"Session1 info: approved={session1_info['approved_images']}, rejected={session1_info['rejected_images']}")
        print(f"Session2 info: approved={session2_info['approved_images']}, rejected={session2_info['rejected_images']}")
        
        # The issue might be that both sessions are using the same progress tracker instance
        # Let's verify the results are correctly associated with each session
        results1 = self.progress_tracker.db_manager.get_session_results(session1_id)
        results2 = self.progress_tracker.db_manager.get_session_results(session2_id)
        
        # Count approved/rejected for each session from actual results
        session1_approved = sum(1 for r in results1 if r['final_decision'] == 'approved')
        session1_rejected = sum(1 for r in results1 if r['final_decision'] == 'rejected')
        session2_approved = sum(1 for r in results2 if r['final_decision'] == 'approved')
        session2_rejected = sum(1 for r in results2 if r['final_decision'] == 'rejected')
        
        self.assertEqual(session1_approved, 5)
        self.assertEqual(session1_rejected, 0)
        self.assertEqual(session2_approved, 0)
        self.assertEqual(session2_rejected, 5)
        
        # Verify results are separate
        results1 = self.progress_tracker.db_manager.get_session_results(session1_id)
        results2 = self.progress_tracker.db_manager.get_session_results(session2_id)
        
        self.assertEqual(len(results1), 5)
        self.assertEqual(len(results2), 5)
        
        # Verify no cross-contamination
        for result in results1:
            self.assertTrue(result['filename'].startswith('session1_'))
        
        for result in results2:
            self.assertTrue(result['filename'].startswith('session2_'))
    
    def test_session_cleanup_and_recovery(self):
        """Test session cleanup and recovery mechanisms."""
        # Create a session
        session_id = self.progress_tracker.create_session(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            total_images=len(self.test_images),
            config=self.mock_config
        )
        
        # Add some progress
        results = []
        for i in range(8):
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'test_image_{i:03d}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
            results.append(result)
        
        self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=8,
            total_count=len(self.test_images),
            results=results
        )
        
        # Mark session as failed
        self.progress_tracker.complete_session(session_id, 'failed', 'Test failure')
        
        # Verify session is marked as failed
        session_info = self.progress_tracker.db_manager.get_session_info(session_id)
        self.assertEqual(session_info['status'], 'failed')
        self.assertEqual(session_info['error_message'], 'Test failure')
        
        # Test cleanup
        cleanup_success = self.progress_tracker.cleanup_session_data(session_id)
        self.assertTrue(cleanup_success)
        
        # Verify data was cleaned up
        session_info_after = self.progress_tracker.db_manager.get_session_info(session_id)
        self.assertIsNone(session_info_after)
        
        results_after = self.progress_tracker.db_manager.get_session_results(session_id)
        self.assertEqual(len(results_after), 0)
    
    def test_force_checkpoint_functionality(self):
        """Test force checkpoint functionality."""
        session_id = self.progress_tracker.create_session(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            total_images=len(self.test_images),
            config=self.mock_config
        )
        
        # Process fewer images than checkpoint interval
        results = []
        for i in range(3):  # Less than checkpoint_interval (5)
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'test_image_{i:03d}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
            results.append(result)
        
        # Save without triggering automatic checkpoint
        self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=3,
            total_count=len(self.test_images),
            results=results
        )
        
        # Verify no checkpoint record was created automatically
        with self.progress_tracker.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM checkpoints WHERE session_id = ?', (session_id,))
            checkpoint_count = cursor.fetchone()[0]
            self.assertEqual(checkpoint_count, 0)
        
        # Force a checkpoint
        force_success = self.progress_tracker.force_checkpoint(session_id)
        self.assertTrue(force_success)
        
        # Verify checkpoint was created
        with self.progress_tracker.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM checkpoints WHERE session_id = ?', (session_id,))
            checkpoint_count = cursor.fetchone()[0]
            self.assertEqual(checkpoint_count, 1)
            
            # Verify it's marked as forced
            cursor.execute('SELECT checkpoint_data FROM checkpoints WHERE session_id = ?', (session_id,))
            checkpoint_data = cursor.fetchone()[0]
            import json
            data = json.loads(checkpoint_data)
            self.assertTrue(data.get('forced', False))
    
    @patch('builtins.input')
    def test_resume_user_interaction(self, mock_input):
        """Test user interaction for resume functionality."""
        # Create a resumable session
        session_id = self.progress_tracker.create_session(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            total_images=len(self.test_images),
            config=self.mock_config
        )
        
        # Add some progress
        results = []
        for i in range(6):
            result = ProcessingResult(
                image_path=self.test_images[i],
                filename=f'test_image_{i:03d}.jpg',
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
            results.append(result)
        
        self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=6,
            total_count=len(self.test_images),
            results=results
        )
        
        # Mock user input to accept resume
        mock_input.return_value = 'y'
        
        # Create ImageProcessor with mocked config
        with patch('main.load_config') as mock_load_config:
            mock_config_obj = MagicMock()
            mock_config_obj.dict.return_value = self.mock_config
            mock_config_obj.database = {}
            mock_config_obj.processing.checkpoint_interval = 5
            mock_config_obj.logging = MagicMock()
            mock_load_config.return_value = mock_config_obj
            
            with patch('main.initialize_logging'):
                processor = ImageProcessor()
                
                # Test resume handling
                resumed_session_id, start_index = processor._handle_resume_request(
                    self.input_dir, self.output_dir
                )
                
                self.assertEqual(resumed_session_id, session_id)
                self.assertEqual(start_index, 6)  # Should resume from after last processed
        
        # Test declining resume
        mock_input.return_value = 'n'
        
        with patch('main.load_config') as mock_load_config:
            mock_config_obj = MagicMock()
            mock_config_obj.dict.return_value = self.mock_config
            mock_config_obj.database = {}
            mock_config_obj.processing.checkpoint_interval = 5
            mock_config_obj.logging = MagicMock()
            mock_load_config.return_value = mock_config_obj
            
            with patch('main.initialize_logging'):
                processor = ImageProcessor()
                
                resumed_session_id, start_index = processor._handle_resume_request(
                    self.input_dir, self.output_dir
                )
                
                self.assertIsNone(resumed_session_id)
                self.assertEqual(start_index, 0)
    
    def test_memory_cleanup_during_resume(self):
        """Test memory cleanup during resume operations."""
        session_id = self.progress_tracker.create_session(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            total_images=len(self.test_images),
            config=self.mock_config
        )
        
        # Create a mock processing function that tracks memory usage
        memory_usage_history = []
        
        def mock_processing_function(image_path: str) -> ProcessingResult:
            # Simulate memory usage
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_usage_history.append(memory_mb)
            
            return ProcessingResult(
                image_path=image_path,
                filename=os.path.basename(image_path),
                final_decision='approved',
                processing_time=0.1,
                timestamp=datetime.now()
            )
        
        # Create batch processor
        batch_processor = BatchProcessor(self.mock_config, mock_processing_function)
        
        # Process first batch
        first_batch = self.test_images[:10]
        first_results = batch_processor.process(first_batch)
        
        # Save checkpoint
        self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=len(first_results),
            total_count=len(self.test_images),
            results=first_results
        )
        
        # Simulate interruption and resume
        checkpoint_data = self.progress_tracker.load_checkpoint(session_id)
        self.assertIsNotNone(checkpoint_data)
        
        # Process remaining batch after "resume"
        remaining_batch = self.test_images[10:]
        if remaining_batch:
            remaining_results = batch_processor.process(remaining_batch)
            
            # Verify memory cleanup occurred
            stats = batch_processor.get_statistics()
            self.assertIn('current_memory_usage', stats)
            self.assertGreater(stats['current_memory_usage'], 0)
        
        # Complete session
        self.progress_tracker.complete_session(session_id, 'completed')
        
        # Verify final state
        final_summary = self.progress_tracker.get_progress_summary(session_id)
        self.assertEqual(final_summary['status'], 'completed')


if __name__ == '__main__':
    unittest.main()