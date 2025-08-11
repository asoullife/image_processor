"""Integration tests for batch processor with other components."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
import tempfile
import os
from unittest.mock import Mock, patch

from backend.core.batch_processor import BatchProcessor
from backend.core.progress_tracker import SQLiteProgressTracker
from backend.core.base import ProcessingResult
from backend.config.config_loader import ConfigLoader


class TestBatchProcessorIntegration(unittest.TestCase):
    """Integration tests for BatchProcessor with other components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        
        # Create test configuration
        self.config = {
            'processing': {
                'batch_size': 3,
                'max_workers': 2,
                'memory_threshold_mb': 100,
                'max_retries': 2,
                'retry_delay': 0.1,
                'enable_memory_monitoring': True,
                'gc_frequency': 2
            }
        }
        
        # Create mock processing function
        def mock_processing_function(image_path: str) -> ProcessingResult:
            return ProcessingResult(
                image_path=image_path,
                filename=os.path.basename(image_path),
                final_decision='approved',
                processing_time=0.01
            )
        
        self.processing_function = mock_processing_function
        self.batch_processor = BatchProcessor(self.config, self.processing_function)
        
        # Create progress tracker
        self.progress_tracker = SQLiteProgressTracker(self.db_path, checkpoint_interval=2)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_batch_processor_with_progress_tracking(self):
        """Test batch processor integration with progress tracking."""
        # Create test image paths
        image_paths = [f"test_image_{i}.jpg" for i in range(7)]
        
        # Create session
        session_id = self.progress_tracker.create_session(
            input_folder="/test/input",
            output_folder="/test/output",
            total_images=len(image_paths)
        )
        
        # Track progress during batch processing
        processed_results = []
        
        def progress_callback(processed, total):
            # Save checkpoint every 2 images
            if processed % 2 == 0 and processed > 0:
                # Get recent results for checkpoint
                recent_results = processed_results[-2:] if len(processed_results) >= 2 else processed_results
                self.progress_tracker.save_checkpoint(
                    session_id=session_id,
                    processed_count=processed,
                    total_count=total,
                    results=recent_results
                )
        
        # Process images with progress tracking
        results = self.batch_processor.process(image_paths, progress_callback=progress_callback)
        processed_results.extend(results)
        
        # Verify results
        self.assertEqual(len(results), 7)
        for result in results:
            self.assertEqual(result.final_decision, 'approved')
        
        # Complete session
        self.progress_tracker.complete_session(session_id, 'completed')
        
        # Verify session status
        status = self.progress_tracker.get_session_status(session_id)
        self.assertEqual(status, 'completed')
        
        # Get progress summary
        summary = self.progress_tracker.get_progress_summary(session_id)
        self.assertIsNotNone(summary)
        self.assertEqual(summary['total_images'], 7)
        self.assertEqual(summary['status'], 'completed')
    
    def test_batch_processor_memory_management(self):
        """Test batch processor memory management features."""
        # Create larger batch to test memory management
        image_paths = [f"test_image_{i}.jpg" for i in range(15)]
        
        # Get initial memory usage
        initial_memory = self.batch_processor.get_memory_usage()
        
        # Process images
        results = self.batch_processor.process(image_paths)
        
        # Verify processing completed
        self.assertEqual(len(results), 15)
        
        # Check memory cleanup was performed
        stats = self.batch_processor.get_statistics()
        self.assertGreater(stats['avg_memory_usage'], 0)
        
        # Force memory cleanup
        before, after = self.batch_processor.cleanup_memory()
        self.assertIsInstance(before, float)
        self.assertIsInstance(after, float)
    
    def test_batch_processor_error_recovery(self):
        """Test batch processor error recovery with progress tracking."""
        # Create processing function that fails on specific images
        def error_prone_function(image_path: str) -> ProcessingResult:
            if "error" in image_path:
                raise RuntimeError(f"Simulated error for {image_path}")
            
            return ProcessingResult(
                image_path=image_path,
                filename=os.path.basename(image_path),
                final_decision='approved',
                processing_time=0.01
            )
        
        # Create processor with error-prone function
        processor = BatchProcessor(self.config, error_prone_function)
        
        # Create session
        session_id = self.progress_tracker.create_session(
            input_folder="/test/input",
            output_folder="/test/output",
            total_images=5
        )
        
        # Mix of good and bad image paths
        image_paths = [
            "good_image_1.jpg",
            "error_image_2.jpg",
            "good_image_3.jpg",
            "error_image_4.jpg",
            "good_image_5.jpg"
        ]
        
        # Process with error handling
        results = processor.process(image_paths)
        
        # Verify results
        self.assertEqual(len(results), 5)
        
        # Check mix of success and error results
        success_count = sum(1 for r in results if r.final_decision == 'approved')
        error_count = sum(1 for r in results if r.final_decision == 'error')
        
        self.assertEqual(success_count, 3)  # 3 good images
        self.assertEqual(error_count, 2)    # 2 error images
        
        # Save final results
        self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=len(results),
            total_count=len(image_paths),
            results=results
        )
        
        # Complete session
        self.progress_tracker.complete_session(session_id, 'completed')
        
        # Verify error statistics
        stats = processor.get_statistics()
        self.assertEqual(stats['total_processed'], 5)
        self.assertEqual(stats['total_errors'], 2)
        self.assertEqual(stats['success_rate'], 60.0)  # 3/5 * 100
    
    def test_batch_processor_stop_and_resume(self):
        """Test stopping batch processor and resuming with progress tracking."""
        # Create session
        session_id = self.progress_tracker.create_session(
            input_folder="/test/input",
            output_folder="/test/output",
            total_images=10
        )
        
        # Create slow processing function for testing stop functionality
        def slow_processing_function(image_path: str) -> ProcessingResult:
            import time
            time.sleep(0.1)  # Simulate slow processing
            return ProcessingResult(
                image_path=image_path,
                filename=os.path.basename(image_path),
                final_decision='approved',
                processing_time=0.1
            )
        
        processor = BatchProcessor(self.config, slow_processing_function)
        
        image_paths = [f"test_image_{i}.jpg" for i in range(10)]
        
        # Start processing in background (simulate)
        # For this test, we'll just process a few images and save checkpoint
        partial_results = []
        for i in range(3):
            result = slow_processing_function(image_paths[i])
            partial_results.append(result)
        
        # Save checkpoint for partial processing
        self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=3,
            total_count=10,
            results=partial_results
        )
        
        # Simulate resuming from checkpoint
        checkpoint_data = self.progress_tracker.load_checkpoint(session_id)
        self.assertIsNotNone(checkpoint_data)
        self.assertEqual(checkpoint_data['processed_count'], 3)
        self.assertTrue(checkpoint_data['can_resume'])
        
        # Process remaining images
        remaining_paths = image_paths[3:]
        remaining_results = processor.process(remaining_paths)
        
        # Verify remaining processing
        self.assertEqual(len(remaining_results), 7)
        
        # Complete session
        all_results = partial_results + remaining_results
        self.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=10,
            total_count=10,
            results=remaining_results
        )
        
        self.progress_tracker.complete_session(session_id, 'completed')
        
        # Verify final state
        summary = self.progress_tracker.get_progress_summary(session_id)
        self.assertEqual(summary['status'], 'completed')
    
    @patch('config.config_loader.ConfigLoader')
    def test_batch_processor_with_config_loader(self, mock_config_loader):
        """Test batch processor integration with configuration loader."""
        # Mock configuration loader
        mock_config = Mock()
        mock_config.processing.batch_size = 5
        mock_config.processing.max_workers = 3
        mock_config.processing.checkpoint_interval = 10
        
        mock_loader_instance = Mock()
        mock_loader_instance.load_config.return_value = mock_config
        mock_config_loader.return_value = mock_loader_instance
        
        # Create config dict from mock
        config = {
            'processing': {
                'batch_size': mock_config.processing.batch_size,
                'max_workers': mock_config.processing.max_workers,
                'memory_threshold_mb': 512,
                'max_retries': 3,
                'retry_delay': 1.0,
                'enable_memory_monitoring': True,
                'gc_frequency': 5
            }
        }
        
        # Create processor with loaded config
        processor = BatchProcessor(config, self.processing_function)
        
        # Verify configuration was applied
        self.assertEqual(processor.batch_config.batch_size, 5)
        self.assertEqual(processor.batch_config.max_workers, 3)
        
        # Test processing with configured settings
        image_paths = [f"test_image_{i}.jpg" for i in range(12)]
        results = processor.process(image_paths)
        
        self.assertEqual(len(results), 12)
        
        # Verify batching worked correctly
        stats = processor.get_statistics()
        expected_batches = (12 + 5 - 1) // 5  # Ceiling division
        self.assertEqual(stats['total_batches'], expected_batches)


if __name__ == '__main__':
    unittest.main()