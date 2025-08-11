"""Unit tests for batch processing engine."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
import tempfile
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import List

from backend.core.batch_processor import BatchProcessor, BatchConfig, BatchResult, MemoryMonitor
from backend.core.base import ProcessingResult


class MockProcessingFunction:
    """Mock processing function for testing."""
    
    def __init__(self, processing_time: float = 0.1, error_rate: float = 0.0):
        """Initialize mock processing function.
        
        Args:
            processing_time: Simulated processing time per image.
            error_rate: Probability of raising an error (0.0 to 1.0).
        """
        self.processing_time = processing_time
        self.error_rate = error_rate
        self.call_count = 0
        self.processed_paths = []
    
    def __call__(self, image_path: str) -> ProcessingResult:
        """Mock processing function."""
        self.call_count += 1
        self.processed_paths.append(image_path)
        
        # Simulate processing time
        time.sleep(self.processing_time)
        
        # Simulate errors
        if self.error_rate > 0 and (self.call_count * self.error_rate) >= 1:
            if self.call_count % int(1 / self.error_rate) == 0:
                raise RuntimeError(f"Simulated error for {image_path}")
        
        return ProcessingResult(
            image_path=image_path,
            filename=image_path.split('/')[-1],
            final_decision='approved',
            processing_time=self.processing_time
        )


class TestMemoryMonitor(unittest.TestCase):
    """Test cases for MemoryMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = MemoryMonitor(threshold_mb=100)
    
    def test_get_memory_usage(self):
        """Test memory usage retrieval."""
        usage = self.monitor.get_memory_usage()
        self.assertIsInstance(usage, float)
        self.assertGreater(usage, 0)
    
    def test_check_memory_threshold(self):
        """Test memory threshold checking."""
        # Set very high threshold
        self.monitor.threshold_mb = 99999
        self.assertFalse(self.monitor.check_memory_threshold())
        
        # Set very low threshold
        self.monitor.threshold_mb = 1
        self.assertTrue(self.monitor.check_memory_threshold())
    
    def test_cleanup_memory(self):
        """Test memory cleanup functionality."""
        # Force cleanup
        before, after = self.monitor.cleanup_memory(force=True)
        
        self.assertIsInstance(before, float)
        self.assertIsInstance(after, float)
        self.assertGreaterEqual(before, 0)
        self.assertGreaterEqual(after, 0)
    
    @patch('psutil.Process')
    def test_memory_usage_error_handling(self, mock_process):
        """Test error handling in memory usage retrieval."""
        mock_process.side_effect = Exception("Process error")
        
        usage = self.monitor.get_memory_usage()
        self.assertEqual(usage, 0.0)


class TestBatchProcessor(unittest.TestCase):
    """Test cases for BatchProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'processing': {
                'batch_size': 5,
                'max_workers': 2,
                'memory_threshold_mb': 100,
                'max_retries': 2,
                'retry_delay': 0.1,
                'enable_memory_monitoring': True,
                'gc_frequency': 2
            }
        }
        
        self.mock_function = MockProcessingFunction(processing_time=0.01)
        self.processor = BatchProcessor(self.config, self.mock_function)
    
    def test_initialization(self):
        """Test BatchProcessor initialization."""
        self.assertEqual(self.processor.batch_config.batch_size, 5)
        self.assertEqual(self.processor.batch_config.max_workers, 2)
        self.assertEqual(self.processor.batch_config.max_retries, 2)
        self.assertIsNotNone(self.processor.error_handler)
        self.assertIsNotNone(self.processor.memory_monitor)
    
    def test_create_batches(self):
        """Test batch creation from image paths."""
        image_paths = [f"image_{i}.jpg" for i in range(12)]
        batches = self.processor._create_batches(image_paths)
        
        self.assertEqual(len(batches), 3)  # 12 images / 5 per batch = 3 batches
        self.assertEqual(len(batches[0]), 5)
        self.assertEqual(len(batches[1]), 5)
        self.assertEqual(len(batches[2]), 2)
    
    def test_process_empty_list(self):
        """Test processing empty image list."""
        results = self.processor.process([])
        self.assertEqual(len(results), 0)
        self.assertEqual(self.mock_function.call_count, 0)
    
    def test_process_single_batch(self):
        """Test processing a single batch."""
        image_paths = [f"image_{i}.jpg" for i in range(3)]
        results = self.processor.process(image_paths)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(self.mock_function.call_count, 3)
        
        # Results may not be in the same order due to threading
        result_paths = [r.image_path for r in results]
        for i in range(3):
            self.assertIn(f"image_{i}.jpg", result_paths)
        
        for result in results:
            self.assertEqual(result.final_decision, 'approved')
    
    def test_process_multiple_batches(self):
        """Test processing multiple batches."""
        image_paths = [f"image_{i}.jpg" for i in range(12)]
        results = self.processor.process(image_paths)
        
        self.assertEqual(len(results), 12)
        self.assertEqual(self.mock_function.call_count, 12)
        self.assertEqual(self.processor.total_batches, 3)
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        progress_updates = []
        
        def progress_callback(processed, total):
            progress_updates.append((processed, total))
        
        image_paths = [f"image_{i}.jpg" for i in range(8)]
        self.processor.process(image_paths, progress_callback=progress_callback)
        
        # Should have progress updates for each batch
        self.assertGreater(len(progress_updates), 0)
        
        # Last update should show completion
        final_processed, final_total = progress_updates[-1]
        self.assertEqual(final_processed, 8)
        self.assertEqual(final_total, 8)
    
    def test_error_handling(self):
        """Test error handling in batch processing."""
        # Create mock function that always errors
        error_function = MockProcessingFunction(error_rate=1.0)
        processor = BatchProcessor(self.config, error_function)
        
        image_paths = [f"image_{i}.jpg" for i in range(3)]
        results = processor.process(image_paths)
        
        self.assertEqual(len(results), 3)
        
        # All results should be errors
        for result in results:
            self.assertEqual(result.final_decision, 'error')
            self.assertIsNotNone(result.error_message)
    
    def test_partial_error_handling(self):
        """Test handling of partial errors in batch."""
        # Create mock function with 25% error rate (every 4th call)
        error_function = MockProcessingFunction(error_rate=0.25)
        processor = BatchProcessor(self.config, error_function)
        
        image_paths = [f"image_{i}.jpg" for i in range(8)]
        results = processor.process(image_paths)
        
        self.assertEqual(len(results), 8)
        
        # Should have mix of success and error results
        success_count = sum(1 for r in results if r.final_decision == 'approved')
        error_count = sum(1 for r in results if r.final_decision == 'error')
        
        self.assertGreater(success_count, 0)
        self.assertGreater(error_count, 0)
        self.assertEqual(success_count + error_count, 8)
    
    def test_memory_monitoring(self):
        """Test memory monitoring during processing."""
        image_paths = [f"image_{i}.jpg" for i in range(10)]
        results = self.processor.process(image_paths)
        
        self.assertEqual(len(results), 10)
        
        # Check that memory monitoring was active
        stats = self.processor.get_statistics()
        self.assertGreater(stats['avg_memory_usage'], 0)
        self.assertGreater(stats['current_memory_usage'], 0)
    
    def test_stop_processing(self):
        """Test stopping processing gracefully."""
        # Create slow processing function
        slow_function = MockProcessingFunction(processing_time=0.5)
        processor = BatchProcessor(self.config, slow_function)
        
        image_paths = [f"image_{i}.jpg" for i in range(20)]
        
        # Start processing in separate thread
        results = []
        def process_thread():
            nonlocal results
            results = processor.process(image_paths)
        
        thread = threading.Thread(target=process_thread)
        thread.start()
        
        # Stop processing after short delay
        time.sleep(0.2)
        processor.stop_processing()
        
        thread.join(timeout=2.0)
        
        # Should have processed fewer than all images
        self.assertLess(len(results), 20)
    
    def test_retry_logic(self):
        """Test retry logic for failed batches."""
        # Test that retry logic is invoked by checking batch history
        # Create function that always fails to trigger retries
        def always_fail_function(image_path: str) -> ProcessingResult:
            raise RuntimeError("Temporary failure")
        
        # Create processor with retry enabled
        config = self.config.copy()
        config['processing']['max_retries'] = 2
        config['processing']['retry_delay'] = 0.01  # Fast retry for testing
        processor = BatchProcessor(config, always_fail_function)
        
        # Process single image (will be in one batch)
        results = processor.process(["test_image.jpg"])
        
        self.assertEqual(len(results), 1)
        # Result should be error after all retries exhausted
        self.assertEqual(results[0].final_decision, 'error')
        self.assertIn("Temporary failure", results[0].error_message)
        
        # Check that batch was processed (even if it failed)
        stats = processor.get_statistics()
        self.assertEqual(stats['total_batches'], 1)
        self.assertEqual(stats['total_errors'], 1)
    
    @patch('core.batch_processor.MemoryMonitor')
    def test_memory_error_handling(self, mock_monitor_class):
        """Test handling of memory errors."""
        # Mock memory monitor to simulate memory error
        mock_monitor = Mock()
        mock_monitor.get_memory_usage.return_value = 500.0
        mock_monitor.cleanup_memory.return_value = (500.0, 400.0)
        mock_monitor_class.return_value = mock_monitor
        
        # Create function that raises MemoryError
        def memory_error_function(image_path: str) -> ProcessingResult:
            raise MemoryError("Out of memory")
        
        processor = BatchProcessor(self.config, memory_error_function)
        
        results = processor.process(["test_image.jpg"])
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].final_decision, 'error')
        self.assertIn("memory", results[0].error_message.lower())
    
    def test_statistics(self):
        """Test statistics collection."""
        image_paths = [f"image_{i}.jpg" for i in range(7)]
        results = self.processor.process(image_paths)
        
        stats = self.processor.get_statistics()
        
        self.assertEqual(stats['total_processed'], 7)
        self.assertGreater(stats['total_batches'], 0)
        self.assertEqual(stats['total_errors'], 0)
        self.assertEqual(stats['success_rate'], 100.0)
        self.assertGreater(stats['avg_batch_time'], 0)
        self.assertGreater(stats['avg_memory_usage'], 0)
        self.assertIsInstance(stats['batch_history'], list)
    
    def test_dynamic_configuration(self):
        """Test dynamic configuration changes."""
        # Test batch size configuration
        original_size = self.processor.batch_config.batch_size
        self.processor.configure_batch_size(10)
        self.assertEqual(self.processor.batch_config.batch_size, 10)
        
        # Test invalid batch size
        with self.assertRaises(ValueError):
            self.processor.configure_batch_size(0)
        
        # Test worker configuration
        original_workers = self.processor.batch_config.max_workers
        self.processor.configure_workers(4)
        self.assertEqual(self.processor.batch_config.max_workers, 4)
        
        # Test invalid worker count
        with self.assertRaises(ValueError):
            self.processor.configure_workers(-1)
    
    def test_thread_safety(self):
        """Test thread safety of batch processor."""
        image_paths = [f"image_{i}.jpg" for i in range(20)]
        results_list = []
        
        def process_thread():
            results = self.processor.process(image_paths[:10])
            results_list.append(results)
        
        # Start multiple threads (though processor should handle one at a time)
        threads = []
        for _ in range(2):
            thread = threading.Thread(target=process_thread)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have results from both threads
        self.assertEqual(len(results_list), 2)
        for results in results_list:
            self.assertEqual(len(results), 10)
    
    def test_cleanup_memory(self):
        """Test manual memory cleanup."""
        before, after = self.processor.cleanup_memory()
        
        self.assertIsInstance(before, float)
        self.assertIsInstance(after, float)
        self.assertGreaterEqual(before, 0)
        self.assertGreaterEqual(after, 0)
    
    def test_get_memory_usage(self):
        """Test memory usage retrieval."""
        usage = self.processor.get_memory_usage()
        self.assertIsInstance(usage, float)
        self.assertGreater(usage, 0)


class TestBatchResult(unittest.TestCase):
    """Test cases for BatchResult dataclass."""
    
    def test_batch_result_creation(self):
        """Test BatchResult creation and attributes."""
        results = [
            ProcessingResult(
                image_path="test1.jpg",
                filename="test1.jpg",
                final_decision='approved'
            )
        ]
        
        batch_result = BatchResult(
            batch_id=1,
            processed_count=1,
            success_count=1,
            error_count=0,
            processing_time=1.5,
            memory_usage_mb=256.0,
            results=results,
            errors=[]
        )
        
        self.assertEqual(batch_result.batch_id, 1)
        self.assertEqual(batch_result.processed_count, 1)
        self.assertEqual(batch_result.success_count, 1)
        self.assertEqual(batch_result.error_count, 0)
        self.assertEqual(batch_result.processing_time, 1.5)
        self.assertEqual(batch_result.memory_usage_mb, 256.0)
        self.assertEqual(len(batch_result.results), 1)
        self.assertEqual(len(batch_result.errors), 0)


class TestBatchConfig(unittest.TestCase):
    """Test cases for BatchConfig dataclass."""
    
    def test_batch_config_defaults(self):
        """Test BatchConfig default values."""
        config = BatchConfig()
        
        self.assertEqual(config.batch_size, 200)
        self.assertEqual(config.max_workers, 4)
        self.assertEqual(config.memory_threshold_mb, 1024)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.retry_delay, 1.0)
        self.assertTrue(config.enable_memory_monitoring)
        self.assertEqual(config.gc_frequency, 10)
    
    def test_batch_config_custom_values(self):
        """Test BatchConfig with custom values."""
        config = BatchConfig(
            batch_size=100,
            max_workers=8,
            memory_threshold_mb=512,
            max_retries=5,
            retry_delay=2.0,
            enable_memory_monitoring=False,
            gc_frequency=5
        )
        
        self.assertEqual(config.batch_size, 100)
        self.assertEqual(config.max_workers, 8)
        self.assertEqual(config.memory_threshold_mb, 512)
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.retry_delay, 2.0)
        self.assertFalse(config.enable_memory_monitoring)
        self.assertEqual(config.gc_frequency, 5)


if __name__ == '__main__':
    unittest.main()