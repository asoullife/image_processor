"""
Unit tests for the comprehensive error handling system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from backend.core.error_handler import (
    ErrorHandler, ErrorCategory, ErrorSeverity, ErrorInfo, RetryConfig,
    get_error_handler, initialize_error_handler, handle_error, retry_on_error,
    handle_errors
)


class TestErrorHandler(unittest.TestCase):
    """Test cases for ErrorHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
        self.test_context = {
            'component': 'test_component',
            'file_path': '/test/path/image.jpg',
            'operation': 'test_operation'
        }
    
    def test_error_categorization(self):
        """Test automatic error categorization."""
        # File system errors
        file_error = FileNotFoundError("File not found")
        error_info = self.error_handler.handle_error(file_error, self.test_context)
        self.assertEqual(error_info.category, ErrorCategory.FILE_SYSTEM)
        
        # Memory errors
        memory_error = MemoryError("Out of memory")
        error_info = self.error_handler.handle_error(memory_error, self.test_context)
        self.assertEqual(error_info.category, ErrorCategory.MEMORY)
        
        # Network errors
        network_error = ConnectionError("Connection failed")
        error_info = self.error_handler.handle_error(network_error, self.test_context)
        self.assertEqual(error_info.category, ErrorCategory.NETWORK)
        
        # Configuration errors
        config_context = {'config': 'test_config'}
        config_error = ValueError("Invalid config value")
        error_info = self.error_handler.handle_error(config_error, config_context)
        self.assertEqual(error_info.category, ErrorCategory.CONFIGURATION)
        
        # Critical errors
        critical_error = SystemError("System error")
        error_info = self.error_handler.handle_error(critical_error, self.test_context)
        self.assertEqual(error_info.category, ErrorCategory.CRITICAL)
    
    def test_severity_assessment(self):
        """Test error severity assessment."""
        # Critical severity
        critical_error = SystemError("System error")
        error_info = self.error_handler.handle_error(critical_error, self.test_context)
        self.assertEqual(error_info.severity, ErrorSeverity.CRITICAL)
        
        # High severity
        memory_error = MemoryError("Out of memory")
        error_info = self.error_handler.handle_error(memory_error, self.test_context)
        self.assertEqual(error_info.severity, ErrorSeverity.HIGH)
        
        # Medium severity
        network_error = ConnectionError("Connection failed")
        error_info = self.error_handler.handle_error(network_error, self.test_context)
        self.assertEqual(error_info.severity, ErrorSeverity.MEDIUM)
        
        # Low severity
        validation_error = ValueError("Invalid value")
        error_info = self.error_handler.handle_error(validation_error, self.test_context)
        self.assertEqual(error_info.severity, ErrorSeverity.LOW)
    
    def test_error_info_creation(self):
        """Test ErrorInfo object creation."""
        test_error = ValueError("Test error")
        error_info = self.error_handler.handle_error(test_error, self.test_context)
        
        self.assertIsNotNone(error_info.error_id)
        self.assertEqual(error_info.exception_type, "ValueError")
        self.assertEqual(error_info.message, "Test error")
        self.assertEqual(error_info.context, self.test_context)
        self.assertIsInstance(error_info.timestamp, datetime)
        self.assertIsNotNone(error_info.stack_trace)
        self.assertEqual(error_info.retry_count, 0)
        self.assertFalse(error_info.resolved)
    
    def test_error_statistics(self):
        """Test error statistics tracking."""
        # Generate some errors
        self.error_handler.handle_error(ValueError("Error 1"), self.test_context)
        self.error_handler.handle_error(FileNotFoundError("Error 2"), self.test_context)
        self.error_handler.handle_error(MemoryError("Error 3"), self.test_context)
        
        stats = self.error_handler.get_error_stats()
        
        self.assertEqual(stats.total_errors, 3)
        self.assertIn(ErrorCategory.VALIDATION, stats.errors_by_category)
        self.assertIn(ErrorCategory.FILE_SYSTEM, stats.errors_by_category)
        self.assertIn(ErrorCategory.MEMORY, stats.errors_by_category)
        self.assertIn("ValueError", stats.errors_by_type)
        self.assertIn("FileNotFoundError", stats.errors_by_type)
        self.assertIn("MemoryError", stats.errors_by_type)
    
    @patch('core.error_handler.psutil.Process')
    def test_memory_usage_tracking(self, mock_process):
        """Test memory usage tracking."""
        # Mock memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 100  # 100MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        test_error = ValueError("Test error")
        error_info = self.error_handler.handle_error(test_error, self.test_context)
        
        self.assertAlmostEqual(error_info.memory_usage_mb, 100.0, places=1)
    
    def test_retry_mechanism(self):
        """Test retry mechanism."""
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection failed")
            return "success"
        
        # Test successful retry
        result = self.error_handler.retry_on_error(
            failing_function,
            category=ErrorCategory.NETWORK
        )
        
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)
        
        # Test retry failure
        call_count = 0
        
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Always fails")
        
        with self.assertRaises(ConnectionError):
            self.error_handler.retry_on_error(
                always_failing_function,
                category=ErrorCategory.NETWORK
            )
        
        # Should have tried max_retries + 1 times
        retry_config = self.error_handler._retry_configs[ErrorCategory.NETWORK]
        self.assertEqual(call_count, retry_config.max_retries + 1)
    
    def test_retry_delay_calculation(self):
        """Test retry delay calculation."""
        retry_config = RetryConfig(
            base_delay=1.0,
            max_delay=10.0,
            exponential_backoff=True,
            jitter=False
        )
        
        # Test exponential backoff
        delay_0 = self.error_handler._calculate_retry_delay(0, retry_config)
        delay_1 = self.error_handler._calculate_retry_delay(1, retry_config)
        delay_2 = self.error_handler._calculate_retry_delay(2, retry_config)
        
        self.assertEqual(delay_0, 1.0)
        self.assertEqual(delay_1, 2.0)
        self.assertEqual(delay_2, 4.0)
        
        # Test max delay
        delay_10 = self.error_handler._calculate_retry_delay(10, retry_config)
        self.assertEqual(delay_10, 10.0)  # Should be capped at max_delay
    
    def test_error_handler_decorator(self):
        """Test error handler decorator."""
        @self.error_handler.handle_errors(
            category=ErrorCategory.PROCESSING,
            graceful_degradation=True,
            default_return="default"
        )
        def failing_function():
            raise ValueError("Function failed")
        
        result = failing_function()
        self.assertEqual(result, "default")
        
        # Check that error was recorded
        stats = self.error_handler.get_error_stats()
        self.assertEqual(stats.total_errors, 1)
    
    def test_error_recovery_memory(self):
        """Test memory error recovery."""
        with patch('gc.collect') as mock_gc:
            mock_gc.return_value = 100  # Objects collected
            
            memory_error = MemoryError("Out of memory")
            context = {'component': 'test'}
            
            # Mock memory usage before and after
            with patch.object(self.error_handler, '_get_memory_usage') as mock_memory:
                mock_memory.side_effect = [1000.0, 800.0]  # Before and after cleanup
                
                error_info = self.error_handler.handle_error(memory_error, context)
                
                # Memory recovery should have been attempted
                mock_gc.assert_called()
    
    def test_file_system_error_recovery(self):
        """Test file system error recovery."""
        with patch('os.path.exists') as mock_exists, \
             patch('builtins.open', create=True) as mock_open:
            
            mock_exists.return_value = True
            mock_open.return_value.__enter__.return_value.read.return_value = b'test'
            
            file_error = FileNotFoundError("File not found")
            context = {'file_path': '/test/file.jpg'}
            
            error_info = self.error_handler.handle_error(file_error, context)
            
            # File recovery should have been attempted
            mock_exists.assert_called_with('/test/file.jpg')
    
    def test_error_reporting(self):
        """Test error reporting functionality."""
        # Generate some test errors
        self.error_handler.handle_error(ValueError("Error 1"), self.test_context)
        self.error_handler.handle_error(FileNotFoundError("Error 2"), self.test_context)
        
        report = self.error_handler.generate_error_report()
        
        self.assertIn('timestamp', report)
        self.assertIn('statistics', report)
        self.assertIn('recent_errors', report)
        self.assertIn('unresolved_errors', report)
        
        self.assertEqual(report['statistics']['total_errors'], 2)
        self.assertEqual(len(report['recent_errors']), 2)
        self.assertEqual(len(report['unresolved_errors']), 2)
    
    def test_error_filtering(self):
        """Test error filtering methods."""
        # Create errors of different categories
        self.error_handler.handle_error(ValueError("Validation error"), self.test_context)
        self.error_handler.handle_error(FileNotFoundError("File error"), self.test_context)
        self.error_handler.handle_error(MemoryError("Memory error"), self.test_context)
        
        # Test filtering by category
        file_errors = self.error_handler.get_errors_by_category(ErrorCategory.FILE_SYSTEM)
        self.assertEqual(len(file_errors), 1)
        self.assertEqual(file_errors[0].exception_type, "FileNotFoundError")
        
        # Test getting recent errors
        recent_errors = self.error_handler.get_recent_errors(2)
        self.assertEqual(len(recent_errors), 2)
        
        # Test getting unresolved errors
        unresolved_errors = self.error_handler.get_unresolved_errors()
        self.assertEqual(len(unresolved_errors), 3)  # All should be unresolved
    
    def test_thread_safety(self):
        """Test thread safety of error handler."""
        errors_generated = []
        
        def generate_errors(thread_id):
            for i in range(10):
                try:
                    raise ValueError(f"Error from thread {thread_id}, iteration {i}")
                except ValueError as e:
                    error_info = self.error_handler.handle_error(
                        e, 
                        {'thread_id': thread_id, 'iteration': i}
                    )
                    errors_generated.append(error_info)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=generate_errors, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all errors were recorded
        stats = self.error_handler.get_error_stats()
        self.assertEqual(stats.total_errors, 50)  # 5 threads * 10 errors each
        self.assertEqual(len(errors_generated), 50)
    
    def test_error_cleanup(self):
        """Test error cleanup functionality."""
        # Generate some errors
        error1 = self.error_handler.handle_error(ValueError("Error 1"), self.test_context)
        error2 = self.error_handler.handle_error(ValueError("Error 2"), self.test_context)
        
        # Mark one as resolved
        error1.resolved = True
        
        # Clear resolved errors
        self.error_handler.clear_resolved_errors()
        
        # Check that only unresolved error remains
        unresolved = self.error_handler.get_unresolved_errors()
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0].error_id, error2.error_id)
    
    def test_custom_retry_config(self):
        """Test custom retry configuration."""
        custom_config = RetryConfig(
            max_retries=5,
            base_delay=0.1,
            exponential_backoff=False
        )
        
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
        
        with self.assertRaises(ValueError):
            self.error_handler.retry_on_error(
                failing_function,
                custom_retry_config=custom_config
            )
        
        # Should have tried max_retries + 1 times
        self.assertEqual(call_count, 6)  # 5 retries + 1 initial attempt


class TestGlobalErrorHandler(unittest.TestCase):
    """Test cases for global error handler functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Reset global error handler
        import backend.core.error_handler
        core.error_handler._global_error_handler = None
    
    def test_get_error_handler(self):
        """Test getting global error handler."""
        handler1 = get_error_handler()
        handler2 = get_error_handler()
        
        # Should return the same instance
        self.assertIs(handler1, handler2)
        self.assertIsInstance(handler1, ErrorHandler)
    
    def test_initialize_error_handler(self):
        """Test initializing global error handler with config."""
        config = {'test_config': 'value'}
        handler = initialize_error_handler(config)
        
        self.assertIsInstance(handler, ErrorHandler)
        self.assertEqual(handler.config, config)
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test handle_error function
        test_error = ValueError("Test error")
        context = {'component': 'test'}
        
        error_info = handle_error(test_error, context)
        self.assertIsInstance(error_info, ErrorInfo)
        
        # Test retry_on_error function
        call_count = 0
        
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection failed")
            return "success"
        
        result = retry_on_error(test_function, category=ErrorCategory.NETWORK)
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 2)
    
    def test_decorator_shortcuts(self):
        """Test decorator shortcuts."""
        @handle_errors(
            category=ErrorCategory.PROCESSING,
            graceful_degradation=True,
            default_return="fallback"
        )
        def failing_function():
            raise ValueError("Function failed")
        
        result = failing_function()
        self.assertEqual(result, "fallback")


class TestErrorRecoveryScenarios(unittest.TestCase):
    """Test cases for specific error recovery scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_memory_pressure_scenario(self):
        """Test handling of memory pressure scenario."""
        with patch('gc.collect') as mock_gc, \
             patch.object(self.error_handler, '_get_memory_usage') as mock_memory:
            
            # Simulate high memory usage, then recovery
            mock_memory.side_effect = [2000.0, 1500.0, 800.0]  # Before, during, after
            mock_gc.return_value = 500  # Objects collected
            
            memory_error = MemoryError("Memory exhausted")
            context = {'component': 'batch_processor', 'batch_size': 1000}
            
            error_info = self.error_handler.handle_error(memory_error, context)
            
            # Should have attempted memory recovery
            self.assertTrue(mock_gc.called)
            self.assertEqual(error_info.category, ErrorCategory.MEMORY)
            self.assertEqual(error_info.severity, ErrorSeverity.HIGH)
    
    def test_file_corruption_scenario(self):
        """Test handling of file corruption scenario."""
        with patch('os.path.exists') as mock_exists, \
             patch('builtins.open', create=True) as mock_open:
            
            # File exists but is corrupted
            mock_exists.return_value = True
            mock_open.side_effect = OSError("File corrupted")
            
            file_error = OSError("Cannot read file")
            context = {'file_path': '/corrupted/image.jpg', 'component': 'quality_analyzer'}
            
            error_info = self.error_handler.handle_error(file_error, context)
            
            self.assertEqual(error_info.category, ErrorCategory.FILE_SYSTEM)
            self.assertFalse(error_info.resolved)  # Should not be resolved
    
    def test_network_timeout_scenario(self):
        """Test handling of network timeout scenario."""
        timeout_error = TimeoutError("Network timeout")
        context = {'url': 'http://example.com/model', 'component': 'model_downloader'}
        
        error_info = self.error_handler.handle_error(timeout_error, context)
        
        self.assertEqual(error_info.category, ErrorCategory.NETWORK)
        self.assertEqual(error_info.severity, ErrorSeverity.MEDIUM)
    
    def test_configuration_validation_scenario(self):
        """Test handling of configuration validation scenario."""
        config_error = ValueError("Invalid threshold value: -0.5")
        context = {'config': 'quality_thresholds', 'parameter': 'min_sharpness'}
        
        error_info = self.error_handler.handle_error(config_error, context)
        
        self.assertEqual(error_info.category, ErrorCategory.CONFIGURATION)
        self.assertEqual(error_info.severity, ErrorSeverity.MEDIUM)
        self.assertFalse(error_info.resolved)  # Config errors are not auto-recoverable


if __name__ == '__main__':
    unittest.main()