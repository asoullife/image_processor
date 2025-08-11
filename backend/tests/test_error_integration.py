"""
Unit tests for error handling integration utilities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from backend.core.error_integration import (
    ErrorIntegration, error_context, safe_execute, batch_safe_execute,
    GracefulDegradation, get_error_integration, get_graceful_degradation,
    wrap_with_error_handling
)
from backend.core.error_handler import ErrorCategory, ErrorSeverity


class TestErrorIntegration(unittest.TestCase):
    """Test cases for ErrorIntegration class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.integration = ErrorIntegration()
    
    def test_wrap_analyzer(self):
        """Test wrapping analyzer class with error handling."""
        class MockAnalyzer:
            def __init__(self, config):
                self.config = config
            
            def analyze(self, image_path):
                if image_path == "error.jpg":
                    raise ValueError("Analysis failed")
                return {"score": 0.8}
        
        WrappedAnalyzer = self.integration.wrap_analyzer(MockAnalyzer)
        analyzer = WrappedAnalyzer({"test": "config"})
        
        # Test successful analysis
        result = analyzer.analyze("good.jpg")
        self.assertEqual(result, {"score": 0.8})
        
        # Test error handling with graceful degradation
        result = analyzer.analyze("error.jpg")
        self.assertIsNone(result)  # Should return None due to graceful degradation
    
    def test_wrap_processor(self):
        """Test wrapping processor class with error handling."""
        class MockProcessor:
            def __init__(self, config):
                self.config = config
            
            def process(self, data):
                if data == "error_data":
                    raise RuntimeError("Processing failed")
                return f"processed_{data}"
            
            def cleanup_memory(self):
                return True
        
        WrappedProcessor = self.integration.wrap_processor(MockProcessor)
        processor = WrappedProcessor({"test": "config"})
        
        # Test successful processing
        result = processor.process("good_data")
        self.assertEqual(result, "processed_good_data")
        
        # Test error handling (should raise since graceful_degradation=False)
        with self.assertRaises(RuntimeError):
            processor.process("error_data")
        
        # Test cleanup with error handling
        result = processor.cleanup_memory()
        self.assertTrue(result)
    
    def test_wrap_file_operations(self):
        """Test wrapping file manager class with error handling."""
        class MockFileManager:
            def __init__(self, config):
                self.config = config
            
            def scan_images(self, folder):
                if folder == "error_folder":
                    raise FileNotFoundError("Folder not found")
                return ["image1.jpg", "image2.jpg"]
            
            def copy_with_verification(self, src, dst):
                if src == "error.jpg":
                    raise OSError("Copy failed")
                return True
            
            def organize_output(self, images, output_folder):
                return True
        
        WrappedFileManager = self.integration.wrap_file_operations(MockFileManager)
        file_manager = WrappedFileManager({"test": "config"})
        
        # Test successful scan
        result = file_manager.scan_images("good_folder")
        self.assertEqual(result, ["image1.jpg", "image2.jpg"])
        
        # Test error handling (should raise since graceful_degradation=False)
        with self.assertRaises(FileNotFoundError):
            file_manager.scan_images("error_folder")
        
        # Test copy with graceful degradation
        result = file_manager.copy_with_verification("error.jpg", "dest.jpg")
        self.assertFalse(result)  # Should return False due to graceful degradation


class TestErrorContext(unittest.TestCase):
    """Test cases for error context manager."""
    
    def test_error_context_success(self):
        """Test error context manager with successful operation."""
        with error_context("test_component", "test_operation", extra_data="test") as ctx:
            self.assertEqual(ctx["component"], "test_component")
            self.assertEqual(ctx["operation"], "test_operation")
            self.assertEqual(ctx["extra_data"], "test")
    
    def test_error_context_with_error(self):
        """Test error context manager with error."""
        with self.assertRaises(ValueError):
            with error_context("test_component", "test_operation") as ctx:
                raise ValueError("Test error")


class TestSafeExecute(unittest.TestCase):
    """Test cases for safe execution utilities."""
    
    def test_safe_execute_success(self):
        """Test safe execute with successful function."""
        def test_function(x, y):
            return x + y
        
        result = safe_execute(test_function, 2, 3)
        self.assertEqual(result, 5)
    
    def test_safe_execute_with_error(self):
        """Test safe execute with error."""
        def failing_function():
            raise ValueError("Function failed")
        
        result = safe_execute(failing_function, default_return="fallback")
        self.assertEqual(result, "fallback")
    
    def test_batch_safe_execute(self):
        """Test batch safe execution."""
        def good_function(x):
            return x * 2
        
        def bad_function(x):
            raise ValueError("Bad function")
        
        functions_and_args = [
            (good_function, (5,), {}),
            (bad_function, (10,), {}),
            (good_function, (15,), {})
        ]
        
        results, errors = batch_safe_execute(functions_and_args, continue_on_error=True)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], 10)  # 5 * 2
        self.assertIsNone(results[1])    # Failed function
        self.assertEqual(results[2], 30)  # 15 * 2
        
        self.assertEqual(len(errors), 1)  # One error occurred


class TestGracefulDegradation(unittest.TestCase):
    """Test cases for graceful degradation utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.degradation = GracefulDegradation()
    
    def test_degrade_quality_analysis_nonexistent_file(self):
        """Test degraded quality analysis for nonexistent file."""
        result = self.degradation.degrade_quality_analysis("nonexistent.jpg")
        
        self.assertFalse(result["passed"])
        self.assertTrue(result["degraded"])
        self.assertIn("File not found", result["degradation_reason"])
    
    def test_degrade_quality_analysis_empty_file(self):
        """Test degraded quality analysis for empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            result = self.degradation.degrade_quality_analysis(tmp_path)
            
            self.assertFalse(result["passed"])
            self.assertTrue(result["degraded"])
            self.assertIn("Empty file", result["degradation_reason"])
        finally:
            os.unlink(tmp_path)
    
    @patch('PIL.Image.open')
    def test_degrade_quality_analysis_valid_image(self, mock_image_open):
        """Test degraded quality analysis for valid image."""
        # Mock PIL Image
        mock_img = Mock()
        mock_img.size = (1920, 1080)
        mock_img.format = "JPEG"
        mock_image_open.return_value.__enter__.return_value = mock_img
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"fake image data")
            tmp_path = tmp_file.name
        
        try:
            result = self.degradation.degrade_quality_analysis(tmp_path)
            
            self.assertTrue(result["passed"])
            self.assertTrue(result["degraded"])
            self.assertEqual(result["resolution"], (1920, 1080))
            self.assertEqual(result["overall_score"], 0.5)
            self.assertIn("Full analysis failed", result["degradation_reason"])
        finally:
            os.unlink(tmp_path)
    
    def test_degrade_defect_detection(self):
        """Test degraded defect detection."""
        result = self.degradation.degrade_defect_detection("test.jpg")
        
        self.assertTrue(result["passed"])
        self.assertTrue(result["degraded"])
        self.assertEqual(result["defect_count"], 0)
        self.assertEqual(len(result["detected_objects"]), 0)
        self.assertIn("Full defect detection failed", result["degradation_reason"])
    
    def test_degrade_similarity_detection(self):
        """Test degraded similarity detection."""
        image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
        result = self.degradation.degrade_similarity_detection(image_paths)
        
        self.assertTrue(result["degraded"])
        self.assertEqual(result["total_groups"], 3)
        self.assertEqual(result["duplicates_found"], 0)
        self.assertEqual(len(result["similarity_groups"]), 3)
        self.assertIn("Full similarity detection failed", result["degradation_reason"])
    
    def test_degrade_compliance_check(self):
        """Test degraded compliance check."""
        result = self.degradation.degrade_compliance_check("test.jpg")
        
        self.assertTrue(result["overall_compliance"])
        self.assertTrue(result["degraded"])
        self.assertEqual(len(result["logo_detections"]), 0)
        self.assertEqual(len(result["privacy_violations"]), 0)
        self.assertIn("Full compliance check failed", result["degradation_reason"])


class TestGlobalInstances(unittest.TestCase):
    """Test cases for global instance functions."""
    
    def test_get_error_integration(self):
        """Test getting global error integration instance."""
        integration1 = get_error_integration()
        integration2 = get_error_integration()
        
        self.assertIs(integration1, integration2)
        self.assertIsInstance(integration1, ErrorIntegration)
    
    def test_get_graceful_degradation(self):
        """Test getting global graceful degradation instance."""
        degradation1 = get_graceful_degradation()
        degradation2 = get_graceful_degradation()
        
        self.assertIs(degradation1, degradation2)
        self.assertIsInstance(degradation1, GracefulDegradation)
    
    def test_wrap_with_error_handling(self):
        """Test convenience function for wrapping components."""
        class MockAnalyzer:
            def analyze(self, data):
                return data
        
        class MockProcessor:
            def process(self, data):
                return data
        
        class MockFileManager:
            def scan_images(self, folder):
                return []
        
        # Test wrapping different component types
        WrappedAnalyzer = wrap_with_error_handling(MockAnalyzer, 'analyzer')
        WrappedProcessor = wrap_with_error_handling(MockProcessor, 'processor')
        WrappedFileManager = wrap_with_error_handling(MockFileManager, 'file_manager')
        
        self.assertTrue(WrappedAnalyzer.__name__.startswith('ErrorHandled'))
        self.assertTrue(WrappedProcessor.__name__.startswith('ErrorHandled'))
        self.assertTrue(WrappedFileManager.__name__.startswith('ErrorHandled'))
        
        # Test invalid component type
        with self.assertRaises(ValueError):
            wrap_with_error_handling(MockAnalyzer, 'invalid_type')


class TestErrorIntegrationScenarios(unittest.TestCase):
    """Test cases for real-world error integration scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.integration = ErrorIntegration()
    
    def test_analyzer_chain_with_errors(self):
        """Test analyzer chain with various error scenarios."""
        class QualityAnalyzer:
            def analyze(self, image_path):
                if "corrupt" in image_path:
                    raise ValueError("Corrupt image")
                return {"quality_score": 0.8}
        
        class DefectDetector:
            def analyze(self, image_path):
                if "memory_error" in image_path:
                    raise MemoryError("Out of memory")
                return {"defects": []}
        
        # Wrap analyzers
        WrappedQuality = self.integration.wrap_analyzer(QualityAnalyzer)
        WrappedDefect = self.integration.wrap_analyzer(DefectDetector)
        
        quality_analyzer = WrappedQuality()
        defect_detector = WrappedDefect()
        
        # Test normal operation
        quality_result = quality_analyzer.analyze("normal.jpg")
        defect_result = defect_detector.analyze("normal.jpg")
        
        self.assertEqual(quality_result["quality_score"], 0.8)
        self.assertEqual(defect_result["defects"], [])
        
        # Test error scenarios with graceful degradation
        quality_result = quality_analyzer.analyze("corrupt.jpg")
        defect_result = defect_detector.analyze("memory_error.jpg")
        
        self.assertIsNone(quality_result)  # Graceful degradation
        self.assertIsNone(defect_result)   # Graceful degradation
    
    def test_batch_processing_with_mixed_errors(self):
        """Test batch processing with mixed error scenarios."""
        def process_image(image_path):
            if "error" in image_path:
                raise RuntimeError(f"Processing failed for {image_path}")
            return f"processed_{image_path}"
        
        image_paths = [
            "good1.jpg",
            "error1.jpg",
            "good2.jpg",
            "error2.jpg",
            "good3.jpg"
        ]
        
        functions_and_args = [(process_image, (path,), {}) for path in image_paths]
        
        results, errors = batch_safe_execute(
            functions_and_args,
            category=ErrorCategory.PROCESSING,
            continue_on_error=True,
            collect_errors=True
        )
        
        # Check results
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0], "processed_good1.jpg")
        self.assertIsNone(results[1])  # Error case
        self.assertEqual(results[2], "processed_good2.jpg")
        self.assertIsNone(results[3])  # Error case
        self.assertEqual(results[4], "processed_good3.jpg")
        
        # Check errors
        self.assertEqual(len(errors), 2)  # Two errors occurred
        self.assertEqual(errors[0].exception_type, "RuntimeError")
        self.assertEqual(errors[1].exception_type, "RuntimeError")


if __name__ == '__main__':
    unittest.main()