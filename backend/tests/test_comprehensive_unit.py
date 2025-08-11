#!/usr/bin/env python3
"""
Comprehensive unit tests for all Adobe Stock Image Processor modules
Ensures complete test coverage of individual functions and classes
"""

import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import all modules to test
from backend.config.config_loader import ConfigLoader
from backend.config.config_validator import ConfigValidator
from backend.core.database import DatabaseManager
from backend.core.progress_tracker import ProgressTracker
from backend.core.batch_processor import BatchProcessor
from backend.core.decision_engine import DecisionEngine
from backend.core.error_handler import ErrorHandler
from backend.core.error_integration import ErrorIntegration
from backend.analyzers.quality_analyzer import QualityAnalyzer
from backend.analyzers.defect_detector import DefectDetector
from backend.analyzers.similarity_finder import SimilarityFinder
from backend.analyzers.compliance_checker import ComplianceChecker
from backend.utils.file_manager import FileManager
from backend.utils.logger import LoggerSetup, get_logger
from backend.utils.report_generator import ReportGenerator


class TestConfigurationModules(unittest.TestCase):
    """Test configuration loading and validation modules"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.json')
        
        # Create test configuration
        self.test_config = {
            "processing": {
                "batch_size": 100,
                "max_workers": 4,
                "checkpoint_interval": 50
            },
            "quality": {
                "min_sharpness": 100.0,
                "max_noise_level": 0.1,
                "min_resolution": [1920, 1080]
            },
            "similarity": {
                "hash_threshold": 5,
                "feature_threshold": 0.85,
                "clustering_eps": 0.3
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_config_loader_load_valid_config(self):
        """Test loading valid configuration"""
        loader = ConfigLoader()
        config = loader.load_config(self.config_file)
        
        self.assertIsInstance(config, dict)
        self.assertIn('processing', config)
        self.assertIn('quality', config)
        self.assertEqual(config['processing']['batch_size'], 100)
    
    def test_config_loader_load_nonexistent_config(self):
        """Test loading non-existent configuration file"""
        loader = ConfigLoader()
        
        with self.assertRaises(FileNotFoundError):
            loader.load_config('/nonexistent/config.json')
    
    def test_config_loader_load_invalid_json(self):
        """Test loading invalid JSON configuration"""
        invalid_config_file = os.path.join(self.test_dir, 'invalid.json')
        with open(invalid_config_file, 'w') as f:
            f.write('{ invalid json }')
        
        loader = ConfigLoader()
        
        with self.assertRaises(json.JSONDecodeError):
            loader.load_config(invalid_config_file)
    
    def test_config_validator_validate_valid_config(self):
        """Test validating valid configuration"""
        validator = ConfigValidator()
        
        is_valid, errors = validator.validate_config(self.test_config)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_config_validator_validate_missing_sections(self):
        """Test validating configuration with missing sections"""
        invalid_config = {"processing": {"batch_size": 100}}
        
        validator = ConfigValidator()
        is_valid, errors = validator.validate_config(invalid_config)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_config_validator_validate_invalid_values(self):
        """Test validating configuration with invalid values"""
        invalid_config = self.test_config.copy()
        invalid_config['processing']['batch_size'] = -1  # Invalid negative value
        
        validator = ConfigValidator()
        is_valid, errors = validator.validate_config(invalid_config)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)


class TestCoreModules(unittest.TestCase):
    """Test core processing modules"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, 'test.db')
        
        # Create test configuration
        self.config = {
            "processing": {"batch_size": 10, "max_workers": 2},
            "database": {"path": self.db_path}
        }
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_database_manager_initialization(self):
        """Test database manager initialization"""
        db_manager = DatabaseManager(self.db_path)
        
        self.assertTrue(os.path.exists(self.db_path))
        self.assertIsNotNone(db_manager.connection)
    
    def test_database_manager_create_tables(self):
        """Test database table creation"""
        db_manager = DatabaseManager(self.db_path)
        db_manager.create_tables()
        
        # Check if tables exist
        cursor = db_manager.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        self.assertIn('processing_sessions', tables)
        self.assertIn('image_results', tables)
    
    def test_progress_tracker_create_session(self):
        """Test progress tracker session creation"""
        db_manager = DatabaseManager(self.db_path)
        tracker = ProgressTracker(db_manager)
        
        session_id = tracker.create_session('/input', '/output', 100)
        
        self.assertIsNotNone(session_id)
        self.assertIsInstance(session_id, str)
    
    def test_progress_tracker_update_progress(self):
        """Test progress tracker progress updates"""
        db_manager = DatabaseManager(self.db_path)
        tracker = ProgressTracker(db_manager)
        
        session_id = tracker.create_session('/input', '/output', 100)
        tracker.update_progress(session_id, 50)
        
        progress = tracker.get_session_progress(session_id)
        self.assertEqual(progress['processed_images'], 50)
    
    def test_error_handler_handle_file_error(self):
        """Test error handler file error handling"""
        handler = ErrorHandler()
        
        # Test with file not found error
        error = FileNotFoundError("Test file not found")
        result = handler.handle_file_error(error, '/test/path.jpg')
        
        self.assertIsInstance(result, bool)
    
    def test_error_handler_handle_processing_error(self):
        """Test error handler processing error handling"""
        handler = ErrorHandler()
        
        # Test with generic processing error
        error = RuntimeError("Test processing error")
        context = {'image_path': '/test/path.jpg', 'step': 'quality_analysis'}
        result = handler.handle_processing_error(error, context)
        
        self.assertIsInstance(result, bool)
    
    def test_decision_engine_make_decision(self):
        """Test decision engine decision making"""
        engine = DecisionEngine(self.config)
        
        # Create mock analysis results
        quality_result = Mock()
        quality_result.passed = True
        quality_result.overall_score = 0.8
        
        defect_result = Mock()
        defect_result.passed = True
        defect_result.anomaly_score = 0.1
        
        compliance_result = Mock()
        compliance_result.overall_compliance = True
        
        decision = engine.make_decision(quality_result, defect_result, compliance_result, 1)
        
        self.assertIn('final_decision', decision)
        self.assertIn('rejection_reasons', decision)
        self.assertIn('confidence_score', decision)


class TestAnalyzerModules(unittest.TestCase):
    """Test analyzer modules"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            "quality": {
                "min_sharpness": 100.0,
                "max_noise_level": 0.1,
                "min_resolution": [800, 600]
            },
            "similarity": {
                "hash_threshold": 5,
                "feature_threshold": 0.85
            },
            "compliance": {
                "logo_detection_confidence": 0.7,
                "face_detection_enabled": True
            }
        }
        
        # Create test image
        self.test_image = self._create_test_image()
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def _create_test_image(self):
        """Create a test image for analyzer testing"""
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        img_array = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        
        image_path = os.path.join(self.test_dir, 'test_image.jpg')
        img.save(image_path, 'JPEG', quality=85)
        
        return image_path
    
    def test_quality_analyzer_initialization(self):
        """Test quality analyzer initialization"""
        analyzer = QualityAnalyzer(self.config)
        
        self.assertIsNotNone(analyzer.config)
        self.assertEqual(analyzer.config['quality']['min_sharpness'], 100.0)
    
    def test_quality_analyzer_analyze_image(self):
        """Test quality analyzer image analysis"""
        analyzer = QualityAnalyzer(self.config)
        
        result = analyzer.analyze(self.test_image)
        
        self.assertIsNotNone(result)
        self.assertHasAttr(result, 'sharpness_score')
        self.assertHasAttr(result, 'noise_level')
        self.assertHasAttr(result, 'overall_score')
        self.assertHasAttr(result, 'passed')
    
    def test_quality_analyzer_check_sharpness(self):
        """Test quality analyzer sharpness checking"""
        analyzer = QualityAnalyzer(self.config)
        
        # Load test image
        import cv2
        image = cv2.imread(self.test_image)
        
        sharpness = analyzer.check_sharpness(image)
        
        self.assertIsInstance(sharpness, float)
        self.assertGreaterEqual(sharpness, 0)
    
    def test_defect_detector_initialization(self):
        """Test defect detector initialization"""
        detector = DefectDetector(self.config)
        
        self.assertIsNotNone(detector.config)
    
    def test_defect_detector_detect_defects(self):
        """Test defect detector defect detection"""
        detector = DefectDetector(self.config)
        
        result = detector.detect_defects(self.test_image)
        
        self.assertIsNotNone(result)
        self.assertHasAttr(result, 'detected_objects')
        self.assertHasAttr(result, 'anomaly_score')
        self.assertHasAttr(result, 'passed')
    
    def test_similarity_finder_initialization(self):
        """Test similarity finder initialization"""
        finder = SimilarityFinder(self.config)
        
        self.assertIsNotNone(finder.config)
        self.assertEqual(finder.similarity_threshold, self.config['similarity']['hash_threshold'])
    
    def test_similarity_finder_compute_hash(self):
        """Test similarity finder hash computation"""
        finder = SimilarityFinder(self.config)
        
        hash_result = finder.compute_hash(self.test_image)
        
        self.assertIsNotNone(hash_result)
        self.assertIsInstance(hash_result, str)
    
    def test_similarity_finder_find_similar_groups(self):
        """Test similarity finder group detection"""
        finder = SimilarityFinder(self.config)
        
        # Create multiple test images
        test_images = [self.test_image]
        for i in range(3):
            additional_image = self._create_test_image()
            test_images.append(additional_image)
        
        groups = finder.find_similar_groups(test_images)
        
        self.assertIsInstance(groups, dict)
    
    def test_compliance_checker_initialization(self):
        """Test compliance checker initialization"""
        checker = ComplianceChecker(self.config)
        
        self.assertIsNotNone(checker.config)
    
    def test_compliance_checker_check_compliance(self):
        """Test compliance checker compliance checking"""
        checker = ComplianceChecker(self.config)
        
        metadata = {'keywords': ['test', 'image'], 'description': 'Test image'}
        result = checker.check_compliance(self.test_image, metadata)
        
        self.assertIsNotNone(result)
        self.assertHasAttr(result, 'logo_detections')
        self.assertHasAttr(result, 'privacy_violations')
        self.assertHasAttr(result, 'overall_compliance')


class TestUtilityModules(unittest.TestCase):
    """Test utility modules"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            "output": {
                "images_per_folder": 200,
                "preserve_metadata": True
            }
        }
        
        # Create test files
        self.test_files = []
        for i in range(5):
            test_file = os.path.join(self.test_dir, f'test_{i}.jpg')
            with open(test_file, 'wb') as f:
                f.write(b'fake image data')
            self.test_files.append(test_file)
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_file_manager_initialization(self):
        """Test file manager initialization"""
        manager = FileManager(self.config)
        
        self.assertIsNotNone(manager.config)
    
    def test_file_manager_scan_images(self):
        """Test file manager image scanning"""
        manager = FileManager(self.config)
        
        # Create some image files with proper extensions
        for ext in ['.jpg', '.jpeg', '.png']:
            test_file = os.path.join(self.test_dir, f'image{ext}')
            with open(test_file, 'wb') as f:
                f.write(b'fake image data')
        
        images = manager.scan_images(self.test_dir)
        
        self.assertIsInstance(images, list)
        self.assertGreaterEqual(len(images), 3)  # At least the 3 we created
    
    def test_file_manager_organize_output(self):
        """Test file manager output organization"""
        manager = FileManager(self.config)
        output_dir = os.path.join(self.test_dir, 'output')
        
        manager.organize_output(self.test_files, output_dir)
        
        self.assertTrue(os.path.exists(output_dir))
        # Check if files were organized into subdirectories
        subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        self.assertGreater(len(subdirs), 0)
    
    def test_logger_initialization(self):
        """Test logger initialization"""
        from backend.config.config_loader import LoggingConfig
        
        log_config = LoggingConfig(
            level='INFO',
            file='test.log',
            max_file_size='10MB',
            backup_count=3
        )
        
        logger_setup = LoggerSetup(log_config)
        logger = logger_setup.setup_logging(self.test_dir)
        
        self.assertIsNotNone(logger)
    
    def test_logger_log_messages(self):
        """Test logger message logging"""
        from backend.config.config_loader import LoggingConfig
        
        log_config = LoggingConfig(
            level='INFO',
            file='test.log',
            max_file_size='10MB',
            backup_count=3
        )
        
        logger_setup = LoggerSetup(log_config)
        logger = logger_setup.setup_logging(self.test_dir)
        
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        # Check if log file was created and contains messages
        log_file = os.path.join(self.test_dir, 'test.log')
        self.assertTrue(os.path.exists(log_file))
        with open(log_file, 'r') as f:
            log_content = f.read()
            self.assertIn("Test info message", log_content)
            self.assertIn("Test warning message", log_content)
            self.assertIn("Test error message", log_content)
    
    def test_report_generator_initialization(self):
        """Test report generator initialization"""
        generator = ReportGenerator(self.config)
        
        self.assertIsNotNone(generator.config)
    
    def test_report_generator_generate_excel_report(self):
        """Test report generator Excel report generation"""
        generator = ReportGenerator(self.config)
        
        # Create mock results
        mock_results = []
        for i in range(3):
            mock_result = Mock()
            mock_result.image_path = f'/test/image_{i}.jpg'
            mock_result.filename = f'image_{i}.jpg'
            mock_result.final_decision = 'approved' if i % 2 == 0 else 'rejected'
            mock_result.quality_result = Mock()
            mock_result.quality_result.overall_score = 0.8
            mock_results.append(mock_result)
        
        output_path = os.path.join(self.test_dir, 'test_report.xlsx')
        generator.generate_excel_report(mock_results, output_path)
        
        self.assertTrue(os.path.exists(output_path))
    
    def test_report_generator_generate_html_dashboard(self):
        """Test report generator HTML dashboard generation"""
        generator = ReportGenerator(self.config)
        
        # Create mock results
        mock_results = []
        for i in range(3):
            mock_result = Mock()
            mock_result.image_path = f'/test/image_{i}.jpg'
            mock_result.filename = f'image_{i}.jpg'
            mock_result.final_decision = 'approved' if i % 2 == 0 else 'rejected'
            mock_results.append(mock_result)
        
        output_path = os.path.join(self.test_dir, 'dashboard.html')
        generator.generate_html_dashboard(mock_results, output_path)
        
        self.assertTrue(os.path.exists(output_path))


class TestBatchProcessorUnit(unittest.TestCase):
    """Detailed unit tests for batch processor"""
    
    def setUp(self):
        self.config = {
            "processing": {
                "batch_size": 5,
                "max_workers": 2,
                "checkpoint_interval": 10
            }
        }
    
    def test_batch_processor_initialization(self):
        """Test batch processor initialization"""
        processor = BatchProcessor(self.config)
        
        self.assertEqual(processor.batch_size, 5)
        self.assertEqual(processor.max_workers, 2)
        self.assertIsNotNone(processor.quality_analyzer)
        self.assertIsNotNone(processor.defect_detector)
    
    def test_batch_processor_cleanup_memory(self):
        """Test batch processor memory cleanup"""
        processor = BatchProcessor(self.config)
        
        # This should not raise an exception
        processor.cleanup_memory()
    
    @patch('core.batch_processor.QualityAnalyzer')
    @patch('core.batch_processor.DefectDetector')
    def test_batch_processor_process_batch_mock(self, mock_defect, mock_quality):
        """Test batch processor batch processing with mocks"""
        # Setup mocks
        mock_quality_instance = Mock()
        mock_quality_instance.analyze.return_value = Mock(passed=True)
        mock_quality.return_value = mock_quality_instance
        
        mock_defect_instance = Mock()
        mock_defect_instance.detect_defects.return_value = Mock(passed=True)
        mock_defect.return_value = mock_defect_instance
        
        processor = BatchProcessor(self.config)
        test_images = ['/test/image1.jpg', '/test/image2.jpg']
        
        results = processor.process_batch(test_images)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_quality_instance.analyze.call_count, 2)
        self.assertEqual(mock_defect_instance.detect_defects.call_count, 2)


if __name__ == '__main__':
    # Run comprehensive unit tests
    unittest.main(verbosity=2, buffer=True)