"""Integration tests for main application orchestration."""

import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import ImageProcessor
from backend.config.config_loader import load_config


class TestMainApplicationIntegration(unittest.TestCase):
    """Test main application orchestration and CLI interface."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.test_dir, 'input')
        self.output_dir = os.path.join(self.test_dir, 'output')
        self.config_dir = os.path.join(self.test_dir, 'config')
        
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
        os.makedirs(self.config_dir)
        
        # Create test images
        self.test_images = []
        for i in range(5):
            image_path = os.path.join(self.input_dir, f'test_image_{i:03d}.jpg')
            # Create dummy image file
            with open(image_path, 'wb') as f:
                f.write(b'fake_image_data')
            self.test_images.append(image_path)
        
        # Create test config
        self.config_path = os.path.join(self.config_dir, 'test_config.json')
        test_config = {
            "processing": {
                "batch_size": 2,
                "max_workers": 1,
                "checkpoint_interval": 2
            },
            "quality": {
                "min_sharpness": 50.0,
                "max_noise_level": 0.2,
                "min_resolution": [800, 600]
            },
            "similarity": {
                "hash_threshold": 10,
                "feature_threshold": 0.8,
                "clustering_eps": 0.5
            },
            "compliance": {
                "logo_detection_confidence": 0.5,
                "face_detection_enabled": True,
                "metadata_validation": True
            },
            "output": {
                "images_per_folder": 200,
                "preserve_metadata": True,
                "generate_thumbnails": False
            },
            "logging": {
                "level": "INFO",
                "file": "test.log"
            }
        }
        
        import json
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f, indent=2)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_processor_initialization(self):
        """Test processor initialization with configuration."""
        processor = ImageProcessor(self.config_path)
        
        # Verify initialization
        self.assertIsNotNone(processor.config)
        self.assertIsNotNone(processor.logger)
        self.assertIsNotNone(processor.progress_tracker)
        self.assertIsNotNone(processor.file_manager)
        self.assertIsNotNone(processor.batch_processor)
        
        # Verify analyzers
        self.assertIsNotNone(processor.quality_analyzer)
        self.assertIsNotNone(processor.defect_detector)
        self.assertIsNotNone(processor.similarity_finder)
        self.assertIsNotNone(processor.compliance_checker)
        
        # Verify decision engine and report generator
        self.assertIsNotNone(processor.decision_engine)
        self.assertIsNotNone(processor.report_generator)
    
    def test_folder_validation(self):
        """Test input and output folder validation."""
        processor = ImageProcessor(self.config_path)
        
        # Test valid folders
        self.assertTrue(processor._validate_folders(self.input_dir, self.output_dir))
        
        # Test invalid input folder
        invalid_input = os.path.join(self.test_dir, 'nonexistent')
        self.assertFalse(processor._validate_folders(invalid_input, self.output_dir))
        
        # Test input is file, not directory
        file_path = os.path.join(self.test_dir, 'test_file.txt')
        with open(file_path, 'w') as f:
            f.write('test')
        self.assertFalse(processor._validate_folders(file_path, self.output_dir))
    
    @patch('main.ImageProcessor._process_single_image')
    @patch('main.ImageProcessor._analyze_similarity')
    def test_processing_pipeline_success(self, mock_similarity, mock_process):
        """Test successful processing pipeline execution."""
        processor = ImageProcessor(self.config_path)
        
        # Mock similarity analysis
        mock_similarity.return_value = {path: 0 for path in self.test_images}
        
        # Mock image processing
        from backend.core.base import ProcessingResult, QualityResult, DefectResult, ComplianceResult
        
        def mock_process_image(image_path, similarity_groups):
            return ProcessingResult(
                image_path=image_path,
                filename=os.path.basename(image_path),
                quality_result=QualityResult(
                    sharpness_score=100.0,
                    noise_level=0.1,
                    exposure_score=0.8,
                    color_balance_score=0.9,
                    resolution=(1920, 1080),
                    file_size=1024000,
                    overall_score=0.85,
                    passed=True
                ),
                defect_result=DefectResult(
                    detected_objects=[],
                    anomaly_score=0.1,
                    defect_count=0,
                    defect_types=[],
                    confidence_scores=[],
                    passed=True
                ),
                similarity_group=0,
                compliance_result=ComplianceResult(
                    logo_detections=[],
                    privacy_violations=[],
                    metadata_issues=[],
                    keyword_relevance=0.8,
                    overall_compliance=True
                ),
                final_decision='approved',
                rejection_reasons=[],
                processing_time=0.5,
                timestamp=datetime.now()
            )
        
        mock_process.side_effect = mock_process_image
        
        # Run processing
        success = processor.run(self.input_dir, self.output_dir, resume=False)
        
        # Verify success
        self.assertTrue(success)
        
        # Verify methods were called
        mock_similarity.assert_called_once()
        self.assertEqual(mock_process.call_count, len(self.test_images))
    
    @patch('main.ImageProcessor._process_single_image')
    @patch('main.ImageProcessor._analyze_similarity')
    def test_processing_pipeline_with_errors(self, mock_similarity, mock_process):
        """Test processing pipeline with some errors."""
        processor = ImageProcessor(self.config_path)
        
        # Mock similarity analysis
        mock_similarity.return_value = {path: 0 for path in self.test_images}
        
        # Mock image processing with some failures
        def mock_process_image(image_path, similarity_groups):
            if 'test_image_002' in image_path:
                raise Exception("Simulated processing error")
            
            from backend.core.base import ProcessingResult, QualityResult, DefectResult, ComplianceResult
            return ProcessingResult(
                image_path=image_path,
                filename=os.path.basename(image_path),
                quality_result=QualityResult(
                    sharpness_score=100.0,
                    noise_level=0.1,
                    exposure_score=0.8,
                    color_balance_score=0.9,
                    resolution=(1920, 1080),
                    file_size=1024000,
                    overall_score=0.85,
                    passed=True
                ),
                defect_result=DefectResult(
                    detected_objects=[],
                    anomaly_score=0.1,
                    defect_count=0,
                    defect_types=[],
                    confidence_scores=[],
                    passed=True
                ),
                similarity_group=0,
                compliance_result=ComplianceResult(
                    logo_detections=[],
                    privacy_violations=[],
                    metadata_issues=[],
                    keyword_relevance=0.8,
                    overall_compliance=True
                ),
                final_decision='approved',
                rejection_reasons=[],
                processing_time=0.5,
                timestamp=datetime.now()
            )
        
        mock_process.side_effect = mock_process_image
        
        # Run processing
        success = processor.run(self.input_dir, self.output_dir, resume=False)
        
        # Should still succeed despite individual image errors
        self.assertTrue(success)
    
    def test_session_management(self):
        """Test session creation and management."""
        processor = ImageProcessor(self.config_path)
        
        # Test session listing (should be empty initially)
        with patch('builtins.print') as mock_print:
            processor.list_sessions()
            mock_print.assert_called_with("No sessions found.")
    
    @patch('builtins.input')
    def test_resume_functionality_no_sessions(self, mock_input):
        """Test resume functionality when no sessions exist."""
        processor = ImageProcessor(self.config_path)
        
        # Test resume with no existing sessions
        session_id, start_index = processor._handle_resume_request(
            self.input_dir, self.output_dir
        )
        
        self.assertIsNone(session_id)
        self.assertEqual(start_index, 0)
    
    @patch('builtins.input')
    def test_resume_functionality_with_session(self, mock_input):
        """Test resume functionality with existing session."""
        processor = ImageProcessor(self.config_path)
        
        # Create a session first
        session_id = processor.progress_tracker.create_session(
            input_folder=self.input_dir,
            output_folder=self.output_dir,
            total_images=len(self.test_images),
            config=processor.config_dict
        )
        
        # Save some progress
        checkpoint_data = {
            'processed_count': 2,
            'approved_count': 1,
            'rejected_count': 1,
            'timestamp': datetime.now().isoformat(),
            'progress_percentage': 40.0
        }
        processor.progress_tracker.save_checkpoint(session_id, checkpoint_data)
        
        # Mock user choosing to resume
        mock_input.return_value = 'r'
        
        # Test resume
        resumed_session_id, start_index = processor._handle_resume_request(
            self.input_dir, self.output_dir
        )
        
        self.assertEqual(resumed_session_id, session_id)
        self.assertEqual(start_index, 2)
    
    def test_error_handling_invalid_config(self):
        """Test error handling with invalid configuration."""
        # Create invalid config
        invalid_config_path = os.path.join(self.config_dir, 'invalid_config.json')
        with open(invalid_config_path, 'w') as f:
            f.write('invalid json content')
        
        # Should exit with error
        with self.assertRaises(SystemExit):
            ImageProcessor(invalid_config_path)
    
    def test_cleanup_old_sessions(self):
        """Test cleanup of old sessions."""
        processor = ImageProcessor(self.config_path)
        
        # Test cleanup (should not crash)
        with patch('builtins.print') as mock_print:
            processor.cleanup_old_sessions(days_old=1)
            # Should print cleanup message
            self.assertTrue(any('Cleaned up' in str(call) for call in mock_print.call_args_list))
    
    def test_session_info_nonexistent(self):
        """Test getting info for nonexistent session."""
        processor = ImageProcessor(self.config_path)
        
        with patch('builtins.print') as mock_print:
            processor.get_session_info('nonexistent_session')
            mock_print.assert_called_with("Session 'nonexistent_session' not found.")
    
    def test_session_recovery_nonexistent(self):
        """Test recovery of nonexistent session."""
        processor = ImageProcessor(self.config_path)
        
        with patch('builtins.print') as mock_print:
            success = processor.recover_session('nonexistent_session')
            self.assertFalse(success)
            mock_print.assert_called_with("Session 'nonexistent_session' not found.")
    
    @patch('main.ImageProcessor._analyze_similarity')
    def test_similarity_analysis_error_handling(self, mock_similarity):
        """Test error handling in similarity analysis."""
        processor = ImageProcessor(self.config_path)
        
        # Mock similarity analysis failure
        mock_similarity.side_effect = Exception("Similarity analysis failed")
        
        # Should handle error gracefully
        result = processor._analyze_similarity(self.test_images)
        
        # Should return empty mapping
        expected = {path: 0 for path in self.test_images}
        self.assertEqual(result, expected)
    
    def test_progress_tracking_variables(self):
        """Test progress tracking variable initialization."""
        processor = ImageProcessor(self.config_path)
        
        # Verify initial state
        self.assertIsNone(processor.start_time)
        self.assertEqual(processor.processed_count, 0)
        self.assertEqual(processor.total_count, 0)
        self.assertEqual(processor.approved_count, 0)
        self.assertEqual(processor.rejected_count, 0)
    
    def test_signal_handler_setup(self):
        """Test signal handler setup."""
        processor = ImageProcessor(self.config_path)
        
        # Verify shutdown event is initialized
        self.assertIsNotNone(processor.shutdown_event)
        self.assertFalse(processor.shutdown_event.is_set())
        
        # Test setting shutdown event
        processor.shutdown_event.set()
        self.assertTrue(processor.shutdown_event.is_set())


class TestMainApplicationCLI(unittest.TestCase):
    """Test CLI interface functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.test_dir, 'input')
        self.output_dir = os.path.join(self.test_dir, 'output')
        
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('main.ImageProcessor')
    @patch('sys.argv', ['backend/main.py', 'process', 'input', 'output'])
    def test_cli_process_command(self, mock_processor_class):
        """Test CLI process command."""
        mock_processor = MagicMock()
        mock_processor.run.return_value = True
        mock_processor_class.return_value = mock_processor
        
        from backend.main import main
        
        # Should not raise exception
        with self.assertRaises(SystemExit) as cm:
            main()
        
        # Should exit with success code
        self.assertEqual(cm.exception.code, 0)
        
        # Verify processor was called correctly
        mock_processor.run.assert_called_once_with('input', 'output', False)
    
    @patch('main.ImageProcessor')
    @patch('sys.argv', ['backend/main.py', 'process', 'input', 'output', '--resume'])
    def test_cli_process_command_with_resume(self, mock_processor_class):
        """Test CLI process command with resume flag."""
        mock_processor = MagicMock()
        mock_processor.run.return_value = True
        mock_processor_class.return_value = mock_processor
        
        from backend.main import main
        
        with self.assertRaises(SystemExit) as cm:
            main()
        
        self.assertEqual(cm.exception.code, 0)
        mock_processor.run.assert_called_once_with('input', 'output', True)
    
    @patch('main.ImageProcessor')
    @patch('sys.argv', ['backend/main.py', 'sessions', '--list'])
    def test_cli_sessions_list_command(self, mock_processor_class):
        """Test CLI sessions list command."""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        from backend.main import main
        main()
        
        mock_processor.list_sessions.assert_called_once_with(None)
    
    @patch('main.ImageProcessor')
    @patch('sys.argv', ['backend/main.py', 'sessions', '--info', 'test_session'])
    def test_cli_sessions_info_command(self, mock_processor_class):
        """Test CLI sessions info command."""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        from backend.main import main
        main()
        
        mock_processor.get_session_info.assert_called_once_with('test_session')
    
    @patch('main.ImageProcessor')
    @patch('sys.argv', ['backend/main.py', 'sessions', '--recover', 'test_session'])
    def test_cli_sessions_recover_command(self, mock_processor_class):
        """Test CLI sessions recover command."""
        mock_processor = MagicMock()
        mock_processor.recover_session.return_value = True
        mock_processor_class.return_value = mock_processor
        
        from backend.main import main
        
        with self.assertRaises(SystemExit) as cm:
            main()
        
        self.assertEqual(cm.exception.code, 0)
        mock_processor.recover_session.assert_called_once_with('test_session')
    
    @patch('main.ImageProcessor')
    @patch('sys.argv', ['backend/main.py', 'sessions', '--cleanup', '7'])
    def test_cli_sessions_cleanup_command(self, mock_processor_class):
        """Test CLI sessions cleanup command."""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        from backend.main import main
        main()
        
        mock_processor.cleanup_old_sessions.assert_called_once_with(7)


if __name__ == '__main__':
    unittest.main()