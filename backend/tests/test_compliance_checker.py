"""
Unit tests for ComplianceChecker module

Tests cover:
- Logo and trademark detection
- Face detection for privacy concerns
- License plate detection
- Metadata validation
- Keyword relevance checking
- Overall compliance determination
"""

import unittest
import os
import sys
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.analyzers.compliance_checker import (
    ComplianceChecker, ComplianceResult, LogoDetection, PrivacyViolation,
    analyze_image_compliance, batch_compliance_check
)


class TestComplianceChecker(unittest.TestCase):
    """Test cases for ComplianceChecker class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            'logo_detection_confidence': 0.7,
            'face_detection_enabled': True,
            'metadata_validation': True
        }
        self.checker = ComplianceChecker(self.test_config)
        
        # Create a simple test image
        self.test_image_path = os.path.join(self.temp_dir, 'test_image.jpg')
        self._create_test_image(self.test_image_path)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_image(self, path: str, width: int = 100, height: int = 100):
        """Create a simple test image"""
        try:
            from PIL import Image
            # Create a simple RGB image
            image = Image.new('RGB', (width, height), color='white')
            image.save(path)
        except ImportError:
            # Create a dummy file if PIL is not available
            with open(path, 'wb') as f:
                f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00d\x00\x00\x00d\x08\x02\x00\x00\x00\xff\x80\x02\x03')
    
    def test_initialization(self):
        """Test ComplianceChecker initialization"""
        # Test with default config
        checker_default = ComplianceChecker()
        self.assertEqual(checker_default.logo_confidence_threshold, 0.7)
        self.assertTrue(checker_default.face_detection_enabled)
        self.assertTrue(checker_default.metadata_validation)
        
        # Test with custom config
        custom_config = {
            'logo_detection_confidence': 0.8,
            'face_detection_enabled': False,
            'metadata_validation': False
        }
        checker_custom = ComplianceChecker(custom_config)
        self.assertEqual(checker_custom.logo_confidence_threshold, 0.8)
        self.assertFalse(checker_custom.face_detection_enabled)
        self.assertFalse(checker_custom.metadata_validation)
    
    def test_check_compliance_file_not_found(self):
        """Test compliance check with non-existent file"""
        result = self.checker.check_compliance('non_existent_file.jpg')
        
        self.assertIsInstance(result, ComplianceResult)
        self.assertFalse(result.overall_compliance)
        self.assertTrue(any("Image file not found" in issue or "Could not load image" in issue for issue in result.metadata_issues))
    
    @patch('analyzers.compliance_checker.cv2')
    @patch('analyzers.compliance_checker.Image')
    def test_check_compliance_success(self, mock_image, mock_cv2):
        """Test successful compliance check"""
        # Mock image loading
        mock_image.open.return_value.__enter__.return_value.convert.return_value = Mock()
        mock_cv2.imread.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Mock the checker methods to return empty results (compliant)
        self.checker.detect_logos = Mock(return_value=[])
        self.checker.check_privacy_elements = Mock(return_value=[])
        self.checker._validate_metadata = Mock(return_value=[])
        self.checker._check_keyword_relevance = Mock(return_value=0.8)
        
        result = self.checker.check_compliance(self.test_image_path)
        
        self.assertIsInstance(result, ComplianceResult)
        self.assertTrue(result.overall_compliance)
        self.assertEqual(len(result.logo_detections), 0)
        self.assertEqual(len(result.privacy_violations), 0)
        self.assertEqual(len(result.metadata_issues), 0)
        self.assertEqual(result.keyword_relevance, 0.8)
    
    @patch('analyzers.compliance_checker.pytesseract')
    def test_detect_logos_ocr(self, mock_pytesseract):
        """Test logo detection using OCR"""
        # Mock OCR data with Nike logo
        mock_ocr_data = {
            'text': ['', 'Nike', 'Just Do It', ''],
            'conf': [0, 85, 75, 0],
            'left': [0, 10, 50, 0],
            'top': [0, 20, 60, 0],
            'width': [0, 40, 80, 0],
            'height': [0, 30, 20, 0]
        }
        mock_pytesseract.image_to_data.return_value = mock_ocr_data
        mock_pytesseract.Output.DICT = 'dict'
        
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = self.checker.detect_logos(test_image)
        
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].text_detected, 'Nike')
        self.assertEqual(detections[0].confidence, 0.85)
        self.assertEqual(detections[0].bounding_box, (10, 20, 40, 30))
        self.assertEqual(detections[0].detection_method, 'ocr')
    
    @patch('analyzers.compliance_checker.cv2')
    def test_detect_faces(self, mock_cv2):
        """Test face detection"""
        # Mock face cascade
        mock_cascade = Mock()
        mock_cascade.detectMultiScale.return_value = [(10, 20, 50, 60), (70, 80, 40, 45)]
        self.checker.face_cascade = mock_cascade
        
        # Mock cv2 functions
        mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
        
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        violations = self.checker.check_privacy_elements(test_image)
        
        # Should detect 2 faces
        face_violations = [v for v in violations if v.violation_type == 'face']
        self.assertEqual(len(face_violations), 2)
        self.assertEqual(face_violations[0].bounding_box, (10, 20, 50, 60))
        self.assertEqual(face_violations[1].bounding_box, (70, 80, 40, 45))
    
    @patch('analyzers.compliance_checker.pytesseract')
    def test_detect_license_plates(self, mock_pytesseract):
        """Test license plate detection"""
        # Mock OCR text with license plate patterns
        mock_pytesseract.image_to_string.return_value = "ABC-123 some text XYZ 456"
        
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        violations = self.checker.check_privacy_elements(test_image)
        
        # Should detect license plate patterns
        plate_violations = [v for v in violations if v.violation_type == 'license_plate']
        self.assertGreater(len(plate_violations), 0)
        self.assertIn('ABC-123', plate_violations[0].description)
    
    @patch('analyzers.compliance_checker.Image')
    @patch('analyzers.compliance_checker.TAGS')
    def test_validate_metadata(self, mock_tags, mock_image):
        """Test metadata validation"""
        # Mock EXIF data with GPS information
        mock_exif = {
            'Make': 'iPhone',
            'Model': 'iPhone 12',
            'Software': 'Watermark Pro Trial',
            'GPS': 'some gps data'
        }
        
        mock_image.open.return_value.__enter__.return_value._getexif.return_value = {
            1: 'iPhone', 2: 'iPhone 12', 3: 'Watermark Pro Trial', 4: 'some gps data'
        }
        mock_tags.get.side_effect = lambda x, default: ['Make', 'Model', 'Software', 'GPS'][x-1] if x <= 4 else default
        
        issues = self.checker._validate_metadata(self.test_image_path, None)
        
        self.assertGreater(len(issues), 0)
        # Should detect multiple issues
        issue_text = ' '.join(issues)
        self.assertIn('GPS', issue_text)
        self.assertIn('Personal device', issue_text)
        self.assertIn('watermark', issue_text.lower())
    
    def test_check_keyword_relevance(self):
        """Test keyword relevance checking"""
        # Test with appropriate keywords
        good_metadata = {
            'keywords': ['nature', 'landscape', 'mountain', 'sky'],
            'description': 'Beautiful mountain landscape'
        }
        relevance = self.checker._check_keyword_relevance(good_metadata)
        self.assertEqual(relevance, 1.0)
        
        # Test with inappropriate keywords
        bad_metadata = {
            'keywords': ['nature', 'nude', 'violence', 'nike'],
            'description': 'Inappropriate content'
        }
        relevance = self.checker._check_keyword_relevance(bad_metadata)
        self.assertLess(relevance, 1.0)
        
        # Test with no keywords
        empty_metadata = {}
        relevance = self.checker._check_keyword_relevance(empty_metadata)
        self.assertEqual(relevance, 0.0)
    
    def test_get_compliance_summary(self):
        """Test compliance summary generation"""
        # Create a result with various issues
        result = ComplianceResult(
            logo_detections=[
                LogoDetection('Nike', 0.8, (10, 20, 30, 40), 'ocr')
            ],
            privacy_violations=[
                PrivacyViolation('face', 0.9, (50, 60, 70, 80), 'Face detected')
            ],
            metadata_issues=['GPS data present'],
            keyword_relevance=0.3,
            overall_compliance=False
        )
        
        summary = self.checker.get_compliance_summary(result)
        
        self.assertFalse(summary['overall_compliance'])
        self.assertEqual(summary['logo_count'], 1)
        self.assertEqual(summary['privacy_violation_count'], 1)
        self.assertEqual(summary['metadata_issue_count'], 1)
        self.assertEqual(summary['keyword_relevance_score'], 0.3)
        self.assertGreater(len(summary['main_issues']), 0)
    
    def test_analyze_image_compliance_function(self):
        """Test standalone analyze_image_compliance function"""
        with patch.object(ComplianceChecker, 'check_compliance') as mock_check:
            mock_result = ComplianceResult([], [], [], 1.0, True)
            mock_check.return_value = mock_result
            
            result = analyze_image_compliance(self.test_image_path, self.test_config)
            
            self.assertEqual(result, mock_result)
            mock_check.assert_called_once_with(self.test_image_path)
    
    def test_batch_compliance_check_function(self):
        """Test batch compliance check function"""
        image_paths = [self.test_image_path, 'another_image.jpg']
        
        with patch.object(ComplianceChecker, 'check_compliance') as mock_check:
            mock_result = ComplianceResult([], [], [], 1.0, True)
            mock_check.return_value = mock_result
            
            results = batch_compliance_check(image_paths, self.test_config)
            
            self.assertEqual(len(results), 2)
            self.assertIn(self.test_image_path, results)
            self.assertIn('another_image.jpg', results)


class TestComplianceDataClasses(unittest.TestCase):
    """Test cases for compliance data classes"""
    
    def test_logo_detection_creation(self):
        """Test LogoDetection dataclass creation"""
        detection = LogoDetection(
            text_detected='Nike',
            confidence=0.85,
            bounding_box=(10, 20, 30, 40),
            detection_method='ocr'
        )
        
        self.assertEqual(detection.text_detected, 'Nike')
        self.assertEqual(detection.confidence, 0.85)
        self.assertEqual(detection.bounding_box, (10, 20, 30, 40))
        self.assertEqual(detection.detection_method, 'ocr')
    
    def test_privacy_violation_creation(self):
        """Test PrivacyViolation dataclass creation"""
        violation = PrivacyViolation(
            violation_type='face',
            confidence=0.9,
            bounding_box=(50, 60, 70, 80),
            description='Face detected'
        )
        
        self.assertEqual(violation.violation_type, 'face')
        self.assertEqual(violation.confidence, 0.9)
        self.assertEqual(violation.bounding_box, (50, 60, 70, 80))
        self.assertEqual(violation.description, 'Face detected')
    
    def test_compliance_result_creation(self):
        """Test ComplianceResult dataclass creation"""
        result = ComplianceResult(
            logo_detections=[],
            privacy_violations=[],
            metadata_issues=['GPS data present'],
            keyword_relevance=0.7,
            overall_compliance=False
        )
        
        self.assertEqual(len(result.logo_detections), 0)
        self.assertEqual(len(result.privacy_violations), 0)
        self.assertEqual(len(result.metadata_issues), 1)
        self.assertEqual(result.keyword_relevance, 0.7)
        self.assertFalse(result.overall_compliance)


if __name__ == '__main__':
    unittest.main()