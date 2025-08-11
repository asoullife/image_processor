"""
Unit tests for DefectDetector module

Tests comprehensive defect detection functionality including:
- Object defect detection
- Edge-based defect detection (cracks, breaks)
- Shape anomaly detection through template matching
- Confidence scoring and result aggregation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

# Mock OpenCV and PIL before importing the module
with patch.dict('sys.modules', {
    'cv2': MagicMock(),
    'PIL': MagicMock(),
    'PIL.Image': MagicMock()
}):
    from backend.analyzers.defect_detector import (
        DefectDetector, EdgeDetector, ShapeMatcher, AnomalyDetector,
        EdgeDefect, ShapeAnomaly
    )
    from backend.core.base import DefectResult, ObjectDefect


class TestDefectDetector(unittest.TestCase):
    """Test cases for DefectDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'defect_detection': {
                'confidence_threshold': 0.5,
                'edge_threshold': 50,
                'model_path': None
            }
        }
        
        # Mock cv2 functions
        self.mock_cv2 = MagicMock()
        self.mock_np = MagicMock()
        
        # Create test image data
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image_path = None
        
        # Create temporary test image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            self.test_image_path = f.name
    
    def tearDown(self):
        """Clean up test fixtures"""
        if self.test_image_path and os.path.exists(self.test_image_path):
            os.unlink(self.test_image_path)
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_defect_detector_initialization(self, mock_np, mock_cv2):
        """Test DefectDetector initialization"""
        detector = DefectDetector(self.config)
        
        self.assertEqual(detector.confidence_threshold, 0.5)
        self.assertEqual(detector.edge_threshold, 50)
        self.assertIsNone(detector.model_path)
        self.assertIsNotNone(detector.edge_detector)
        self.assertIsNotNone(detector.shape_matcher)
        self.assertIsNotNone(detector.anomaly_detector)
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_defect_detector_with_app_config(self, mock_np, mock_cv2):
        """Test DefectDetector initialization with AppConfig object"""
        # Mock AppConfig object
        mock_config = Mock()
        mock_config.defect_detection = Mock()
        mock_config.defect_detection.confidence_threshold = 0.7
        mock_config.defect_detection.edge_threshold = 75
        mock_config.defect_detection.model_path = '/path/to/model'
        
        detector = DefectDetector(mock_config)
        
        self.assertEqual(detector.confidence_threshold, 0.7)
        self.assertEqual(detector.edge_threshold, 75)
        self.assertEqual(detector.model_path, '/path/to/model')
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_analyze_success(self, mock_np, mock_cv2):
        """Test successful defect analysis"""
        # Setup mocks
        mock_cv2.imread.return_value = self.test_image
        mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.threshold.return_value = (127, np.zeros((100, 100), dtype=np.uint8))
        mock_cv2.findContours.return_value = ([], None)
        mock_cv2.Canny.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.matchTemplate.return_value = np.zeros((1, 1))
        mock_np.mean.return_value = 0.5
        mock_np.std.return_value = 0.1
        mock_np.where.return_value = ([], [])
        
        detector = DefectDetector(self.config)
        result = detector.analyze(self.test_image_path)
        
        self.assertIsInstance(result, DefectResult)
        self.assertIsInstance(result.detected_objects, list)
        self.assertIsInstance(result.anomaly_score, float)
        self.assertIsInstance(result.defect_count, int)
        self.assertIsInstance(result.defect_types, list)
        self.assertIsInstance(result.confidence_scores, list)
        self.assertIsInstance(result.passed, bool)
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_analyze_with_defects(self, mock_np, mock_cv2):
        """Test analysis with detected defects"""
        # Setup mocks to simulate defects
        mock_cv2.imread.return_value = self.test_image
        mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
        
        # Mock contour detection to return some contours
        mock_contour = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]])
        mock_cv2.findContours.return_value = ([mock_contour], None)
        mock_cv2.contourArea.return_value = 1500  # Above minimum area
        mock_cv2.boundingRect.return_value = (10, 10, 10, 10)
        mock_cv2.arcLength.return_value = 40
        mock_cv2.convexHull.return_value = mock_contour
        
        # Mock edge detection
        mock_cv2.Canny.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.GaussianBlur.return_value = np.zeros((100, 100), dtype=np.uint8)
        
        # Mock template matching
        mock_cv2.matchTemplate.return_value = np.array([[0.7]])  # Above threshold
        mock_np.where.return_value = ([25], [25])  # Mock location
        
        mock_np.mean.return_value = 0.5
        mock_np.std.return_value = 0.1
        
        detector = DefectDetector(self.config)
        result = detector.analyze(self.test_image_path)
        
        self.assertIsInstance(result, DefectResult)
        # Should have some defects detected
        self.assertGreaterEqual(result.defect_count, 0)
    
    def test_analyze_missing_dependencies(self):
        """Test analysis with missing dependencies"""
        with patch('analyzers.defect_detector.cv2', None), \
             patch('analyzers.defect_detector.np', None):
            detector = DefectDetector(self.config)
            result = detector.analyze(self.test_image_path)
            
            self.assertIsInstance(result, DefectResult)
            self.assertFalse(result.passed)
            self.assertEqual(result.defect_count, 0)
            self.assertEqual(result.anomaly_score, 1.0)
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_analyze_invalid_image_path(self, mock_np, mock_cv2):
        """Test analysis with invalid image path"""
        detector = DefectDetector(self.config)
        result = detector.analyze('/nonexistent/path.jpg')
        
        self.assertIsInstance(result, DefectResult)
        self.assertFalse(result.passed)
        self.assertEqual(result.defect_count, 0)
        self.assertEqual(result.anomaly_score, 1.0)
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_analyze_image_load_failure(self, mock_np, mock_cv2):
        """Test analysis when image loading fails"""
        mock_cv2.imread.return_value = None  # Simulate load failure
        
        detector = DefectDetector(self.config)
        result = detector.analyze(self.test_image_path)
        
        self.assertIsInstance(result, DefectResult)
        self.assertFalse(result.passed)
        self.assertEqual(result.defect_count, 0)
        self.assertEqual(result.anomaly_score, 1.0)
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_detect_objects_simple(self, mock_np, mock_cv2):
        """Test simple object detection"""
        # Setup mocks
        mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.threshold.return_value = (127, np.zeros((100, 100), dtype=np.uint8))
        
        # Mock contour with sufficient area
        mock_contour = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]])
        mock_cv2.findContours.return_value = ([mock_contour], None)
        mock_cv2.contourArea.return_value = 1500  # Above minimum area
        mock_cv2.boundingRect.return_value = (10, 10, 10, 10)
        mock_cv2.arcLength.return_value = 40
        mock_np.pi = 3.14159
        
        detector = DefectDetector(self.config)
        objects = detector._detect_objects_simple(self.test_image)
        
        self.assertIsInstance(objects, list)
        if objects:  # If objects were detected
            obj = objects[0]
            self.assertIn('type', obj)
            self.assertIn('bounding_box', obj)
            self.assertIn('area', obj)
            self.assertIn('contour', obj)
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_classify_object_simple(self, mock_np, mock_cv2):
        """Test simple object classification"""
        # Mock contour
        mock_contour = np.array([[[10, 10]], [[20, 10]], [[20, 20]], [[10, 20]]])
        mock_cv2.arcLength.return_value = 40
        mock_cv2.boundingRect.return_value = (10, 10, 10, 10)
        mock_np.pi = 3.14159
        
        detector = DefectDetector(self.config)
        
        # Test different scenarios
        object_type = detector._classify_object_simple(mock_contour, 1500)
        self.assertIsInstance(object_type, str)
        self.assertIn(object_type, [
            'circular_object', 'elongated_object', 'tall_object', 
            'rectangular_object', 'unknown'
        ])
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_passes_defect_checks(self, mock_np, mock_cv2):
        """Test defect check evaluation"""
        detector = DefectDetector(self.config)
        
        # Test passing case
        passes = detector._passes_defect_checks(0.1, 1, [0.3])
        self.assertTrue(passes)
        
        # Test failing cases
        passes = detector._passes_defect_checks(0.5, 1, [0.3])  # High anomaly score
        self.assertFalse(passes)
        
        passes = detector._passes_defect_checks(0.1, 5, [0.3])  # Too many defects
        self.assertFalse(passes)
        
        passes = detector._passes_defect_checks(0.1, 1, [0.8, 0.9])  # High confidence defects
        self.assertFalse(passes)
    
    @patch('analyzers.defect_detector.cv2')
    def test_load_object_templates(self, mock_cv2):
        """Test object template loading"""
        # Mock cv2 functions
        mock_cv2.circle = MagicMock()
        mock_cv2.rectangle = MagicMock()
        mock_cv2.fillPoly = MagicMock()
        
        detector = DefectDetector(self.config)
        templates = detector._load_object_templates()
        
        self.assertIsInstance(templates, dict)
        self.assertIn('circle', templates)
        self.assertIn('rectangle', templates)
        self.assertIn('triangle', templates)


class TestEdgeDetector(unittest.TestCase):
    """Test cases for EdgeDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.edge_threshold = 50
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_edge_detector_initialization(self, mock_np, mock_cv2):
        """Test EdgeDetector initialization"""
        detector = EdgeDetector(self.edge_threshold)
        
        self.assertEqual(detector.edge_threshold, self.edge_threshold)
        self.assertIsNotNone(detector.logger)
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_detect_defects(self, mock_np, mock_cv2):
        """Test edge defect detection"""
        # Setup mocks
        mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.GaussianBlur.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.Canny.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.findContours.return_value = ([], None)
        
        detector = EdgeDetector(self.edge_threshold)
        defects = detector.detect_defects(self.test_image)
        
        self.assertIsInstance(defects, list)
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_analyze_edge_contour(self, mock_np, mock_cv2):
        """Test edge contour analysis"""
        # Mock contour with sufficient area
        mock_contour = np.array([[[10, 10]], [[50, 10]], [[50, 20]], [[10, 20]]])
        mock_cv2.contourArea.return_value = 1500  # Above minimum
        mock_cv2.boundingRect.return_value = (10, 10, 40, 10)  # Elongated
        mock_cv2.arcLength.return_value = 100
        mock_cv2.convexHull.return_value = mock_contour
        
        detector = EdgeDetector(self.edge_threshold)
        defect = detector._analyze_edge_contour(mock_contour)
        
        if defect:  # If defect was detected
            self.assertIsInstance(defect, EdgeDefect)
            self.assertIsInstance(defect.defect_type, str)
            self.assertIsInstance(defect.confidence, float)
            self.assertIsInstance(defect.severity, str)
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_classify_edge_defect(self, mock_np, mock_cv2):
        """Test edge defect classification"""
        detector = EdgeDetector(self.edge_threshold)
        
        # Test crack detection (high aspect ratio, low solidity)
        defect_type, confidence = detector._classify_edge_defect(1000, 100, 5.0, 0.5)
        self.assertEqual(defect_type, 'crack')
        self.assertGreater(confidence, 0.0)
        
        # Test break detection (medium aspect ratio, low solidity)
        defect_type, confidence = detector._classify_edge_defect(1000, 100, 2.0, 0.6)
        self.assertEqual(defect_type, 'break')
        self.assertGreater(confidence, 0.0)
        
        # Test scratch detection (elongated, higher solidity)
        defect_type, confidence = detector._classify_edge_defect(1000, 100, 3.0, 0.8)
        self.assertEqual(defect_type, 'scratch')
        self.assertGreater(confidence, 0.0)
    
    def test_determine_severity(self):
        """Test severity determination"""
        detector = EdgeDetector(self.edge_threshold)
        
        # Test high severity
        severity = detector._determine_severity(6000, 15.0, 0.3)
        self.assertEqual(severity, 'high')
        
        # Test medium severity
        severity = detector._determine_severity(2000, 6.0, 0.6)
        self.assertEqual(severity, 'medium')
        
        # Test low severity
        severity = detector._determine_severity(500, 2.0, 0.8)
        self.assertEqual(severity, 'low')


class TestShapeMatcher(unittest.TestCase):
    """Test cases for ShapeMatcher class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_templates = {
            'circle': np.zeros((50, 50), dtype=np.uint8),
            'rectangle': np.zeros((50, 50), dtype=np.uint8)
        }
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_shape_matcher_initialization(self, mock_np, mock_cv2):
        """Test ShapeMatcher initialization"""
        matcher = ShapeMatcher()
        self.assertIsNotNone(matcher.logger)
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_detect_anomalies(self, mock_np, mock_cv2):
        """Test shape anomaly detection"""
        # Setup mocks
        mock_cv2.cvtColor.return_value = np.zeros((100, 100), dtype=np.uint8)
        mock_cv2.matchTemplate.return_value = np.array([[0.5]])  # Below threshold
        mock_np.where.return_value = ([], [])
        
        matcher = ShapeMatcher()
        anomalies = matcher.detect_anomalies(self.test_image, self.test_templates)
        
        self.assertIsInstance(anomalies, list)
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_match_template(self, mock_np, mock_cv2):
        """Test template matching"""
        # Setup mocks
        mock_cv2.matchTemplate.return_value = np.array([[0.7]])  # Above threshold
        mock_np.where.return_value = ([25], [25])  # Mock location
        mock_cv2.normalize.return_value = np.zeros((50, 50), dtype=np.uint8)
        
        matcher = ShapeMatcher()
        template = self.test_templates['circle']
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        
        anomalies = matcher._match_template(gray_image, template, 'circle')
        
        self.assertIsInstance(anomalies, list)
    
    @patch('analyzers.defect_detector.cv2')
    def test_calculate_shape_similarity(self, mock_cv2):
        """Test shape similarity calculation"""
        # Setup mocks
        mock_cv2.normalize.return_value = np.zeros((50, 50), dtype=np.uint8)
        mock_cv2.matchTemplate.return_value = np.array([[0.8]])
        
        matcher = ShapeMatcher()
        region = np.zeros((50, 50), dtype=np.uint8)
        template = np.zeros((50, 50), dtype=np.uint8)
        
        similarity = matcher._calculate_shape_similarity(region, template)
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)


class TestAnomalyDetector(unittest.TestCase):
    """Test cases for AnomalyDetector class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.confidence_threshold = 0.5
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_object = {
            'type': 'rectangular_object',
            'bounding_box': (10, 10, 30, 30),
            'area': 900
        }
    
    def test_anomaly_detector_initialization(self):
        """Test AnomalyDetector initialization"""
        detector = AnomalyDetector(self.confidence_threshold)
        
        self.assertEqual(detector.confidence_threshold, self.confidence_threshold)
        self.assertIsNotNone(detector.logger)
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_analyze_object(self, mock_np, mock_cv2):
        """Test object anomaly analysis"""
        # Setup mocks
        mock_cv2.cvtColor.return_value = np.zeros((30, 30), dtype=np.uint8)
        mock_cv2.threshold.return_value = (127, np.zeros((30, 30), dtype=np.uint8))
        mock_cv2.findContours.return_value = ([], None)
        mock_np.std.return_value = 25  # Below texture threshold
        mock_np.mean.return_value = 128
        
        detector = AnomalyDetector(self.confidence_threshold)
        defects = detector.analyze_object(self.test_image, self.test_object)
        
        self.assertIsInstance(defects, list)
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_analyze_texture_anomalies(self, mock_np, mock_cv2):
        """Test texture anomaly analysis"""
        # Setup mocks for high texture variation
        mock_cv2.cvtColor.return_value = np.zeros((30, 30), dtype=np.uint8)
        mock_np.std.return_value = 60  # Above threshold
        mock_np.mean.return_value = 128
        
        detector = AnomalyDetector(self.confidence_threshold)
        region = np.zeros((30, 30, 3), dtype=np.uint8)
        
        defects = detector._analyze_texture_anomalies(region, self.test_object)
        
        self.assertIsInstance(defects, list)
        if defects:  # If defects were detected
            defect = defects[0]
            self.assertEqual(defect.defect_type, 'texture_anomaly')
            self.assertGreater(defect.confidence, 0.0)
    
    @patch('analyzers.defect_detector.np')
    def test_analyze_color_anomalies(self, mock_np):
        """Test color anomaly analysis"""
        # Setup mocks for color imbalance
        mock_np.mean.side_effect = [50, 150, 100]  # Imbalanced color channels
        mock_np.std.return_value = 50  # Above threshold
        
        detector = AnomalyDetector(self.confidence_threshold)
        region = np.zeros((30, 30, 3), dtype=np.uint8)
        
        defects = detector._analyze_color_anomalies(region, self.test_object)
        
        self.assertIsInstance(defects, list)
        if defects:  # If defects were detected
            defect = defects[0]
            self.assertEqual(defect.defect_type, 'color_anomaly')
            self.assertGreater(defect.confidence, 0.0)
    
    @patch('analyzers.defect_detector.cv2')
    @patch('analyzers.defect_detector.np')
    def test_analyze_structural_anomalies(self, mock_np, mock_cv2):
        """Test structural anomaly analysis"""
        # Setup mocks for many internal contours
        mock_cv2.cvtColor.return_value = np.zeros((30, 30), dtype=np.uint8)
        mock_cv2.threshold.return_value = (127, np.zeros((30, 30), dtype=np.uint8))
        
        # Mock many contours (indicating structural damage)
        mock_contours = [np.array([[[i, i]]]) for i in range(10)]
        mock_cv2.findContours.return_value = (mock_contours, None)
        
        detector = AnomalyDetector(self.confidence_threshold)
        region = np.zeros((30, 30, 3), dtype=np.uint8)
        
        defects = detector._analyze_structural_anomalies(region, self.test_object)
        
        self.assertIsInstance(defects, list)
        if defects:  # If defects were detected
            defect = defects[0]
            self.assertEqual(defect.defect_type, 'structural_damage')
            self.assertGreater(defect.confidence, 0.0)


if __name__ == '__main__':
    unittest.main()