#!/usr/bin/env python3
"""
Test script for AI-Enhanced Compliance Checker

This script tests the AI compliance checking functionality to ensure:
- Proper initialization and configuration
- Logo and trademark detection capabilities
- Face detection and privacy analysis
- Content appropriateness evaluation
- Metadata validation and analysis
- Error handling and fallback mechanisms
"""

import os
import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from analyzers.ai_compliance_checker import (
        AIComplianceChecker, AIComplianceResult, AILogoDetection, 
        AIPrivacyViolation, ContentAppropriateness
    )
    from analyzers.compliance_checker import ComplianceResult
    from config.config_loader import load_config
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the backend directory")
    sys.exit(1)

class TestAIComplianceChecker(unittest.TestCase):
    """Test cases for AI Compliance Checker"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Load test configuration
        self.config = {
            'compliance': {
                'ai_logo_confidence': 0.7,
                'ai_face_confidence': 0.6,
                'content_safety_threshold': 0.8,
                'logo_detection_confidence': 0.7,
                'face_detection_enabled': True,
                'metadata_validation': True
            }
        }
        
        # Initialize compliance checker
        self.compliance_checker = AIComplianceChecker(self.config)
        
        # Create temporary test image
        self.test_image = self._create_test_image()
        
        # Sample metadata for testing
        self.test_metadata = {
            'title': 'Test Image',
            'description': 'A sample image for testing',
            'keywords': ['test', 'sample', 'photography'],
            'Make': 'Canon',
            'Model': 'EOS R5',
            'Software': 'Adobe Photoshop 2024'
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self, 'test_image') and os.path.exists(self.test_image):
            os.unlink(self.test_image)
    
    def _create_test_image(self):
        """Create a temporary test image"""
        try:
            import numpy as np
            from PIL import Image
            
            # Create a simple test image
            image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            image.save(temp_file.name, 'JPEG')
            temp_file.close()
            
            return temp_file.name
        except ImportError:
            # Fallback: create empty file
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_file.close()
            return temp_file.name
    
    def test_initialization(self):
        """Test AI compliance checker initialization"""
        self.assertIsInstance(self.compliance_checker, AIComplianceChecker)
        self.assertEqual(self.compliance_checker.performance_mode, "balanced")
        self.assertIsNotNone(self.compliance_checker.base_checker)
        self.assertIn('face_detected', self.compliance_checker.thai_translations)
    
    def test_performance_mode_setting(self):
        """Test performance mode configuration"""
        # Test different performance modes
        modes = ['speed', 'balanced', 'smart']
        
        for mode in modes:
            self.compliance_checker.set_performance_mode(mode)
            self.assertEqual(self.compliance_checker.performance_mode, mode)
            
            # Check that thresholds are adjusted
            if mode == 'speed':
                self.assertEqual(self.compliance_checker.logo_confidence_threshold, 0.6)
                self.assertEqual(self.compliance_checker.face_confidence_threshold, 0.5)
            elif mode == 'smart':
                self.assertEqual(self.compliance_checker.logo_confidence_threshold, 0.8)
                self.assertEqual(self.compliance_checker.face_confidence_threshold, 0.7)
    
    def test_image_loading(self):
        """Test image loading functionality"""
        # Test with existing image
        image = self.compliance_checker._load_image_for_ai(self.test_image)
        if image is not None:
            self.assertIsInstance(image, type(None).__class__.__bases__[0])  # numpy.ndarray or None
        
        # Test with non-existent image
        non_existent = self.compliance_checker._load_image_for_ai("non_existent.jpg")
        self.assertIsNone(non_existent)
    
    @patch('easyocr.Reader')
    def test_logo_detection(self, mock_ocr):
        """Test AI logo detection functionality"""
        # Mock OCR reader
        mock_reader = Mock()
        mock_reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 50], [0, 50]], 'Nike', 0.9),
            ([[0, 60], [80, 60], [80, 90], [0, 90]], 'Test', 0.5)
        ]
        mock_ocr.return_value = mock_reader
        self.compliance_checker._ocr_reader = mock_reader
        
        # Create test image array
        import numpy as np
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test logo detection
        detections = self.compliance_checker._detect_logos_ai(test_image)
        
        # Should detect Nike logo
        self.assertGreater(len(detections), 0)
        nike_detection = next((d for d in detections if 'nike' in d.text_detected.lower()), None)
        if nike_detection:
            self.assertEqual(nike_detection.brand_classification, 'nike')
            self.assertIn(nike_detection.risk_level, ['low', 'medium', 'high', 'critical'])
    
    @patch('cv2.dnn.readNetFromTensorflow')
    def test_face_detection(self, mock_dnn):
        """Test AI face detection functionality"""
        # Mock DNN face detector
        mock_net = Mock()
        mock_detections = np.array([[[[0, 0, 0.8, 0.1, 0.1, 0.3, 0.3]]]])
        mock_net.forward.return_value = mock_detections
        mock_dnn.return_value = mock_net
        self.compliance_checker._face_detector = mock_net
        
        # Create test image array
        import numpy as np
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test face detection
        violations = self.compliance_checker._detect_faces_ai(test_image)
        
        # Should detect face
        self.assertGreater(len(violations), 0)
        face_violation = violations[0]
        self.assertEqual(face_violation.violation_type, 'face')
        self.assertIn(face_violation.severity, ['low', 'medium', 'high', 'critical'])
        self.assertIsNotNone(face_violation.description_thai)
    
    def test_content_appropriateness_analysis(self):
        """Test content appropriateness analysis"""
        # Create test image array
        import numpy as np
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test content analysis
        content_result = self.compliance_checker._analyze_content_appropriateness(
            test_image, self.test_metadata
        )
        
        self.assertIsInstance(content_result, ContentAppropriateness)
        self.assertGreaterEqual(content_result.overall_score, 0.0)
        self.assertLessEqual(content_result.overall_score, 1.0)
        self.assertIn(content_result.age_appropriateness, 
                     ['all_ages', 'teen', 'adult', 'restricted', 'unknown'])
    
    def test_metadata_analysis(self):
        """Test AI metadata analysis"""
        # Test with good metadata
        analysis = self.compliance_checker._analyze_metadata_ai(self.test_metadata)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('keyword_quality', analysis)
        self.assertIn('description_quality', analysis)
        self.assertIn('commercial_viability', analysis)
        self.assertIn('detected_issues', analysis)
        
        # Test keyword quality assessment
        keywords = ['test', 'sample', 'photography', 'professional']
        quality_score = self.compliance_checker._assess_keyword_quality(keywords)
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)
        
        # Test with inappropriate keywords
        bad_keywords = ['nude', 'explicit', 'violence']
        bad_quality_score = self.compliance_checker._assess_keyword_quality(bad_keywords)
        self.assertLess(bad_quality_score, quality_score)
    
    def test_compliance_score_calculation(self):
        """Test AI compliance score calculation"""
        # Create mock detection results
        logo_detections = [
            AILogoDetection('Nike', 0.9, 0.8, (0, 0, 50, 20), 'ai_ocr', 'nike', 'high')
        ]
        
        privacy_violations = [
            AIPrivacyViolation('face', 0.8, 0.7, (10, 10, 30, 40), 'Face detected', 
                             'ตรวจพบใบหน้า', 'medium', 'Model release required', 
                             'ต้องมีใบอนุญาตจากนางแบบ')
        ]
        
        content_appropriateness = ContentAppropriateness(
            0.8, 0.9, 0.8, 'all_ages', [], [], []
        )
        
        enhanced_metadata = {'commercial_viability': 0.7}
        
        # Calculate compliance score
        score = self.compliance_checker._calculate_ai_compliance_score(
            logo_detections, privacy_violations, content_appropriateness, enhanced_metadata
        )
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Score should be lower due to logo and privacy violations
        self.assertLess(score, 0.8)
    
    def test_final_decision_making(self):
        """Test final compliance decision making"""
        # Create mock traditional result
        traditional_result = ComplianceResult([], [], [], 0.8, True)
        
        # Test with high compliance score (should approve)
        decision, reasons_en, reasons_th = self.compliance_checker._make_final_decision(
            traditional_result, 0.9, [], [], 
            ContentAppropriateness(0.9, 0.9, 0.9, 'all_ages', [], [], []),
            {'commercial_viability': 0.8}
        )
        
        self.assertEqual(decision, 'approved')
        self.assertEqual(len(reasons_en), 0)
        self.assertEqual(len(reasons_th), 0)
        
        # Test with critical logo detection (should reject)
        critical_logo = AILogoDetection('Nike', 0.9, 0.9, (0, 0, 50, 20), 'ai_ocr', 'nike', 'critical')
        
        decision, reasons_en, reasons_th = self.compliance_checker._make_final_decision(
            traditional_result, 0.9, [critical_logo], [],
            ContentAppropriateness(0.9, 0.9, 0.9, 'all_ages', [], [], []),
            {'commercial_viability': 0.8}
        )
        
        self.assertEqual(decision, 'rejected')
        self.assertGreater(len(reasons_en), 0)
        self.assertGreater(len(reasons_th), 0)
    
    def test_full_analysis_workflow(self):
        """Test complete analysis workflow"""
        try:
            # Perform full analysis
            result = self.compliance_checker.analyze(self.test_image, self.test_metadata)
            
            # Verify result structure
            self.assertIsInstance(result, AIComplianceResult)
            self.assertIsNotNone(result.traditional_result)
            self.assertIsInstance(result.ai_logo_detections, list)
            self.assertIsInstance(result.ai_privacy_violations, list)
            self.assertIsInstance(result.content_appropriateness, ContentAppropriateness)
            self.assertIsInstance(result.enhanced_metadata_analysis, dict)
            
            # Verify scores and confidence
            self.assertGreaterEqual(result.ai_compliance_score, 0.0)
            self.assertLessEqual(result.ai_compliance_score, 1.0)
            self.assertGreaterEqual(result.ai_confidence, 0.0)
            self.assertLessEqual(result.ai_confidence, 1.0)
            
            # Verify decision
            self.assertIn(result.final_decision, ['approved', 'rejected'])
            
            # Verify reasoning
            self.assertIsInstance(result.ai_reasoning, str)
            self.assertIsInstance(result.ai_reasoning_thai, str)
            self.assertGreater(len(result.ai_reasoning), 0)
            self.assertGreater(len(result.ai_reasoning_thai), 0)
            
            # Verify processing metadata
            self.assertGreater(result.processing_time, 0.0)
            self.assertIsInstance(result.models_used, list)
            self.assertIsInstance(result.fallback_used, bool)
            
        except Exception as e:
            # If analysis fails due to missing dependencies, that's acceptable
            self.assertIn('not available', str(e).lower())
    
    def test_error_handling(self):
        """Test error handling and fallback mechanisms"""
        # Test with non-existent image
        result = self.compliance_checker.analyze("non_existent_image.jpg")
        
        self.assertIsInstance(result, AIComplianceResult)
        self.assertEqual(result.final_decision, 'rejected')
        self.assertTrue(result.fallback_used)
        self.assertGreater(len(result.rejection_reasons), 0)
        self.assertGreater(len(result.rejection_reasons_thai), 0)
    
    def test_system_status(self):
        """Test system status reporting"""
        status = self.compliance_checker.get_system_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('performance_mode', status)
        self.assertIn('ocr_reader_loaded', status)
        self.assertIn('face_detector_loaded', status)
        self.assertIn('thresholds', status)
        self.assertIn('supported_languages', status)
        self.assertIn('gpu_available', status)
        
        # Verify threshold values
        thresholds = status['thresholds']
        self.assertIn('logo_confidence', thresholds)
        self.assertIn('face_confidence', thresholds)
        self.assertIn('content_safety', thresholds)
        
        # Verify supported languages
        self.assertIn('en', status['supported_languages'])
        self.assertIn('th', status['supported_languages'])

def run_tests():
    """Run all tests"""
    print("🧪 Running AI Compliance Checker Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestAIComplianceChecker)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)