"""
Unit tests for QualityAnalyzer module

Tests all quality analysis functionality including sharpness detection,
noise analysis, exposure analysis, color balance, and overall scoring.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
import numpy as np
import cv2
import os
import tempfile
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.analyzers.quality_analyzer import QualityAnalyzer, QualityResult, ExposureResult


class TestQualityAnalyzer(unittest.TestCase):
    """Test cases for QualityAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'quality': {
                'min_sharpness': 100.0,
                'max_noise_level': 0.1,
                'min_resolution': [1920, 1080]
            }
        }
        self.analyzer = QualityAnalyzer(self.config)
        
        # Create temporary directory for test images
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_image(self, width=1920, height=1080, noise_level=0, blur_level=0):
        """Create a test image with specified characteristics"""
        # Create base image with gradient
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient pattern
        for i in range(height):
            for j in range(width):
                image[i, j] = [
                    int(255 * j / width),  # Blue gradient
                    int(255 * i / height),  # Green gradient
                    int(255 * (i + j) / (width + height))  # Red gradient
                ]
        
        # Add noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 255, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Add blur if specified
        if blur_level > 0:
            image = cv2.GaussianBlur(image, (blur_level * 2 + 1, blur_level * 2 + 1), 0)
        
        return image
    
    def save_test_image(self, image, filename):
        """Save test image to temporary directory"""
        filepath = os.path.join(self.temp_dir, filename)
        cv2.imwrite(filepath, image)
        return filepath
    
    def test_initialization(self):
        """Test QualityAnalyzer initialization"""
        self.assertEqual(self.analyzer.min_sharpness, 100.0)
        self.assertEqual(self.analyzer.max_noise_level, 0.1)
        self.assertEqual(self.analyzer.min_resolution, (1920, 1080))
        
        # Test with default config
        analyzer_default = QualityAnalyzer({})
        self.assertEqual(analyzer_default.min_sharpness, 100.0)
        self.assertEqual(analyzer_default.max_noise_level, 0.1)
        self.assertEqual(analyzer_default.min_resolution, (1920, 1080))
    
    def test_check_sharpness_sharp_image(self):
        """Test sharpness detection with sharp image"""
        # Create sharp image with high contrast edges
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :50] = 255  # Half white, half black
        
        sharpness = self.analyzer.check_sharpness(image)
        self.assertGreater(sharpness, 1000)  # Sharp edge should give high score
    
    def test_check_sharpness_blurry_image(self):
        """Test sharpness detection with blurry image"""
        # Create blurry image
        image = self.create_test_image(blur_level=10)
        
        sharpness = self.analyzer.check_sharpness(image)
        self.assertLess(sharpness, 100)  # Blurry image should give low score
    
    def test_check_sharpness_grayscale(self):
        """Test sharpness detection with grayscale image"""
        # Create grayscale image
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        gray_image[:, :50] = 255
        
        sharpness = self.analyzer.check_sharpness(gray_image)
        self.assertGreater(sharpness, 1000)
    
    def test_detect_noise_clean_image(self):
        """Test noise detection with clean image"""
        image = self.create_test_image(noise_level=0)
        
        noise_level = self.analyzer.detect_noise(image)
        self.assertLess(noise_level, 0.05)  # Clean image should have low noise
    
    def test_detect_noise_noisy_image(self):
        """Test noise detection with noisy image"""
        image = self.create_test_image(noise_level=0.3)
        
        noise_level = self.analyzer.detect_noise(image)
        self.assertGreater(noise_level, 0.1)  # Noisy image should have high noise
    
    def test_analyze_exposure_normal_image(self):
        """Test exposure analysis with normal exposure"""
        # Create image with normal exposure (mid-gray)
        image = np.full((100, 100, 3), 128, dtype=np.uint8)
        
        exposure_result = self.analyzer.analyze_exposure(image)
        
        self.assertIsInstance(exposure_result, ExposureResult)
        self.assertAlmostEqual(exposure_result.brightness_score, 0.5, delta=0.1)
        self.assertLess(exposure_result.overexposed_pixels, 0.01)
        self.assertLess(exposure_result.underexposed_pixels, 0.01)
    
    def test_analyze_exposure_overexposed_image(self):
        """Test exposure analysis with overexposed image"""
        # Create overexposed image (mostly white)
        image = np.full((100, 100, 3), 250, dtype=np.uint8)
        
        exposure_result = self.analyzer.analyze_exposure(image)
        
        self.assertGreater(exposure_result.brightness_score, 0.9)
        self.assertGreater(exposure_result.overexposed_pixels, 0.5)
        self.assertFalse(exposure_result.passed)
    
    def test_analyze_exposure_underexposed_image(self):
        """Test exposure analysis with underexposed image"""
        # Create underexposed image (mostly black)
        image = np.full((100, 100, 3), 10, dtype=np.uint8)
        
        exposure_result = self.analyzer.analyze_exposure(image)
        
        self.assertLess(exposure_result.brightness_score, 0.1)
        self.assertGreater(exposure_result.underexposed_pixels, 0.5)
        self.assertFalse(exposure_result.passed)
    
    def test_check_color_balance_neutral_image(self):
        """Test color balance with neutral image"""
        # Create neutral gray image
        image = np.full((100, 100, 3), [128, 128, 128], dtype=np.uint8)
        
        color_balance = self.analyzer.check_color_balance(image)
        self.assertGreater(color_balance, 0.8)  # Neutral should have good balance
    
    def test_check_color_balance_color_cast(self):
        """Test color balance with color cast"""
        # Create image with blue color cast
        image = np.full((100, 100, 3), [200, 100, 100], dtype=np.uint8)
        
        color_balance = self.analyzer.check_color_balance(image)
        self.assertLess(color_balance, 0.7)  # Color cast should reduce score
    
    def test_check_color_balance_grayscale(self):
        """Test color balance with grayscale image"""
        # Create grayscale image
        gray_image = np.full((100, 100), 128, dtype=np.uint8)
        
        color_balance = self.analyzer.check_color_balance(gray_image)
        self.assertEqual(color_balance, 0.8)  # Grayscale should return neutral score
    
    def test_analyze_high_quality_image(self):
        """Test complete analysis with high quality image"""
        # Create high quality image
        image = self.create_test_image(width=2000, height=1200, noise_level=0, blur_level=0)
        filepath = self.save_test_image(image, 'high_quality.jpg')
        
        result = self.analyzer.analyze(filepath)
        
        self.assertIsInstance(result, QualityResult)
        self.assertGreater(result.sharpness_score, 50)
        self.assertLess(result.noise_level, 0.1)
        self.assertGreater(result.overall_score, 0.6)
        self.assertEqual(result.resolution, (2000, 1200))
        self.assertGreater(result.file_size, 0)
        self.assertTrue(result.passed)
    
    def test_analyze_low_quality_image(self):
        """Test complete analysis with low quality image"""
        # Create low quality image (small, noisy, blurry)
        image = self.create_test_image(width=800, height=600, noise_level=0.3, blur_level=5)
        filepath = self.save_test_image(image, 'low_quality.jpg')
        
        result = self.analyzer.analyze(filepath)
        
        self.assertIsInstance(result, QualityResult)
        self.assertLess(result.sharpness_score, 100)
        self.assertGreater(result.noise_level, 0.1)
        self.assertLess(result.overall_score, 0.5)
        self.assertEqual(result.resolution, (800, 600))
        self.assertFalse(result.passed)
    
    def test_analyze_nonexistent_file(self):
        """Test analysis with nonexistent file"""
        result = self.analyzer.analyze('nonexistent.jpg')
        
        self.assertIsInstance(result, QualityResult)
        self.assertEqual(result.sharpness_score, 0.0)
        self.assertEqual(result.noise_level, 1.0)
        self.assertEqual(result.overall_score, 0.0)
        self.assertFalse(result.passed)
    
    def test_calculate_overall_score(self):
        """Test overall score calculation"""
        # Test with good values
        score = self.analyzer._calculate_overall_score(
            sharpness=200.0,
            noise=0.05,
            exposure=0.8,
            color_balance=0.9,
            resolution=(2000, 1200)
        )
        self.assertGreater(score, 0.8)
        
        # Test with poor values
        score = self.analyzer._calculate_overall_score(
            sharpness=50.0,
            noise=0.2,
            exposure=0.3,
            color_balance=0.4,
            resolution=(800, 600)
        )
        self.assertLess(score, 0.5)
    
    def test_passes_quality_checks(self):
        """Test quality check pass/fail logic"""
        # Create good exposure result
        good_exposure = ExposureResult(0.5, 0.3, 0.8, 0.01, 0.01, True)
        
        # Test passing case
        passes = self.analyzer._passes_quality_checks(
            sharpness=150.0,
            noise=0.08,
            exposure_result=good_exposure,
            color_balance=0.7,
            resolution=(2000, 1200)
        )
        self.assertTrue(passes)
        
        # Test failing case (low sharpness)
        passes = self.analyzer._passes_quality_checks(
            sharpness=50.0,
            noise=0.08,
            exposure_result=good_exposure,
            color_balance=0.7,
            resolution=(2000, 1200)
        )
        self.assertFalse(passes)
        
        # Test failing case (high noise)
        passes = self.analyzer._passes_quality_checks(
            sharpness=150.0,
            noise=0.15,
            exposure_result=good_exposure,
            color_balance=0.7,
            resolution=(2000, 1200)
        )
        self.assertFalse(passes)
        
        # Test failing case (low resolution)
        passes = self.analyzer._passes_quality_checks(
            sharpness=150.0,
            noise=0.08,
            exposure_result=good_exposure,
            color_balance=0.7,
            resolution=(800, 600)
        )
        self.assertFalse(passes)
    
    def test_error_handling(self):
        """Test error handling in various methods"""
        # Test with invalid image data
        invalid_image = np.array([])
        
        # These should not raise exceptions
        sharpness = self.analyzer.check_sharpness(invalid_image)
        self.assertEqual(sharpness, 0.0)
        
        noise = self.analyzer.detect_noise(invalid_image)
        self.assertEqual(noise, 1.0)
        
        exposure = self.analyzer.analyze_exposure(invalid_image)
        self.assertFalse(exposure.passed)
        
        color_balance = self.analyzer.check_color_balance(invalid_image)
        self.assertEqual(color_balance, 0.0)


class TestQualityAnalyzerIntegration(unittest.TestCase):
    """Integration tests with real image scenarios"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.config = {
            'quality': {
                'min_sharpness': 100.0,
                'max_noise_level': 0.1,
                'min_resolution': [1920, 1080]
            }
        }
        self.analyzer = QualityAnalyzer(self.config)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_realistic_test_image(self, quality='high'):
        """Create realistic test images for integration testing"""
        if quality == 'high':
            # High quality: sharp, low noise, good exposure, balanced colors
            image = np.random.randint(50, 200, (1200, 1920, 3), dtype=np.uint8)
            # Add some structure (not just noise)
            for i in range(0, 1200, 100):
                for j in range(0, 1920, 100):
                    cv2.rectangle(image, (j, i), (j+50, i+50), (255, 255, 255), -1)
        
        elif quality == 'medium':
            # Medium quality: moderate sharpness, some noise
            image = np.random.randint(30, 220, (1080, 1920, 3), dtype=np.uint8)
            image = cv2.GaussianBlur(image, (3, 3), 0)
        
        else:  # low quality
            # Low quality: blurry, noisy, poor exposure
            image = np.random.randint(0, 100, (600, 800, 3), dtype=np.uint8)
            image = cv2.GaussianBlur(image, (15, 15), 0)
            # Add heavy noise
            noise = np.random.normal(0, 50, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return image
    
    def test_realistic_high_quality_image(self):
        """Test with realistic high quality image"""
        image = self.create_realistic_test_image('high')
        filepath = os.path.join(self.temp_dir, 'realistic_high.jpg')
        cv2.imwrite(filepath, image)
        
        result = self.analyzer.analyze(filepath)
        
        # High quality image should generally pass
        self.assertGreater(result.overall_score, 0.4)  # Reasonable threshold
        self.assertEqual(result.resolution, (1920, 1200))
    
    def test_realistic_medium_quality_image(self):
        """Test with realistic medium quality image"""
        image = self.create_realistic_test_image('medium')
        filepath = os.path.join(self.temp_dir, 'realistic_medium.jpg')
        cv2.imwrite(filepath, image)
        
        result = self.analyzer.analyze(filepath)
        
        # Medium quality should have moderate scores
        self.assertGreater(result.overall_score, 0.2)
        self.assertLess(result.overall_score, 0.8)
    
    def test_realistic_low_quality_image(self):
        """Test with realistic low quality image"""
        image = self.create_realistic_test_image('low')
        filepath = os.path.join(self.temp_dir, 'realistic_low.jpg')
        cv2.imwrite(filepath, image)
        
        result = self.analyzer.analyze(filepath)
        
        # Low quality should fail quality checks
        self.assertLess(result.overall_score, 0.6)
        self.assertFalse(result.passed)  # Should fail due to resolution and quality


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)