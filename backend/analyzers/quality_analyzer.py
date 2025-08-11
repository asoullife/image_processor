"""
Quality Analyzer Module for Adobe Stock Image Processor

This module implements comprehensive image quality analysis including:
- Sharpness detection using Laplacian variance method
- Noise level analysis using standard deviation calculations
- Exposure analysis using histogram-based methods
- Color balance checking algorithms
- Resolution and dimension validation functions
"""

try:
    import cv2
    import numpy as np
    from PIL import Image
    from PIL.ExifTags import TAGS
except ImportError:
    # Handle missing dependencies gracefully for testing
    cv2 = None
    np = None
    Image = None
    TAGS = None

import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExposureResult:
    """Result of exposure analysis"""
    brightness_score: float
    contrast_score: float
    histogram_balance: float
    overexposed_pixels: float
    underexposed_pixels: float
    passed: bool


@dataclass
class QualityResult:
    """Comprehensive quality analysis result"""
    sharpness_score: float
    noise_level: float
    exposure_score: float
    color_balance_score: float
    resolution: Tuple[int, int]
    file_size: int
    overall_score: float
    passed: bool
    exposure_result: Optional[ExposureResult] = None


class QualityAnalyzer:
    """
    Comprehensive image quality analyzer for Adobe Stock submissions
    
    Analyzes images for technical quality including sharpness, noise,
    exposure, color balance, and resolution compliance.
    """
    
    def __init__(self, config: Union[Dict[str, Any], Any]):
        """
        Initialize QualityAnalyzer with configuration
        
        Args:
            config: Configuration dictionary or AppConfig object with quality thresholds
        """
        # Handle both dictionary and AppConfig object
        if hasattr(config, 'quality'):
            # AppConfig object
            quality_config = config.quality
            self.min_sharpness = quality_config.min_sharpness
            self.max_noise_level = quality_config.max_noise_level
            self.min_resolution = quality_config.min_resolution
        else:
            # Dictionary config
            self.config = config.get('quality', {})
            self.min_sharpness = self.config.get('min_sharpness', 100.0)
            self.max_noise_level = self.config.get('max_noise_level', 0.1)
            self.min_resolution = tuple(self.config.get('min_resolution', [1920, 1080]))
        
        # Quality scoring weights
        self.weights = {
            'sharpness': 0.3,
            'noise': 0.2,
            'exposure': 0.25,
            'color_balance': 0.15,
            'resolution': 0.1
        }
        
        logger.info(f"QualityAnalyzer initialized with min_sharpness={self.min_sharpness}, "
                   f"max_noise={self.max_noise_level}, min_resolution={self.min_resolution}")
    
    def analyze(self, image_path: str) -> QualityResult:
        """
        Perform comprehensive quality analysis on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            QualityResult with all analysis metrics
        """
        try:
            # Check dependencies
            if cv2 is None or np is None:
                raise ImportError("OpenCV and NumPy are required for image analysis")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Get file size
            file_size = os.path.getsize(image_path)
            
            # Get resolution
            height, width = image.shape[:2]
            resolution = (width, height)
            
            # Perform individual analyses
            sharpness_score = self.check_sharpness(image)
            noise_level = self.detect_noise(image)
            exposure_result = self.analyze_exposure(image)
            color_balance_score = self.check_color_balance(image)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                sharpness_score, noise_level, exposure_result.brightness_score,
                color_balance_score, resolution
            )
            
            # Determine if image passes quality checks
            passed = self._passes_quality_checks(
                sharpness_score, noise_level, exposure_result,
                color_balance_score, resolution
            )
            
            result = QualityResult(
                sharpness_score=sharpness_score,
                noise_level=noise_level,
                exposure_score=exposure_result.brightness_score,
                color_balance_score=color_balance_score,
                resolution=resolution,
                file_size=file_size,
                overall_score=overall_score,
                passed=passed,
                exposure_result=exposure_result
            )
            
            logger.debug(f"Quality analysis complete for {image_path}: "
                        f"score={overall_score:.2f}, passed={passed}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {str(e)}")
            # Return failed result
            return QualityResult(
                sharpness_score=0.0,
                noise_level=1.0,
                exposure_score=0.0,
                color_balance_score=0.0,
                resolution=(0, 0),
                file_size=0,
                overall_score=0.0,
                passed=False
            )
    
    def check_sharpness(self, image: Union[list, Any]) -> float:
        """
        Detect image sharpness using Laplacian variance method
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Sharpness score (higher is sharper)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply Laplacian operator
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            
            # Calculate variance of Laplacian (measure of sharpness)
            sharpness = laplacian.var()
            
            logger.debug(f"Sharpness score: {sharpness:.2f}")
            return float(sharpness)
            
        except Exception as e:
            logger.error(f"Error calculating sharpness: {str(e)}")
            return 0.0
    
    def detect_noise(self, image: Union[list, Any]) -> float:
        """
        Analyze noise level using standard deviation calculations
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Noise level (0.0 to 1.0, lower is better)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply Gaussian blur to get smooth version
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate difference (noise)
            noise = cv2.absdiff(gray, blurred)
            
            # Calculate noise level as normalized standard deviation
            noise_level = np.std(noise) / 255.0
            
            logger.debug(f"Noise level: {noise_level:.4f}")
            return float(noise_level)
            
        except Exception as e:
            logger.error(f"Error calculating noise level: {str(e)}")
            return 1.0
    
    def analyze_exposure(self, image: Union[list, Any]) -> ExposureResult:
        """
        Analyze exposure using histogram-based methods
        
        Args:
            image: Input image as numpy array
            
        Returns:
            ExposureResult with detailed exposure analysis
        """
        try:
            # Convert to grayscale for luminance analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist.flatten()
            
            # Calculate brightness (mean luminance)
            brightness = np.mean(gray) / 255.0
            
            # Calculate contrast (standard deviation)
            contrast = np.std(gray) / 255.0
            
            # Calculate histogram balance (distribution across tonal range)
            # Good images should have reasonable distribution
            total_pixels = gray.shape[0] * gray.shape[1]
            
            # Check for overexposure (pixels > 240)
            overexposed = np.sum(gray > 240) / total_pixels
            
            # Check for underexposure (pixels < 15)
            underexposed = np.sum(gray < 15) / total_pixels
            
            # Calculate histogram balance score
            # Divide histogram into 5 regions and check distribution
            regions = np.array_split(hist, 5)
            region_sums = [np.sum(region) for region in regions]
            region_percentages = np.array(region_sums) / total_pixels
            
            # Good balance means no region is completely empty and no region dominates
            histogram_balance = 1.0 - np.std(region_percentages)
            
            # Calculate overall exposure score
            # Penalize extreme brightness, low contrast, and poor balance
            brightness_penalty = abs(brightness - 0.5) * 2  # Optimal around 0.5
            contrast_penalty = max(0, 0.2 - contrast) * 5   # Minimum contrast needed
            overexposure_penalty = max(0, overexposed - 0.02) * 10  # Max 2% overexposed
            underexposure_penalty = max(0, underexposed - 0.02) * 10  # Max 2% underexposed
            
            exposure_score = max(0, 1.0 - brightness_penalty - contrast_penalty - 
                               overexposure_penalty - underexposure_penalty)
            
            # Determine if exposure passes
            passed = (exposure_score > 0.6 and overexposed < 0.05 and 
                     underexposed < 0.05 and contrast > 0.1)
            
            result = ExposureResult(
                brightness_score=brightness,
                contrast_score=contrast,
                histogram_balance=histogram_balance,
                overexposed_pixels=overexposed,
                underexposed_pixels=underexposed,
                passed=passed
            )
            
            logger.debug(f"Exposure analysis: brightness={brightness:.3f}, "
                        f"contrast={contrast:.3f}, score={exposure_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing exposure: {str(e)}")
            return ExposureResult(0.0, 0.0, 0.0, 1.0, 1.0, False)
    
    def check_color_balance(self, image: Union[list, Any]) -> float:
        """
        Check color balance of the image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Color balance score (0.0 to 1.0, higher is better)
        """
        try:
            if len(image.shape) != 3:
                # Grayscale image, return neutral score
                return 0.8
            
            # Calculate mean values for each channel
            b_mean = np.mean(image[:, :, 0])
            g_mean = np.mean(image[:, :, 1])
            r_mean = np.mean(image[:, :, 2])
            
            # Calculate color balance
            # Good color balance means channels are reasonably close
            channel_means = np.array([b_mean, g_mean, r_mean])
            overall_mean = np.mean(channel_means)
            
            if overall_mean == 0:
                return 0.0
            
            # Calculate deviation from neutral
            deviations = np.abs(channel_means - overall_mean) / overall_mean
            max_deviation = np.max(deviations)
            
            # Convert to score (lower deviation = higher score)
            color_balance_score = max(0, 1.0 - max_deviation * 2)
            
            # Additional check for color cast
            # Calculate ratios between channels
            if overall_mean > 10:  # Avoid division by very small numbers
                rg_ratio = r_mean / g_mean if g_mean > 0 else 1.0
                bg_ratio = b_mean / g_mean if g_mean > 0 else 1.0
                
                # Healthy ratios should be close to 1.0
                ratio_penalty = (abs(rg_ratio - 1.0) + abs(bg_ratio - 1.0)) * 0.5
                color_balance_score = max(0, color_balance_score - ratio_penalty)
            
            logger.debug(f"Color balance score: {color_balance_score:.3f} "
                        f"(R:{r_mean:.1f}, G:{g_mean:.1f}, B:{b_mean:.1f})")
            
            return float(color_balance_score)
            
        except Exception as e:
            logger.error(f"Error checking color balance: {str(e)}")
            return 0.0
    
    def _calculate_overall_score(self, sharpness: float, noise: float, 
                               exposure: float, color_balance: float,
                               resolution: Tuple[int, int]) -> float:
        """Calculate weighted overall quality score"""
        try:
            # Normalize sharpness (typical range 0-500, good > 100)
            sharpness_norm = min(1.0, sharpness / 200.0)
            
            # Normalize noise (0-1, lower is better, so invert)
            noise_norm = max(0, 1.0 - noise * 10)
            
            # Exposure is already 0-1
            exposure_norm = exposure
            
            # Color balance is already 0-1
            color_balance_norm = color_balance
            
            # Resolution score
            width, height = resolution
            min_width, min_height = self.min_resolution
            resolution_norm = min(1.0, (width * height) / (min_width * min_height))
            
            # Calculate weighted score
            overall_score = (
                self.weights['sharpness'] * sharpness_norm +
                self.weights['noise'] * noise_norm +
                self.weights['exposure'] * exposure_norm +
                self.weights['color_balance'] * color_balance_norm +
                self.weights['resolution'] * resolution_norm
            )
            
            return float(overall_score)
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {str(e)}")
            return 0.0
    
    def _passes_quality_checks(self, sharpness: float, noise: float,
                             exposure_result: ExposureResult, color_balance: float,
                             resolution: Tuple[int, int]) -> bool:
        """Determine if image passes all quality checks"""
        try:
            # Check individual thresholds
            sharpness_pass = sharpness >= self.min_sharpness
            noise_pass = noise <= self.max_noise_level
            exposure_pass = exposure_result.passed
            color_balance_pass = color_balance >= 0.5
            
            # Check resolution
            width, height = resolution
            min_width, min_height = self.min_resolution
            resolution_pass = width >= min_width and height >= min_height
            
            # All checks must pass
            overall_pass = (sharpness_pass and noise_pass and exposure_pass and
                          color_balance_pass and resolution_pass)
            
            logger.debug(f"Quality checks: sharpness={sharpness_pass}, "
                        f"noise={noise_pass}, exposure={exposure_pass}, "
                        f"color={color_balance_pass}, resolution={resolution_pass}, "
                        f"overall={overall_pass}")
            
            return overall_pass
            
        except Exception as e:
            logger.error(f"Error in quality checks: {str(e)}")
            return False