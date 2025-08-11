"""
Compliance Checker Module for Adobe Stock Image Processor

This module implements comprehensive compliance checking including:
- Logo and trademark detection using OCR and template matching
- Face detection system for privacy concern identification
- License plate detection algorithms
- Metadata validation and keyword relevance checking
- Adobe Stock guideline enforcement
"""

try:
    import cv2
    import numpy as np
    from PIL import Image
    from PIL.ExifTags import TAGS
    import pytesseract
except ImportError:
    # Handle missing dependencies gracefully for testing
    cv2 = None
    np = None
    Image = None
    TAGS = None
    pytesseract = None

import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class LogoDetection:
    """Result of logo/trademark detection"""
    text_detected: str
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    detection_method: str  # 'ocr' or 'template'


@dataclass
class PrivacyViolation:
    """Result of privacy element detection"""
    violation_type: str  # 'face', 'license_plate', 'personal_info'
    confidence: float
    bounding_box: Tuple[int, int, int, int]
    description: str


@dataclass
class ComplianceResult:
    """Comprehensive compliance analysis result"""
    logo_detections: List[LogoDetection]
    privacy_violations: List[PrivacyViolation]
    metadata_issues: List[str]
    keyword_relevance: float
    overall_compliance: bool


class ComplianceChecker:
    """
    Comprehensive compliance checker for Adobe Stock guidelines
    
    This class implements various compliance checks including:
    - Logo and trademark detection
    - Privacy element detection (faces, license plates)
    - Metadata validation
    - Keyword relevance checking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ComplianceChecker with configuration
        
        Args:
            config: Configuration dictionary with thresholds and settings
        """
        self.config = config or {}
        self.logo_confidence_threshold = self.config.get('logo_detection_confidence', 0.7)
        self.face_detection_enabled = self.config.get('face_detection_enabled', True)
        self.metadata_validation = self.config.get('metadata_validation', True)
        
        # Initialize face cascade classifier if available
        self.face_cascade = None
        if cv2 is not None and self.face_detection_enabled:
            try:
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            except Exception as e:
                logger.warning(f"Could not load face cascade classifier: {e}")
        
        # Common brand/logo keywords to detect
        self.brand_keywords = [
            'nike', 'adidas', 'apple', 'google', 'microsoft', 'coca-cola', 'pepsi',
            'mcdonalds', 'starbucks', 'amazon', 'facebook', 'twitter', 'instagram',
            'youtube', 'netflix', 'disney', 'marvel', 'dc comics', 'batman', 'superman'
        ]
        
        # License plate patterns (various formats)
        self.license_plate_patterns = [
            r'[A-Z]{2,3}[-\s]?\d{3,4}',  # ABC-123, AB 1234
            r'\d{3}[-\s]?[A-Z]{3}',      # 123-ABC
            r'[A-Z]\d{3}[-\s]?[A-Z]{3}', # A123-BCD
            r'\d{1,3}[-\s]?[A-Z]{1,3}[-\s]?\d{1,4}'  # 12-AB-345
        ]
    
    def check_compliance(self, image_path: str, metadata: Optional[Dict[str, Any]] = None) -> ComplianceResult:
        """
        Perform comprehensive compliance check on an image
        
        Args:
            image_path: Path to the image file
            metadata: Optional metadata dictionary
            
        Returns:
            ComplianceResult with all compliance check results
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load image
            image = self._load_image(image_path)
            if image is None:
                return ComplianceResult(
                    logo_detections=[],
                    privacy_violations=[],
                    metadata_issues=["Could not load image"],
                    keyword_relevance=0.0,
                    overall_compliance=False
                )
            
            # Perform all compliance checks
            logo_detections = self.detect_logos(image)
            privacy_violations = self.check_privacy_elements(image)
            metadata_issues = self._validate_metadata(image_path, metadata)
            keyword_relevance = self._check_keyword_relevance(metadata or {})
            
            # Determine overall compliance
            overall_compliance = (
                len(logo_detections) == 0 and
                len(privacy_violations) == 0 and
                len(metadata_issues) == 0 and
                keyword_relevance >= 0.5
            )
            
            return ComplianceResult(
                logo_detections=logo_detections,
                privacy_violations=privacy_violations,
                metadata_issues=metadata_issues,
                keyword_relevance=keyword_relevance,
                overall_compliance=overall_compliance
            )
            
        except Exception as e:
            logger.error(f"Error in compliance check for {image_path}: {e}")
            return ComplianceResult(
                logo_detections=[],
                privacy_violations=[],
                metadata_issues=[f"Processing error: {str(e)}"],
                keyword_relevance=0.0,
                overall_compliance=False
            )
    
    def detect_logos(self, image: Union[Any, None]) -> List[LogoDetection]:
        """
        Detect logos and trademarks in the image using OCR and template matching
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of LogoDetection objects
        """
        detections = []
        
        try:
            # OCR-based logo detection
            ocr_detections = self._detect_logos_ocr(image)
            detections.extend(ocr_detections)
            
            # Template matching could be added here for specific logos
            # template_detections = self._detect_logos_template(image)
            # detections.extend(template_detections)
            
        except Exception as e:
            logger.error(f"Error in logo detection: {e}")
        
        return detections
    
    def check_privacy_elements(self, image: Union[Any, None]) -> List[PrivacyViolation]:
        """
        Check for privacy-related elements like faces and license plates
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of PrivacyViolation objects
        """
        violations = []
        
        try:
            # Face detection
            if self.face_detection_enabled:
                face_violations = self._detect_faces(image)
                violations.extend(face_violations)
            
            # License plate detection
            plate_violations = self._detect_license_plates(image)
            violations.extend(plate_violations)
            
        except Exception as e:
            logger.error(f"Error in privacy element detection: {e}")
        
        return violations
    
    def _load_image(self, image_path: str) -> Optional[Any]:
        """Load image from file path"""
        try:
            if cv2 is not None:
                image = cv2.imread(image_path)
                if image is not None:
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Fallback to PIL
            if Image is not None:
                pil_image = Image.open(image_path)
                return np.array(pil_image.convert('RGB'))
            
            return None
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def _detect_logos_ocr(self, image: Union[Any, None]) -> List[LogoDetection]:
        """Detect logos using OCR text recognition"""
        detections = []
        
        if pytesseract is None:
            logger.warning("pytesseract not available, skipping OCR logo detection")
            return detections
        
        try:
            # Convert to grayscale for better OCR
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if cv2 is not None else image[:,:,0]
            else:
                gray = image
            
            # Get OCR data with bounding boxes
            ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
            
            # Check each detected text for brand keywords
            for i, text in enumerate(ocr_data['text']):
                if text.strip():
                    text_lower = text.lower().strip()
                    confidence = float(ocr_data['conf'][i]) / 100.0
                    
                    # Check if text matches any brand keywords
                    for brand in self.brand_keywords:
                        if brand in text_lower and confidence >= self.logo_confidence_threshold:
                            x = ocr_data['left'][i]
                            y = ocr_data['top'][i]
                            w = ocr_data['width'][i]
                            h = ocr_data['height'][i]
                            
                            detections.append(LogoDetection(
                                text_detected=text.strip(),
                                confidence=confidence,
                                bounding_box=(x, y, w, h),
                                detection_method='ocr'
                            ))
                            break
            
        except Exception as e:
            logger.error(f"Error in OCR logo detection: {e}")
        
        return detections
    
    def _detect_faces(self, image: Union[Any, None]) -> List[PrivacyViolation]:
        """Detect faces in the image"""
        violations = []
        
        if self.face_cascade is None or cv2 is None:
            return violations
        
        try:
            # Convert to grayscale for face detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Create violations for each detected face
            for (x, y, w, h) in faces:
                violations.append(PrivacyViolation(
                    violation_type='face',
                    confidence=0.8,  # Haar cascades don't provide confidence scores
                    bounding_box=(x, y, w, h),
                    description='Human face detected - may require model release'
                ))
                
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
        
        return violations
    
    def _detect_license_plates(self, image: Union[Any, None]) -> List[PrivacyViolation]:
        """Detect license plates using OCR and pattern matching"""
        violations = []
        
        if pytesseract is None:
            return violations
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if cv2 is not None else image[:,:,0]
            else:
                gray = image
            
            # Extract text from image
            text = pytesseract.image_to_string(gray)
            
            # Check for license plate patterns
            for pattern in self.license_plate_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    violations.append(PrivacyViolation(
                        violation_type='license_plate',
                        confidence=0.7,
                        bounding_box=(0, 0, 0, 0),  # Would need more sophisticated detection for exact location
                        description=f'Potential license plate detected: {match.group()}'
                    ))
            
        except Exception as e:
            logger.error(f"Error in license plate detection: {e}")
        
        return violations
    
    def _validate_metadata(self, image_path: str, metadata: Optional[Dict[str, Any]]) -> List[str]:
        """Validate image metadata for compliance issues"""
        issues = []
        
        if not self.metadata_validation:
            return issues
        
        try:
            # Extract EXIF data if not provided
            if metadata is None:
                metadata = self._extract_exif_data(image_path)
            
            # Check for required metadata fields
            if not metadata.get('description') and not metadata.get('title'):
                issues.append("Missing image description or title")
            
            # Check for inappropriate EXIF data
            if 'GPS' in str(metadata):
                issues.append("GPS location data present - privacy concern")
            
            # Check for camera/device information that might identify photographer
            camera_info = metadata.get('Make') or metadata.get('Model')
            if camera_info and any(brand in str(camera_info).lower() for brand in ['iphone', 'samsung', 'pixel']):
                issues.append("Personal device information in EXIF data")
            
            # Check for software watermarks
            software = metadata.get('Software', '')
            if any(watermark in software.lower() for watermark in ['watermark', 'trial', 'demo']):
                issues.append("Software watermark detected in metadata")
            
        except Exception as e:
            logger.error(f"Error validating metadata: {e}")
            issues.append(f"Metadata validation error: {str(e)}")
        
        return issues
    
    def _extract_exif_data(self, image_path: str) -> Dict[str, Any]:
        """Extract EXIF data from image file"""
        metadata = {}
        
        if Image is None or TAGS is None:
            return metadata
        
        try:
            with Image.open(image_path) as img:
                exif_data = img._getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        metadata[tag] = value
        except Exception as e:
            logger.error(f"Error extracting EXIF data from {image_path}: {e}")
        
        return metadata
    
    def _check_keyword_relevance(self, metadata: Dict[str, Any]) -> float:
        """Check keyword relevance and appropriateness"""
        try:
            # Get keywords from various metadata fields
            keywords = []
            
            # Extract keywords from different fields
            for field in ['keywords', 'tags', 'description', 'title', 'subject']:
                value = metadata.get(field, '')
                if isinstance(value, str):
                    keywords.extend(value.lower().split())
                elif isinstance(value, list):
                    keywords.extend([str(k).lower() for k in value])
            
            if not keywords:
                return 0.0
            
            # Check for inappropriate keywords
            inappropriate_keywords = [
                'nude', 'naked', 'sex', 'porn', 'adult', 'explicit',
                'violence', 'weapon', 'gun', 'knife', 'blood',
                'drug', 'alcohol', 'cigarette', 'smoking',
                'hate', 'racist', 'discrimination'
            ]
            
            inappropriate_count = sum(1 for keyword in keywords if keyword in inappropriate_keywords)
            
            # Check for brand-related keywords
            brand_count = sum(1 for keyword in keywords if keyword in self.brand_keywords)
            
            # Calculate relevance score
            total_keywords = len(keywords)
            if total_keywords == 0:
                return 0.0
            
            # Penalize inappropriate and brand keywords
            penalty = (inappropriate_count + brand_count) / total_keywords
            relevance_score = max(0.0, 1.0 - penalty)
            
            return relevance_score
            
        except Exception as e:
            logger.error(f"Error checking keyword relevance: {e}")
            return 0.0
    
    def get_compliance_summary(self, result: ComplianceResult) -> Dict[str, Any]:
        """Get a summary of compliance check results"""
        return {
            'overall_compliance': result.overall_compliance,
            'logo_count': len(result.logo_detections),
            'privacy_violation_count': len(result.privacy_violations),
            'metadata_issue_count': len(result.metadata_issues),
            'keyword_relevance_score': result.keyword_relevance,
            'main_issues': self._get_main_issues(result)
        }
    
    def _get_main_issues(self, result: ComplianceResult) -> List[str]:
        """Get list of main compliance issues"""
        issues = []
        
        if result.logo_detections:
            issues.append(f"Logo/trademark detected: {', '.join([d.text_detected for d in result.logo_detections])}")
        
        if result.privacy_violations:
            violation_types = [v.violation_type for v in result.privacy_violations]
            issues.append(f"Privacy violations: {', '.join(set(violation_types))}")
        
        if result.metadata_issues:
            issues.extend(result.metadata_issues)
        
        if result.keyword_relevance < 0.5:
            issues.append(f"Low keyword relevance score: {result.keyword_relevance:.2f}")
        
        return issues


# Utility functions for testing and standalone usage
def analyze_image_compliance(image_path: str, config: Optional[Dict[str, Any]] = None) -> ComplianceResult:
    """
    Standalone function to analyze image compliance
    
    Args:
        image_path: Path to the image file
        config: Optional configuration dictionary
        
    Returns:
        ComplianceResult object
    """
    checker = ComplianceChecker(config)
    return checker.check_compliance(image_path)


def batch_compliance_check(image_paths: List[str], config: Optional[Dict[str, Any]] = None) -> Dict[str, ComplianceResult]:
    """
    Perform compliance check on multiple images
    
    Args:
        image_paths: List of image file paths
        config: Optional configuration dictionary
        
    Returns:
        Dictionary mapping image paths to ComplianceResult objects
    """
    checker = ComplianceChecker(config)
    results = {}
    
    for image_path in image_paths:
        try:
            results[image_path] = checker.check_compliance(image_path)
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            results[image_path] = ComplianceResult(
                logo_detections=[],
                privacy_violations=[],
                metadata_issues=[f"Processing error: {str(e)}"],
                keyword_relevance=0.0,
                overall_compliance=False
            )
    
    return results