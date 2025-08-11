"""
Defect Detection Module for Adobe Stock Image Processor

This module implements comprehensive defect detection using computer vision:
- Object detection using pre-trained YOLO or SSD model
- Anomaly detection algorithms for detected objects
- Edge detection methods for identifying breaks and cracks
- Template matching for shape irregularities
- Confidence scoring for all detections
"""

try:
    import cv2
    import numpy as np
    from PIL import Image
except ImportError:
    # Handle missing dependencies gracefully for testing
    cv2 = None
    np = None
    Image = None

import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Tuple, TYPE_CHECKING
from backend.core.base import BaseAnalyzer, DefectResult, ObjectDefect

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EdgeDefect:
    """Detected edge-based defect (crack, break, etc.)"""
    defect_type: str  # 'crack', 'break', 'scratch'
    confidence: float
    contour_area: float
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    severity: str  # 'low', 'medium', 'high'


@dataclass
class ShapeAnomaly:
    """Detected shape irregularity"""
    object_type: str
    anomaly_type: str  # 'deformation', 'missing_part', 'extra_part'
    confidence: float
    expected_shape: str
    actual_shape_score: float
    bounding_box: Tuple[int, int, int, int]


class DefectDetector(BaseAnalyzer):
    """
    Comprehensive defect detection system using computer vision
    
    Detects various types of defects in images including:
    - Object defects using deep learning models
    - Physical damage through edge detection
    - Shape irregularities through template matching
    """
    
    def __init__(self, config: Union[Dict[str, Any], Any]):
        """
        Initialize DefectDetector with configuration
        
        Args:
            config: Configuration dictionary or AppConfig object
        """
        super().__init__(config if isinstance(config, dict) else {})
        
        # Handle both dictionary and AppConfig object
        if hasattr(config, 'defect_detection'):
            # AppConfig object
            defect_config = config.defect_detection
            self.confidence_threshold = defect_config.confidence_threshold
            self.edge_threshold = defect_config.edge_threshold
            self.model_path = defect_config.model_path
        else:
            # Dictionary config
            defect_config = config.get('defect_detection', {})
            self.confidence_threshold = defect_config.get('confidence_threshold', 0.5)
            self.edge_threshold = defect_config.get('edge_threshold', 50)
            self.model_path = defect_config.get('model_path', None)
        
        # Initialize detection components
        self.object_detector = None
        self.edge_detector = EdgeDetector(self.edge_threshold)
        self.shape_matcher = ShapeMatcher()
        self.anomaly_detector = AnomalyDetector(self.confidence_threshold)
        
        # Common object templates for shape matching
        self.object_templates = self._load_object_templates()
        
        logger.info(f"DefectDetector initialized with confidence_threshold={self.confidence_threshold}")
    
    def analyze(self, image_path: str) -> DefectResult:
        """
        Perform comprehensive defect detection on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            DefectResult with all detected defects
        """
        try:
            # Check dependencies
            if cv2 is None or np is None:
                logger.error("OpenCV and NumPy are required for defect detection")
                return DefectResult(
                    detected_objects=[],
                    anomaly_score=1.0,
                    defect_count=0,
                    defect_types=[],
                    confidence_scores=[],
                    passed=False
                )
            
            # Validate image path
            if not self.validate_image_path(image_path):
                logger.error(f"Invalid image path: {image_path}")
                return DefectResult(
                    detected_objects=[],
                    anomaly_score=1.0,
                    defect_count=0,
                    defect_types=[],
                    confidence_scores=[],
                    passed=False
                )
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return DefectResult(
                    detected_objects=[],
                    anomaly_score=1.0,
                    defect_count=0,
                    defect_types=[],
                    confidence_scores=[],
                    passed=False
                )
            
            # Perform different types of defect detection
            object_defects = self._detect_object_defects(image)
            edge_defects = self._detect_edge_defects(image)
            shape_anomalies = self._detect_shape_anomalies(image)
            
            # Combine all defects
            all_defects = object_defects + self._convert_edge_defects(edge_defects) + \
                         self._convert_shape_anomalies(shape_anomalies)
            
            # Calculate overall scores
            defect_count = len(all_defects)
            confidence_scores = [defect.confidence for defect in all_defects]
            
            # Calculate anomaly score (0.0 = no defects, 1.0 = many high-confidence defects)
            if confidence_scores:
                avg_confidence = np.mean(confidence_scores)
                anomaly_score = min(1.0, (defect_count * avg_confidence) / 10.0)
            else:
                anomaly_score = 0.0
            
            # Extract defect types
            defect_types = list(set([defect.defect_type for defect in all_defects]))
            
            # Determine if image passes defect checks
            passed = self._passes_defect_checks(anomaly_score, defect_count, confidence_scores)
            
            result = DefectResult(
                detected_objects=all_defects,
                anomaly_score=anomaly_score,
                defect_count=defect_count,
                defect_types=defect_types,
                confidence_scores=confidence_scores,
                passed=passed
            )
            
            logger.debug(f"Defect detection complete for {image_path}: "
                        f"defects={defect_count}, anomaly_score={anomaly_score:.3f}, passed={passed}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in defect detection for {image_path}: {str(e)}")
            # Return failed result
            return DefectResult(
                detected_objects=[],
                anomaly_score=1.0,
                defect_count=0,
                defect_types=[],
                confidence_scores=[],
                passed=False
            )
    
    def _detect_object_defects(self, image: Any) -> List[ObjectDefect]:
        """
        Detect object-level defects using computer vision
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected object defects
        """
        defects = []
        
        try:
            # Use YOLO-style object detection (simplified implementation)
            # In a real implementation, this would use a pre-trained model
            objects = self._detect_objects_simple(image)
            
            for obj in objects:
                # Analyze each detected object for defects
                object_defects = self.anomaly_detector.analyze_object(image, obj)
                defects.extend(object_defects)
            
            logger.debug(f"Detected {len(defects)} object defects")
            
        except Exception as e:
            logger.error(f"Error in object defect detection: {str(e)}")
        
        return defects
    
    def _detect_edge_defects(self, image: Any) -> List[EdgeDefect]:
        """
        Detect edge-based defects like cracks and breaks
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected edge defects
        """
        try:
            return self.edge_detector.detect_defects(image)
        except Exception as e:
            logger.error(f"Error in edge defect detection: {str(e)}")
            return []
    
    def _detect_shape_anomalies(self, image: Any) -> List[ShapeAnomaly]:
        """
        Detect shape irregularities using template matching
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected shape anomalies
        """
        try:
            return self.shape_matcher.detect_anomalies(image, self.object_templates)
        except Exception as e:
            logger.error(f"Error in shape anomaly detection: {str(e)}")
            return []
    
    def _detect_objects_simple(self, image: Any) -> List[Dict[str, Any]]:
        """
        Simple object detection using contour analysis
        (In production, this would use YOLO/SSD)
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected objects with bounding boxes
        """
        objects = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size and create object representations
            min_area = 1000  # Minimum object area
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Simple object classification based on shape
                    object_type = self._classify_object_simple(contour, area)
                    
                    objects.append({
                        'type': object_type,
                        'bounding_box': (x, y, w, h),
                        'area': area,
                        'contour': contour
                    })
            
            logger.debug(f"Detected {len(objects)} objects using simple detection")
            
        except Exception as e:
            logger.error(f"Error in simple object detection: {str(e)}")
        
        return objects
    
    def _classify_object_simple(self, contour: Any, area: float) -> str:
        """
        Simple object classification based on contour properties
        
        Args:
            contour: Object contour
            area: Contour area
            
        Returns:
            Object type string
        """
        try:
            # Calculate shape properties
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                return 'unknown'
            
            # Circularity (4π*area/perimeter²)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 1.0
            
            # Simple classification rules
            if circularity > 0.7:
                return 'circular_object'
            elif aspect_ratio > 2.0:
                return 'elongated_object'
            elif aspect_ratio < 0.5:
                return 'tall_object'
            else:
                return 'rectangular_object'
                
        except Exception as e:
            logger.error(f"Error in object classification: {str(e)}")
            return 'unknown'
    
    def _convert_edge_defects(self, edge_defects: List[EdgeDefect]) -> List[ObjectDefect]:
        """Convert EdgeDefect objects to ObjectDefect objects"""
        converted = []
        for edge_defect in edge_defects:
            obj_defect = ObjectDefect(
                object_type='edge_feature',
                defect_type=edge_defect.defect_type,
                confidence=edge_defect.confidence,
                bounding_box=edge_defect.bounding_box,
                description=f"{edge_defect.defect_type} with {edge_defect.severity} severity"
            )
            converted.append(obj_defect)
        return converted
    
    def _convert_shape_anomalies(self, shape_anomalies: List[ShapeAnomaly]) -> List[ObjectDefect]:
        """Convert ShapeAnomaly objects to ObjectDefect objects"""
        converted = []
        for anomaly in shape_anomalies:
            obj_defect = ObjectDefect(
                object_type=anomaly.object_type,
                defect_type=anomaly.anomaly_type,
                confidence=anomaly.confidence,
                bounding_box=anomaly.bounding_box,
                description=f"{anomaly.anomaly_type} in {anomaly.object_type}"
            )
            converted.append(obj_defect)
        return converted
    
    def _passes_defect_checks(self, anomaly_score: float, defect_count: int, 
                            confidence_scores: List[float]) -> bool:
        """
        Determine if image passes defect detection checks
        
        Args:
            anomaly_score: Overall anomaly score
            defect_count: Number of detected defects
            confidence_scores: List of confidence scores for defects
            
        Returns:
            True if image passes, False otherwise
        """
        try:
            # Thresholds for passing
            max_anomaly_score = 0.3
            max_defect_count = 3
            max_high_confidence_defects = 1
            
            # Count high confidence defects
            high_confidence_count = sum(1 for score in confidence_scores 
                                      if score > self.confidence_threshold)
            
            # All conditions must be met to pass
            passes = (
                anomaly_score <= max_anomaly_score and
                defect_count <= max_defect_count and
                high_confidence_count <= max_high_confidence_defects
            )
            
            logger.debug(f"Defect checks: anomaly={anomaly_score:.3f}, "
                        f"count={defect_count}, high_conf={high_confidence_count}, "
                        f"passes={passes}")
            
            return passes
            
        except Exception as e:
            logger.error(f"Error in defect checks: {str(e)}")
            return False
    
    def _load_object_templates(self) -> Dict[str, Any]:
        """
        Load object templates for shape matching
        
        Returns:
            Dictionary of object templates
        """
        templates = {}
        
        try:
            # In a real implementation, these would be loaded from files
            # For now, create simple geometric templates
            
            # Circle template
            circle_template = np.zeros((50, 50), dtype=np.uint8)
            cv2.circle(circle_template, (25, 25), 20, 255, -1)
            templates['circle'] = circle_template
            
            # Rectangle template
            rect_template = np.zeros((50, 50), dtype=np.uint8)
            cv2.rectangle(rect_template, (10, 15), (40, 35), 255, -1)
            templates['rectangle'] = rect_template
            
            # Triangle template
            triangle_template = np.zeros((50, 50), dtype=np.uint8)
            points = np.array([[25, 10], [10, 40], [40, 40]], np.int32)
            cv2.fillPoly(triangle_template, [points], 255)
            templates['triangle'] = triangle_template
            
            logger.debug(f"Loaded {len(templates)} object templates")
            
        except Exception as e:
            logger.error(f"Error loading object templates: {str(e)}")
        
        return templates


class EdgeDetector:
    """Edge-based defect detection for cracks, breaks, and scratches"""
    
    def __init__(self, edge_threshold: int = 50):
        """
        Initialize EdgeDetector
        
        Args:
            edge_threshold: Threshold for edge detection
        """
        self.edge_threshold = edge_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_defects(self, image: Any) -> List[EdgeDefect]:
        """
        Detect edge-based defects in image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected edge defects
        """
        defects = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection using Canny
            edges = cv2.Canny(blurred, self.edge_threshold, self.edge_threshold * 2)
            
            # Find contours of edges
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze each contour for defect characteristics
            for contour in contours:
                defect = self._analyze_edge_contour(contour)
                if defect:
                    defects.append(defect)
            
            self.logger.debug(f"Detected {len(defects)} edge defects")
            
        except Exception as e:
            self.logger.error(f"Error in edge defect detection: {str(e)}")
        
        return defects
    
    def _analyze_edge_contour(self, contour: Any) -> Optional[EdgeDefect]:
        """
        Analyze edge contour to determine if it represents a defect
        
        Args:
            contour: Edge contour to analyze
            
        Returns:
            EdgeDefect if defect detected, None otherwise
        """
        try:
            area = cv2.contourArea(contour)
            
            # Filter out very small contours
            if area < 100:
                return None
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate contour properties
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                return None
            
            # Aspect ratio
            aspect_ratio = float(w) / h if h > 0 else 1.0
            
            # Solidity (area / convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Classify defect type based on properties
            defect_type, confidence = self._classify_edge_defect(
                area, perimeter, aspect_ratio, solidity
            )
            
            if confidence > 0.3:  # Minimum confidence threshold
                # Determine severity based on size and shape
                severity = self._determine_severity(area, aspect_ratio, solidity)
                
                return EdgeDefect(
                    defect_type=defect_type,
                    confidence=confidence,
                    contour_area=area,
                    bounding_box=(x, y, w, h),
                    severity=severity
                )
            
        except Exception as e:
            self.logger.error(f"Error analyzing edge contour: {str(e)}")
        
        return None
    
    def _classify_edge_defect(self, area: float, perimeter: float, 
                            aspect_ratio: float, solidity: float) -> Tuple[str, float]:
        """
        Classify edge defect type and confidence
        
        Args:
            area: Contour area
            perimeter: Contour perimeter
            aspect_ratio: Width/height ratio
            solidity: Area/convex hull area ratio
            
        Returns:
            Tuple of (defect_type, confidence)
        """
        try:
            # Crack detection: long, thin, low solidity
            if aspect_ratio > 3.0 and solidity < 0.7:
                confidence = min(0.9, aspect_ratio / 10.0 + (1.0 - solidity))
                return 'crack', confidence
            
            # Break detection: irregular shape, medium solidity
            elif solidity < 0.8 and aspect_ratio > 1.5:
                confidence = min(0.8, (1.0 - solidity) * 2.0)
                return 'break', confidence
            
            # Scratch detection: elongated, higher solidity than crack
            elif aspect_ratio > 2.0 and solidity > 0.7:
                confidence = min(0.7, aspect_ratio / 5.0)
                return 'scratch', confidence
            
            # General edge anomaly
            else:
                confidence = 0.4
                return 'edge_anomaly', confidence
                
        except Exception as e:
            self.logger.error(f"Error classifying edge defect: {str(e)}")
            return 'unknown', 0.0
    
    def _determine_severity(self, area: float, aspect_ratio: float, solidity: float) -> str:
        """
        Determine defect severity based on properties
        
        Args:
            area: Contour area
            aspect_ratio: Width/height ratio
            solidity: Area/convex hull area ratio
            
        Returns:
            Severity level: 'low', 'medium', 'high'
        """
        try:
            # Large area or extreme aspect ratio indicates high severity
            if area > 5000 or aspect_ratio > 10.0:
                return 'high'
            elif area > 1000 or aspect_ratio > 5.0 or solidity < 0.5:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            self.logger.error(f"Error determining severity: {str(e)}")
            return 'low'


class ShapeMatcher:
    """Template matching for shape irregularities"""
    
    def __init__(self):
        """Initialize ShapeMatcher"""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_anomalies(self, image: Any, 
                        templates: Dict[str, Any]) -> List[ShapeAnomaly]:
        """
        Detect shape anomalies using template matching
        
        Args:
            image: Input image as numpy array
            templates: Dictionary of template images
            
        Returns:
            List of detected shape anomalies
        """
        anomalies = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # For each template, find matches and analyze them
            for template_name, template in templates.items():
                template_anomalies = self._match_template(gray, template, template_name)
                anomalies.extend(template_anomalies)
            
            self.logger.debug(f"Detected {len(anomalies)} shape anomalies")
            
        except Exception as e:
            self.logger.error(f"Error in shape anomaly detection: {str(e)}")
        
        return anomalies
    
    def _match_template(self, image: Any, template: Any, 
                       template_name: str) -> List[ShapeAnomaly]:
        """
        Match template against image and detect anomalies
        
        Args:
            image: Input grayscale image
            template: Template to match
            template_name: Name of the template
            
        Returns:
            List of shape anomalies for this template
        """
        anomalies = []
        
        try:
            # Perform template matching
            result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            
            # Find locations where matching score is above threshold
            threshold = 0.6
            locations = np.where(result >= threshold)
            
            # Analyze each match
            for pt in zip(*locations[::-1]):  # Switch x and y coordinates
                x, y = pt
                h, w = template.shape
                
                # Extract the matched region
                matched_region = image[y:y+h, x:x+w]
                
                if matched_region.shape == template.shape:
                    # Calculate shape similarity score
                    similarity_score = self._calculate_shape_similarity(matched_region, template)
                    
                    # If similarity is low despite template match, it might be an anomaly
                    if similarity_score < 0.8:
                        anomaly = ShapeAnomaly(
                            object_type=template_name,
                            anomaly_type='deformation',
                            confidence=1.0 - similarity_score,
                            expected_shape=template_name,
                            actual_shape_score=similarity_score,
                            bounding_box=(x, y, w, h)
                        )
                        anomalies.append(anomaly)
            
        except Exception as e:
            self.logger.error(f"Error in template matching for {template_name}: {str(e)}")
        
        return anomalies
    
    def _calculate_shape_similarity(self, region: Any, template: Any) -> float:
        """
        Calculate shape similarity between region and template
        
        Args:
            region: Image region to compare
            template: Template image
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            if region.shape != template.shape:
                return 0.0
            
            # Normalize both images
            region_norm = cv2.normalize(region, None, 0, 255, cv2.NORM_MINMAX)
            template_norm = cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX)
            
            # Calculate structural similarity
            # Simple implementation using correlation coefficient
            correlation = cv2.matchTemplate(region_norm, template_norm, cv2.TM_CCOEFF_NORMED)
            similarity = correlation[0, 0] if correlation.size > 0 else 0.0
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            self.logger.error(f"Error calculating shape similarity: {str(e)}")
            return 0.0


class AnomalyDetector:
    """Anomaly detection for detected objects"""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize AnomalyDetector
        
        Args:
            confidence_threshold: Minimum confidence for anomaly detection
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def analyze_object(self, image: Any, obj: Dict[str, Any]) -> List[ObjectDefect]:
        """
        Analyze detected object for anomalies
        
        Args:
            image: Full image as numpy array
            obj: Object information from detection
            
        Returns:
            List of detected object defects
        """
        defects = []
        
        try:
            # Extract object region
            x, y, w, h = obj['bounding_box']
            object_region = image[y:y+h, x:x+w]
            
            # Analyze object for various types of defects
            texture_defects = self._analyze_texture_anomalies(object_region, obj)
            color_defects = self._analyze_color_anomalies(object_region, obj)
            structural_defects = self._analyze_structural_anomalies(object_region, obj)
            
            # Combine all defects
            all_defects = texture_defects + color_defects + structural_defects
            
            # Filter by confidence threshold
            filtered_defects = [d for d in all_defects if d.confidence >= self.confidence_threshold]
            
            defects.extend(filtered_defects)
            
        except Exception as e:
            self.logger.error(f"Error analyzing object anomalies: {str(e)}")
        
        return defects
    
    def _analyze_texture_anomalies(self, region: Any, obj: Dict[str, Any]) -> List[ObjectDefect]:
        """Analyze texture-based anomalies"""
        defects = []
        
        try:
            # Convert to grayscale for texture analysis
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region
            
            # Calculate texture uniformity
            texture_std = np.std(gray)
            texture_mean = np.mean(gray)
            
            # High standard deviation might indicate texture anomalies
            if texture_std > 50:  # Threshold for texture variation
                confidence = min(0.9, texture_std / 100.0)
                
                defect = ObjectDefect(
                    object_type=obj['type'],
                    defect_type='texture_anomaly',
                    confidence=confidence,
                    bounding_box=obj['bounding_box'],
                    description=f"Irregular texture pattern in {obj['type']}"
                )
                defects.append(defect)
            
        except Exception as e:
            self.logger.error(f"Error in texture anomaly analysis: {str(e)}")
        
        return defects
    
    def _analyze_color_anomalies(self, region: Any, obj: Dict[str, Any]) -> List[ObjectDefect]:
        """Analyze color-based anomalies"""
        defects = []
        
        try:
            if len(region.shape) != 3:
                return defects  # Skip grayscale images
            
            # Calculate color distribution
            b_mean = np.mean(region[:, :, 0])
            g_mean = np.mean(region[:, :, 1])
            r_mean = np.mean(region[:, :, 2])
            
            # Check for extreme color imbalance
            color_means = [b_mean, g_mean, r_mean]
            color_std = np.std(color_means)
            
            if color_std > 30:  # Threshold for color imbalance
                confidence = min(0.8, color_std / 50.0)
                
                defect = ObjectDefect(
                    object_type=obj['type'],
                    defect_type='color_anomaly',
                    confidence=confidence,
                    bounding_box=obj['bounding_box'],
                    description=f"Color imbalance in {obj['type']}"
                )
                defects.append(defect)
            
        except Exception as e:
            self.logger.error(f"Error in color anomaly analysis: {str(e)}")
        
        return defects
    
    def _analyze_structural_anomalies(self, region: Any, obj: Dict[str, Any]) -> List[ObjectDefect]:
        """Analyze structural anomalies"""
        defects = []
        
        try:
            # Convert to grayscale
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region
            
            # Find contours within the object
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_INTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze internal structure
            if len(contours) > 5:  # Too many internal contours might indicate damage
                confidence = min(0.7, len(contours) / 20.0)
                
                defect = ObjectDefect(
                    object_type=obj['type'],
                    defect_type='structural_damage',
                    confidence=confidence,
                    bounding_box=obj['bounding_box'],
                    description=f"Internal structural irregularities in {obj['type']}"
                )
                defects.append(defect)
            
        except Exception as e:
            self.logger.error(f"Error in structural anomaly analysis: {str(e)}")
        
        return defects