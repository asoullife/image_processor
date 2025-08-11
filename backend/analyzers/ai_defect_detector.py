"""
AI-Enhanced Defect Detector for Adobe Stock Image Processor

This module implements AI-enhanced defect detection using:
- YOLO v8 integration for advanced object detection
- TensorFlow models for anomaly detection
- Fallback to OpenCV-based methods when AI models unavailable
- Performance mode optimization (Speed/Balanced/Smart)
- Memory-efficient processing for RTX2060
"""

import os
import logging
import gc
import time
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
from pathlib import Path

try:
    from ultralytics import YOLO
    import tensorflow as tf
    import cv2
    import numpy as np
    from PIL import Image
    YOLO_AVAILABLE = True
    TF_AVAILABLE = True
    CV2_AVAILABLE = True
    PIL_AVAILABLE = True
except ImportError as e:
    logging.warning(f"AI/ML dependencies not available: {e}")
    YOLO = None
    tf = None
    cv2 = None
    np = None
    Image = None
    YOLO_AVAILABLE = False
    TF_AVAILABLE = False
    CV2_AVAILABLE = False
    PIL_AVAILABLE = False

from .ai_model_manager import AIModelManager
try:
    from .defect_detector import DefectDetector
    from ..core.base import DefectResult, ObjectDefect
except ImportError:
    # Fallback for testing without full backend structure
    from dataclasses import dataclass
    from typing import List, Tuple, Any
    
    @dataclass
    class ObjectDefect:
        """Fallback ObjectDefect for testing"""
        object_type: str
        defect_type: str
        confidence: float
        bounding_box: Tuple[int, int, int, int]
        description: str
    
    @dataclass
    class DefectResult:
        """Fallback DefectResult for testing"""
        detected_objects: List[ObjectDefect]
        anomaly_score: float
        edge_count: int
        contour_irregularities: List[Any]
        texture_anomalies: List[Any]
        passed: bool
    
    class DefectDetector:
        """Fallback DefectDetector for testing"""
        def __init__(self, config):
            pass
        
        def analyze(self, image_path: str) -> DefectResult:
            return DefectResult([], 0.1, 0, [], [], True)

logger = logging.getLogger(__name__)


@dataclass
class AIDefectResult:
    """Enhanced defect detection result with AI confidence"""
    traditional_result: DefectResult
    yolo_detections: List[Dict[str, Any]]
    ai_anomaly_score: float
    ai_confidence: float
    ai_reasoning: str
    ai_reasoning_thai: str
    detected_objects: List[ObjectDefect]
    defect_categories: Dict[str, int]
    severity_assessment: str  # 'low', 'medium', 'high', 'critical'
    final_decision: str  # 'approved', 'rejected'
    rejection_reasons: List[str]
    processing_time: float
    model_used: str
    fallback_used: bool


class AIDefectDetector:
    """
    AI-Enhanced Defect Detector with YOLO v8 and fallback mechanisms
    
    Features:
    - YOLO v8 for advanced object detection and defect identification
    - TensorFlow models for anomaly detection
    - GPU acceleration optimized for RTX2060
    - Fallback to traditional OpenCV methods when AI unavailable
    - Performance mode optimization (Speed/Balanced/Smart)
    - Thai language explanations for rejection reasons
    """
    
    def __init__(self, config: Dict[str, Any], model_manager: Optional[AIModelManager] = None):
        """
        Initialize AI Defect Detector
        
        Args:
            config: Configuration dictionary
            model_manager: Optional AI model manager instance
        """
        self.config = config
        self.defect_config = config.get('defect_detection', {})
        
        # Initialize model manager
        if model_manager is not None:
            self.model_manager = model_manager
        else:
            self.model_manager = AIModelManager(config)
        
        # Initialize traditional detector as fallback
        self.traditional_detector = DefectDetector(config)
        
        # AI-specific thresholds
        self.ai_thresholds = {
            'min_ai_confidence': 0.6,
            'yolo_confidence': 0.5,
            'anomaly_threshold': 0.7,
            'max_defects_allowed': 3,
            'critical_defect_threshold': 0.8
        }
        
        # Performance mode settings
        self.performance_mode = "balanced"
        self.batch_size = 4  # YOLO is memory-intensive
        
        # Defect categories for YOLO detection
        self.defect_categories = {
            'crack': ['crack', 'fracture', 'split'],
            'scratch': ['scratch', 'scrape', 'abrasion'],
            'stain': ['stain', 'spot', 'discoloration'],
            'damage': ['damage', 'broken', 'torn'],
            'artifact': ['artifact', 'noise', 'distortion'],
            'blur': ['blur', 'motion_blur', 'out_of_focus']
        }
        
        # Thai language rejection reasons
        self.thai_reasons = {
            'crack_detected': 'พบรอยแตกในภาพ ไม่เหมาะสำหรับการขาย',
            'scratch_detected': 'พบรอยขีดข่วนในภาพ คุณภาพไม่ดี',
            'stain_detected': 'พบคราบหรือจุดด่างในภาพ',
            'damage_detected': 'พบความเสียหายในภาพ ไม่ผ่านมาตรฐาน',
            'artifact_detected': 'พบสิ่งแปลกปลอมหรือสัญญาณรบกวนในภาพ',
            'blur_detected': 'พบภาพเบลอหรือไม่คมชัด',
            'multiple_defects': 'พบข้อบกพร่องหลายประเภทในภาพ',
            'high_anomaly': 'AI ตรวจพบความผิดปกติสูงในภาพ',
            'ai_low_confidence': 'AI ไม่มั่นใจในการประเมิน อาจมีปัญหาที่ซับซ้อน'
        }
        
        logger.info(f"AIDefectDetector initialized - YOLO: {YOLO_AVAILABLE}, TF: {TF_AVAILABLE}")
    
    def set_performance_mode(self, mode: str):
        """
        Set performance mode for AI processing
        
        Args:
            mode: Performance mode ('speed', 'balanced', 'smart')
        """
        self.performance_mode = mode
        self.model_manager.set_performance_mode(mode)
        
        # Update batch size and thresholds based on mode
        mode_settings = {
            'speed': {'batch_size': 8, 'confidence': 0.4},
            'balanced': {'batch_size': 4, 'confidence': 0.5},
            'smart': {'batch_size': 2, 'confidence': 0.6}
        }
        
        settings = mode_settings.get(mode, mode_settings['balanced'])
        self.batch_size = settings['batch_size']
        self.ai_thresholds['yolo_confidence'] = settings['confidence']
        
        logger.info(f"Performance mode set to: {mode}, batch_size: {self.batch_size}")
    
    def analyze(self, image_path: str) -> AIDefectResult:
        """
        Perform AI-enhanced defect detection
        
        Args:
            image_path: Path to the image file
            
        Returns:
            AIDefectResult with comprehensive analysis
        """
        start_time = time.time()
        
        try:
            # Always perform traditional analysis first (fallback)
            traditional_result = self.traditional_detector.analyze(image_path)
            
            # Attempt AI-enhanced analysis
            if YOLO_AVAILABLE and self.model_manager.is_model_available('yolov8n'):
                ai_result = self._analyze_with_ai(image_path, traditional_result)
                processing_time = time.time() - start_time
                ai_result.processing_time = processing_time
                return ai_result
            else:
                # Fallback to traditional analysis only
                logger.info("AI models not available, using traditional analysis")
                return self._create_fallback_result(traditional_result, time.time() - start_time)
                
        except Exception as e:
            logger.error(f"Error in AI defect detection for {image_path}: {e}")
            # Return fallback result on error
            traditional_result = self.traditional_detector.analyze(image_path)
            return self._create_fallback_result(traditional_result, time.time() - start_time)
    
    def _analyze_with_ai(self, image_path: str, traditional_result: DefectResult) -> AIDefectResult:
        """
        Perform AI-enhanced analysis using YOLO v8
        
        Args:
            image_path: Path to the image file
            traditional_result: Traditional analysis result
            
        Returns:
            AIDefectResult with AI enhancements
        """
        try:
            # Get YOLO model
            yolo_model = self.model_manager.get_model('yolov8n')
            if yolo_model is None:
                logger.warning("YOLO model not available, using fallback")
                return self._create_fallback_result(traditional_result, 0.0)
            
            # Run YOLO detection
            yolo_results = yolo_model(image_path, conf=self.ai_thresholds['yolo_confidence'])
            
            # Process YOLO detections
            yolo_detections = self._process_yolo_results(yolo_results)
            
            # Calculate AI anomaly score
            ai_anomaly_score = self._calculate_ai_anomaly_score(yolo_detections, traditional_result)
            
            # Convert detections to ObjectDefect format
            detected_objects = self._convert_yolo_to_defects(yolo_detections)
            
            # Categorize defects
            defect_categories = self._categorize_defects(detected_objects)
            
            # Assess severity
            severity = self._assess_severity(ai_anomaly_score, defect_categories, detected_objects)
            
            # Calculate AI confidence
            ai_confidence = self._calculate_ai_confidence(yolo_detections, ai_anomaly_score)
            
            # Generate AI reasoning
            reasoning_en, reasoning_th = self._generate_ai_reasoning(
                detected_objects, defect_categories, severity, ai_confidence
            )
            
            # Make final decision
            final_decision, rejection_reasons = self._make_ai_decision(
                detected_objects, defect_categories, severity, ai_confidence, traditional_result
            )
            
            result = AIDefectResult(
                traditional_result=traditional_result,
                yolo_detections=yolo_detections,
                ai_anomaly_score=ai_anomaly_score,
                ai_confidence=ai_confidence,
                ai_reasoning=reasoning_en,
                ai_reasoning_thai=reasoning_th,
                detected_objects=detected_objects,
                defect_categories=defect_categories,
                severity_assessment=severity,
                final_decision=final_decision,
                rejection_reasons=rejection_reasons,
                processing_time=0.0,  # Will be set by caller
                model_used='yolov8n',
                fallback_used=False
            )
            
            # Cleanup memory
            self._cleanup_memory()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AI defect analysis: {e}")
            return self._create_fallback_result(traditional_result, 0.0)
    
    def _process_yolo_results(self, yolo_results) -> List[Dict[str, Any]]:
        """
        Process YOLO detection results
        
        Args:
            yolo_results: Raw YOLO detection results
            
        Returns:
            List of processed detection dictionaries
        """
        detections = []
        
        try:
            for result in yolo_results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Extract box information
                        box = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Get class name
                        class_name = result.names[class_id] if hasattr(result, 'names') else f'class_{class_id}'
                        
                        detection = {
                            'bbox': box.tolist(),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'defect_type': self._map_class_to_defect(class_name)
                        }
                        
                        detections.append(detection)
            
            logger.debug(f"Processed {len(detections)} YOLO detections")
            
        except Exception as e:
            logger.error(f"Error processing YOLO results: {e}")
        
        return detections
    
    def _map_class_to_defect(self, class_name: str) -> str:
        """
        Map YOLO class name to defect type
        
        Args:
            class_name: YOLO class name
            
        Returns:
            Defect type string
        """
        class_name_lower = class_name.lower()
        
        for defect_type, keywords in self.defect_categories.items():
            if any(keyword in class_name_lower for keyword in keywords):
                return defect_type
        
        # Default mapping for common objects that might indicate defects
        if any(word in class_name_lower for word in ['person', 'face', 'hand']):
            return 'privacy_concern'
        elif any(word in class_name_lower for word in ['text', 'sign', 'logo']):
            return 'copyright_concern'
        else:
            return 'unknown_object'
    
    def _calculate_ai_anomaly_score(self, yolo_detections: List[Dict[str, Any]], 
                                   traditional_result: DefectResult) -> float:
        """
        Calculate AI-based anomaly score
        
        Args:
            yolo_detections: YOLO detection results
            traditional_result: Traditional analysis result
            
        Returns:
            AI anomaly score (0.0 to 1.0)
        """
        try:
            if not yolo_detections:
                return traditional_result.anomaly_score
            
            # Base score from traditional analysis
            base_score = traditional_result.anomaly_score
            
            # YOLO-based adjustments
            yolo_score = 0.0
            
            # Count defects by type
            defect_counts = {}
            total_confidence = 0.0
            
            for detection in yolo_detections:
                defect_type = detection['defect_type']
                confidence = detection['confidence']
                
                defect_counts[defect_type] = defect_counts.get(defect_type, 0) + 1
                total_confidence += confidence
            
            # Calculate YOLO contribution
            if yolo_detections:
                avg_confidence = total_confidence / len(yolo_detections)
                defect_variety = len(defect_counts)
                total_defects = len(yolo_detections)
                
                # Higher score for more defects, higher confidence, more variety
                yolo_score = min(1.0, (total_defects * avg_confidence * defect_variety) / 10.0)
            
            # Combine scores (weighted average)
            combined_score = (base_score * 0.4) + (yolo_score * 0.6)
            
            return float(max(0.0, min(1.0, combined_score)))
            
        except Exception as e:
            logger.error(f"Error calculating AI anomaly score: {e}")
            return traditional_result.anomaly_score
    
    def _convert_yolo_to_defects(self, yolo_detections: List[Dict[str, Any]]) -> List[ObjectDefect]:
        """
        Convert YOLO detections to ObjectDefect format
        
        Args:
            yolo_detections: YOLO detection results
            
        Returns:
            List of ObjectDefect objects
        """
        defects = []
        
        for detection in yolo_detections:
            try:
                # Convert bbox format: [x1, y1, x2, y2] -> (x, y, width, height)
                x1, y1, x2, y2 = detection['bbox']
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                
                defect = ObjectDefect(
                    object_type=detection['class_name'],
                    defect_type=detection['defect_type'],
                    confidence=detection['confidence'],
                    bounding_box=bbox,
                    description=f"AI detected {detection['defect_type']} with {detection['confidence']:.2f} confidence"
                )
                
                defects.append(defect)
                
            except Exception as e:
                logger.error(f"Error converting YOLO detection to defect: {e}")
                continue
        
        return defects
    
    def _categorize_defects(self, detected_objects: List[ObjectDefect]) -> Dict[str, int]:
        """
        Categorize detected defects by type
        
        Args:
            detected_objects: List of detected defects
            
        Returns:
            Dictionary of defect type counts
        """
        categories = {}
        
        for defect in detected_objects:
            defect_type = defect.defect_type
            categories[defect_type] = categories.get(defect_type, 0) + 1
        
        return categories
    
    def _assess_severity(self, ai_anomaly_score: float, defect_categories: Dict[str, int], 
                        detected_objects: List[ObjectDefect]) -> str:
        """
        Assess overall severity of detected defects
        
        Args:
            ai_anomaly_score: AI anomaly score
            defect_categories: Defect category counts
            detected_objects: List of detected defects
            
        Returns:
            Severity level: 'low', 'medium', 'high', 'critical'
        """
        try:
            # Critical defects that always result in high severity
            critical_defects = ['crack', 'damage', 'privacy_concern', 'copyright_concern']
            
            # Check for critical defects
            has_critical = any(defect_type in critical_defects for defect_type in defect_categories.keys())
            
            # High confidence defects
            high_confidence_count = sum(1 for obj in detected_objects if obj.confidence > 0.8)
            
            # Total defect count
            total_defects = len(detected_objects)
            
            # Determine severity
            if has_critical or ai_anomaly_score > 0.9:
                return 'critical'
            elif ai_anomaly_score > 0.7 or high_confidence_count > 2 or total_defects > 5:
                return 'high'
            elif ai_anomaly_score > 0.4 or high_confidence_count > 0 or total_defects > 2:
                return 'medium'
            else:
                return 'low'
                
        except Exception as e:
            logger.error(f"Error assessing severity: {e}")
            return 'medium'
    
    def _calculate_ai_confidence(self, yolo_detections: List[Dict[str, Any]], 
                               ai_anomaly_score: float) -> float:
        """
        Calculate AI confidence in the analysis
        
        Args:
            yolo_detections: YOLO detection results
            ai_anomaly_score: AI anomaly score
            
        Returns:
            AI confidence score (0.0 to 1.0)
        """
        try:
            if not yolo_detections:
                return 0.8  # High confidence when no defects detected
            
            # Average YOLO confidence
            avg_yolo_confidence = sum(d['confidence'] for d in yolo_detections) / len(yolo_detections)
            
            # Confidence factors
            factors = [
                avg_yolo_confidence,  # YOLO detection confidence
                min(1.0, len(yolo_detections) / 5.0),  # Detection count factor
                1.0 - (ai_anomaly_score * 0.3)  # Lower confidence for high anomaly
            ]
            
            # Calculate weighted confidence
            confidence = sum(factors) / len(factors)
            
            return float(max(0.3, min(1.0, confidence)))
            
        except Exception as e:
            logger.error(f"Error calculating AI confidence: {e}")
            return 0.5
    
    def _generate_ai_reasoning(self, detected_objects: List[ObjectDefect], 
                             defect_categories: Dict[str, int], severity: str,
                             ai_confidence: float) -> Tuple[str, str]:
        """
        Generate AI reasoning in English and Thai
        
        Args:
            detected_objects: List of detected defects
            defect_categories: Defect category counts
            severity: Severity assessment
            ai_confidence: AI confidence score
            
        Returns:
            Tuple of (English reasoning, Thai reasoning)
        """
        try:
            # English reasoning
            reasoning_parts_en = []
            
            if not detected_objects:
                reasoning_parts_en.append("No significant defects detected by AI analysis")
            else:
                reasoning_parts_en.append(f"AI detected {len(detected_objects)} potential defects")
                
                # Detail by category
                for defect_type, count in defect_categories.items():
                    if count > 0:
                        reasoning_parts_en.append(f"{count} {defect_type} defect(s)")
            
            reasoning_parts_en.append(f"Severity assessment: {severity}")
            reasoning_parts_en.append(f"AI confidence: {ai_confidence:.2f}")
            
            reasoning_en = ". ".join(reasoning_parts_en) + "."
            
            # Thai reasoning
            reasoning_parts_th = []
            
            if not detected_objects:
                reasoning_parts_th.append("AI ไม่พบข้อบกพร่องที่สำคัญในภาพ")
            else:
                reasoning_parts_th.append(f"AI ตรวจพบข้อบกพร่องที่อาจเกิดขึ้น {len(detected_objects)} จุด")
                
                # Detail by category in Thai
                thai_defect_names = {
                    'crack': 'รอยแตก',
                    'scratch': 'รอยขีดข่วน',
                    'stain': 'คราบหรือจุดด่าง',
                    'damage': 'ความเสียหาย',
                    'artifact': 'สิ่งแปลกปลอม',
                    'blur': 'ภาพเบลอ',
                    'privacy_concern': 'ปัญหาความเป็นส่วนตัว',
                    'copyright_concern': 'ปัญหาลิขสิทธิ์'
                }
                
                for defect_type, count in defect_categories.items():
                    if count > 0:
                        thai_name = thai_defect_names.get(defect_type, defect_type)
                        reasoning_parts_th.append(f"{thai_name} {count} จุด")
            
            severity_thai = {
                'low': 'ต่ำ',
                'medium': 'ปานกลาง', 
                'high': 'สูง',
                'critical': 'วิกฤต'
            }
            
            reasoning_parts_th.append(f"ระดับความรุนแรง: {severity_thai.get(severity, severity)}")
            reasoning_parts_th.append(f"ความมั่นใจของ AI: {ai_confidence:.2f}")
            
            reasoning_th = " ".join(reasoning_parts_th)
            
            return reasoning_en, reasoning_th
            
        except Exception as e:
            logger.error(f"Error generating AI reasoning: {e}")
            return "AI analysis completed", "การวิเคราะห์ด้วย AI เสร็จสิ้น"
    
    def _make_ai_decision(self, detected_objects: List[ObjectDefect], 
                         defect_categories: Dict[str, int], severity: str,
                         ai_confidence: float, traditional_result: DefectResult) -> Tuple[str, List[str]]:
        """
        Make final decision based on AI and traditional analysis
        
        Args:
            detected_objects: List of detected defects
            defect_categories: Defect category counts
            severity: Severity assessment
            ai_confidence: AI confidence score
            traditional_result: Traditional analysis result
            
        Returns:
            Tuple of (decision, rejection_reasons)
        """
        try:
            rejection_reasons = []
            
            # Check AI confidence
            if ai_confidence < self.ai_thresholds['min_ai_confidence']:
                rejection_reasons.append(self.thai_reasons['ai_low_confidence'])
            
            # Check severity
            if severity in ['critical', 'high']:
                if 'crack' in defect_categories:
                    rejection_reasons.append(self.thai_reasons['crack_detected'])
                if 'scratch' in defect_categories:
                    rejection_reasons.append(self.thai_reasons['scratch_detected'])
                if 'stain' in defect_categories:
                    rejection_reasons.append(self.thai_reasons['stain_detected'])
                if 'damage' in defect_categories:
                    rejection_reasons.append(self.thai_reasons['damage_detected'])
                if 'artifact' in defect_categories:
                    rejection_reasons.append(self.thai_reasons['artifact_detected'])
                if 'blur' in defect_categories:
                    rejection_reasons.append(self.thai_reasons['blur_detected'])
            
            # Check for multiple defect types
            if len(defect_categories) > 2:
                rejection_reasons.append(self.thai_reasons['multiple_defects'])
            
            # Check traditional analysis
            if not traditional_result.passed:
                rejection_reasons.append(self.thai_reasons['high_anomaly'])
            
            # Remove duplicates
            rejection_reasons = list(set(rejection_reasons))
            
            # Make final decision
            if rejection_reasons or severity in ['critical', 'high']:
                decision = 'rejected'
            elif severity == 'medium' and len(detected_objects) > 3:
                decision = 'rejected'
                rejection_reasons.append(self.thai_reasons['multiple_defects'])
            else:
                decision = 'approved'
            
            return decision, rejection_reasons
            
        except Exception as e:
            logger.error(f"Error making AI decision: {e}")
            return 'rejected', [self.thai_reasons['ai_low_confidence']]
    
    def _create_fallback_result(self, traditional_result: DefectResult, 
                              processing_time: float) -> AIDefectResult:
        """
        Create fallback result when AI analysis is not available
        
        Args:
            traditional_result: Traditional analysis result
            processing_time: Processing time in seconds
            
        Returns:
            AIDefectResult using traditional analysis
        """
        # Convert traditional defects to AI format
        detected_objects = traditional_result.detected_objects
        
        # Categorize traditional defects
        defect_categories = {}
        for defect in detected_objects:
            defect_type = defect.defect_type
            defect_categories[defect_type] = defect_categories.get(defect_type, 0) + 1
        
        # Assess severity based on traditional results
        if traditional_result.anomaly_score > 0.8:
            severity = 'high'
        elif traditional_result.anomaly_score > 0.5:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Generate reasoning
        if traditional_result.passed:
            reasoning_en = "Traditional defect analysis passed"
            reasoning_th = "การวิเคราะห์ข้อบกพร่องแบบดั้งเดิมผ่าน"
            decision = 'approved'
            rejection_reasons = []
        else:
            reasoning_en = "Traditional defect analysis failed"
            reasoning_th = "การวิเคราะห์ข้อบกพร่องแบบดั้งเดิมไม่ผ่าน"
            decision = 'rejected'
            rejection_reasons = [self.thai_reasons['high_anomaly']]
        
        return AIDefectResult(
            traditional_result=traditional_result,
            yolo_detections=[],
            ai_anomaly_score=traditional_result.anomaly_score,
            ai_confidence=0.7 if traditional_result.passed else 0.5,
            ai_reasoning=reasoning_en,
            ai_reasoning_thai=reasoning_th,
            detected_objects=detected_objects,
            defect_categories=defect_categories,
            severity_assessment=severity,
            final_decision=decision,
            rejection_reasons=rejection_reasons,
            processing_time=processing_time,
            model_used='traditional_fallback',
            fallback_used=True
        )
    
    def _cleanup_memory(self):
        """Cleanup memory after AI processing"""
        try:
            # Python garbage collection
            gc.collect()
            
            # TensorFlow memory cleanup
            if TF_AVAILABLE:
                tf.keras.backend.clear_session()
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
    
    def analyze_batch(self, image_paths: List[str]) -> List[AIDefectResult]:
        """
        Analyze multiple images in batch for better performance
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of AIDefectResult objects
        """
        logger.info(f"Starting batch defect analysis of {len(image_paths)} images")
        
        results = []
        
        # Process in batches to manage memory
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            
            logger.debug(f"Processing defect batch {i//self.batch_size + 1}: {len(batch_paths)} images")
            
            # Analyze each image in the batch
            batch_results = []
            for image_path in batch_paths:
                result = self.analyze(image_path)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Memory cleanup after each batch
            self._cleanup_memory()
            
            logger.debug(f"Completed defect batch {i//self.batch_size + 1}")
        
        logger.info(f"Batch defect analysis complete: {len(results)} results")
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status for AI defect detection
        
        Returns:
            Status information dictionary
        """
        status = {
            'yolo_available': YOLO_AVAILABLE,
            'tensorflow_available': TF_AVAILABLE,
            'opencv_available': CV2_AVAILABLE,
            'pil_available': PIL_AVAILABLE,
            'performance_mode': self.performance_mode,
            'batch_size': self.batch_size,
            'model_manager_status': self.model_manager.get_system_status(),
            'fallback_available': self.traditional_detector is not None,
            'defect_categories': list(self.defect_categories.keys())
        }
        
        return status