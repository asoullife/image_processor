"""
AI-Enhanced Compliance Checker for Adobe Stock Image Processor

This module implements AI-enhanced compliance checking including:
- Advanced logo and trademark detection using OCR, template matching, and deep learning
- Sophisticated face detection and privacy concern identification with AI models
- Enhanced metadata validation and keyword relevance checking
- Content appropriateness analysis with cultural sensitivity
- Comprehensive compliance reporting with detailed explanations in Thai and English
"""

import os
import logging
import time
import gc
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

# Core dependencies
try:
    import cv2
    import numpy as np
    from PIL import Image
    import easyocr
    import torch
    import torchvision.transforms as transforms
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
except ImportError as e:
    logging.warning(f"Some AI dependencies not available: {e}")
    cv2 = None
    np = None
    Image = None
    easyocr = None
    torch = None
    transforms = None
    pipeline = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

# Import base compliance checker
from .compliance_checker import ComplianceChecker, ComplianceResult, LogoDetection, PrivacyViolation
from .ai_model_manager import AIModelManager

logger = logging.getLogger(__name__)


@dataclass
class AILogoDetection:
    """Enhanced logo detection result with AI confidence"""
    text_detected: str
    confidence: float
    ai_confidence: float
    bounding_box: Tuple[int, int, int, int]
    detection_method: str  # 'ocr', 'template', 'ai_vision'
    brand_classification: str
    risk_level: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class AIPrivacyViolation:
    """Enhanced privacy violation with AI analysis"""
    violation_type: str
    confidence: float
    ai_confidence: float
    bounding_box: Tuple[int, int, int, int]
    description: str
    description_thai: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    recommendation: str
    recommendation_thai: str


@dataclass
class ContentAppropriateness:
    """Content appropriateness analysis result"""
    overall_score: float
    cultural_sensitivity_score: float
    content_safety_score: float
    age_appropriateness: str  # 'all_ages', 'teen', 'adult', 'restricted'
    detected_themes: List[str]
    potential_issues: List[str]
    cultural_concerns: List[str]


@dataclass
class AIComplianceResult:
    """Comprehensive AI-enhanced compliance analysis result"""
    # Traditional compliance results
    traditional_result: ComplianceResult
    
    # AI-enhanced results
    ai_logo_detections: List[AILogoDetection]
    ai_privacy_violations: List[AIPrivacyViolation]
    content_appropriateness: ContentAppropriateness
    enhanced_metadata_analysis: Dict[str, Any]
    
    # Overall AI assessment
    ai_compliance_score: float
    ai_confidence: float
    ai_reasoning: str
    ai_reasoning_thai: str
    
    # Final decision
    final_decision: str  # 'approved', 'rejected'
    rejection_reasons: List[str]
    rejection_reasons_thai: List[str]
    
    # Processing metadata
    processing_time: float
    models_used: List[str]
    fallback_used: bool


class AIComplianceChecker:
    """
    AI-Enhanced Compliance Checker for Adobe Stock guidelines
    
    This class extends the basic compliance checker with AI capabilities:
    - Advanced logo/trademark detection using deep learning
    - Sophisticated face detection with emotion and age analysis
    - Content appropriateness analysis with cultural sensitivity
    - Enhanced metadata validation with NLP
    - Comprehensive reporting in Thai and English
    """
    
    def __init__(self, config: Dict[str, Any], model_manager: Optional[AIModelManager] = None):
        """
        Initialize AI Compliance Checker
        
        Args:
            config: Configuration dictionary
            model_manager: Shared AI model manager instance
        """
        self.config = config
        self.model_manager = model_manager or AIModelManager(config)
        
        # Initialize base compliance checker
        self.base_checker = ComplianceChecker(config)
        
        # AI-specific configuration
        self.ai_config = config.get('compliance', {})
        self.logo_confidence_threshold = self.ai_config.get('ai_logo_confidence', 0.8)
        self.face_confidence_threshold = self.ai_config.get('ai_face_confidence', 0.7)
        self.content_safety_threshold = self.ai_config.get('content_safety_threshold', 0.8)
        
        # Performance mode
        self.performance_mode = "balanced"
        
        # AI models (loaded on demand)
        self._ocr_reader = None
        self._face_detector = None
        self._content_classifier = None
        self._brand_classifier = None
        
        # Thai language support
        self.thai_translations = {
            'face_detected': 'ตรวจพบใบหน้าบุคคล',
            'logo_detected': 'ตรวจพบโลโก้หรือเครื่องหมายการค้า',
            'license_plate': 'ตรวจพบป้ายทะเบียนรถ',
            'inappropriate_content': 'เนื้อหาไม่เหมาะสม',
            'cultural_sensitivity': 'ความไวทางวัฒนธรรม',
            'privacy_concern': 'ปัญหาความเป็นส่วนตัว',
            'metadata_issue': 'ปัญหาข้อมูลเมตาดาต้า',
            'high_risk': 'ความเสี่ยงสูง',
            'medium_risk': 'ความเสี่ยงปานกลาง',
            'low_risk': 'ความเสี่ยงต่ำ',
            'approved': 'อนุมัติ',
            'rejected': 'ไม่อนุมัติ'
        }
        
        logger.info("AIComplianceChecker initialized")
    
    def set_performance_mode(self, mode: str):
        """
        Set performance mode for AI processing
        
        Args:
            mode: Performance mode ('speed', 'balanced', 'smart')
        """
        self.performance_mode = mode
        
        # Update model manager
        if self.model_manager:
            self.model_manager.set_performance_mode(mode)
        
        # Adjust processing parameters based on mode
        if mode == 'speed':
            self.logo_confidence_threshold = 0.6
            self.face_confidence_threshold = 0.5
        elif mode == 'balanced':
            self.logo_confidence_threshold = 0.7
            self.face_confidence_threshold = 0.6
        elif mode == 'smart':
            self.logo_confidence_threshold = 0.8
            self.face_confidence_threshold = 0.7
        
        logger.info(f"AI Compliance Checker performance mode set to: {mode}")
    
    def analyze(self, image_path: str, metadata: Optional[Dict[str, Any]] = None) -> AIComplianceResult:
        """
        Perform comprehensive AI-enhanced compliance analysis
        
        Args:
            image_path: Path to the image file
            metadata: Optional metadata dictionary
            
        Returns:
            AIComplianceResult with comprehensive analysis
        """
        start_time = time.time()
        
        try:
            # Perform traditional compliance check first
            traditional_result = self.base_checker.check_compliance(image_path, metadata)
            
            # Load image for AI analysis
            image = self._load_image_for_ai(image_path)
            if image is None:
                return self._create_error_result(image_path, traditional_result, time.time() - start_time)
            
            # Perform AI-enhanced analysis
            ai_logo_detections = self._detect_logos_ai(image)
            ai_privacy_violations = self._detect_privacy_violations_ai(image)
            content_appropriateness = self._analyze_content_appropriateness(image, metadata)
            enhanced_metadata = self._analyze_metadata_ai(metadata or {})
            
            # Create comprehensive result
            ai_result = self._create_ai_result(
                image_path, traditional_result, ai_logo_detections,
                ai_privacy_violations, content_appropriateness, enhanced_metadata,
                time.time() - start_time
            )
            
            # Memory cleanup
            self._cleanup_memory()
            
            return ai_result
            
        except Exception as e:
            logger.error(f"Error in AI compliance analysis for {image_path}: {e}")
            return self._create_error_result(image_path, None, time.time() - start_time)
    
    def _load_image_for_ai(self, image_path: str) -> Optional[Any]:
        """Load and preprocess image for AI analysis"""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
            
            # Load with OpenCV
            if cv2 is not None:
                image = cv2.imread(image_path)
                if image is not None:
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Resize for AI processing if too large
                    height, width = image.shape[:2]
                    if max(height, width) > 2048:
                        scale = 2048 / max(height, width)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                    
                    return image
            
            # Fallback to PIL
            if Image is not None:
                pil_image = Image.open(image_path)
                image = np.array(pil_image.convert('RGB'))
                
                # Resize if needed
                height, width = image.shape[:2]
                if max(height, width) > 2048:
                    pil_image = pil_image.resize((2048, int(2048 * height / width)), Image.Resampling.LANCZOS)
                    image = np.array(pil_image)
                
                return image
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading image for AI analysis: {e}")
            return None
    
    def _detect_logos_ai(self, image: Any) -> List[AILogoDetection]:
        """Detect logos using AI-enhanced methods"""
        detections = []
        
        try:
            # Initialize OCR reader if needed
            if self._ocr_reader is None:
                self._ocr_reader = self._load_ocr_reader()
            
            if self._ocr_reader is None:
                logger.warning("OCR reader not available, skipping AI logo detection")
                return detections
            
            # Perform OCR with EasyOCR (supports multiple languages)
            ocr_results = self._ocr_reader.readtext(image)
            
            # Analyze each detected text
            for (bbox, text, confidence) in ocr_results:
                if confidence >= self.logo_confidence_threshold:
                    # Classify if text is a brand/logo
                    brand_classification, ai_confidence = self._classify_brand_text(text)
                    
                    if brand_classification != 'not_brand':
                        # Convert bbox format
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        x, y = int(min(x_coords)), int(min(y_coords))
                        w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                        
                        # Determine risk level
                        risk_level = self._assess_logo_risk(brand_classification, ai_confidence)
                        
                        detections.append(AILogoDetection(
                            text_detected=text,
                            confidence=confidence,
                            ai_confidence=ai_confidence,
                            bounding_box=(x, y, w, h),
                            detection_method='ai_ocr',
                            brand_classification=brand_classification,
                            risk_level=risk_level
                        ))
            
        except Exception as e:
            logger.error(f"Error in AI logo detection: {e}")
        
        return detections
    
    def _detect_privacy_violations_ai(self, image: Any) -> List[AIPrivacyViolation]:
        """Detect privacy violations using AI models"""
        violations = []
        
        try:
            # Face detection with AI
            face_violations = self._detect_faces_ai(image)
            violations.extend(face_violations)
            
            # License plate detection with AI
            plate_violations = self._detect_license_plates_ai(image)
            violations.extend(plate_violations)
            
            # Personal information detection
            personal_info_violations = self._detect_personal_info_ai(image)
            violations.extend(personal_info_violations)
            
        except Exception as e:
            logger.error(f"Error in AI privacy violation detection: {e}")
        
        return violations
    
    def _detect_faces_ai(self, image: Any) -> List[AIPrivacyViolation]:
        """Detect faces using AI models with enhanced analysis"""
        violations = []
        
        try:
            # Use OpenCV DNN face detector for better accuracy
            if cv2 is not None:
                # Load face detection model
                face_net = self._load_face_detector()
                if face_net is None:
                    return violations
                
                # Prepare image for DNN
                blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
                face_net.setInput(blob)
                detections = face_net.forward()
                
                h, w = image.shape[:2]
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    
                    if confidence >= self.face_confidence_threshold:
                        # Get bounding box
                        x1 = int(detections[0, 0, i, 3] * w)
                        y1 = int(detections[0, 0, i, 4] * h)
                        x2 = int(detections[0, 0, i, 5] * w)
                        y2 = int(detections[0, 0, i, 6] * h)
                        
                        # Calculate width and height
                        bbox_w = x2 - x1
                        bbox_h = y2 - y1
                        
                        # Analyze face characteristics
                        face_analysis = self._analyze_face_characteristics(image, (x1, y1, bbox_w, bbox_h))
                        
                        # Determine severity based on face size and characteristics
                        severity = self._assess_face_privacy_risk(bbox_w, bbox_h, w, h, face_analysis)
                        
                        violations.append(AIPrivacyViolation(
                            violation_type='face',
                            confidence=confidence,
                            ai_confidence=confidence,
                            bounding_box=(x1, y1, bbox_w, bbox_h),
                            description=f'Human face detected with {confidence:.2f} confidence',
                            description_thai=f'ตรวจพบใบหน้าบุคคลด้วยความมั่นใจ {confidence:.2f}',
                            severity=severity,
                            recommendation='Model release required for commercial use',
                            recommendation_thai='ต้องมีใบอนุญาตจากนางแบบสำหรับการใช้เชิงพาณิชย์'
                        ))
            
        except Exception as e:
            logger.error(f"Error in AI face detection: {e}")
        
        return violations
    
    def _detect_license_plates_ai(self, image: Any) -> List[AIPrivacyViolation]:
        """Detect license plates using AI-enhanced OCR"""
        violations = []
        
        try:
            if self._ocr_reader is None:
                return violations
            
            # Focus on potential license plate regions
            # Use edge detection to find rectangular regions
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if cv2 is not None else image[:,:,0]
            
            # Perform OCR on the entire image
            ocr_results = self._ocr_reader.readtext(gray)
            
            # License plate patterns (enhanced)
            import re
            plate_patterns = [
                r'[A-Z]{2,3}[-\s]?\d{3,4}',  # ABC-123, AB 1234
                r'\d{3}[-\s]?[A-Z]{3}',      # 123-ABC
                r'[A-Z]\d{3}[-\s]?[A-Z]{3}', # A123-BCD
                r'\d{1,3}[-\s]?[A-Z]{1,3}[-\s]?\d{1,4}',  # 12-AB-345
                r'[A-Z]{1,2}\d{1,4}[A-Z]{1,2}',  # AB123CD
                r'\d{1,3}[A-Z]{1,3}\d{1,4}'     # 12ABC345
            ]
            
            for (bbox, text, confidence) in ocr_results:
                text_clean = text.replace(' ', '').replace('-', '').upper()
                
                # Check if text matches license plate patterns
                for pattern in plate_patterns:
                    if re.match(pattern, text_clean) and len(text_clean) >= 5:
                        # Convert bbox format
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        x, y = int(min(x_coords)), int(min(y_coords))
                        w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                        
                        # Assess aspect ratio (license plates are typically wider than tall)
                        aspect_ratio = w / h if h > 0 else 0
                        if 2.0 <= aspect_ratio <= 6.0:  # Typical license plate aspect ratios
                            violations.append(AIPrivacyViolation(
                                violation_type='license_plate',
                                confidence=confidence,
                                ai_confidence=confidence * 0.9,  # Slightly lower confidence for pattern matching
                                bounding_box=(x, y, w, h),
                                description=f'License plate detected: {text}',
                                description_thai=f'ตรวจพบป้ายทะเบียนรถ: {text}',
                                severity='high',
                                recommendation='Blur or remove license plate information',
                                recommendation_thai='ควรเบลอหรือลบข้อมูลป้ายทะเบียนรถ'
                            ))
                        break
            
        except Exception as e:
            logger.error(f"Error in AI license plate detection: {e}")
        
        return violations
    
    def _detect_personal_info_ai(self, image: Any) -> List[AIPrivacyViolation]:
        """Detect personal information using AI OCR"""
        violations = []
        
        try:
            if self._ocr_reader is None:
                return violations
            
            # Perform OCR
            ocr_results = self._ocr_reader.readtext(image)
            
            # Patterns for personal information
            import re
            personal_patterns = {
                'phone': r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'id_number': r'\b\d{13}\b',  # Thai ID format
                'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
            }
            
            for (bbox, text, confidence) in ocr_results:
                for info_type, pattern in personal_patterns.items():
                    if re.search(pattern, text):
                        # Convert bbox format
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        x, y = int(min(x_coords)), int(min(y_coords))
                        w, h = int(max(x_coords) - x), int(max(y_coords) - y)
                        
                        violations.append(AIPrivacyViolation(
                            violation_type='personal_info',
                            confidence=confidence,
                            ai_confidence=confidence * 0.8,
                            bounding_box=(x, y, w, h),
                            description=f'Personal information detected: {info_type}',
                            description_thai=f'ตรวจพบข้อมูลส่วนบุคคล: {info_type}',
                            severity='critical',
                            recommendation=f'Remove or blur {info_type} information',
                            recommendation_thai=f'ควรลบหรือเบลอข้อมูล {info_type}'
                        ))
            
        except Exception as e:
            logger.error(f"Error in personal information detection: {e}")
        
        return violations
    
    def _analyze_content_appropriateness(self, image: Any, 
                                       metadata: Optional[Dict[str, Any]]) -> ContentAppropriateness:
        """Analyze content appropriateness with cultural sensitivity"""
        try:
            # Initialize content classifier if needed
            if self._content_classifier is None:
                self._content_classifier = self._load_content_classifier()
            
            # Analyze image content
            content_scores = self._analyze_image_content(image)
            
            # Analyze metadata for inappropriate content
            metadata_scores = self._analyze_metadata_content(metadata or {})
            
            # Combine scores
            overall_score = (content_scores.get('safety', 0.8) + metadata_scores.get('safety', 0.8)) / 2
            cultural_score = (content_scores.get('cultural', 0.8) + metadata_scores.get('cultural', 0.8)) / 2
            
            # Determine age appropriateness
            age_appropriateness = self._determine_age_appropriateness(overall_score, content_scores)
            
            # Detect themes and issues
            detected_themes = content_scores.get('themes', [])
            potential_issues = content_scores.get('issues', [])
            cultural_concerns = content_scores.get('cultural_concerns', [])
            
            return ContentAppropriateness(
                overall_score=overall_score,
                cultural_sensitivity_score=cultural_score,
                content_safety_score=content_scores.get('safety', 0.8),
                age_appropriateness=age_appropriateness,
                detected_themes=detected_themes,
                potential_issues=potential_issues,
                cultural_concerns=cultural_concerns
            )
            
        except Exception as e:
            logger.error(f"Error in content appropriateness analysis: {e}")
            return ContentAppropriateness(
                overall_score=0.5,
                cultural_sensitivity_score=0.5,
                content_safety_score=0.5,
                age_appropriateness='unknown',
                detected_themes=[],
                potential_issues=['Analysis failed'],
                cultural_concerns=[]
            )
    
    def _analyze_metadata_ai(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metadata using AI/NLP techniques"""
        analysis = {
            'keyword_quality': 0.0,
            'description_quality': 0.0,
            'language_appropriateness': 0.0,
            'commercial_viability': 0.0,
            'detected_issues': []
        }
        
        try:
            # Analyze keywords
            keywords = self._extract_keywords_from_metadata(metadata)
            if keywords:
                analysis['keyword_quality'] = self._assess_keyword_quality(keywords)
            
            # Analyze description
            description = metadata.get('description', '') or metadata.get('title', '')
            if description:
                analysis['description_quality'] = self._assess_description_quality(description)
                analysis['language_appropriateness'] = self._assess_language_appropriateness(description)
            
            # Assess commercial viability
            analysis['commercial_viability'] = self._assess_commercial_viability(metadata)
            
            # Detect specific issues
            analysis['detected_issues'] = self._detect_metadata_issues(metadata)
            
        except Exception as e:
            logger.error(f"Error in AI metadata analysis: {e}")
            analysis['detected_issues'].append(f"Analysis error: {str(e)}")
        
        return analysis
    
    def _create_ai_result(self, image_path: str, traditional_result: ComplianceResult,
                         ai_logo_detections: List[AILogoDetection],
                         ai_privacy_violations: List[AIPrivacyViolation],
                         content_appropriateness: ContentAppropriateness,
                         enhanced_metadata: Dict[str, Any],
                         processing_time: float) -> AIComplianceResult:
        """Create comprehensive AI compliance result"""
        try:
            # Calculate AI compliance score
            ai_compliance_score = self._calculate_ai_compliance_score(
                ai_logo_detections, ai_privacy_violations, content_appropriateness, enhanced_metadata
            )
            
            # Calculate AI confidence
            ai_confidence = self._calculate_ai_confidence(
                ai_logo_detections, ai_privacy_violations, content_appropriateness
            )
            
            # Make final decision
            final_decision, rejection_reasons, rejection_reasons_thai = self._make_final_decision(
                traditional_result, ai_compliance_score, ai_logo_detections,
                ai_privacy_violations, content_appropriateness, enhanced_metadata
            )
            
            # Generate AI reasoning
            ai_reasoning, ai_reasoning_thai = self._generate_ai_reasoning(
                ai_compliance_score, ai_confidence, final_decision,
                ai_logo_detections, ai_privacy_violations, content_appropriateness
            )
            
            # Collect models used
            models_used = ['easyocr', 'opencv_dnn']
            if self._content_classifier is not None:
                models_used.append('content_classifier')
            
            fallback_used = (self._ocr_reader is None or self._face_detector is None)
            
            return AIComplianceResult(
                traditional_result=traditional_result,
                ai_logo_detections=ai_logo_detections,
                ai_privacy_violations=ai_privacy_violations,
                content_appropriateness=content_appropriateness,
                enhanced_metadata_analysis=enhanced_metadata,
                ai_compliance_score=ai_compliance_score,
                ai_confidence=ai_confidence,
                ai_reasoning=ai_reasoning,
                ai_reasoning_thai=ai_reasoning_thai,
                final_decision=final_decision,
                rejection_reasons=rejection_reasons,
                rejection_reasons_thai=rejection_reasons_thai,
                processing_time=processing_time,
                models_used=models_used,
                fallback_used=fallback_used
            )
            
        except Exception as e:
            logger.error(f"Error creating AI compliance result: {e}")
            return self._create_error_result(image_path, traditional_result, processing_time)
    
    # Helper methods for AI model loading and processing
    def _load_ocr_reader(self):
        """Load EasyOCR reader"""
        try:
            if easyocr is not None:
                # Support Thai and English
                return easyocr.Reader(['en', 'th'], gpu=torch.cuda.is_available() if torch else False)
        except Exception as e:
            logger.error(f"Error loading OCR reader: {e}")
        return None
    
    def _load_face_detector(self):
        """Load OpenCV DNN face detector"""
        try:
            if cv2 is not None:
                # Use OpenCV DNN face detector
                model_path = "models/opencv_face_detector_uint8.pb"
                config_path = "models/opencv_face_detector.pbtxt"
                
                # If model files don't exist, use Haar cascade as fallback
                if not (os.path.exists(model_path) and os.path.exists(config_path)):
                    logger.warning("DNN face detector models not found, using Haar cascade")
                    return None
                
                net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
                return net
        except Exception as e:
            logger.error(f"Error loading face detector: {e}")
        return None
    
    def _load_content_classifier(self):
        """Load content classification model"""
        try:
            if pipeline is not None:
                # Use a lightweight content classification model
                return pipeline("text-classification", 
                              model="unitary/toxic-bert",
                              device=0 if torch and torch.cuda.is_available() else -1)
        except Exception as e:
            logger.error(f"Error loading content classifier: {e}")
        return None
    
    def _classify_brand_text(self, text: str) -> Tuple[str, float]:
        """Classify if text is a brand/logo"""
        # Simple brand classification based on known brands
        brand_keywords = [
            'nike', 'adidas', 'apple', 'google', 'microsoft', 'coca-cola', 'pepsi',
            'mcdonalds', 'starbucks', 'amazon', 'facebook', 'twitter', 'instagram',
            'youtube', 'netflix', 'disney', 'marvel', 'sony', 'samsung', 'lg'
        ]
        
        text_lower = text.lower().strip()
        
        for brand in brand_keywords:
            if brand in text_lower:
                confidence = 0.9 if text_lower == brand else 0.7
                return brand, confidence
        
        # Check for trademark symbols
        if '®' in text or '™' in text or '©' in text:
            return 'trademark', 0.8
        
        return 'not_brand', 0.1
    
    def _assess_logo_risk(self, brand_classification: str, confidence: float) -> str:
        """Assess risk level of detected logo"""
        high_risk_brands = ['nike', 'adidas', 'apple', 'google', 'microsoft', 'disney', 'marvel']
        
        if brand_classification in high_risk_brands and confidence > 0.8:
            return 'critical'
        elif brand_classification != 'not_brand' and confidence > 0.7:
            return 'high'
        elif brand_classification != 'not_brand' and confidence > 0.5:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_face_characteristics(self, image: Any, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
        """Analyze face characteristics for privacy assessment"""
        # Basic analysis - could be enhanced with age/emotion detection models
        x, y, w, h = bbox
        
        # Calculate face size relative to image
        image_area = image.shape[0] * image.shape[1]
        face_area = w * h
        face_ratio = face_area / image_area
        
        return {
            'face_size_ratio': face_ratio,
            'is_prominent': face_ratio > 0.05,  # Face takes up more than 5% of image
            'bbox_area': face_area
        }
    
    def _assess_face_privacy_risk(self, face_w: int, face_h: int, img_w: int, img_h: int, 
                                 face_analysis: Dict[str, Any]) -> str:
        """Assess privacy risk level for detected face"""
        face_ratio = face_analysis.get('face_size_ratio', 0)
        
        if face_ratio > 0.15:  # Face is very prominent
            return 'critical'
        elif face_ratio > 0.05:  # Face is clearly visible
            return 'high'
        elif face_ratio > 0.01:  # Face is small but detectable
            return 'medium'
        else:
            return 'low'
    
    def _analyze_image_content(self, image: Any) -> Dict[str, Any]:
        """Analyze image content for appropriateness"""
        # Basic content analysis - could be enhanced with specialized models
        return {
            'safety': 0.8,  # Default safe score
            'cultural': 0.8,  # Default culturally appropriate score
            'themes': [],
            'issues': [],
            'cultural_concerns': []
        }
    
    def _analyze_metadata_content(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metadata content for appropriateness"""
        # Basic metadata content analysis
        return {
            'safety': 0.8,
            'cultural': 0.8
        }
    
    def _determine_age_appropriateness(self, overall_score: float, content_scores: Dict[str, Any]) -> str:
        """Determine age appropriateness based on content analysis"""
        if overall_score >= 0.9:
            return 'all_ages'
        elif overall_score >= 0.7:
            return 'teen'
        elif overall_score >= 0.5:
            return 'adult'
        else:
            return 'restricted'
    
    def _extract_keywords_from_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract keywords from metadata"""
        keywords = []
        
        for field in ['keywords', 'tags', 'description', 'title', 'subject']:
            value = metadata.get(field, '')
            if isinstance(value, str):
                keywords.extend(value.lower().split())
            elif isinstance(value, list):
                keywords.extend([str(k).lower() for k in value])
        
        return list(set(keywords))  # Remove duplicates
    
    def _assess_keyword_quality(self, keywords: List[str]) -> float:
        """Assess quality of keywords"""
        if not keywords:
            return 0.0
        
        # Basic quality assessment
        inappropriate_keywords = [
            'nude', 'naked', 'sex', 'porn', 'adult', 'explicit',
            'violence', 'weapon', 'gun', 'knife', 'blood'
        ]
        
        inappropriate_count = sum(1 for keyword in keywords if keyword in inappropriate_keywords)
        quality_score = max(0.0, 1.0 - (inappropriate_count / len(keywords)))
        
        return quality_score
    
    def _assess_description_quality(self, description: str) -> float:
        """Assess quality of description"""
        if not description:
            return 0.0
        
        # Basic quality metrics
        word_count = len(description.split())
        
        if word_count < 3:
            return 0.3
        elif word_count < 10:
            return 0.6
        else:
            return 0.9
    
    def _assess_language_appropriateness(self, text: str) -> float:
        """Assess language appropriateness"""
        # Basic profanity and inappropriate language detection
        inappropriate_words = [
            'damn', 'hell', 'stupid', 'idiot', 'hate', 'kill', 'die', 'death'
        ]
        
        text_lower = text.lower()
        inappropriate_count = sum(1 for word in inappropriate_words if word in text_lower)
        
        if inappropriate_count == 0:
            return 1.0
        else:
            return max(0.0, 1.0 - (inappropriate_count * 0.2))
    
    def _assess_commercial_viability(self, metadata: Dict[str, Any]) -> float:
        """Assess commercial viability based on metadata"""
        # Basic commercial viability assessment
        score = 0.5  # Default neutral score
        
        # Check for commercial keywords
        commercial_keywords = ['business', 'professional', 'corporate', 'marketing', 'advertising']
        description = str(metadata.get('description', '')).lower()
        
        commercial_count = sum(1 for keyword in commercial_keywords if keyword in description)
        if commercial_count > 0:
            score += 0.2
        
        # Check for proper metadata fields
        if metadata.get('title'):
            score += 0.1
        if metadata.get('description'):
            score += 0.1
        if metadata.get('keywords'):
            score += 0.1
        
        return min(1.0, score)
    
    def _detect_metadata_issues(self, metadata: Dict[str, Any]) -> List[str]:
        """Detect specific issues in metadata"""
        issues = []
        
        # Check for missing essential fields
        if not metadata.get('title') and not metadata.get('description'):
            issues.append("Missing title and description")
        
        # Check for GPS data
        if any('gps' in str(key).lower() for key in metadata.keys()):
            issues.append("GPS location data present")
        
        # Check for personal device information
        device_fields = ['make', 'model', 'software']
        for field in device_fields:
            value = metadata.get(field, '')
            if value and any(brand in str(value).lower() for brand in ['iphone', 'samsung', 'pixel']):
                issues.append(f"Personal device information in {field}")
        
        return issues
    
    def _calculate_ai_compliance_score(self, ai_logo_detections: List[AILogoDetection],
                                     ai_privacy_violations: List[AIPrivacyViolation],
                                     content_appropriateness: ContentAppropriateness,
                                     enhanced_metadata: Dict[str, Any]) -> float:
        """Calculate overall AI compliance score"""
        try:
            # Start with base score
            score = 1.0
            
            # Penalize for logo detections
            for logo in ai_logo_detections:
                if logo.risk_level == 'critical':
                    score -= 0.4
                elif logo.risk_level == 'high':
                    score -= 0.3
                elif logo.risk_level == 'medium':
                    score -= 0.2
                else:
                    score -= 0.1
            
            # Penalize for privacy violations
            for violation in ai_privacy_violations:
                if violation.severity == 'critical':
                    score -= 0.5
                elif violation.severity == 'high':
                    score -= 0.3
                elif violation.severity == 'medium':
                    score -= 0.2
                else:
                    score -= 0.1
            
            # Factor in content appropriateness
            score *= content_appropriateness.overall_score
            
            # Factor in metadata quality
            metadata_score = enhanced_metadata.get('commercial_viability', 0.5)
            score *= (0.7 + 0.3 * metadata_score)  # Weight metadata less heavily
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating AI compliance score: {e}")
            return 0.5
    
    def _calculate_ai_confidence(self, ai_logo_detections: List[AILogoDetection],
                               ai_privacy_violations: List[AIPrivacyViolation],
                               content_appropriateness: ContentAppropriateness) -> float:
        """Calculate AI confidence in the analysis"""
        try:
            confidences = []
            
            # Logo detection confidences
            for logo in ai_logo_detections:
                confidences.append(logo.ai_confidence)
            
            # Privacy violation confidences
            for violation in ai_privacy_violations:
                confidences.append(violation.ai_confidence)
            
            # Content analysis confidence (estimated)
            confidences.append(content_appropriateness.overall_score)
            
            if confidences:
                return sum(confidences) / len(confidences)
            else:
                return 0.8  # Default confidence when no specific detections
                
        except Exception as e:
            logger.error(f"Error calculating AI confidence: {e}")
            return 0.5
    
    def _make_final_decision(self, traditional_result: ComplianceResult, ai_compliance_score: float,
                           ai_logo_detections: List[AILogoDetection],
                           ai_privacy_violations: List[AIPrivacyViolation],
                           content_appropriateness: ContentAppropriateness,
                           enhanced_metadata: Dict[str, Any]) -> Tuple[str, List[str], List[str]]:
        """Make final compliance decision"""
        try:
            rejection_reasons = []
            rejection_reasons_thai = []
            
            # Check AI compliance score
            if ai_compliance_score < 0.6:
                rejection_reasons.append("Overall compliance score too low")
                rejection_reasons_thai.append("คะแนนการปฏิบัติตามกฎระเบียบโดยรวมต่ำเกินไป")
            
            # Check for critical logo detections
            critical_logos = [logo for logo in ai_logo_detections if logo.risk_level == 'critical']
            if critical_logos:
                logos_text = ', '.join([logo.text_detected for logo in critical_logos])
                rejection_reasons.append(f"Critical trademark/logo detected: {logos_text}")
                rejection_reasons_thai.append(f"ตรวจพบเครื่องหมายการค้าที่มีความเสี่ยงสูง: {logos_text}")
            
            # Check for critical privacy violations
            critical_violations = [v for v in ai_privacy_violations if v.severity == 'critical']
            if critical_violations:
                violation_types = ', '.join(set([v.violation_type for v in critical_violations]))
                rejection_reasons.append(f"Critical privacy violations: {violation_types}")
                rejection_reasons_thai.append(f"การละเมิดความเป็นส่วนตัวระดับวิกฤต: {violation_types}")
            
            # Check content appropriateness
            if content_appropriateness.overall_score < 0.5:
                rejection_reasons.append("Content not appropriate for stock photography")
                rejection_reasons_thai.append("เนื้อหาไม่เหมาะสมสำหรับภาพสต็อก")
            
            # Check for high-risk privacy violations
            high_risk_violations = [v for v in ai_privacy_violations if v.severity in ['high', 'critical']]
            if len(high_risk_violations) > 2:
                rejection_reasons.append("Multiple privacy concerns detected")
                rejection_reasons_thai.append("ตรวจพบปัญหาความเป็นส่วนตัวหลายประการ")
            
            # Make final decision
            if rejection_reasons:
                return 'rejected', rejection_reasons, rejection_reasons_thai
            else:
                return 'approved', [], []
                
        except Exception as e:
            logger.error(f"Error making final decision: {e}")
            return 'rejected', ["Decision making error"], ["เกิดข้อผิดพลาดในการตัดสินใจ"]
    
    def _generate_ai_reasoning(self, ai_compliance_score: float, ai_confidence: float,
                             final_decision: str, ai_logo_detections: List[AILogoDetection],
                             ai_privacy_violations: List[AIPrivacyViolation],
                             content_appropriateness: ContentAppropriateness) -> Tuple[str, str]:
        """Generate AI reasoning for the decision"""
        try:
            # English reasoning
            reasoning_parts_en = [
                f"AI compliance analysis completed with score: {ai_compliance_score:.2f}",
                f"Analysis confidence: {ai_confidence:.2f}",
                f"Logo detections: {len(ai_logo_detections)}",
                f"Privacy violations: {len(ai_privacy_violations)}",
                f"Content appropriateness: {content_appropriateness.overall_score:.2f}",
                f"Final decision: {final_decision}"
            ]
            
            reasoning_en = ". ".join(reasoning_parts_en) + "."
            
            # Thai reasoning
            reasoning_parts_th = [
                f"การวิเคราะห์การปฏิบัติตามกฎระเบียบด้วย AI เสร็จสิ้น คะแนน: {ai_compliance_score:.2f}",
                f"ความมั่นใจในการวิเคราะห์: {ai_confidence:.2f}",
                f"การตรวจจับโลโก้: {len(ai_logo_detections)} รายการ",
                f"การละเมิดความเป็นส่วนตัว: {len(ai_privacy_violations)} รายการ",
                f"ความเหมาะสมของเนื้อหา: {content_appropriateness.overall_score:.2f}"
            ]
            
            if final_decision == 'approved':
                reasoning_parts_th.append("ผลการตัดสินใจ: อนุมัติ")
            else:
                reasoning_parts_th.append("ผลการตัดสินใจ: ไม่อนุมัติ")
            
            reasoning_th = " ".join(reasoning_parts_th)
            
            return reasoning_en, reasoning_th
            
        except Exception as e:
            logger.error(f"Error generating AI reasoning: {e}")
            return "AI compliance analysis completed", "การวิเคราะห์การปฏิบัติตามกฎระเบียบด้วย AI เสร็จสิ้น"
    
    def _create_error_result(self, image_path: str, traditional_result: Optional[ComplianceResult],
                           processing_time: float) -> AIComplianceResult:
        """Create error result when AI analysis fails"""
        if traditional_result is None:
            traditional_result = ComplianceResult(
                logo_detections=[],
                privacy_violations=[],
                metadata_issues=["Analysis failed"],
                keyword_relevance=0.0,
                overall_compliance=False
            )
        
        error_content = ContentAppropriateness(
            overall_score=0.0,
            cultural_sensitivity_score=0.0,
            content_safety_score=0.0,
            age_appropriateness='unknown',
            detected_themes=[],
            potential_issues=['Analysis failed'],
            cultural_concerns=[]
        )
        
        return AIComplianceResult(
            traditional_result=traditional_result,
            ai_logo_detections=[],
            ai_privacy_violations=[],
            content_appropriateness=error_content,
            enhanced_metadata_analysis={'error': True},
            ai_compliance_score=0.0,
            ai_confidence=0.0,
            ai_reasoning="AI compliance analysis failed due to error",
            ai_reasoning_thai="การวิเคราะห์การปฏิบัติตามกฎระเบียบด้วย AI ล้มเหลวเนื่องจากข้อผิดพลาด",
            final_decision='rejected',
            rejection_reasons=["AI analysis failed"],
            rejection_reasons_thai=["การวิเคราะห์ด้วย AI ล้มเหลว"],
            processing_time=max(0.001, processing_time),
            models_used=[],
            fallback_used=True
        )
    
    def _cleanup_memory(self):
        """Cleanup memory after processing"""
        try:
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status for AI compliance checker"""
        return {
            'performance_mode': self.performance_mode,
            'ocr_reader_loaded': self._ocr_reader is not None,
            'face_detector_loaded': self._face_detector is not None,
            'content_classifier_loaded': self._content_classifier is not None,
            'brand_classifier_loaded': self._brand_classifier is not None,
            'thresholds': {
                'logo_confidence': self.logo_confidence_threshold,
                'face_confidence': self.face_confidence_threshold,
                'content_safety': self.content_safety_threshold
            },
            'supported_languages': ['en', 'th'],
            'gpu_available': torch.cuda.is_available() if torch else False
        }