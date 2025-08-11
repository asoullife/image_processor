"""
Unified AI Analyzer for Adobe Stock Image Processor

This module provides a unified interface for all AI-enhanced analysis:
- Integrates AI Quality Analyzer, AI Defect Detector, and AI Similarity Finder
- Manages performance modes (Speed/Balanced/Smart) across all AI components
- Provides fallback mechanisms when AI models are unavailable
- Optimizes memory usage and GPU acceleration for RTX2060
- Coordinates batch processing for maximum efficiency
"""

import os
import logging
import gc
import time
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .ai_model_manager import AIModelManager
try:
    from .ai_quality_analyzer import AIQualityAnalyzer, AIQualityResult
    from .ai_defect_detector import AIDefectDetector, AIDefectResult
    from .ai_similarity_finder import AISimilarityFinder, AIGroupResult
    from .ai_compliance_checker import AIComplianceChecker, AIComplianceResult
except ImportError:
    # Fallback for testing without full backend structure
    AIQualityAnalyzer = None
    AIQualityResult = None
    AIDefectDetector = None
    AIDefectResult = None
    AISimilarityFinder = None
    AIGroupResult = None
    AIComplianceChecker = None
    AIComplianceResult = None

logger = logging.getLogger(__name__)


@dataclass
class UnifiedAIResult:
    """Unified result from all AI analyzers"""
    image_path: str
    quality_result: AIQualityResult
    defect_result: AIDefectResult
    compliance_result: AIComplianceResult
    overall_score: float
    final_decision: str  # 'approved', 'rejected'
    rejection_reasons: List[str]
    confidence_score: float
    processing_time: float
    ai_reasoning: str
    ai_reasoning_thai: str
    models_used: List[str]
    fallback_used: bool


@dataclass
class BatchProcessingResult:
    """Result from batch processing multiple images"""
    total_images: int
    processed_images: int
    approved_images: int
    rejected_images: int
    individual_results: List[UnifiedAIResult]
    similarity_analysis: Optional[AIGroupResult]
    batch_statistics: Dict[str, Any]
    total_processing_time: float
    average_time_per_image: float
    models_used: List[str]
    fallback_used: bool


class UnifiedAIAnalyzer:
    """
    Unified AI Analyzer coordinating all AI-enhanced analysis components
    
    Features:
    - Integrates quality analysis, defect detection, and similarity finding
    - Manages performance modes across all AI components
    - Optimizes batch processing for memory efficiency
    - Provides comprehensive fallback mechanisms
    - Coordinates GPU memory usage for RTX2060
    - Generates unified decisions and reasoning
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Unified AI Analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize AI model manager (shared across all analyzers)
        self.model_manager = AIModelManager(config)
        
        # Initialize individual AI analyzers
        self.quality_analyzer = AIQualityAnalyzer(config, self.model_manager)
        self.defect_detector = AIDefectDetector(config, self.model_manager)
        self.similarity_finder = AISimilarityFinder(config, self.model_manager)
        self.compliance_checker = AIComplianceChecker(config, self.model_manager)
        
        # Performance settings
        self.performance_mode = "balanced"
        self.batch_size = 8  # Conservative for RTX2060
        self.max_workers = 2  # Limit concurrent processing
        
        # Decision weights for combining AI results
        self.decision_weights = {
            'quality': 0.25,
            'defects': 0.25,
            'compliance': 0.30,
            'confidence': 0.20
        }
        
        # Unified thresholds
        self.unified_thresholds = {
            'approval_threshold': 0.75,
            'rejection_threshold': 0.40,
            'min_confidence_threshold': 0.60
        }
        
        logger.info("UnifiedAIAnalyzer initialized with all AI components")
    
    def set_performance_mode(self, mode: str):
        """
        Set performance mode for all AI components
        
        Args:
            mode: Performance mode ('speed', 'balanced', 'smart')
        """
        self.performance_mode = mode
        
        # Update model manager
        self.model_manager.set_performance_mode(mode)
        
        # Update individual analyzers
        self.quality_analyzer.set_performance_mode(mode)
        self.defect_detector.set_performance_mode(mode)
        self.similarity_finder.set_performance_mode(mode)
        self.compliance_checker.set_performance_mode(mode)
        
        # Update batch processing settings
        mode_settings = {
            'speed': {'batch_size': 16, 'max_workers': 3},
            'balanced': {'batch_size': 8, 'max_workers': 2},
            'smart': {'batch_size': 4, 'max_workers': 1}
        }
        
        settings = mode_settings.get(mode, mode_settings['balanced'])
        self.batch_size = settings['batch_size']
        self.max_workers = settings['max_workers']
        
        logger.info(f"Performance mode set to: {mode} across all AI components")
    
    def analyze_single_image(self, image_path: str, metadata: Optional[Dict[str, Any]] = None) -> UnifiedAIResult:
        """
        Perform unified AI analysis on a single image
        
        Args:
            image_path: Path to the image file
            metadata: Optional metadata dictionary
            
        Returns:
            UnifiedAIResult with comprehensive analysis
        """
        start_time = time.time()
        
        try:
            # Perform quality analysis
            quality_result = self.quality_analyzer.analyze(image_path)
            
            # Perform defect detection
            defect_result = self.defect_detector.analyze(image_path)
            
            # Perform compliance checking
            compliance_result = self.compliance_checker.analyze(image_path, metadata)
            
            # Calculate unified scores and decision
            unified_result = self._create_unified_result(
                image_path, quality_result, defect_result, compliance_result, time.time() - start_time
            )
            
            # Memory cleanup
            self._cleanup_memory()
            
            return unified_result
            
        except Exception as e:
            logger.error(f"Error in unified AI analysis for {image_path}: {e}")
            return self._create_error_result(image_path, time.time() - start_time)
    
    def analyze_batch(self, image_paths: List[str], 
                     include_similarity: bool = True) -> BatchProcessingResult:
        """
        Perform unified AI analysis on a batch of images
        
        Args:
            image_paths: List of image file paths
            include_similarity: Whether to include similarity analysis
            
        Returns:
            BatchProcessingResult with comprehensive batch analysis
        """
        start_time = time.time()
        
        logger.info(f"Starting unified AI batch analysis of {len(image_paths)} images")
        
        try:
            # Process images in batches for memory efficiency
            individual_results = []
            
            for i in range(0, len(image_paths), self.batch_size):
                batch_paths = image_paths[i:i + self.batch_size]
                
                logger.debug(f"Processing unified batch {i//self.batch_size + 1}: {len(batch_paths)} images")
                
                # Process batch with threading for I/O operations
                batch_results = self._process_batch_threaded(batch_paths)
                individual_results.extend(batch_results)
                
                # Memory cleanup after each batch
                self._cleanup_memory()
                
                logger.debug(f"Completed unified batch {i//self.batch_size + 1}")
            
            # Perform similarity analysis if requested
            similarity_result = None
            if include_similarity and len(image_paths) > 1:
                logger.info("Performing similarity analysis on batch")
                similarity_result = self.similarity_finder.analyze_similarity(image_paths)
            
            # Calculate batch statistics
            batch_result = self._create_batch_result(
                image_paths, individual_results, similarity_result, time.time() - start_time
            )
            
            logger.info(f"Unified batch analysis complete: {batch_result.approved_images} approved, "
                       f"{batch_result.rejected_images} rejected")
            
            return batch_result
            
        except Exception as e:
            logger.error(f"Error in unified batch analysis: {e}")
            return self._create_error_batch_result(image_paths, time.time() - start_time)
    
    def _process_batch_threaded(self, batch_paths: List[str]) -> List[UnifiedAIResult]:
        """
        Process a batch of images using threading for I/O operations
        
        Args:
            batch_paths: List of image paths in the batch
            
        Returns:
            List of UnifiedAIResult objects
        """
        results = []
        
        try:
            # Use ThreadPoolExecutor for I/O bound operations
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(self.analyze_single_image, path): path 
                    for path in batch_paths
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {path}: {e}")
                        # Create error result
                        error_result = self._create_error_result(path, 0.0)
                        results.append(error_result)
            
        except Exception as e:
            logger.error(f"Error in threaded batch processing: {e}")
            # Fallback to sequential processing
            for path in batch_paths:
                try:
                    result = self.analyze_single_image(path)
                    results.append(result)
                except Exception as path_error:
                    logger.error(f"Error processing {path}: {path_error}")
                    error_result = self._create_error_result(path, 0.0)
                    results.append(error_result)
        
        return results
    
    def _create_unified_result(self, image_path: str, quality_result: AIQualityResult,
                              defect_result: AIDefectResult, compliance_result: AIComplianceResult, 
                              processing_time: float) -> UnifiedAIResult:
        """
        Create unified result from individual AI analyzer results
        
        Args:
            image_path: Path to the image file
            quality_result: AI quality analysis result
            defect_result: AI defect detection result
            compliance_result: AI compliance analysis result
            processing_time: Total processing time
            
        Returns:
            UnifiedAIResult with combined analysis
        """
        try:
            # Calculate overall score using weighted combination
            quality_score = quality_result.overall_ai_score
            defect_score = 1.0 - defect_result.ai_anomaly_score  # Invert anomaly score
            compliance_score = compliance_result.ai_compliance_score
            confidence_score = (
                quality_result.ai_confidence + 
                defect_result.ai_confidence + 
                compliance_result.ai_confidence
            ) / 3.0
            
            overall_score = (
                self.decision_weights['quality'] * quality_score +
                self.decision_weights['defects'] * defect_score +
                self.decision_weights['compliance'] * compliance_score +
                self.decision_weights['confidence'] * confidence_score
            )
            
            # Make unified decision
            final_decision, rejection_reasons = self._make_unified_decision(
                overall_score, confidence_score, quality_result, defect_result, compliance_result
            )
            
            # Generate unified reasoning
            ai_reasoning, ai_reasoning_thai = self._generate_unified_reasoning(
                quality_result, defect_result, compliance_result, overall_score, final_decision
            )
            
            # Collect models used
            models_used = []
            if not quality_result.fallback_used:
                models_used.append(quality_result.model_used)
            if not defect_result.fallback_used:
                models_used.append(defect_result.model_used)
            if not compliance_result.fallback_used:
                models_used.extend(compliance_result.models_used)
            
            fallback_used = (quality_result.fallback_used and 
                           defect_result.fallback_used and 
                           compliance_result.fallback_used)
            
            return UnifiedAIResult(
                image_path=image_path,
                quality_result=quality_result,
                defect_result=defect_result,
                compliance_result=compliance_result,
                overall_score=overall_score,
                final_decision=final_decision,
                rejection_reasons=rejection_reasons,
                confidence_score=confidence_score,
                processing_time=processing_time,
                ai_reasoning=ai_reasoning,
                ai_reasoning_thai=ai_reasoning_thai,
                models_used=models_used,
                fallback_used=fallback_used
            )
            
        except Exception as e:
            logger.error(f"Error creating unified result: {e}")
            return self._create_error_result(image_path, processing_time)
    
    def _make_unified_decision(self, overall_score: float, confidence_score: float,
                              quality_result: AIQualityResult, 
                              defect_result: AIDefectResult,
                              compliance_result: AIComplianceResult) -> Tuple[str, List[str]]:
        """
        Make unified decision based on all AI analysis results
        
        Args:
            overall_score: Combined overall score
            confidence_score: Combined confidence score
            quality_result: Quality analysis result
            defect_result: Defect detection result
            compliance_result: Compliance analysis result
            
        Returns:
            Tuple of (decision, rejection_reasons)
        """
        try:
            rejection_reasons = []
            
            # Check confidence threshold
            if confidence_score < self.unified_thresholds['min_confidence_threshold']:
                rejection_reasons.append("AI ไม่มั่นใจในการประเมิน ควรตรวจสอบด้วยตนเอง")
            
            # Collect rejection reasons from individual analyzers
            if quality_result.final_decision == 'rejected':
                rejection_reasons.extend(quality_result.rejection_reasons)
            
            if defect_result.final_decision == 'rejected':
                rejection_reasons.extend(defect_result.rejection_reasons)
            
            if compliance_result.final_decision == 'rejected':
                rejection_reasons.extend(compliance_result.rejection_reasons_thai)
            
            # Check for critical compliance issues (override other decisions)
            critical_compliance_issues = [
                reason for reason in compliance_result.rejection_reasons_thai
                if any(critical in reason for critical in ['วิกฤต', 'ความเสี่ยงสูง', 'เครื่องหมายการค้า'])
            ]
            
            if critical_compliance_issues:
                decision = 'rejected'
                rejection_reasons = critical_compliance_issues
                return decision, rejection_reasons
            
            # Check overall score thresholds
            if overall_score >= self.unified_thresholds['approval_threshold']:
                # High score - approve if no critical issues
                critical_issues = [
                    reason for reason in rejection_reasons 
                    if any(critical in reason for critical in ['รอยแตก', 'ความเสียหาย', 'วิกฤต', 'ความเสี่ยงสูง'])
                ]
                
                if critical_issues:
                    decision = 'rejected'
                    rejection_reasons = critical_issues
                else:
                    decision = 'approved'
                    rejection_reasons = []
                    
            elif overall_score <= self.unified_thresholds['rejection_threshold']:
                # Low score - reject
                decision = 'rejected'
                if not rejection_reasons:
                    rejection_reasons = ["คุณภาพโดยรวมต่ำกว่ามาตรฐาน Adobe Stock"]
                    
            else:
                # Medium score - depends on specific issues
                if rejection_reasons:
                    decision = 'rejected'
                else:
                    decision = 'approved'
            
            # Remove duplicates
            rejection_reasons = list(set(rejection_reasons))
            
            return decision, rejection_reasons
            
        except Exception as e:
            logger.error(f"Error making unified decision: {e}")
            return 'rejected', ["เกิดข้อผิดพลาดในการประเมิน"]
    
    def _generate_unified_reasoning(self, quality_result: AIQualityResult,
                                   defect_result: AIDefectResult, compliance_result: AIComplianceResult,
                                   overall_score: float, final_decision: str) -> Tuple[str, str]:
        """
        Generate unified reasoning combining all AI analysis results
        
        Args:
            quality_result: Quality analysis result
            defect_result: Defect detection result
            compliance_result: Compliance analysis result
            overall_score: Combined overall score
            final_decision: Final decision
            
        Returns:
            Tuple of (English reasoning, Thai reasoning)
        """
        try:
            # English reasoning
            reasoning_parts_en = [
                f"Unified AI analysis completed with overall score: {overall_score:.2f}",
                f"Quality analysis: {quality_result.ai_reasoning}",
                f"Defect detection: {defect_result.ai_reasoning}",
                f"Compliance check: {compliance_result.ai_reasoning}",
                f"Final decision: {final_decision}"
            ]
            
            reasoning_en = ". ".join(reasoning_parts_en) + "."
            
            # Thai reasoning
            reasoning_parts_th = [
                f"การวิเคราะห์ด้วย AI แบบรวมเสร็จสิ้น คะแนนรวม: {overall_score:.2f}",
                f"การวิเคราะห์คุณภาพ: {quality_result.ai_reasoning_thai}",
                f"การตรวจจับข้อบกพร่อง: {defect_result.ai_reasoning_thai}",
                f"การตรวจสอบการปฏิบัติตามกฎระเบียบ: {compliance_result.ai_reasoning_thai}"
            ]
            
            if final_decision == 'approved':
                reasoning_parts_th.append("ผลการตัดสินใจ: อนุมัติสำหรับการขาย")
            else:
                reasoning_parts_th.append("ผลการตัดสินใจ: ไม่อนุมัติสำหรับการขาย")
            
            reasoning_th = " ".join(reasoning_parts_th)
            
            return reasoning_en, reasoning_th
            
        except Exception as e:
            logger.error(f"Error generating unified reasoning: {e}")
            return "Unified AI analysis completed", "การวิเคราะห์ด้วย AI แบบรวมเสร็จสิ้น"
    
    def _create_batch_result(self, image_paths: List[str], individual_results: List[UnifiedAIResult],
                            similarity_result: Optional[AIGroupResult], 
                            total_time: float) -> BatchProcessingResult:
        """
        Create batch processing result from individual results
        
        Args:
            image_paths: List of image paths
            individual_results: List of individual analysis results
            similarity_result: Similarity analysis result
            total_time: Total processing time
            
        Returns:
            BatchProcessingResult with batch statistics
        """
        try:
            # Calculate statistics
            total_images = len(image_paths)
            processed_images = len(individual_results)
            approved_images = len([r for r in individual_results if r.final_decision == 'approved'])
            rejected_images = len([r for r in individual_results if r.final_decision == 'rejected'])
            
            # Collect all models used
            all_models = set()
            fallback_count = 0
            
            for result in individual_results:
                all_models.update(result.models_used)
                if result.fallback_used:
                    fallback_count += 1
            
            # Calculate batch statistics
            batch_statistics = {
                'approval_rate': approved_images / processed_images if processed_images > 0 else 0.0,
                'rejection_rate': rejected_images / processed_images if processed_images > 0 else 0.0,
                'average_overall_score': sum(r.overall_score for r in individual_results) / processed_images if processed_images > 0 else 0.0,
                'average_confidence': sum(r.confidence_score for r in individual_results) / processed_images if processed_images > 0 else 0.0,
                'fallback_usage_rate': fallback_count / processed_images if processed_images > 0 else 0.0,
                'models_used_count': len(all_models)
            }
            
            # Add similarity statistics if available
            if similarity_result:
                batch_statistics.update({
                    'similarity_clusters': similarity_result.traditional_result.total_groups,
                    'duplicate_groups': len(similarity_result.traditional_result.duplicate_groups),
                    'similar_groups': len(similarity_result.traditional_result.similar_groups),
                    'recommended_removes': len(similarity_result.traditional_result.recommended_removes)
                })
            
            return BatchProcessingResult(
                total_images=total_images,
                processed_images=processed_images,
                approved_images=approved_images,
                rejected_images=rejected_images,
                individual_results=individual_results,
                similarity_analysis=similarity_result,
                batch_statistics=batch_statistics,
                total_processing_time=total_time,
                average_time_per_image=total_time / processed_images if processed_images > 0 else 0.0,
                models_used=list(all_models),
                fallback_used=fallback_count > 0
            )
            
        except Exception as e:
            logger.error(f"Error creating batch result: {e}")
            return self._create_error_batch_result(image_paths, total_time)
    
    def _create_error_result(self, image_path: str, processing_time: float) -> UnifiedAIResult:
        """Create error result when analysis fails"""
        from .ai_quality_analyzer import AIQualityResult
        from .ai_defect_detector import AIDefectResult
        from .ai_compliance_checker import AIComplianceResult, ContentAppropriateness
        from .quality_analyzer import QualityResult
        from .compliance_checker import ComplianceResult
        from ..core.base import DefectResult
        
        # Create minimal error results
        error_quality = QualityResult(0.0, 1.0, 0.0, 0.0, (0, 0), 0, 0.0, False)
        error_defect = DefectResult([], 1.0, 0, [], [], False)
        error_compliance = ComplianceResult([], [], ["Analysis failed"], 0.0, False)
        
        error_ai_quality = AIQualityResult(
            traditional_result=error_quality,
            ai_sharpness_score=0.0, ai_noise_score=0.0, ai_aesthetic_score=0.0,
            ai_technical_score=0.0, ai_confidence=0.0,
            ai_reasoning="Analysis failed", ai_reasoning_thai="การวิเคราะห์ล้มเหลว",
            overall_ai_score=0.0, final_decision='rejected',
            rejection_reasons=["เกิดข้อผิดพลาดในการวิเคราะห์"],
            processing_time=processing_time, model_used='error', fallback_used=True
        )
        
        error_ai_defect = AIDefectResult(
            traditional_result=error_defect, yolo_detections=[], ai_anomaly_score=1.0,
            ai_confidence=0.0, ai_reasoning="Analysis failed", 
            ai_reasoning_thai="การวิเคราะห์ล้มเหลว", detected_objects=[],
            defect_categories={}, severity_assessment='critical', final_decision='rejected',
            rejection_reasons=["เกิดข้อผิดพลาดในการวิเคราะห์"],
            processing_time=processing_time, model_used='error', fallback_used=True
        )
        
        error_content = ContentAppropriateness(
            overall_score=0.0, cultural_sensitivity_score=0.0, content_safety_score=0.0,
            age_appropriateness='unknown', detected_themes=[], potential_issues=['Analysis failed'],
            cultural_concerns=[]
        )
        
        error_ai_compliance = AIComplianceResult(
            traditional_result=error_compliance, ai_logo_detections=[], ai_privacy_violations=[],
            content_appropriateness=error_content, enhanced_metadata_analysis={'error': True},
            ai_compliance_score=0.0, ai_confidence=0.0,
            ai_reasoning="Analysis failed", ai_reasoning_thai="การวิเคราะห์ล้มเหลว",
            final_decision='rejected', rejection_reasons=["Analysis failed"],
            rejection_reasons_thai=["การวิเคราะห์ล้มเหลว"],
            processing_time=processing_time, models_used=[], fallback_used=True
        )
        
        return UnifiedAIResult(
            image_path=image_path,
            quality_result=error_ai_quality,
            defect_result=error_ai_defect,
            compliance_result=error_ai_compliance,
            overall_score=0.0,
            final_decision='rejected',
            rejection_reasons=["เกิดข้อผิดพลาดในการวิเคราะห์"],
            confidence_score=0.0,
            processing_time=processing_time,
            ai_reasoning="Analysis failed due to error",
            ai_reasoning_thai="การวิเคราะห์ล้มเหลวเนื่องจากข้อผิดพลาด",
            models_used=[],
            fallback_used=True
        )
    
    def _create_error_batch_result(self, image_paths: List[str], 
                                  total_time: float) -> BatchProcessingResult:
        """Create error batch result when batch processing fails"""
        return BatchProcessingResult(
            total_images=len(image_paths),
            processed_images=0,
            approved_images=0,
            rejected_images=0,
            individual_results=[],
            similarity_analysis=None,
            batch_statistics={'error': True},
            total_processing_time=total_time,
            average_time_per_image=0.0,
            models_used=[],
            fallback_used=True
        )
    
    def _cleanup_memory(self):
        """Cleanup memory after processing"""
        try:
            # Cleanup individual analyzers
            self.quality_analyzer._cleanup_memory()
            self.defect_detector._cleanup_memory()
            self.similarity_finder._cleanup_memory()
            self.compliance_checker._cleanup_memory()
            
            # Python garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status for all AI components
        
        Returns:
            Status information dictionary
        """
        status = {
            'unified_analyzer': {
                'performance_mode': self.performance_mode,
                'batch_size': self.batch_size,
                'max_workers': self.max_workers,
                'decision_weights': self.decision_weights,
                'unified_thresholds': self.unified_thresholds
            },
            'model_manager': self.model_manager.get_system_status(),
            'quality_analyzer': self.quality_analyzer.get_system_status(),
            'defect_detector': self.defect_detector.get_system_status(),
            'similarity_finder': self.similarity_finder.get_system_status(),
            'compliance_checker': self.compliance_checker.get_system_status()
        }
        
        return status
    
    def preload_models(self):
        """Preload all AI models for faster processing"""
        try:
            logger.info("Preloading AI models...")
            self.model_manager.preload_models(['resnet50', 'yolov8n'])
            self.similarity_finder._load_clip_model()
            logger.info("AI models preloaded successfully")
        except Exception as e:
            logger.error(f"Error preloading models: {e}")
    
    def get_performance_recommendations(self) -> Dict[str, Any]:
        """
        Get performance recommendations based on system capabilities
        
        Returns:
            Performance recommendations dictionary
        """
        system_status = self.model_manager.get_system_status()
        
        recommendations = {
            'recommended_mode': 'balanced',
            'recommended_batch_size': 8,
            'gpu_optimization': False,
            'memory_optimization': True,
            'reasons': []
        }
        
        try:
            # Check GPU availability
            if system_status.get('gpu_available', False):
                recommendations['gpu_optimization'] = True
                recommendations['reasons'].append("GPU available for acceleration")
                
                # Check GPU memory
                gpu_info = system_status.get('gpu_info', {})
                if gpu_info:
                    memory_free = gpu_info.get('memory_free', '0MB')
                    memory_mb = int(memory_free.replace('MB', ''))
                    
                    if memory_mb > 3000:  # > 3GB free
                        recommendations['recommended_mode'] = 'smart'
                        recommendations['recommended_batch_size'] = 4
                        recommendations['reasons'].append("High GPU memory available")
                    elif memory_mb > 1500:  # > 1.5GB free
                        recommendations['recommended_mode'] = 'balanced'
                        recommendations['recommended_batch_size'] = 8
                        recommendations['reasons'].append("Moderate GPU memory available")
                    else:
                        recommendations['recommended_mode'] = 'speed'
                        recommendations['recommended_batch_size'] = 16
                        recommendations['reasons'].append("Limited GPU memory, using speed mode")
            else:
                recommendations['recommended_mode'] = 'speed'
                recommendations['recommended_batch_size'] = 16
                recommendations['reasons'].append("No GPU available, using CPU optimization")
            
            # Check system memory
            system_memory = system_status.get('system_memory', {})
            if system_memory:
                available_gb = int(system_memory.get('available', '0GB').replace('GB', ''))
                if available_gb < 4:
                    recommendations['memory_optimization'] = True
                    recommendations['recommended_batch_size'] = min(4, recommendations['recommended_batch_size'])
                    recommendations['reasons'].append("Low system memory, reducing batch size")
            
        except Exception as e:
            logger.error(f"Error generating performance recommendations: {e}")
            recommendations['reasons'].append("Error analyzing system, using safe defaults")
        
        return recommendations