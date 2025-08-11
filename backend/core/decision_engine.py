"""
Decision Engine Module for Adobe Stock Image Processor

This module implements the decision engine and result aggregation system:
- Scoring algorithms that combine all analysis results
- Decision logic based on configurable thresholds
- Result aggregation and final approval/rejection determination
- Rejection reason tracking and categorization
- Comprehensive result data structures and validation
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

from backend.core.base import (
    ProcessingResult, QualityResult, DefectResult, 
    ComplianceResult, BaseProcessor
)

logger = logging.getLogger(__name__)


class DecisionCategory(Enum):
    """Categories for decision reasons"""
    QUALITY = "quality"
    DEFECTS = "defects"
    SIMILARITY = "similarity"
    COMPLIANCE = "compliance"
    TECHNICAL = "technical"


@dataclass
class RejectionReason:
    """Detailed rejection reason with category and severity"""
    category: DecisionCategory
    reason: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    score: float
    threshold: float
    description: str


@dataclass
class DecisionScores:
    """Comprehensive scoring breakdown for decision making"""
    quality_score: float = 0.0
    defect_score: float = 0.0
    similarity_score: float = 0.0
    compliance_score: float = 0.0
    technical_score: float = 0.0
    overall_score: float = 0.0
    weighted_score: float = 0.0


@dataclass
class DecisionResult:
    """Comprehensive decision result with detailed breakdown"""
    image_path: str
    filename: str
    decision: str  # 'approved', 'rejected', 'review_required'
    confidence: float
    scores: DecisionScores
    rejection_reasons: List[RejectionReason] = field(default_factory=list)
    approval_factors: List[str] = field(default_factory=list)
    recommendation: str = ""
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AggregatedResults:
    """Aggregated results for batch processing"""
    total_images: int
    approved_count: int
    rejected_count: int
    review_required_count: int
    approval_rate: float
    avg_quality_score: float
    avg_overall_score: float
    rejection_breakdown: Dict[str, int]
    top_rejection_reasons: List[Tuple[str, int]]
    processing_statistics: Dict[str, Any]


class DecisionEngine(BaseProcessor):
    """
    Decision engine for Adobe Stock image approval/rejection
    
    This class implements comprehensive decision logic that combines
    results from all analyzers to make final approval decisions.
    """
    
    def __init__(self, config):
        """
        Initialize DecisionEngine with configuration
        
        Args:
            config: Configuration dictionary or AppConfig object with decision thresholds
        """
        super().__init__(config if isinstance(config, dict) else {})
        
        # Handle both dictionary and AppConfig object
        if hasattr(config, 'decision'):
            # AppConfig object
            decision_config = config.decision
            self.weights = {
                'quality': decision_config.quality_weight,
                'defects': decision_config.defect_weight,
                'similarity': decision_config.similarity_weight,
                'compliance': decision_config.compliance_weight,
                'technical': decision_config.technical_weight
            }
            
            self.thresholds = {
                'approval_threshold': decision_config.approval_threshold,
                'rejection_threshold': decision_config.rejection_threshold,
                'quality_min': decision_config.quality_min_threshold,
                'defect_max': decision_config.defect_max_threshold,
                'similarity_max': decision_config.similarity_max_threshold,
                'compliance_min': decision_config.compliance_min_threshold
            }
            
            self.critical_thresholds = {
                'quality_critical': decision_config.quality_critical_threshold,
                'defect_critical': decision_config.defect_critical_threshold,
                'compliance_critical': decision_config.compliance_critical_threshold
            }
        else:
            # Dictionary config
            decision_config = config.get('decision', {})
            
            # Score weights for different analysis components
            self.weights = {
                'quality': decision_config.get('quality_weight', 0.35),
                'defects': decision_config.get('defect_weight', 0.25),
                'similarity': decision_config.get('similarity_weight', 0.20),
                'compliance': decision_config.get('compliance_weight', 0.15),
                'technical': decision_config.get('technical_weight', 0.05)
            }
            
            # Decision thresholds
            self.thresholds = {
                'approval_threshold': decision_config.get('approval_threshold', 0.75),
                'rejection_threshold': decision_config.get('rejection_threshold', 0.40),
                'quality_min': decision_config.get('quality_min_threshold', 0.60),
                'defect_max': decision_config.get('defect_max_threshold', 0.30),
                'similarity_max': decision_config.get('similarity_max_threshold', 0.85),
                'compliance_min': decision_config.get('compliance_min_threshold', 0.80)
            }
            
            # Critical failure thresholds (automatic rejection)
            self.critical_thresholds = {
                'quality_critical': decision_config.get('quality_critical_threshold', 0.30),
                'defect_critical': decision_config.get('defect_critical_threshold', 0.70),
                'compliance_critical': decision_config.get('compliance_critical_threshold', 0.50)
            }
        
        logger.info(f"DecisionEngine initialized with weights: {self.weights}")
        logger.info(f"Decision thresholds: {self.thresholds}")
    
    def process(self, processing_result: ProcessingResult) -> DecisionResult:
        """
        Process a single image result and make approval decision
        
        Args:
            processing_result: ProcessingResult from all analyzers
            
        Returns:
            DecisionResult with final decision and detailed breakdown
        """
        start_time = datetime.now()
        
        try:
            # Calculate individual component scores
            scores = self._calculate_scores(processing_result)
            
            # Check for critical failures first
            critical_reasons = self._check_critical_failures(processing_result, scores)
            
            if critical_reasons:
                # Automatic rejection due to critical failures
                decision = 'rejected'
                confidence = 0.95
                rejection_reasons = critical_reasons
                approval_factors = []
                recommendation = "Image rejected due to critical quality or compliance issues"
            else:
                # Normal decision logic
                decision, confidence, rejection_reasons, approval_factors, recommendation = \
                    self._make_decision(processing_result, scores)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create decision result
            result = DecisionResult(
                image_path=processing_result.image_path,
                filename=processing_result.filename,
                decision=decision,
                confidence=confidence,
                scores=scores,
                rejection_reasons=rejection_reasons,
                approval_factors=approval_factors,
                recommendation=recommendation,
                processing_time=processing_time
            )
            
            logger.debug(f"Decision for {processing_result.filename}: {decision} "
                        f"(confidence: {confidence:.3f}, score: {scores.overall_score:.3f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in decision processing for {processing_result.image_path}: {e}")
            
            # Return failed decision result
            return DecisionResult(
                image_path=processing_result.image_path,
                filename=processing_result.filename,
                decision='rejected',
                confidence=1.0,
                scores=DecisionScores(),
                rejection_reasons=[RejectionReason(
                    category=DecisionCategory.TECHNICAL,
                    reason="processing_error",
                    severity="critical",
                    score=0.0,
                    threshold=1.0,
                    description=f"Processing error: {str(e)}"
                )],
                recommendation="Image rejected due to processing error"
            )
    
    def _calculate_scores(self, result: ProcessingResult) -> DecisionScores:
        """
        Calculate comprehensive scores from all analysis results
        
        Args:
            result: ProcessingResult with all analyzer outputs
            
        Returns:
            DecisionScores with detailed scoring breakdown
        """
        scores = DecisionScores()
        
        try:
            # Quality score (0.0 to 1.0, higher is better)
            if result.quality_result:
                scores.quality_score = result.quality_result.overall_score
            
            # Defect score (0.0 to 1.0, lower is better - invert for consistency)
            if result.defect_result:
                scores.defect_score = max(0.0, 1.0 - result.defect_result.anomaly_score)
            
            # Similarity score (0.0 to 1.0, higher means less similar/duplicate)
            if result.similarity_group is not None:
                # If in similarity group, penalize based on group size
                # This is a simplified approach - in practice, you'd get this from SimilarityFinder
                scores.similarity_score = 0.5 if result.similarity_group > 0 else 1.0
            else:
                scores.similarity_score = 1.0
            
            # Compliance score (0.0 to 1.0, higher is better)
            if result.compliance_result:
                scores.compliance_score = self._calculate_compliance_score(result.compliance_result)
            
            # Technical score (based on file size, resolution, etc.)
            scores.technical_score = self._calculate_technical_score(result)
            
            # Overall score (simple average)
            component_scores = [
                scores.quality_score,
                scores.defect_score,
                scores.similarity_score,
                scores.compliance_score,
                scores.technical_score
            ]
            scores.overall_score = sum(component_scores) / len(component_scores)
            
            # Weighted score (using configured weights)
            scores.weighted_score = (
                self.weights['quality'] * scores.quality_score +
                self.weights['defects'] * scores.defect_score +
                self.weights['similarity'] * scores.similarity_score +
                self.weights['compliance'] * scores.compliance_score +
                self.weights['technical'] * scores.technical_score
            )
            
        except Exception as e:
            logger.error(f"Error calculating scores: {e}")
        
        return scores
    
    def _calculate_compliance_score(self, compliance_result: ComplianceResult) -> float:
        """Calculate compliance score from ComplianceResult"""
        try:
            if compliance_result.overall_compliance:
                return 1.0
            
            # Calculate penalty based on violations
            penalty = 0.0
            
            # Logo detections penalty
            penalty += len(compliance_result.logo_detections) * 0.3
            
            # Privacy violations penalty
            penalty += len(compliance_result.privacy_violations) * 0.4
            
            # Metadata issues penalty
            penalty += len(compliance_result.metadata_issues) * 0.1
            
            # Keyword relevance penalty
            penalty += max(0, 0.5 - compliance_result.keyword_relevance) * 0.5
            
            # Calculate final score
            score = max(0.0, 1.0 - penalty)
            return score
            
        except Exception as e:
            logger.error(f"Error calculating compliance score: {e}")
            return 0.0
    
    def _calculate_technical_score(self, result: ProcessingResult) -> float:
        """Calculate technical score based on file properties"""
        try:
            score = 1.0
            
            # Check file size (penalize very small files)
            if result.quality_result and result.quality_result.file_size:
                file_size_mb = result.quality_result.file_size / (1024 * 1024)
                if file_size_mb < 0.5:  # Less than 500KB
                    score -= 0.3
                elif file_size_mb < 1.0:  # Less than 1MB
                    score -= 0.1
            
            # Check resolution
            if result.quality_result and result.quality_result.resolution:
                width, height = result.quality_result.resolution
                pixel_count = width * height
                
                # Penalize low resolution images
                if pixel_count < 1920 * 1080:  # Less than Full HD
                    score -= 0.4
                elif pixel_count < 2560 * 1440:  # Less than QHD
                    score -= 0.2
            
            return max(0.0, score)
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 0.5
    
    def _check_critical_failures(self, result: ProcessingResult, 
                                scores: DecisionScores) -> List[RejectionReason]:
        """
        Check for critical failures that result in automatic rejection
        
        Args:
            result: ProcessingResult to check
            scores: Calculated scores
            
        Returns:
            List of critical rejection reasons
        """
        critical_reasons = []
        
        try:
            # Critical quality failure
            if scores.quality_score < self.critical_thresholds['quality_critical']:
                critical_reasons.append(RejectionReason(
                    category=DecisionCategory.QUALITY,
                    reason="critical_quality_failure",
                    severity="critical",
                    score=scores.quality_score,
                    threshold=self.critical_thresholds['quality_critical'],
                    description=f"Image quality score {scores.quality_score:.3f} below critical threshold {self.critical_thresholds['quality_critical']}"
                ))
            
            # Critical defect failure
            if result.defect_result and result.defect_result.anomaly_score > self.critical_thresholds['defect_critical']:
                critical_reasons.append(RejectionReason(
                    category=DecisionCategory.DEFECTS,
                    reason="critical_defect_level",
                    severity="critical",
                    score=result.defect_result.anomaly_score,
                    threshold=self.critical_thresholds['defect_critical'],
                    description=f"Defect anomaly score {result.defect_result.anomaly_score:.3f} exceeds critical threshold"
                ))
            
            # Critical compliance failure
            if scores.compliance_score < self.critical_thresholds['compliance_critical']:
                critical_reasons.append(RejectionReason(
                    category=DecisionCategory.COMPLIANCE,
                    reason="critical_compliance_failure",
                    severity="critical",
                    score=scores.compliance_score,
                    threshold=self.critical_thresholds['compliance_critical'],
                    description=f"Compliance score {scores.compliance_score:.3f} below critical threshold"
                ))
            
        except Exception as e:
            logger.error(f"Error checking critical failures: {e}")
        
        return critical_reasons
    
    def _make_decision(self, result: ProcessingResult, 
                      scores: DecisionScores) -> Tuple[str, float, List[RejectionReason], List[str], str]:
        """
        Make final approval/rejection decision based on scores and thresholds
        
        Args:
            result: ProcessingResult to evaluate
            scores: Calculated scores
            
        Returns:
            Tuple of (decision, confidence, rejection_reasons, approval_factors, recommendation)
        """
        rejection_reasons = []
        approval_factors = []
        
        try:
            # Check individual component thresholds
            
            # Quality check
            if scores.quality_score < self.thresholds['quality_min']:
                rejection_reasons.append(RejectionReason(
                    category=DecisionCategory.QUALITY,
                    reason="low_quality_score",
                    severity="medium" if scores.quality_score > 0.4 else "high",
                    score=scores.quality_score,
                    threshold=self.thresholds['quality_min'],
                    description=f"Quality score {scores.quality_score:.3f} below minimum threshold {self.thresholds['quality_min']}"
                ))
            else:
                approval_factors.append(f"Good quality score: {scores.quality_score:.3f}")
            
            # Defect check
            defect_score_inverted = 1.0 - scores.defect_score  # Convert back to anomaly score for threshold comparison
            if defect_score_inverted > self.thresholds['defect_max']:
                rejection_reasons.append(RejectionReason(
                    category=DecisionCategory.DEFECTS,
                    reason="high_defect_level",
                    severity="medium" if defect_score_inverted < 0.5 else "high",
                    score=defect_score_inverted,
                    threshold=self.thresholds['defect_max'],
                    description=f"Defect level {defect_score_inverted:.3f} exceeds maximum threshold {self.thresholds['defect_max']}"
                ))
            else:
                approval_factors.append(f"Low defect level: {defect_score_inverted:.3f}")
            
            # Similarity check
            if scores.similarity_score < (1.0 - self.thresholds['similarity_max']):
                rejection_reasons.append(RejectionReason(
                    category=DecisionCategory.SIMILARITY,
                    reason="high_similarity_detected",
                    severity="medium",
                    score=1.0 - scores.similarity_score,
                    threshold=self.thresholds['similarity_max'],
                    description=f"Image appears to be similar/duplicate to existing images"
                ))
            else:
                approval_factors.append("Unique image with low similarity to others")
            
            # Compliance check
            if scores.compliance_score < self.thresholds['compliance_min']:
                rejection_reasons.append(RejectionReason(
                    category=DecisionCategory.COMPLIANCE,
                    reason="compliance_issues",
                    severity="high",
                    score=scores.compliance_score,
                    threshold=self.thresholds['compliance_min'],
                    description=f"Compliance score {scores.compliance_score:.3f} below minimum threshold"
                ))
            else:
                approval_factors.append(f"Good compliance score: {scores.compliance_score:.3f}")
            
            # Make final decision based on weighted score and rejection reasons
            if scores.weighted_score >= self.thresholds['approval_threshold'] and not rejection_reasons:
                decision = 'approved'
                confidence = min(0.95, scores.weighted_score)
                recommendation = "Image approved for Adobe Stock submission"
                
            elif scores.weighted_score <= self.thresholds['rejection_threshold'] or len(rejection_reasons) >= 3:
                decision = 'rejected'
                confidence = min(0.95, 1.0 - scores.weighted_score)
                recommendation = f"Image rejected due to {len(rejection_reasons)} quality/compliance issues"
                
            else:
                decision = 'review_required'
                confidence = 0.6
                recommendation = "Image requires manual review - mixed quality indicators"
            
            return decision, confidence, rejection_reasons, approval_factors, recommendation
            
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            return 'rejected', 1.0, [RejectionReason(
                category=DecisionCategory.TECHNICAL,
                reason="decision_error",
                severity="critical",
                score=0.0,
                threshold=1.0,
                description=f"Error in decision logic: {str(e)}"
            )], [], "Image rejected due to decision processing error"
    
    def aggregate_results(self, decision_results: List[DecisionResult]) -> AggregatedResults:
        """
        Aggregate multiple decision results for batch analysis
        
        Args:
            decision_results: List of DecisionResult objects
            
        Returns:
            AggregatedResults with comprehensive statistics
        """
        if not decision_results:
            return AggregatedResults(
                total_images=0,
                approved_count=0,
                rejected_count=0,
                review_required_count=0,
                approval_rate=0.0,
                avg_quality_score=0.0,
                avg_overall_score=0.0,
                rejection_breakdown={},
                top_rejection_reasons=[],
                processing_statistics={}
            )
        
        try:
            total_images = len(decision_results)
            
            # Count decisions
            approved_count = sum(1 for r in decision_results if r.decision == 'approved')
            rejected_count = sum(1 for r in decision_results if r.decision == 'rejected')
            review_required_count = sum(1 for r in decision_results if r.decision == 'review_required')
            
            # Calculate rates
            approval_rate = approved_count / total_images if total_images > 0 else 0.0
            
            # Calculate average scores
            avg_quality_score = sum(r.scores.quality_score for r in decision_results) / total_images
            avg_overall_score = sum(r.scores.overall_score for r in decision_results) / total_images
            
            # Analyze rejection reasons
            rejection_breakdown = {}
            all_rejection_reasons = []
            
            for result in decision_results:
                for reason in result.rejection_reasons:
                    category = reason.category.value
                    rejection_breakdown[category] = rejection_breakdown.get(category, 0) + 1
                    all_rejection_reasons.append(reason.reason)
            
            # Top rejection reasons
            reason_counts = {}
            for reason in all_rejection_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            top_rejection_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Processing statistics
            processing_times = [r.processing_time for r in decision_results if r.processing_time > 0]
            processing_statistics = {
                'avg_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0.0,
                'min_processing_time': min(processing_times) if processing_times else 0.0,
                'max_processing_time': max(processing_times) if processing_times else 0.0,
                'total_processing_time': sum(processing_times)
            }
            
            return AggregatedResults(
                total_images=total_images,
                approved_count=approved_count,
                rejected_count=rejected_count,
                review_required_count=review_required_count,
                approval_rate=approval_rate,
                avg_quality_score=avg_quality_score,
                avg_overall_score=avg_overall_score,
                rejection_breakdown=rejection_breakdown,
                top_rejection_reasons=top_rejection_reasons,
                processing_statistics=processing_statistics
            )
            
        except Exception as e:
            logger.error(f"Error aggregating results: {e}")
            return AggregatedResults(
                total_images=len(decision_results),
                approved_count=0,
                rejected_count=len(decision_results),
                review_required_count=0,
                approval_rate=0.0,
                avg_quality_score=0.0,
                avg_overall_score=0.0,
                rejection_breakdown={'error': len(decision_results)},
                top_rejection_reasons=[('aggregation_error', len(decision_results))],
                processing_statistics={'error': str(e)}
            )
    
    def validate_decision_result(self, result: DecisionResult) -> List[str]:
        """
        Validate a decision result for consistency and completeness
        
        Args:
            result: DecisionResult to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        try:
            # Check required fields
            if not result.image_path:
                errors.append("Missing image_path")
            
            if not result.filename:
                errors.append("Missing filename")
            
            if result.decision not in ['approved', 'rejected', 'review_required']:
                errors.append(f"Invalid decision: {result.decision}")
            
            # Check confidence range
            if not (0.0 <= result.confidence <= 1.0):
                errors.append(f"Invalid confidence value: {result.confidence}")
            
            # Check score ranges
            scores = result.scores
            score_fields = [
                ('quality_score', scores.quality_score),
                ('defect_score', scores.defect_score),
                ('similarity_score', scores.similarity_score),
                ('compliance_score', scores.compliance_score),
                ('technical_score', scores.technical_score),
                ('overall_score', scores.overall_score),
                ('weighted_score', scores.weighted_score)
            ]
            
            for field_name, score_value in score_fields:
                if not (0.0 <= score_value <= 1.0):
                    errors.append(f"Invalid {field_name}: {score_value}")
            
            # Check rejection reasons consistency
            if result.decision == 'rejected' and not result.rejection_reasons:
                errors.append("Rejected decision must have rejection reasons")
            
            if result.decision == 'approved' and result.rejection_reasons:
                errors.append("Approved decision should not have rejection reasons")
            
            # Validate rejection reasons
            for reason in result.rejection_reasons:
                if reason.severity not in ['low', 'medium', 'high', 'critical']:
                    errors.append(f"Invalid rejection reason severity: {reason.severity}")
                
                if not (0.0 <= reason.score <= 1.0):
                    errors.append(f"Invalid rejection reason score: {reason.score}")
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors
    
    def get_decision_summary(self, result: DecisionResult) -> Dict[str, Any]:
        """
        Get a summary of decision result for reporting
        
        Args:
            result: DecisionResult to summarize
            
        Returns:
            Dictionary with summary information
        """
        return {
            'filename': result.filename,
            'decision': result.decision,
            'confidence': round(result.confidence, 3),
            'overall_score': round(result.scores.overall_score, 3),
            'weighted_score': round(result.scores.weighted_score, 3),
            'quality_score': round(result.scores.quality_score, 3),
            'defect_score': round(result.scores.defect_score, 3),
            'similarity_score': round(result.scores.similarity_score, 3),
            'compliance_score': round(result.scores.compliance_score, 3),
            'rejection_count': len(result.rejection_reasons),
            'main_rejection_reasons': [r.reason for r in result.rejection_reasons[:3]],
            'recommendation': result.recommendation,
            'processing_time': round(result.processing_time, 4)
        }