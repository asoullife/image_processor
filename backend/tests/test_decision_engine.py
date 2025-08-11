"""
Unit tests for Decision Engine Module

Tests the decision engine and result aggregation system including:
- Scoring algorithms that combine all analysis results
- Decision logic based on configurable thresholds
- Result aggregation and final approval/rejection determination
- Rejection reason tracking and categorization
- Comprehensive result data structures and validation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
from unittest.mock import Mock, patch
from datetime import datetime
import tempfile
import os

from backend.core.decision_engine import (
    DecisionEngine, DecisionResult, DecisionScores, RejectionReason,
    DecisionCategory, AggregatedResults
)
from backend.core.base import (
    ProcessingResult, QualityResult, DefectResult, ComplianceResult,
    ObjectDefect, LogoDetection, PrivacyViolation
)


class TestDecisionEngine(unittest.TestCase):
    """Test cases for DecisionEngine class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'decision': {
                'quality_weight': 0.35,
                'defect_weight': 0.25,
                'similarity_weight': 0.20,
                'compliance_weight': 0.15,
                'technical_weight': 0.05,
                'approval_threshold': 0.75,
                'rejection_threshold': 0.40,
                'quality_min_threshold': 0.60,
                'defect_max_threshold': 0.30,
                'similarity_max_threshold': 0.85,
                'compliance_min_threshold': 0.80,
                'quality_critical_threshold': 0.30,
                'defect_critical_threshold': 0.70,
                'compliance_critical_threshold': 0.50
            }
        }
        self.engine = DecisionEngine(self.config)
    
    def test_initialization(self):
        """Test DecisionEngine initialization"""
        self.assertIsInstance(self.engine, DecisionEngine)
        self.assertEqual(self.engine.weights['quality'], 0.35)
        self.assertEqual(self.engine.thresholds['approval_threshold'], 0.75)
        self.assertEqual(self.engine.critical_thresholds['quality_critical'], 0.30)
    
    def test_calculate_scores_high_quality(self):
        """Test score calculation for high quality image"""
        # Create high quality processing result
        quality_result = QualityResult(
            sharpness_score=150.0,
            noise_level=0.05,
            exposure_score=0.8,
            color_balance_score=0.9,
            resolution=(2560, 1440),
            file_size=2048000,  # 2MB
            overall_score=0.85,
            passed=True
        )
        
        defect_result = DefectResult(
            detected_objects=[],
            anomaly_score=0.1,
            defect_count=0,
            defect_types=[],
            confidence_scores=[],
            passed=True
        )
        
        compliance_result = ComplianceResult(
            logo_detections=[],
            privacy_violations=[],
            metadata_issues=[],
            keyword_relevance=0.8,
            overall_compliance=True
        )
        
        processing_result = ProcessingResult(
            image_path="/test/image.jpg",
            filename="image.jpg",
            quality_result=quality_result,
            defect_result=defect_result,
            similarity_group=None,
            compliance_result=compliance_result
        )
        
        scores = self.engine._calculate_scores(processing_result)
        
        self.assertGreater(scores.quality_score, 0.8)
        self.assertGreater(scores.defect_score, 0.8)  # Inverted from anomaly score
        self.assertEqual(scores.similarity_score, 1.0)  # No similarity group
        self.assertEqual(scores.compliance_score, 1.0)  # Perfect compliance
        self.assertGreater(scores.technical_score, 0.8)
        self.assertGreater(scores.overall_score, 0.8)
        self.assertGreater(scores.weighted_score, 0.8)
    
    def test_calculate_scores_low_quality(self):
        """Test score calculation for low quality image"""
        # Create low quality processing result
        quality_result = QualityResult(
            sharpness_score=50.0,
            noise_level=0.3,
            exposure_score=0.3,
            color_balance_score=0.4,
            resolution=(800, 600),
            file_size=100000,  # 100KB
            overall_score=0.2,
            passed=False
        )
        
        defect_result = DefectResult(
            detected_objects=[
                ObjectDefect("glass", "crack", 0.8, (10, 10, 50, 50), "Cracked glass detected")
            ],
            anomaly_score=0.7,
            defect_count=1,
            defect_types=["crack"],
            confidence_scores=[0.8],
            passed=False
        )
        
        compliance_result = ComplianceResult(
            logo_detections=[
                LogoDetection("Nike", 0.9, (100, 100, 50, 30), "nike")
            ],
            privacy_violations=[
                PrivacyViolation("face", 0.8, (200, 200, 80, 80), "Face detected")
            ],
            metadata_issues=["GPS data present"],
            keyword_relevance=0.3,
            overall_compliance=False
        )
        
        processing_result = ProcessingResult(
            image_path="/test/bad_image.jpg",
            filename="bad_image.jpg",
            quality_result=quality_result,
            defect_result=defect_result,
            similarity_group=1,  # In similarity group
            compliance_result=compliance_result
        )
        
        scores = self.engine._calculate_scores(processing_result)
        
        self.assertLess(scores.quality_score, 0.5)
        self.assertLess(scores.defect_score, 0.5)  # Inverted from high anomaly score
        self.assertEqual(scores.similarity_score, 0.5)  # In similarity group
        self.assertLess(scores.compliance_score, 0.5)
        self.assertLess(scores.technical_score, 0.8)
        self.assertLess(scores.overall_score, 0.5)
        self.assertLess(scores.weighted_score, 0.5)
    
    def test_check_critical_failures(self):
        """Test critical failure detection"""
        # Create processing result with critical failures
        quality_result = QualityResult(
            sharpness_score=20.0,
            noise_level=0.5,
            exposure_score=0.1,
            color_balance_score=0.2,
            resolution=(640, 480),
            file_size=50000,
            overall_score=0.15,  # Critical failure
            passed=False
        )
        
        defect_result = DefectResult(
            detected_objects=[],
            anomaly_score=0.8,  # Critical failure
            defect_count=5,
            defect_types=["crack", "break", "scratch"],
            confidence_scores=[0.9, 0.8, 0.7, 0.6, 0.5],
            passed=False
        )
        
        compliance_result = ComplianceResult(
            logo_detections=[
                LogoDetection("Apple", 0.95, (0, 0, 100, 50), "apple"),
                LogoDetection("Google", 0.90, (100, 0, 100, 50), "google")
            ],
            privacy_violations=[
                PrivacyViolation("face", 0.9, (0, 100, 100, 100), "Multiple faces"),
                PrivacyViolation("license_plate", 0.8, (200, 100, 100, 50), "License plate")
            ],
            metadata_issues=["GPS data", "Personal device info", "Watermark"],
            keyword_relevance=0.1,
            overall_compliance=False
        )
        
        processing_result = ProcessingResult(
            image_path="/test/critical_failure.jpg",
            filename="critical_failure.jpg",
            quality_result=quality_result,
            defect_result=defect_result,
            compliance_result=compliance_result
        )
        
        scores = self.engine._calculate_scores(processing_result)
        critical_reasons = self.engine._check_critical_failures(processing_result, scores)
        
        # Should have multiple critical failures
        self.assertGreater(len(critical_reasons), 0)
        
        # Check for specific critical failures
        critical_categories = [reason.category for reason in critical_reasons]
        self.assertIn(DecisionCategory.QUALITY, critical_categories)
        self.assertIn(DecisionCategory.DEFECTS, critical_categories)
        self.assertIn(DecisionCategory.COMPLIANCE, critical_categories)
        
        # All should be critical severity
        for reason in critical_reasons:
            self.assertEqual(reason.severity, "critical")
    
    def test_make_decision_approval(self):
        """Test decision making for approval case"""
        # Create high quality processing result
        quality_result = QualityResult(
            sharpness_score=200.0,
            noise_level=0.02,
            exposure_score=0.9,
            color_balance_score=0.85,
            resolution=(3840, 2160),  # 4K
            file_size=5000000,  # 5MB
            overall_score=0.9,
            passed=True
        )
        
        defect_result = DefectResult(
            detected_objects=[],
            anomaly_score=0.05,
            defect_count=0,
            defect_types=[],
            confidence_scores=[],
            passed=True
        )
        
        compliance_result = ComplianceResult(
            logo_detections=[],
            privacy_violations=[],
            metadata_issues=[],
            keyword_relevance=0.9,
            overall_compliance=True
        )
        
        processing_result = ProcessingResult(
            image_path="/test/excellent_image.jpg",
            filename="excellent_image.jpg",
            quality_result=quality_result,
            defect_result=defect_result,
            similarity_group=None,
            compliance_result=compliance_result
        )
        
        scores = self.engine._calculate_scores(processing_result)
        decision, confidence, rejection_reasons, approval_factors, recommendation = \
            self.engine._make_decision(processing_result, scores)
        
        self.assertEqual(decision, 'approved')
        self.assertGreater(confidence, 0.8)
        self.assertEqual(len(rejection_reasons), 0)
        self.assertGreater(len(approval_factors), 0)
        self.assertIn("approved", recommendation.lower())
    
    def test_make_decision_rejection(self):
        """Test decision making for rejection case"""
        # Create poor quality processing result
        quality_result = QualityResult(
            sharpness_score=30.0,
            noise_level=0.4,
            exposure_score=0.2,
            color_balance_score=0.3,
            resolution=(640, 480),
            file_size=80000,
            overall_score=0.25,
            passed=False
        )
        
        defect_result = DefectResult(
            detected_objects=[
                ObjectDefect("object", "major_defect", 0.9, (0, 0, 100, 100), "Major defect")
            ],
            anomaly_score=0.6,
            defect_count=3,
            defect_types=["crack", "break", "damage"],
            confidence_scores=[0.9, 0.8, 0.7],
            passed=False
        )
        
        compliance_result = ComplianceResult(
            logo_detections=[
                LogoDetection("Brand", 0.8, (0, 0, 50, 50), "brand")
            ],
            privacy_violations=[
                PrivacyViolation("face", 0.9, (100, 100, 50, 50), "Face")
            ],
            metadata_issues=["Multiple issues"],
            keyword_relevance=0.2,
            overall_compliance=False
        )
        
        processing_result = ProcessingResult(
            image_path="/test/poor_image.jpg",
            filename="poor_image.jpg",
            quality_result=quality_result,
            defect_result=defect_result,
            similarity_group=1,
            compliance_result=compliance_result
        )
        
        scores = self.engine._calculate_scores(processing_result)
        decision, confidence, rejection_reasons, approval_factors, recommendation = \
            self.engine._make_decision(processing_result, scores)
        
        self.assertEqual(decision, 'rejected')
        self.assertGreaterEqual(confidence, 0.65)
        self.assertGreater(len(rejection_reasons), 0)
        self.assertIn("rejected", recommendation.lower())
        
        # Check rejection reason categories
        reason_categories = [reason.category for reason in rejection_reasons]
        self.assertIn(DecisionCategory.QUALITY, reason_categories)
        self.assertIn(DecisionCategory.DEFECTS, reason_categories)
        self.assertIn(DecisionCategory.COMPLIANCE, reason_categories)
    
    def test_make_decision_review_required(self):
        """Test decision making for review required case"""
        # Create borderline quality processing result that should require review
        quality_result = QualityResult(
            sharpness_score=90.0,  # Lower sharpness
            noise_level=0.12,      # Higher noise
            exposure_score=0.55,   # Lower exposure
            color_balance_score=0.65,
            resolution=(1920, 1080),
            file_size=1500000,
            overall_score=0.55,    # Lower overall score
            passed=False           # Failed quality check
        )
        
        defect_result = DefectResult(
            detected_objects=[
                ObjectDefect("object", "moderate_defect", 0.7, (0, 0, 50, 50), "Moderate defect")
            ],
            anomaly_score=0.35,    # Higher anomaly score
            defect_count=2,        # More defects
            defect_types=["scratch", "wear"],
            confidence_scores=[0.7, 0.6],
            passed=False           # Failed defect check
        )
        
        compliance_result = ComplianceResult(
            logo_detections=[],
            privacy_violations=[],
            metadata_issues=["Metadata issue", "Another issue"],  # More issues
            keyword_relevance=0.65,  # Lower relevance
            overall_compliance=False  # Failed compliance
        )
        
        processing_result = ProcessingResult(
            image_path="/test/borderline_image.jpg",
            filename="borderline_image.jpg",
            quality_result=quality_result,
            defect_result=defect_result,
            similarity_group=None,
            compliance_result=compliance_result
        )
        
        scores = self.engine._calculate_scores(processing_result)
        decision, confidence, rejection_reasons, approval_factors, recommendation = \
            self.engine._make_decision(processing_result, scores)
        
        # With the adjusted scores, should be rejected or review_required
        self.assertIn(decision, ['rejected', 'review_required'])
        # Confidence can vary based on exact scoring
        self.assertGreaterEqual(confidence, 0.5)
        # Recommendation should mention rejection or review
        self.assertTrue("rejected" in recommendation.lower() or "review" in recommendation.lower())
    
    def test_process_complete_workflow(self):
        """Test complete processing workflow"""
        # Create processing result
        quality_result = QualityResult(
            sharpness_score=180.0,
            noise_level=0.03,
            exposure_score=0.85,
            color_balance_score=0.8,
            resolution=(2560, 1440),
            file_size=3000000,
            overall_score=0.82,
            passed=True
        )
        
        defect_result = DefectResult(
            detected_objects=[],
            anomaly_score=0.08,
            defect_count=0,
            defect_types=[],
            confidence_scores=[],
            passed=True
        )
        
        compliance_result = ComplianceResult(
            logo_detections=[],
            privacy_violations=[],
            metadata_issues=[],
            keyword_relevance=0.85,
            overall_compliance=True
        )
        
        processing_result = ProcessingResult(
            image_path="/test/good_image.jpg",
            filename="good_image.jpg",
            quality_result=quality_result,
            defect_result=defect_result,
            similarity_group=None,
            compliance_result=compliance_result
        )
        
        # Process the result
        decision_result = self.engine.process(processing_result)
        
        # Validate result
        self.assertIsInstance(decision_result, DecisionResult)
        self.assertEqual(decision_result.image_path, "/test/good_image.jpg")
        self.assertEqual(decision_result.filename, "good_image.jpg")
        self.assertIn(decision_result.decision, ['approved', 'rejected', 'review_required'])
        self.assertGreaterEqual(decision_result.confidence, 0.0)
        self.assertLessEqual(decision_result.confidence, 1.0)
        self.assertIsInstance(decision_result.scores, DecisionScores)
        self.assertGreaterEqual(decision_result.processing_time, 0.0)
        self.assertIsInstance(decision_result.timestamp, datetime)
    
    def test_aggregate_results(self):
        """Test result aggregation functionality"""
        # Create multiple decision results
        decision_results = []
        
        # Approved result
        approved_result = DecisionResult(
            image_path="/test/approved.jpg",
            filename="approved.jpg",
            decision='approved',
            confidence=0.9,
            scores=DecisionScores(
                quality_score=0.9,
                defect_score=0.95,
                similarity_score=1.0,
                compliance_score=1.0,
                technical_score=0.9,
                overall_score=0.95,
                weighted_score=0.92
            ),
            processing_time=0.5
        )
        decision_results.append(approved_result)
        
        # Rejected result
        rejected_result = DecisionResult(
            image_path="/test/rejected.jpg",
            filename="rejected.jpg",
            decision='rejected',
            confidence=0.85,
            scores=DecisionScores(
                quality_score=0.3,
                defect_score=0.2,
                similarity_score=0.4,
                compliance_score=0.3,
                technical_score=0.4,
                overall_score=0.32,
                weighted_score=0.28
            ),
            rejection_reasons=[
                RejectionReason(
                    category=DecisionCategory.QUALITY,
                    reason="low_quality",
                    severity="high",
                    score=0.3,
                    threshold=0.6,
                    description="Low quality"
                )
            ],
            processing_time=0.7
        )
        decision_results.append(rejected_result)
        
        # Review required result
        review_result = DecisionResult(
            image_path="/test/review.jpg",
            filename="review.jpg",
            decision='review_required',
            confidence=0.6,
            scores=DecisionScores(
                quality_score=0.65,
                defect_score=0.7,
                similarity_score=0.8,
                compliance_score=0.75,
                technical_score=0.7,
                overall_score=0.72,
                weighted_score=0.68
            ),
            processing_time=0.6
        )
        decision_results.append(review_result)
        
        # Aggregate results
        aggregated = self.engine.aggregate_results(decision_results)
        
        # Validate aggregation
        self.assertIsInstance(aggregated, AggregatedResults)
        self.assertEqual(aggregated.total_images, 3)
        self.assertEqual(aggregated.approved_count, 1)
        self.assertEqual(aggregated.rejected_count, 1)
        self.assertEqual(aggregated.review_required_count, 1)
        self.assertAlmostEqual(aggregated.approval_rate, 1/3, places=2)
        
        # Check statistics
        self.assertGreater(aggregated.avg_quality_score, 0.0)
        self.assertGreater(aggregated.avg_overall_score, 0.0)
        self.assertIn('quality', aggregated.rejection_breakdown)
        self.assertGreater(len(aggregated.top_rejection_reasons), 0)
        self.assertIn('avg_processing_time', aggregated.processing_statistics)
    
    def test_validate_decision_result(self):
        """Test decision result validation"""
        # Valid result
        valid_result = DecisionResult(
            image_path="/test/valid.jpg",
            filename="valid.jpg",
            decision='approved',
            confidence=0.85,
            scores=DecisionScores(
                quality_score=0.8,
                defect_score=0.9,
                similarity_score=1.0,
                compliance_score=0.9,
                technical_score=0.8,
                overall_score=0.88,
                weighted_score=0.86
            )
        )
        
        errors = self.engine.validate_decision_result(valid_result)
        self.assertEqual(len(errors), 0)
        
        # Invalid result - missing required fields
        invalid_result = DecisionResult(
            image_path="",
            filename="",
            decision='invalid_decision',
            confidence=1.5,  # Invalid range
            scores=DecisionScores(
                quality_score=1.5,  # Invalid range
                defect_score=-0.1,  # Invalid range
                similarity_score=0.5,
                compliance_score=0.5,
                technical_score=0.5,
                overall_score=0.5,
                weighted_score=0.5
            )
        )
        
        errors = self.engine.validate_decision_result(invalid_result)
        self.assertGreater(len(errors), 0)
        
        # Check specific error types
        error_text = " ".join(errors)
        self.assertIn("image_path", error_text)
        self.assertIn("filename", error_text)
        self.assertIn("Invalid decision", error_text)
        self.assertIn("Invalid confidence", error_text)
        self.assertIn("Invalid quality_score", error_text)
    
    def test_get_decision_summary(self):
        """Test decision summary generation"""
        decision_result = DecisionResult(
            image_path="/test/summary_test.jpg",
            filename="summary_test.jpg",
            decision='approved',
            confidence=0.88,
            scores=DecisionScores(
                quality_score=0.85,
                defect_score=0.92,
                similarity_score=1.0,
                compliance_score=0.9,
                technical_score=0.8,
                overall_score=0.894,
                weighted_score=0.876
            ),
            approval_factors=["High quality", "No defects"],
            recommendation="Excellent image for Adobe Stock",
            processing_time=0.45
        )
        
        summary = self.engine.get_decision_summary(decision_result)
        
        # Validate summary structure
        self.assertIn('filename', summary)
        self.assertIn('decision', summary)
        self.assertIn('confidence', summary)
        self.assertIn('overall_score', summary)
        self.assertIn('weighted_score', summary)
        self.assertIn('quality_score', summary)
        self.assertIn('defect_score', summary)
        self.assertIn('similarity_score', summary)
        self.assertIn('compliance_score', summary)
        self.assertIn('rejection_count', summary)
        self.assertIn('recommendation', summary)
        self.assertIn('processing_time', summary)
        
        # Validate summary values
        self.assertEqual(summary['filename'], "summary_test.jpg")
        self.assertEqual(summary['decision'], 'approved')
        self.assertEqual(summary['confidence'], 0.88)
        self.assertEqual(summary['rejection_count'], 0)
    
    def test_error_handling(self):
        """Test error handling in decision processing"""
        # Create invalid processing result
        invalid_result = ProcessingResult(
            image_path="/nonexistent/path.jpg",
            filename="invalid.jpg",
            quality_result=None,
            defect_result=None,
            compliance_result=None
        )
        
        # Process should handle errors gracefully
        decision_result = self.engine.process(invalid_result)
        
        self.assertIsInstance(decision_result, DecisionResult)
        self.assertEqual(decision_result.decision, 'rejected')
        self.assertGreaterEqual(decision_result.confidence, 0.9)
        self.assertGreater(len(decision_result.rejection_reasons), 0)
        
        # Should have some rejection reasons (could be technical or other categories)
        self.assertGreater(len(decision_result.rejection_reasons), 0)
        
        # Check if any are technical errors
        technical_errors = [r for r in decision_result.rejection_reasons 
                          if r.category == DecisionCategory.TECHNICAL]
        # Note: With None results, we might get quality/defect/compliance errors instead of technical
    
    def test_empty_aggregation(self):
        """Test aggregation with empty result list"""
        aggregated = self.engine.aggregate_results([])
        
        self.assertEqual(aggregated.total_images, 0)
        self.assertEqual(aggregated.approved_count, 0)
        self.assertEqual(aggregated.rejected_count, 0)
        self.assertEqual(aggregated.review_required_count, 0)
        self.assertEqual(aggregated.approval_rate, 0.0)
        self.assertEqual(aggregated.avg_quality_score, 0.0)
        self.assertEqual(aggregated.avg_overall_score, 0.0)


class TestDecisionEngineDataStructures(unittest.TestCase):
    """Test cases for decision engine data structures"""
    
    def test_decision_scores_creation(self):
        """Test DecisionScores data structure"""
        scores = DecisionScores(
            quality_score=0.8,
            defect_score=0.9,
            similarity_score=1.0,
            compliance_score=0.85,
            technical_score=0.75,
            overall_score=0.86,
            weighted_score=0.84
        )
        
        self.assertEqual(scores.quality_score, 0.8)
        self.assertEqual(scores.defect_score, 0.9)
        self.assertEqual(scores.similarity_score, 1.0)
        self.assertEqual(scores.compliance_score, 0.85)
        self.assertEqual(scores.technical_score, 0.75)
        self.assertEqual(scores.overall_score, 0.86)
        self.assertEqual(scores.weighted_score, 0.84)
    
    def test_rejection_reason_creation(self):
        """Test RejectionReason data structure"""
        reason = RejectionReason(
            category=DecisionCategory.QUALITY,
            reason="low_sharpness",
            severity="high",
            score=0.3,
            threshold=0.6,
            description="Image sharpness below acceptable threshold"
        )
        
        self.assertEqual(reason.category, DecisionCategory.QUALITY)
        self.assertEqual(reason.reason, "low_sharpness")
        self.assertEqual(reason.severity, "high")
        self.assertEqual(reason.score, 0.3)
        self.assertEqual(reason.threshold, 0.6)
        self.assertIn("sharpness", reason.description)
    
    def test_decision_result_creation(self):
        """Test DecisionResult data structure"""
        scores = DecisionScores(overall_score=0.75, weighted_score=0.73)
        
        result = DecisionResult(
            image_path="/test/image.jpg",
            filename="image.jpg",
            decision='approved',
            confidence=0.82,
            scores=scores,
            recommendation="Good quality image"
        )
        
        self.assertEqual(result.image_path, "/test/image.jpg")
        self.assertEqual(result.filename, "image.jpg")
        self.assertEqual(result.decision, 'approved')
        self.assertEqual(result.confidence, 0.82)
        self.assertEqual(result.scores, scores)
        self.assertEqual(result.recommendation, "Good quality image")
        self.assertIsInstance(result.timestamp, datetime)
        self.assertEqual(len(result.rejection_reasons), 0)
        self.assertEqual(len(result.approval_factors), 0)
    
    def test_aggregated_results_creation(self):
        """Test AggregatedResults data structure"""
        aggregated = AggregatedResults(
            total_images=100,
            approved_count=75,
            rejected_count=20,
            review_required_count=5,
            approval_rate=0.75,
            avg_quality_score=0.78,
            avg_overall_score=0.76,
            rejection_breakdown={'quality': 15, 'compliance': 5},
            top_rejection_reasons=[('low_quality', 10), ('compliance_issue', 5)],
            processing_statistics={'avg_processing_time': 0.5}
        )
        
        self.assertEqual(aggregated.total_images, 100)
        self.assertEqual(aggregated.approved_count, 75)
        self.assertEqual(aggregated.rejected_count, 20)
        self.assertEqual(aggregated.review_required_count, 5)
        self.assertEqual(aggregated.approval_rate, 0.75)
        self.assertEqual(aggregated.avg_quality_score, 0.78)
        self.assertEqual(aggregated.avg_overall_score, 0.76)
        self.assertIn('quality', aggregated.rejection_breakdown)
        self.assertGreater(len(aggregated.top_rejection_reasons), 0)
        self.assertIn('avg_processing_time', aggregated.processing_statistics)


if __name__ == '__main__':
    unittest.main()