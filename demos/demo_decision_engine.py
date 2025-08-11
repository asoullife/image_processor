#!/usr/bin/env python3
"""
Demo script for Decision Engine Module

This script demonstrates the decision engine and result aggregation system:
- Scoring algorithms that combine all analysis results
- Decision logic based on configurable thresholds
- Result aggregation and final approval/rejection determination
- Rejection reason tracking and categorization
"""

import os
import sys
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.decision_engine import DecisionEngine, DecisionCategory
from backend.core.base import (
    ProcessingResult, QualityResult, DefectResult, ComplianceResult,
    ObjectDefect, LogoDetection, PrivacyViolation
)
from backend.config.config_loader import load_config


def create_sample_processing_results():
    """Create sample processing results for demonstration"""
    results = []
    
    # 1. High quality approved image
    quality_result_1 = QualityResult(
        sharpness_score=180.0,
        noise_level=0.03,
        exposure_score=0.88,
        color_balance_score=0.85,
        resolution=(3840, 2160),  # 4K
        file_size=4500000,  # 4.5MB
        overall_score=0.89,
        passed=True
    )
    
    defect_result_1 = DefectResult(
        detected_objects=[],
        anomaly_score=0.05,
        defect_count=0,
        defect_types=[],
        confidence_scores=[],
        passed=True
    )
    
    compliance_result_1 = ComplianceResult(
        logo_detections=[],
        privacy_violations=[],
        metadata_issues=[],
        keyword_relevance=0.92,
        overall_compliance=True
    )
    
    result_1 = ProcessingResult(
        image_path="/demo/excellent_landscape.jpg",
        filename="excellent_landscape.jpg",
        quality_result=quality_result_1,
        defect_result=defect_result_1,
        similarity_group=None,
        compliance_result=compliance_result_1
    )
    results.append(result_1)
    
    # 2. Poor quality rejected image
    quality_result_2 = QualityResult(
        sharpness_score=45.0,
        noise_level=0.35,
        exposure_score=0.25,
        color_balance_score=0.3,
        resolution=(800, 600),
        file_size=120000,  # 120KB
        overall_score=0.22,
        passed=False
    )
    
    defect_result_2 = DefectResult(
        detected_objects=[
            ObjectDefect("glass", "crack", 0.85, (100, 100, 50, 50), "Cracked glass detected"),
            ObjectDefect("surface", "scratch", 0.75, (200, 150, 30, 80), "Surface scratch visible")
        ],
        anomaly_score=0.65,
        defect_count=2,
        defect_types=["crack", "scratch"],
        confidence_scores=[0.85, 0.75],
        passed=False
    )
    
    compliance_result_2 = ComplianceResult(
        logo_detections=[
            LogoDetection("Nike", 0.88, (50, 50, 100, 40), "nike")
        ],
        privacy_violations=[
            PrivacyViolation("face", 0.82, (300, 200, 80, 80), "Person's face clearly visible")
        ],
        metadata_issues=["GPS location data present", "Personal device information"],
        keyword_relevance=0.25,
        overall_compliance=False
    )
    
    result_2 = ProcessingResult(
        image_path="/demo/poor_quality_image.jpg",
        filename="poor_quality_image.jpg",
        quality_result=quality_result_2,
        defect_result=defect_result_2,
        similarity_group=1,  # In similarity group
        compliance_result=compliance_result_2
    )
    results.append(result_2)
    
    # 3. Borderline image requiring review
    quality_result_3 = QualityResult(
        sharpness_score=125.0,
        noise_level=0.08,
        exposure_score=0.68,
        color_balance_score=0.72,
        resolution=(1920, 1080),
        file_size=1800000,  # 1.8MB
        overall_score=0.67,
        passed=True
    )
    
    defect_result_3 = DefectResult(
        detected_objects=[
            ObjectDefect("object", "minor_wear", 0.55, (150, 100, 40, 40), "Minor wear visible")
        ],
        anomaly_score=0.22,
        defect_count=1,
        defect_types=["minor_wear"],
        confidence_scores=[0.55],
        passed=True
    )
    
    compliance_result_3 = ComplianceResult(
        logo_detections=[],
        privacy_violations=[],
        metadata_issues=["Minor metadata inconsistency"],
        keyword_relevance=0.78,
        overall_compliance=True
    )
    
    result_3 = ProcessingResult(
        image_path="/demo/borderline_image.jpg",
        filename="borderline_image.jpg",
        quality_result=quality_result_3,
        defect_result=defect_result_3,
        similarity_group=None,
        compliance_result=compliance_result_3
    )
    results.append(result_3)
    
    # 4. Critical failure image
    quality_result_4 = QualityResult(
        sharpness_score=25.0,
        noise_level=0.55,
        exposure_score=0.15,
        color_balance_score=0.2,
        resolution=(640, 480),
        file_size=80000,  # 80KB
        overall_score=0.18,  # Critical failure
        passed=False
    )
    
    defect_result_4 = DefectResult(
        detected_objects=[
            ObjectDefect("multiple", "severe_damage", 0.92, (0, 0, 640, 480), "Severe damage throughout image")
        ],
        anomaly_score=0.85,  # Critical failure
        defect_count=5,
        defect_types=["severe_damage", "corruption", "artifacts"],
        confidence_scores=[0.92, 0.88, 0.85, 0.80, 0.75],
        passed=False
    )
    
    compliance_result_4 = ComplianceResult(
        logo_detections=[
            LogoDetection("Apple", 0.95, (100, 50, 80, 30), "apple"),
            LogoDetection("Google", 0.90, (200, 50, 90, 35), "google")
        ],
        privacy_violations=[
            PrivacyViolation("face", 0.95, (300, 100, 100, 100), "Multiple faces visible"),
            PrivacyViolation("license_plate", 0.85, (400, 300, 120, 40), "License plate readable")
        ],
        metadata_issues=["GPS coordinates", "Personal information", "Copyrighted content"],
        keyword_relevance=0.1,
        overall_compliance=False
    )
    
    result_4 = ProcessingResult(
        image_path="/demo/critical_failure.jpg",
        filename="critical_failure.jpg",
        quality_result=quality_result_4,
        defect_result=defect_result_4,
        similarity_group=2,
        compliance_result=compliance_result_4
    )
    results.append(result_4)
    
    # 5. Good quality approved image
    quality_result_5 = QualityResult(
        sharpness_score=165.0,
        noise_level=0.04,
        exposure_score=0.82,
        color_balance_score=0.88,
        resolution=(2560, 1440),
        file_size=3200000,  # 3.2MB
        overall_score=0.84,
        passed=True
    )
    
    defect_result_5 = DefectResult(
        detected_objects=[],
        anomaly_score=0.08,
        defect_count=0,
        defect_types=[],
        confidence_scores=[],
        passed=True
    )
    
    compliance_result_5 = ComplianceResult(
        logo_detections=[],
        privacy_violations=[],
        metadata_issues=[],
        keyword_relevance=0.86,
        overall_compliance=True
    )
    
    result_5 = ProcessingResult(
        image_path="/demo/good_portrait.jpg",
        filename="good_portrait.jpg",
        quality_result=quality_result_5,
        defect_result=defect_result_5,
        similarity_group=None,
        compliance_result=compliance_result_5
    )
    results.append(result_5)
    
    return results


def print_separator(title=""):
    """Print a separator line with optional title"""
    if title:
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
    else:
        print("-" * 60)


def print_decision_result(decision_result):
    """Print detailed decision result"""
    print(f"\nImage: {decision_result.filename}")
    print(f"Decision: {decision_result.decision.upper()}")
    print(f"Confidence: {decision_result.confidence:.3f}")
    print(f"Processing Time: {decision_result.processing_time:.4f}s")
    
    print(f"\nScores:")
    scores = decision_result.scores
    print(f"  Quality:     {scores.quality_score:.3f}")
    print(f"  Defects:     {scores.defect_score:.3f}")
    print(f"  Similarity:  {scores.similarity_score:.3f}")
    print(f"  Compliance:  {scores.compliance_score:.3f}")
    print(f"  Technical:   {scores.technical_score:.3f}")
    print(f"  Overall:     {scores.overall_score:.3f}")
    print(f"  Weighted:    {scores.weighted_score:.3f}")
    
    if decision_result.rejection_reasons:
        print(f"\nRejection Reasons ({len(decision_result.rejection_reasons)}):")
        for i, reason in enumerate(decision_result.rejection_reasons, 1):
            print(f"  {i}. [{reason.category.value.upper()}] {reason.reason}")
            print(f"     Severity: {reason.severity}, Score: {reason.score:.3f}, Threshold: {reason.threshold:.3f}")
            print(f"     {reason.description}")
    
    if decision_result.approval_factors:
        print(f"\nApproval Factors ({len(decision_result.approval_factors)}):")
        for i, factor in enumerate(decision_result.approval_factors, 1):
            print(f"  {i}. {factor}")
    
    print(f"\nRecommendation: {decision_result.recommendation}")


def print_aggregated_results(aggregated):
    """Print aggregated results summary"""
    print(f"\nTotal Images Processed: {aggregated.total_images}")
    print(f"Approved: {aggregated.approved_count} ({aggregated.approval_rate:.1%})")
    print(f"Rejected: {aggregated.rejected_count} ({aggregated.rejected_count/aggregated.total_images:.1%})")
    print(f"Review Required: {aggregated.review_required_count} ({aggregated.review_required_count/aggregated.total_images:.1%})")
    
    print(f"\nAverage Scores:")
    print(f"  Quality Score: {aggregated.avg_quality_score:.3f}")
    print(f"  Overall Score: {aggregated.avg_overall_score:.3f}")
    
    print(f"\nRejection Breakdown by Category:")
    for category, count in aggregated.rejection_breakdown.items():
        print(f"  {category.title()}: {count}")
    
    print(f"\nTop Rejection Reasons:")
    for reason, count in aggregated.top_rejection_reasons[:5]:
        print(f"  {reason}: {count}")
    
    print(f"\nProcessing Statistics:")
    stats = aggregated.processing_statistics
    print(f"  Average Processing Time: {stats.get('avg_processing_time', 0):.4f}s")
    print(f"  Total Processing Time: {stats.get('total_processing_time', 0):.4f}s")
    print(f"  Min Processing Time: {stats.get('min_processing_time', 0):.4f}s")
    print(f"  Max Processing Time: {stats.get('max_processing_time', 0):.4f}s")


def demonstrate_decision_validation():
    """Demonstrate decision result validation"""
    print_separator("Decision Result Validation Demo")
    
    # Load configuration
    config = load_config()
    engine = DecisionEngine(config)
    
    # Create a sample processing result
    sample_results = create_sample_processing_results()
    decision_result = engine.process(sample_results[0])
    
    print("Validating a good decision result:")
    errors = engine.validate_decision_result(decision_result)
    if errors:
        print("Validation errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ Decision result is valid")
    
    # Create an invalid decision result for testing
    from backend.core.decision_engine import DecisionResult, DecisionScores
    
    invalid_result = DecisionResult(
        image_path="",  # Missing path
        filename="",    # Missing filename
        decision='invalid_decision',  # Invalid decision
        confidence=1.5,  # Invalid range
        scores=DecisionScores(
            quality_score=1.2,  # Invalid range
            defect_score=-0.1,  # Invalid range
            similarity_score=0.5,
            compliance_score=0.5,
            technical_score=0.5,
            overall_score=0.5,
            weighted_score=0.5
        )
    )
    
    print("\nValidating an invalid decision result:")
    errors = engine.validate_decision_result(invalid_result)
    if errors:
        print("Validation errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("No validation errors (unexpected)")


def demonstrate_decision_summary():
    """Demonstrate decision summary generation"""
    print_separator("Decision Summary Demo")
    
    # Load configuration
    config = load_config()
    engine = DecisionEngine(config)
    
    # Process sample results
    sample_results = create_sample_processing_results()
    decision_results = []
    
    for processing_result in sample_results:
        decision_result = engine.process(processing_result)
        decision_results.append(decision_result)
    
    print("Decision Summaries:")
    for decision_result in decision_results:
        summary = engine.get_decision_summary(decision_result)
        print(f"\n{summary['filename']}:")
        print(f"  Decision: {summary['decision']} (confidence: {summary['confidence']})")
        print(f"  Scores: Overall={summary['overall_score']}, Weighted={summary['weighted_score']}")
        print(f"  Quality={summary['quality_score']}, Defects={summary['defect_score']}")
        print(f"  Similarity={summary['similarity_score']}, Compliance={summary['compliance_score']}")
        print(f"  Rejections: {summary['rejection_count']}")
        if summary['main_rejection_reasons']:
            print(f"  Main Issues: {', '.join(summary['main_rejection_reasons'])}")
        print(f"  Processing Time: {summary['processing_time']}s")


def main():
    """Main demonstration function"""
    print_separator("Adobe Stock Decision Engine Demo")
    print("This demo showcases the decision engine and result aggregation system")
    print("for the Adobe Stock Image Processor.")
    
    try:
        # Load configuration
        print("\nLoading configuration...")
        config = load_config()
        
        # Initialize decision engine
        print("Initializing Decision Engine...")
        engine = DecisionEngine(config)
        
        print(f"Decision Engine Configuration:")
        print(f"  Weights: {engine.weights}")
        print(f"  Thresholds: {engine.thresholds}")
        print(f"  Critical Thresholds: {engine.critical_thresholds}")
        
        # Create sample processing results
        print("\nCreating sample processing results...")
        sample_results = create_sample_processing_results()
        print(f"Created {len(sample_results)} sample processing results")
        
        # Process each result through decision engine
        print_separator("Individual Decision Processing")
        decision_results = []
        
        for i, processing_result in enumerate(sample_results, 1):
            print(f"\nProcessing Image {i}/{len(sample_results)}: {processing_result.filename}")
            
            # Process through decision engine
            decision_result = engine.process(processing_result)
            decision_results.append(decision_result)
            
            # Print detailed results
            print_decision_result(decision_result)
            
            if i < len(sample_results):
                print_separator()
        
        # Aggregate results
        print_separator("Result Aggregation")
        print("Aggregating all decision results...")
        
        aggregated_results = engine.aggregate_results(decision_results)
        print_aggregated_results(aggregated_results)
        
        # Demonstrate validation
        demonstrate_decision_validation()
        
        # Demonstrate summary generation
        demonstrate_decision_summary()
        
        print_separator("Demo Complete")
        print("Decision engine demonstration completed successfully!")
        print(f"Processed {len(decision_results)} images with {aggregated_results.approved_count} approved, "
              f"{aggregated_results.rejected_count} rejected, and {aggregated_results.review_required_count} requiring review.")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)