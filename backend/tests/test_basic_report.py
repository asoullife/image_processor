#!/usr/bin/env python3
"""Basic test for report generator functionality"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tempfile
import shutil
from datetime import datetime

from backend.utils.report_generator import ReportGenerator
from backend.core.base import ProcessingResult, QualityResult
from backend.core.decision_engine import DecisionResult, DecisionScores, AggregatedResults

def test_basic_functionality():
    """Test basic report generator functionality"""
    print("Testing ReportGenerator basic functionality...")
    
    # Create test directory
    test_dir = tempfile.mkdtemp()
    
    try:
        # Initialize report generator
        config = {
            'reports': {
                'excel_enabled': True,
                'html_enabled': True,
                'charts_enabled': True,
                'thumbnails_enabled': False
            }
        }
        
        generator = ReportGenerator(config)
        assert generator.initialize(), "Failed to initialize ReportGenerator"
        print("✅ ReportGenerator initialized")
        
        # Create minimal test data
        processing_result = ProcessingResult(
            image_path="/test/image.jpg",
            filename="image.jpg",
            quality_result=QualityResult(
                sharpness_score=0.8,
                noise_level=0.1,
                exposure_score=0.7,
                color_balance_score=0.9,
                resolution=(1920, 1080),
                file_size=2048000,
                overall_score=0.8,
                passed=True
            ),
            final_decision='approved',
            processing_time=1.0,
            timestamp=datetime.now()
        )
        
        decision_result = DecisionResult(
            image_path="/test/image.jpg",
            filename="image.jpg",
            decision='approved',
            confidence=0.9,
            scores=DecisionScores(
                quality_score=0.8,
                defect_score=0.9,
                similarity_score=0.95,
                compliance_score=0.9,
                technical_score=0.85,
                overall_score=0.85,
                weighted_score=0.83
            ),
            processing_time=0.1
        )
        
        aggregated_results = AggregatedResults(
            total_images=1,
            approved_count=1,
            rejected_count=0,
            review_required_count=0,
            approval_rate=1.0,
            avg_quality_score=0.8,
            avg_overall_score=0.85,
            rejection_breakdown={},
            top_rejection_reasons=[],
            processing_statistics={'total_processing_time': 1.0}
        )
        
        # Test summary statistics
        stats = generator.create_summary_statistics([decision_result])
        assert isinstance(stats, dict), "Statistics should be a dictionary"
        assert stats['total_images'] == 1, "Should have 1 image"
        print("✅ Summary statistics created")
        
        # Test CSV export
        csv_path = f"{test_dir}/test_results.csv"
        success = generator.export_results_to_csv([processing_result], [decision_result], csv_path)
        assert success, "CSV export should succeed"
        print("✅ CSV export successful")
        
        # Test HTML generation (without thumbnails)
        html_path = generator.generate_html_dashboard(
            "test_session", [processing_result], [decision_result], 
            aggregated_results, test_dir
        )
        assert html_path is not None, "HTML generation should succeed"
        print("✅ HTML dashboard generated")
        
        # Test Excel generation
        excel_path = generator.generate_excel_report(
            "test_session", [processing_result], [decision_result],
            aggregated_results, test_dir
        )
        assert excel_path is not None, "Excel generation should succeed"
        print("✅ Excel report generated")
        
        # Cleanup
        generator.cleanup()
        print("✅ All basic tests passed!")
        
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    test_basic_functionality()