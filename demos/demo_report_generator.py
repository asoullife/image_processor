#!/usr/bin/env python3
"""
Demo script for ReportGenerator module

This script demonstrates the comprehensive report generation capabilities:
- Excel report generation with multiple sheets
- HTML dashboard with visual summaries
- Chart and graph creation
- Statistical analysis
- CSV export functionality
"""

import os
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.utils.report_generator import ReportGenerator
from backend.core.base import ProcessingResult, QualityResult, DefectResult, ComplianceResult, ObjectDefect, LogoDetection, PrivacyViolation
from backend.core.decision_engine import DecisionResult, DecisionScores, RejectionReason, DecisionCategory, AggregatedResults
from backend.config.config_loader import load_config


def create_demo_processing_results(count=50):
    """Create demo processing results for testing"""
    results = []
    
    for i in range(count):
        # Vary quality scores to create realistic distribution
        base_quality = 0.4 + (i / count) * 0.6  # Range from 0.4 to 1.0
        
        # Create quality result with some randomness
        quality_result = QualityResult(
            sharpness_score=min(1.0, base_quality + (i % 5) * 0.05),
            noise_level=max(0.0, 0.2 - (i % 7) * 0.02),
            exposure_score=min(1.0, base_quality + (i % 3) * 0.1),
            color_balance_score=min(1.0, base_quality + (i % 4) * 0.08),
            resolution=(1920 + (i % 10) * 100, 1080 + (i % 8) * 50),
            file_size=1500000 + (i % 20) * 100000,
            overall_score=base_quality,
            passed=base_quality > 0.6
        )
        
        # Create defect result
        defect_count = max(0, 3 - (i % 4))
        defect_types = ['blur', 'noise', 'artifact', 'distortion'][:defect_count]
        
        defect_result = DefectResult(
            detected_objects=[
                ObjectDefect(
                    object_type='glass',
                    defect_type='crack',
                    confidence=0.8,
                    bounding_box=(100, 100, 200, 200),
                    description='Small crack detected'
                )
            ] if i % 8 == 0 else [],
            anomaly_score=max(0.0, 0.5 - base_quality),
            defect_count=defect_count,
            defect_types=defect_types,
            confidence_scores=[0.8, 0.7, 0.6][:defect_count],
            passed=defect_count < 2
        )
        
        # Create compliance result
        logo_detections = []
        privacy_violations = []
        metadata_issues = []
        
        if i % 15 == 0:  # Some images have logo detections
            logo_detections.append(LogoDetection(
                logo_type='brand_logo',
                confidence=0.85,
                bounding_box=(50, 50, 150, 100),
                brand_name='SampleBrand'
            ))
        
        if i % 20 == 0:  # Some images have privacy violations
            privacy_violations.append(PrivacyViolation(
                violation_type='face',
                confidence=0.9,
                bounding_box=(200, 150, 300, 250),
                description='Identifiable face detected'
            ))
        
        if i % 12 == 0:  # Some images have metadata issues
            metadata_issues.append('missing_keywords')
        
        compliance_result = ComplianceResult(
            logo_detections=logo_detections,
            privacy_violations=privacy_violations,
            metadata_issues=metadata_issues,
            keyword_relevance=0.7 + (i % 10) * 0.03,
            overall_compliance=len(logo_detections) == 0 and len(privacy_violations) == 0
        )
        
        # Determine final decision based on various factors
        decision_factors = [
            quality_result.passed,
            defect_result.passed,
            compliance_result.overall_compliance,
            base_quality > 0.7
        ]
        
        if sum(decision_factors) >= 3:
            final_decision = 'approved'
        elif sum(decision_factors) <= 1:
            final_decision = 'rejected'
        else:
            final_decision = 'review_required' if i % 10 == 0 else 'rejected'
        
        # Create rejection reasons for rejected images
        rejection_reasons = []
        if final_decision == 'rejected':
            if not quality_result.passed:
                rejection_reasons.append('low_quality')
            if not defect_result.passed:
                rejection_reasons.append('defects_detected')
            if not compliance_result.overall_compliance:
                rejection_reasons.append('compliance_issues')
        
        # Create processing result
        result = ProcessingResult(
            image_path=f"/demo/images/sample_image_{i:03d}.jpg",
            filename=f"sample_image_{i:03d}.jpg",
            quality_result=quality_result,
            defect_result=defect_result,
            similarity_group=i // 5 if i % 7 == 0 else None,  # Some images in similarity groups
            compliance_result=compliance_result,
            final_decision=final_decision,
            rejection_reasons=rejection_reasons,
            processing_time=0.5 + (i % 10) * 0.1,
            timestamp=datetime.now(),
            error_message=f"Processing warning for image {i}" if i % 25 == 0 else None
        )
        
        results.append(result)
    
    return results


def create_demo_decision_results(processing_results):
    """Create demo decision results based on processing results"""
    decision_results = []
    
    for i, proc_result in enumerate(processing_results):
        # Calculate component scores
        quality_score = proc_result.quality_result.overall_score
        defect_score = max(0.0, 1.0 - proc_result.defect_result.anomaly_score)
        similarity_score = 0.8 if proc_result.similarity_group is None else 0.5
        compliance_score = 0.9 if proc_result.compliance_result.overall_compliance else 0.3
        technical_score = 0.85
        
        # Calculate overall and weighted scores
        overall_score = (quality_score + defect_score + similarity_score + compliance_score + technical_score) / 5
        weighted_score = (
            0.35 * quality_score +
            0.25 * defect_score +
            0.20 * similarity_score +
            0.15 * compliance_score +
            0.05 * technical_score
        )
        
        scores = DecisionScores(
            quality_score=quality_score,
            defect_score=defect_score,
            similarity_score=similarity_score,
            compliance_score=compliance_score,
            technical_score=technical_score,
            overall_score=overall_score,
            weighted_score=weighted_score
        )
        
        # Create rejection reasons for rejected images
        rejection_reasons = []
        approval_factors = []
        
        if proc_result.final_decision == 'rejected':
            if quality_score < 0.6:
                rejection_reasons.append(RejectionReason(
                    category=DecisionCategory.QUALITY,
                    reason="low_quality_score",
                    severity="high" if quality_score < 0.4 else "medium",
                    score=quality_score,
                    threshold=0.6,
                    description=f"Quality score {quality_score:.3f} below minimum threshold 0.6"
                ))
            
            if defect_score < 0.7:
                rejection_reasons.append(RejectionReason(
                    category=DecisionCategory.DEFECTS,
                    reason="high_defect_level",
                    severity="high",
                    score=1.0 - defect_score,
                    threshold=0.3,
                    description=f"High defect level detected: {1.0 - defect_score:.3f}"
                ))
            
            if compliance_score < 0.8:
                rejection_reasons.append(RejectionReason(
                    category=DecisionCategory.COMPLIANCE,
                    reason="compliance_issues",
                    severity="critical",
                    score=compliance_score,
                    threshold=0.8,
                    description="Image contains compliance violations"
                ))
        
        elif proc_result.final_decision == 'approved':
            if quality_score > 0.8:
                approval_factors.append("Excellent image quality")
            if defect_score > 0.9:
                approval_factors.append("No significant defects detected")
            if compliance_score > 0.9:
                approval_factors.append("Full compliance with guidelines")
        
        # Determine confidence based on score clarity
        if proc_result.final_decision == 'approved':
            confidence = min(0.95, weighted_score + 0.1)
        elif proc_result.final_decision == 'rejected':
            confidence = min(0.95, 1.0 - weighted_score + 0.1)
        else:  # review_required
            confidence = 0.6
        
        # Create recommendation
        if proc_result.final_decision == 'approved':
            recommendation = "Image approved for Adobe Stock submission"
        elif proc_result.final_decision == 'rejected':
            recommendation = f"Image rejected due to {len(rejection_reasons)} issues"
        else:
            recommendation = "Image requires manual review - mixed quality indicators"
        
        decision_result = DecisionResult(
            image_path=proc_result.image_path,
            filename=proc_result.filename,
            decision=proc_result.final_decision,
            confidence=confidence,
            scores=scores,
            rejection_reasons=rejection_reasons,
            approval_factors=approval_factors,
            recommendation=recommendation,
            processing_time=0.05 + (i % 5) * 0.01,
            timestamp=datetime.now()
        )
        
        decision_results.append(decision_result)
    
    return decision_results


def create_demo_aggregated_results(decision_results):
    """Create demo aggregated results from decision results"""
    total_images = len(decision_results)
    approved_count = sum(1 for dr in decision_results if dr.decision == 'approved')
    rejected_count = sum(1 for dr in decision_results if dr.decision == 'rejected')
    review_required_count = sum(1 for dr in decision_results if dr.decision == 'review_required')
    
    approval_rate = approved_count / total_images if total_images > 0 else 0.0
    
    # Calculate average scores
    quality_scores = [dr.scores.quality_score for dr in decision_results]
    overall_scores = [dr.scores.overall_score for dr in decision_results]
    
    avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    avg_overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    
    # Analyze rejection reasons
    rejection_breakdown = {}
    all_rejection_reasons = []
    
    for dr in decision_results:
        for reason in dr.rejection_reasons:
            category = reason.category.value
            rejection_breakdown[category] = rejection_breakdown.get(category, 0) + 1
            all_rejection_reasons.append(reason.reason)
    
    # Count reason occurrences
    reason_counts = {}
    for reason in all_rejection_reasons:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    
    top_rejection_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Processing statistics
    processing_times = [dr.processing_time for dr in decision_results if dr.processing_time > 0]
    processing_statistics = {
        'total_processing_time': sum(processing_times),
        'avg_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0.0,
        'min_processing_time': min(processing_times) if processing_times else 0.0,
        'max_processing_time': max(processing_times) if processing_times else 0.0
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


def main():
    """Main demo function"""
    print("=" * 60)
    print("Adobe Stock Image Processor - Report Generator Demo")
    print("=" * 60)
    
    # Create temporary directory for demo output
    demo_dir = tempfile.mkdtemp(prefix="adobe_stock_report_demo_")
    print(f"Demo output directory: {demo_dir}")
    
    try:
        # Load configuration
        print("\n1. Loading configuration...")
        app_config = load_config()
        
        # Create dictionary config for ReportGenerator
        config = {
            'reports': {
                'excel_enabled': True,
                'html_enabled': True,
                'charts_enabled': True,
                'thumbnails_enabled': False,  # Disable for demo (no real images)
                'thumbnail_size': [150, 150],
                'max_thumbnails': 20,
                'chart_style': 'default',
                'chart_dpi': 100,
                'chart_figsize': [12, 8]
            }
        }
        
        # Initialize report generator
        print("2. Initializing ReportGenerator...")
        report_generator = ReportGenerator(config)
        
        if not report_generator.initialize():
            print("❌ Failed to initialize ReportGenerator")
            return
        
        print("✅ ReportGenerator initialized successfully")
        
        # Create demo data
        print("\n3. Creating demo processing data...")
        processing_results = create_demo_processing_results(50)
        print(f"✅ Created {len(processing_results)} processing results")
        
        print("4. Creating demo decision data...")
        decision_results = create_demo_decision_results(processing_results)
        print(f"✅ Created {len(decision_results)} decision results")
        
        print("5. Creating aggregated statistics...")
        aggregated_results = create_demo_aggregated_results(decision_results)
        print(f"✅ Aggregated results: {aggregated_results.approved_count} approved, "
              f"{aggregated_results.rejected_count} rejected, "
              f"{aggregated_results.review_required_count} review required")
        
        # Generate comprehensive report
        print("\n6. Generating comprehensive report...")
        session_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        report_files = report_generator.generate_comprehensive_report(
            session_id=session_id,
            processing_results=processing_results,
            decision_results=decision_results,
            aggregated_results=aggregated_results,
            output_dir=demo_dir
        )
        
        if report_files:
            print("✅ Comprehensive report generated successfully!")
            print("\nGenerated files:")
            for report_type, file_path in report_files.items():
                full_path = os.path.join(demo_dir, file_path) if not os.path.isabs(file_path) else file_path
                print(f"  📄 {report_type.upper()}: {full_path}")
        else:
            print("❌ Failed to generate comprehensive report")
            return
        
        # Test individual report components
        print("\n7. Testing individual report components...")
        
        # Test Excel report
        print("   Testing Excel report generation...")
        excel_path = report_generator.generate_excel_report(
            session_id=f"{session_id}_excel",
            processing_results=processing_results,
            decision_results=decision_results,
            aggregated_results=aggregated_results,
            output_dir=demo_dir
        )
        
        if excel_path:
            print(f"   ✅ Excel report: {excel_path}")
        else:
            print("   ❌ Excel report generation failed")
        
        # Test HTML dashboard
        print("   Testing HTML dashboard generation...")
        html_path = report_generator.generate_html_dashboard(
            session_id=f"{session_id}_html",
            processing_results=processing_results,
            decision_results=decision_results,
            aggregated_results=aggregated_results,
            output_dir=demo_dir
        )
        
        if html_path:
            print(f"   ✅ HTML dashboard: {html_path}")
        else:
            print("   ❌ HTML dashboard generation failed")
        
        # Test chart generation
        print("   Testing chart generation...")
        charts_dir = report_generator.generate_charts(
            decision_results=decision_results,
            aggregated_results=aggregated_results,
            charts_dir=os.path.join(demo_dir, 'individual_charts')
        )
        
        if charts_dir:
            print(f"   ✅ Charts directory: {charts_dir}")
            chart_files = [f for f in os.listdir(charts_dir) if f.endswith('.png')]
            print(f"   📊 Generated {len(chart_files)} chart files")
        else:
            print("   ❌ Chart generation failed")
        
        # Test CSV export
        print("   Testing CSV export...")
        csv_path = os.path.join(demo_dir, f"results_{session_id}.csv")
        csv_success = report_generator.export_results_to_csv(
            processing_results=processing_results,
            decision_results=decision_results,
            output_path=csv_path
        )
        
        if csv_success:
            print(f"   ✅ CSV export: {csv_path}")
        else:
            print("   ❌ CSV export failed")
        
        # Test statistical analysis
        print("   Testing statistical analysis...")
        stats = report_generator.create_summary_statistics(decision_results)
        
        if stats:
            print("   ✅ Statistical analysis completed")
            print(f"   📊 Total images: {stats['total_images']}")
            print(f"   📊 Approval rate: {stats['approval_rate']:.1%}")
            print(f"   📊 Avg quality score: {stats['score_statistics']['quality_score']['mean']:.3f}")
        else:
            print("   ❌ Statistical analysis failed")
        
        # Display summary
        print("\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        print(f"Session ID: {session_id}")
        print(f"Output Directory: {demo_dir}")
        print(f"Total Images Processed: {len(processing_results)}")
        print(f"Approved: {aggregated_results.approved_count}")
        print(f"Rejected: {aggregated_results.rejected_count}")
        print(f"Review Required: {aggregated_results.review_required_count}")
        print(f"Approval Rate: {aggregated_results.approval_rate:.1%}")
        print(f"Average Quality Score: {aggregated_results.avg_quality_score:.3f}")
        
        if aggregated_results.top_rejection_reasons:
            print("\nTop Rejection Reasons:")
            for reason, count in aggregated_results.top_rejection_reasons[:5]:
                print(f"  • {reason}: {count} images")
        
        print(f"\nProcessing Statistics:")
        print(f"  • Total processing time: {aggregated_results.processing_statistics['total_processing_time']:.2f}s")
        print(f"  • Average time per image: {aggregated_results.processing_statistics['avg_processing_time']:.3f}s")
        
        print("\n📁 Generated Files:")
        for root, dirs, files in os.walk(demo_dir):
            level = root.replace(demo_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_size = os.path.getsize(os.path.join(root, file))
                print(f"{subindent}{file} ({file_size:,} bytes)")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📂 All files saved to: {demo_dir}")
        print("\nYou can now:")
        print("  • Open the HTML dashboard in your web browser")
        print("  • View the Excel report with detailed analysis")
        print("  • Examine the generated charts")
        print("  • Import the CSV data into other tools")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            report_generator.cleanup()
        except NameError:
            pass  # report_generator wasn't initialized
        
        # Ask user if they want to keep the demo files
        try:
            keep_files = input(f"\nKeep demo files in {demo_dir}? (y/N): ").lower().strip()
            if keep_files != 'y':
                shutil.rmtree(demo_dir, ignore_errors=True)
                print("Demo files cleaned up.")
            else:
                print(f"Demo files preserved in: {demo_dir}")
        except KeyboardInterrupt:
            print(f"\nDemo files preserved in: {demo_dir}")


if __name__ == "__main__":
    main()