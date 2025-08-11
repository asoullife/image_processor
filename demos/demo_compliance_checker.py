#!/usr/bin/env python3
"""
Demo script for ComplianceChecker module

This script demonstrates the compliance checking functionality including:
- Logo and trademark detection
- Face detection for privacy concerns
- License plate detection
- Metadata validation
- Keyword relevance checking
"""

import os
import sys
import json
from pathlib import Path

# Add the current directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.analyzers.compliance_checker import ComplianceChecker, analyze_image_compliance, batch_compliance_check
from backend.utils.logger import get_logger


def create_demo_config():
    """Create demo configuration for compliance checking"""
    return {
        'logo_detection_confidence': 0.7,
        'face_detection_enabled': True,
        'metadata_validation': True
    }


def demo_single_image_compliance():
    """Demonstrate compliance checking on a single image"""
    print("=== Single Image Compliance Check Demo ===")
    
    # Use test images if available
    test_image_paths = [
        'test_input/main_image_01.jpg',
        'test_input/main_image_02.png',
        'test_input/fake_image.jpg'
    ]
    
    config = create_demo_config()
    checker = ComplianceChecker(config)
    
    for image_path in test_image_paths:
        if os.path.exists(image_path):
            print(f"\nAnalyzing: {image_path}")
            
            try:
                # Test with sample metadata
                sample_metadata = {
                    'description': 'A beautiful landscape photo',
                    'keywords': ['nature', 'landscape', 'mountain', 'sky'],
                    'title': 'Mountain Landscape',
                    'Make': 'Canon',
                    'Model': 'EOS 5D'
                }
                
                result = checker.check_compliance(image_path, sample_metadata)
                
                print(f"Overall Compliance: {'✓ PASS' if result.overall_compliance else '✗ FAIL'}")
                print(f"Logo Detections: {len(result.logo_detections)}")
                print(f"Privacy Violations: {len(result.privacy_violations)}")
                print(f"Metadata Issues: {len(result.metadata_issues)}")
                print(f"Keyword Relevance: {result.keyword_relevance:.2f}")
                
                # Show detailed results
                if result.logo_detections:
                    print("  Logo Detections:")
                    for detection in result.logo_detections:
                        print(f"    - {detection.text_detected} (confidence: {detection.confidence:.2f})")
                
                if result.privacy_violations:
                    print("  Privacy Violations:")
                    for violation in result.privacy_violations:
                        print(f"    - {violation.violation_type}: {violation.description}")
                
                if result.metadata_issues:
                    print("  Metadata Issues:")
                    for issue in result.metadata_issues:
                        print(f"    - {issue}")
                
                # Get summary
                summary = checker.get_compliance_summary(result)
                if summary['main_issues']:
                    print("  Main Issues:")
                    for issue in summary['main_issues']:
                        print(f"    - {issue}")
                
            except Exception as e:
                print(f"Error analyzing {image_path}: {e}")
            
            print("-" * 50)
            break  # Only demo first available image
    else:
        print("No test images found. Please ensure test images exist in test_input/ directory.")


def demo_batch_compliance_check():
    """Demonstrate batch compliance checking"""
    print("\n=== Batch Compliance Check Demo ===")
    
    # Find available test images
    test_dir = Path('backend/data/input')
    if test_dir.exists():
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(list(test_dir.glob(ext)))
        
        if image_paths:
            # Limit to first 3 images for demo
            image_paths = [str(p) for p in image_paths[:3]]
            
            print(f"Processing {len(image_paths)} images...")
            
            config = create_demo_config()
            results = batch_compliance_check(image_paths, config)
            
            # Summary statistics
            total_images = len(results)
            compliant_images = sum(1 for r in results.values() if r.overall_compliance)
            
            print(f"\nBatch Processing Results:")
            print(f"Total Images: {total_images}")
            print(f"Compliant Images: {compliant_images}")
            print(f"Non-compliant Images: {total_images - compliant_images}")
            print(f"Compliance Rate: {(compliant_images/total_images)*100:.1f}%")
            
            # Detailed results
            print("\nDetailed Results:")
            for image_path, result in results.items():
                status = "✓ PASS" if result.overall_compliance else "✗ FAIL"
                print(f"  {os.path.basename(image_path)}: {status}")
                
                if not result.overall_compliance:
                    issues = []
                    if result.logo_detections:
                        issues.append(f"{len(result.logo_detections)} logo(s)")
                    if result.privacy_violations:
                        issues.append(f"{len(result.privacy_violations)} privacy violation(s)")
                    if result.metadata_issues:
                        issues.append(f"{len(result.metadata_issues)} metadata issue(s)")
                    if result.keyword_relevance < 0.5:
                        issues.append(f"low keyword relevance ({result.keyword_relevance:.2f})")
                    
                    if issues:
                        print(f"    Issues: {', '.join(issues)}")
        else:
            print("No image files found in test_input/ directory.")
    else:
        print("test_input/ directory not found.")


def demo_compliance_with_problematic_metadata():
    """Demonstrate compliance checking with problematic metadata"""
    print("\n=== Problematic Metadata Demo ===")
    
    # Test with various problematic metadata scenarios
    test_scenarios = [
        {
            'name': 'GPS Location Data',
            'metadata': {
                'description': 'Beach photo',
                'GPS': {'latitude': 40.7128, 'longitude': -74.0060},
                'keywords': ['beach', 'ocean', 'sunset']
            }
        },
        {
            'name': 'Personal Device Info',
            'metadata': {
                'description': 'Street photography',
                'Make': 'Apple',
                'Model': 'iPhone 12 Pro',
                'keywords': ['street', 'urban', 'people']
            }
        },
        {
            'name': 'Software Watermark',
            'metadata': {
                'description': 'Portrait photo',
                'Software': 'Photoshop Trial Version with Watermark',
                'keywords': ['portrait', 'person', 'studio']
            }
        },
        {
            'name': 'Inappropriate Keywords',
            'metadata': {
                'description': 'Fashion photo',
                'keywords': ['fashion', 'nude', 'adult', 'explicit'],
                'title': 'Fashion shoot'
            }
        },
        {
            'name': 'Brand Keywords',
            'metadata': {
                'description': 'Sports photo',
                'keywords': ['sports', 'nike', 'adidas', 'running'],
                'title': 'Athletic wear'
            }
        }
    ]
    
    config = create_demo_config()
    checker = ComplianceChecker(config)
    
    # Use first available test image
    test_image = None
    for path in ['test_input/main_image_01.jpg', 'test_input/fake_image.jpg']:
        if os.path.exists(path):
            test_image = path
            break
    
    if not test_image:
        print("No test image available for metadata demo.")
        return
    
    print(f"Using test image: {test_image}")
    
    for scenario in test_scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        try:
            result = checker.check_compliance(test_image, scenario['metadata'])
            
            print(f"Overall Compliance: {'✓ PASS' if result.overall_compliance else '✗ FAIL'}")
            print(f"Keyword Relevance: {result.keyword_relevance:.2f}")
            
            if result.metadata_issues:
                print("Metadata Issues:")
                for issue in result.metadata_issues:
                    print(f"  - {issue}")
            
            if result.keyword_relevance < 0.5:
                print(f"Low keyword relevance score: {result.keyword_relevance:.2f}")
            
        except Exception as e:
            print(f"Error in scenario '{scenario['name']}': {e}")


def demo_logo_detection():
    """Demonstrate logo detection capabilities"""
    print("\n=== Logo Detection Demo ===")
    
    # This would require images with actual logos for real testing
    # For demo purposes, we'll show the detection logic
    
    config = create_demo_config()
    checker = ComplianceChecker(config)
    
    print("Logo Detection Configuration:")
    print(f"  Confidence Threshold: {checker.logo_confidence_threshold}")
    print(f"  Brand Keywords Monitored: {len(checker.brand_keywords)}")
    print(f"  Sample Brand Keywords: {', '.join(checker.brand_keywords[:10])}")
    
    # Note about OCR requirements
    try:
        import pytesseract
        print("  OCR Engine: Available (pytesseract)")
    except ImportError:
        print("  OCR Engine: Not available (pytesseract not installed)")
        print("  Note: Install pytesseract for logo detection functionality")


def main():
    """Main demo function"""
    print("ComplianceChecker Demo")
    print("=" * 50)
    
    # Setup logging
    logger = get_logger('compliance_demo')
    
    try:
        # Run all demos
        demo_single_image_compliance()
        demo_batch_compliance_check()
        demo_compliance_with_problematic_metadata()
        demo_logo_detection()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nNote: For full functionality, ensure the following are installed:")
        print("  - opencv-python (for face detection)")
        print("  - pytesseract (for OCR-based logo detection)")
        print("  - Pillow (for image processing)")
        
    except Exception as e:
        print(f"Demo error: {e}")
        logger.error(f"Demo error: {e}")


if __name__ == '__main__':
    main()