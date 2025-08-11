#!/usr/bin/env python3
"""
Demo script for AI-Enhanced Compliance Checker

This script demonstrates the AI compliance checking capabilities including:
- Logo and trademark detection using OCR and AI
- Face detection and privacy concern identification
- Content appropriateness analysis with cultural sensitivity
- Enhanced metadata validation
- Comprehensive compliance reporting in Thai and English
"""

import os
import sys
import logging
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from analyzers.ai_compliance_checker import AIComplianceChecker
    from config.config_loader import load_config
    from utils.demo_utils import create_demo_images, setup_demo_logging
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the backend directory")
    sys.exit(1)

def main():
    """Main demo function"""
    print("🔍 AI-Enhanced Compliance Checker Demo")
    print("=" * 50)
    
    # Setup logging
    setup_demo_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize AI compliance checker
        print("\n📋 Initializing AI Compliance Checker...")
        compliance_checker = AIComplianceChecker(config)
        
        # Test different performance modes
        performance_modes = ['speed', 'balanced', 'smart']
        
        for mode in performance_modes:
            print(f"\n🚀 Testing Performance Mode: {mode.upper()}")
            print("-" * 30)
            
            # Set performance mode
            compliance_checker.set_performance_mode(mode)
            
            # Get system status
            status = compliance_checker.get_system_status()
            print(f"Performance Mode: {status['performance_mode']}")
            print(f"OCR Reader: {'✅' if status['ocr_reader_loaded'] else '❌'}")
            print(f"Face Detector: {'✅' if status['face_detector_loaded'] else '❌'}")
            print(f"GPU Available: {'✅' if status['gpu_available'] else '❌'}")
            
            # Test with sample images
            test_images = [
                "data/input/test_image_1.jpg",
                "data/input/test_image_2.jpg",
                "data/input/test_image_3.jpg"
            ]
            
            # Create test images if they don't exist
            for image_path in test_images:
                if not os.path.exists(image_path):
                    print(f"Creating test image: {image_path}")
                    create_demo_images([image_path])
            
            # Test compliance analysis
            for i, image_path in enumerate(test_images, 1):
                if os.path.exists(image_path):
                    print(f"\n📸 Analyzing Image {i}: {os.path.basename(image_path)}")
                    
                    # Sample metadata for testing
                    test_metadata = {
                        'title': f'Test Image {i}',
                        'description': 'A sample image for testing compliance checking',
                        'keywords': ['test', 'sample', 'demo', 'stock', 'photography'],
                        'Make': 'Canon' if i == 1 else 'Nikon',
                        'Model': 'EOS R5' if i == 1 else 'D850',
                        'Software': 'Adobe Photoshop 2024'
                    }
                    
                    # Perform AI compliance analysis
                    try:
                        result = compliance_checker.analyze(image_path, test_metadata)
                        
                        # Display results
                        print(f"   AI Compliance Score: {result.ai_compliance_score:.2f}")
                        print(f"   AI Confidence: {result.ai_confidence:.2f}")
                        print(f"   Final Decision: {result.final_decision}")
                        print(f"   Processing Time: {result.processing_time:.2f}s")
                        print(f"   Models Used: {', '.join(result.models_used)}")
                        print(f"   Fallback Used: {'Yes' if result.fallback_used else 'No'}")
                        
                        # Logo detections
                        if result.ai_logo_detections:
                            print(f"   🏷️  Logo Detections: {len(result.ai_logo_detections)}")
                            for logo in result.ai_logo_detections:
                                print(f"      - {logo.text_detected} (Risk: {logo.risk_level}, Confidence: {logo.ai_confidence:.2f})")
                        
                        # Privacy violations
                        if result.ai_privacy_violations:
                            print(f"   🔒 Privacy Violations: {len(result.ai_privacy_violations)}")
                            for violation in result.ai_privacy_violations:
                                print(f"      - {violation.violation_type} (Severity: {violation.severity}, Confidence: {violation.ai_confidence:.2f})")
                                print(f"        Thai: {violation.description_thai}")
                        
                        # Content appropriateness
                        content = result.content_appropriateness
                        print(f"   📋 Content Appropriateness:")
                        print(f"      - Overall Score: {content.overall_score:.2f}")
                        print(f"      - Cultural Sensitivity: {content.cultural_sensitivity_score:.2f}")
                        print(f"      - Age Appropriateness: {content.age_appropriateness}")
                        
                        # Rejection reasons
                        if result.rejection_reasons:
                            print(f"   ❌ Rejection Reasons:")
                            for reason in result.rejection_reasons:
                                print(f"      - {reason}")
                        
                        if result.rejection_reasons_thai:
                            print(f"   ❌ เหตุผลการปฏิเสธ:")
                            for reason in result.rejection_reasons_thai:
                                print(f"      - {reason}")
                        
                        # AI reasoning
                        print(f"   🤖 AI Reasoning (EN): {result.ai_reasoning}")
                        print(f"   🤖 AI Reasoning (TH): {result.ai_reasoning_thai}")
                        
                    except Exception as e:
                        logger.error(f"Error analyzing {image_path}: {e}")
                        print(f"   ❌ Analysis failed: {e}")
                else:
                    print(f"   ⚠️  Image not found: {image_path}")
            
            print(f"\n✅ Performance mode '{mode}' testing completed")
        
        # Test batch processing capabilities
        print(f"\n📦 Testing Batch Processing...")
        existing_images = [img for img in test_images if os.path.exists(img)]
        
        if existing_images:
            print(f"Processing {len(existing_images)} images in batch...")
            
            # Process all images
            batch_results = []
            for image_path in existing_images:
                try:
                    result = compliance_checker.analyze(image_path)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Batch processing error for {image_path}: {e}")
            
            # Batch statistics
            if batch_results:
                total_images = len(batch_results)
                approved_count = len([r for r in batch_results if r.final_decision == 'approved'])
                rejected_count = len([r for r in batch_results if r.final_decision == 'rejected'])
                avg_score = sum(r.ai_compliance_score for r in batch_results) / total_images
                avg_confidence = sum(r.ai_confidence for r in batch_results) / total_images
                total_time = sum(r.processing_time for r in batch_results)
                
                print(f"\n📊 Batch Processing Results:")
                print(f"   Total Images: {total_images}")
                print(f"   Approved: {approved_count} ({approved_count/total_images*100:.1f}%)")
                print(f"   Rejected: {rejected_count} ({rejected_count/total_images*100:.1f}%)")
                print(f"   Average Compliance Score: {avg_score:.2f}")
                print(f"   Average Confidence: {avg_confidence:.2f}")
                print(f"   Total Processing Time: {total_time:.2f}s")
                print(f"   Average Time per Image: {total_time/total_images:.2f}s")
        
        # Test error handling
        print(f"\n🧪 Testing Error Handling...")
        try:
            # Test with non-existent image
            error_result = compliance_checker.analyze("non_existent_image.jpg")
            print(f"   Error handling test: {error_result.final_decision}")
            print(f"   Fallback used: {error_result.fallback_used}")
        except Exception as e:
            print(f"   Error handling test failed: {e}")
        
        print(f"\n🎉 AI Compliance Checker Demo Completed Successfully!")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)