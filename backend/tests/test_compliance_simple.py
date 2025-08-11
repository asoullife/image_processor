#!/usr/bin/env python3
"""
Simple test for AI Compliance Checker

This script tests the AI compliance checker independently to verify it works correctly.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def create_test_image():
    """Create a simple test image"""
    try:
        import numpy as np
        from PIL import Image
        
        # Create a simple test image
        image_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        image.save(temp_file.name, 'JPEG')
        temp_file.close()
        
        return temp_file.name
    except ImportError:
        # Fallback: create empty file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_file.close()
        return temp_file.name

def test_compliance_checker():
    """Test AI compliance checker independently"""
    print("🔍 Testing AI Compliance Checker (Simple)")
    print("=" * 50)
    
    try:
        # Import compliance checker directly
        from analyzers.ai_compliance_checker import AIComplianceChecker
        
        # Test configuration
        config = {
            'compliance': {
                'ai_logo_confidence': 0.7,
                'ai_face_confidence': 0.6,
                'content_safety_threshold': 0.8,
                'logo_detection_confidence': 0.7,
                'face_detection_enabled': True,
                'metadata_validation': True
            }
        }
        
        print("✅ Successfully imported AIComplianceChecker")
        
        # Initialize compliance checker
        compliance_checker = AIComplianceChecker(config)
        print("✅ Successfully initialized AIComplianceChecker")
        
        # Test system status
        status = compliance_checker.get_system_status()
        print(f"📊 System Status:")
        print(f"   Performance Mode: {status['performance_mode']}")
        print(f"   OCR Reader: {'✅' if status['ocr_reader_loaded'] else '❌'}")
        print(f"   Face Detector: {'✅' if status['face_detector_loaded'] else '❌'}")
        print(f"   GPU Available: {'✅' if status['gpu_available'] else '❌'}")
        print(f"   Supported Languages: {status['supported_languages']}")
        
        # Test performance mode setting
        print("\n🚀 Testing Performance Modes:")
        for mode in ['speed', 'balanced', 'smart']:
            compliance_checker.set_performance_mode(mode)
            print(f"   {mode}: ✅")
        
        # Create test image
        test_image = create_test_image()
        print(f"\n📸 Created test image: {os.path.basename(test_image)}")
        
        # Test metadata
        test_metadata = {
            'title': 'Simple Test Image',
            'description': 'Testing AI compliance checker',
            'keywords': ['test', 'simple', 'compliance'],
            'Make': 'Test Camera',
            'Model': 'Simple Model'
        }
        
        # Perform analysis
        print("🔍 Performing compliance analysis...")
        result = compliance_checker.analyze(test_image, test_metadata)
        
        print("✅ Analysis completed successfully!")
        print(f"📊 Results:")
        print(f"   AI Compliance Score: {result.ai_compliance_score:.2f}")
        print(f"   AI Confidence: {result.ai_confidence:.2f}")
        print(f"   Final Decision: {result.final_decision}")
        print(f"   Processing Time: {result.processing_time:.3f}s")
        print(f"   Models Used: {result.models_used}")
        print(f"   Fallback Used: {'Yes' if result.fallback_used else 'No'}")
        
        # Check result structure
        print(f"\n🔍 Result Structure Validation:")
        print(f"   Traditional Result: {'✅' if result.traditional_result else '❌'}")
        print(f"   Logo Detections: {len(result.ai_logo_detections)} items")
        print(f"   Privacy Violations: {len(result.ai_privacy_violations)} items")
        print(f"   Content Appropriateness: {'✅' if result.content_appropriateness else '❌'}")
        print(f"   Enhanced Metadata: {'✅' if result.enhanced_metadata_analysis else '❌'}")
        
        # Check reasoning
        print(f"\n💭 AI Reasoning:")
        print(f"   English: {result.ai_reasoning}")
        print(f"   Thai: {result.ai_reasoning_thai}")
        
        if result.rejection_reasons:
            print(f"\n❌ Rejection Reasons (EN): {result.rejection_reasons}")
        
        if result.rejection_reasons_thai:
            print(f"❌ Rejection Reasons (TH): {result.rejection_reasons_thai}")
        
        # Test error handling
        print(f"\n🧪 Testing Error Handling:")
        error_result = compliance_checker.analyze("non_existent_image.jpg")
        print(f"   Error handling: {'✅' if error_result.fallback_used else '❌'}")
        print(f"   Error decision: {error_result.final_decision}")
        
        # Cleanup
        if os.path.exists(test_image):
            os.unlink(test_image)
        
        print(f"\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    success = test_compliance_checker()
    
    if success:
        print("\n✅ Simple compliance checker test completed successfully!")
        return 0
    else:
        print("\n❌ Simple compliance checker test failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)