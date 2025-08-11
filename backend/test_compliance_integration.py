#!/usr/bin/env python3
"""
Integration test for AI Compliance Checker with Unified AI Analyzer

This script tests the integration of the AI compliance checker with the unified analyzer
to ensure proper workflow and data flow.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from analyzers.analyzer_factory import AnalyzerFactory
    from analyzers.unified_ai_analyzer import UnifiedAIAnalyzer
    from config.config_loader import load_config
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you're running from the backend directory")
    sys.exit(1)

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

def test_compliance_integration():
    """Test AI compliance checker integration"""
    print("🔗 Testing AI Compliance Checker Integration")
    print("=" * 50)
    
    try:
        # Load configuration
        config = load_config()
        
        # Test 1: Analyzer Factory Integration
        print("\n1️⃣ Testing Analyzer Factory Integration...")
        factory = AnalyzerFactory(config)
        
        # Get AI compliance checker from factory
        compliance_checker = factory.get_ai_compliance_checker()
        print(f"   ✅ AI Compliance Checker created: {type(compliance_checker).__name__}")
        
        # Test system status
        status = compliance_checker.get_system_status()
        print(f"   📊 Performance Mode: {status['performance_mode']}")
        print(f"   🔧 OCR Reader: {'✅' if status['ocr_reader_loaded'] else '❌'}")
        print(f"   👤 Face Detector: {'✅' if status['face_detector_loaded'] else '❌'}")
        
        # Test 2: Unified AI Analyzer Integration
        print("\n2️⃣ Testing Unified AI Analyzer Integration...")
        unified_analyzer = factory.get_unified_ai_analyzer()
        print(f"   ✅ Unified AI Analyzer created: {type(unified_analyzer).__name__}")
        
        # Check that compliance checker is integrated
        has_compliance = hasattr(unified_analyzer, 'compliance_checker')
        print(f"   🔗 Compliance Checker integrated: {'✅' if has_compliance else '❌'}")
        
        if has_compliance:
            compliance_type = type(unified_analyzer.compliance_checker).__name__
            print(f"   📋 Compliance Checker type: {compliance_type}")
        
        # Test 3: Performance Mode Synchronization
        print("\n3️⃣ Testing Performance Mode Synchronization...")
        test_modes = ['speed', 'balanced', 'smart']
        
        for mode in test_modes:
            print(f"   🚀 Setting mode to: {mode}")
            unified_analyzer.set_performance_mode(mode)
            
            # Check that all analyzers have the same mode
            quality_mode = unified_analyzer.quality_analyzer.performance_mode
            defect_mode = unified_analyzer.defect_detector.performance_mode
            similarity_mode = unified_analyzer.similarity_finder.performance_mode
            compliance_mode = unified_analyzer.compliance_checker.performance_mode
            
            modes_match = all(m == mode for m in [quality_mode, defect_mode, similarity_mode, compliance_mode])
            print(f"      Modes synchronized: {'✅' if modes_match else '❌'}")
            
            if not modes_match:
                print(f"      Quality: {quality_mode}, Defect: {defect_mode}")
                print(f"      Similarity: {similarity_mode}, Compliance: {compliance_mode}")
        
        # Test 4: End-to-End Analysis
        print("\n4️⃣ Testing End-to-End Analysis...")
        
        # Create test image
        test_image = create_test_image()
        print(f"   📸 Created test image: {os.path.basename(test_image)}")
        
        # Sample metadata
        test_metadata = {
            'title': 'Integration Test Image',
            'description': 'Testing AI compliance integration',
            'keywords': ['test', 'integration', 'compliance'],
            'Make': 'Test Camera',
            'Model': 'Integration Model'
        }
        
        try:
            # Perform unified analysis
            print("   🔍 Performing unified AI analysis...")
            result = unified_analyzer.analyze_single_image(test_image, test_metadata)
            
            # Verify result structure
            print(f"   📊 Analysis completed:")
            print(f"      Overall Score: {result.overall_score:.2f}")
            print(f"      Final Decision: {result.final_decision}")
            print(f"      Confidence: {result.confidence_score:.2f}")
            print(f"      Processing Time: {result.processing_time:.3f}s")
            print(f"      Models Used: {len(result.models_used)}")
            print(f"      Fallback Used: {'Yes' if result.fallback_used else 'No'}")
            
            # Verify compliance result is included
            has_compliance_result = hasattr(result, 'compliance_result')
            print(f"   🔗 Compliance Result included: {'✅' if has_compliance_result else '❌'}")
            
            if has_compliance_result:
                compliance_result = result.compliance_result
                print(f"      Compliance Score: {compliance_result.ai_compliance_score:.2f}")
                print(f"      Logo Detections: {len(compliance_result.ai_logo_detections)}")
                print(f"      Privacy Violations: {len(compliance_result.ai_privacy_violations)}")
                print(f"      Content Score: {compliance_result.content_appropriateness.overall_score:.2f}")
            
            # Verify reasoning includes compliance
            reasoning_includes_compliance = 'compliance' in result.ai_reasoning.lower()
            print(f"   📝 Reasoning includes compliance: {'✅' if reasoning_includes_compliance else '❌'}")
            
            print("   ✅ End-to-end analysis successful!")
            
        except Exception as e:
            print(f"   ❌ End-to-end analysis failed: {e}")
        
        finally:
            # Cleanup test image
            if os.path.exists(test_image):
                os.unlink(test_image)
        
        # Test 5: System Status Integration
        print("\n5️⃣ Testing System Status Integration...")
        
        system_status = unified_analyzer.get_system_status()
        has_compliance_status = 'compliance_checker' in system_status
        print(f"   📊 Compliance status included: {'✅' if has_compliance_status else '❌'}")
        
        if has_compliance_status:
            compliance_status = system_status['compliance_checker']
            print(f"   🔧 Compliance performance mode: {compliance_status.get('performance_mode', 'unknown')}")
            print(f"   🌐 Supported languages: {compliance_status.get('supported_languages', [])}")
        
        print("\n✅ Integration testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    success = test_compliance_integration()
    
    if success:
        print("\n🎉 All integration tests passed!")
        return 0
    else:
        print("\n💥 Some integration tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)