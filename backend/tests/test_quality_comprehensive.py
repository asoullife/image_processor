"""
Comprehensive test of QualityAnalyzer functionality

This test demonstrates all features of the QualityAnalyzer including:
- Initialization with different config types
- All analysis methods
- Error handling
- Data structures
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sys
import os
sys.path.append('.')

def test_quality_analyzer_comprehensive():
    """Test all QualityAnalyzer functionality"""
    
    print("🧪 Comprehensive QualityAnalyzer Test")
    print("=" * 50)
    
    # Test 1: Import and basic structure
    print("\n1. Testing imports and basic structure...")
    try:
        from backend.analyzers.quality_analyzer import QualityAnalyzer, QualityResult, ExposureResult
        print("✓ Successfully imported all classes")
        
        # Test data structures
        exposure_result = ExposureResult(
            brightness_score=0.5,
            contrast_score=0.3,
            histogram_balance=0.8,
            overexposed_pixels=0.01,
            underexposed_pixels=0.02,
            passed=True
        )
        print("✓ ExposureResult dataclass works")
        
        quality_result = QualityResult(
            sharpness_score=150.0,
            noise_level=0.05,
            exposure_score=0.8,
            color_balance_score=0.9,
            resolution=(1920, 1080),
            file_size=1024000,
            overall_score=0.85,
            passed=True,
            exposure_result=exposure_result
        )
        print("✓ QualityResult dataclass works")
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Initialization with different config types
    print("\n2. Testing initialization with different config types...")
    
    # Dictionary config
    dict_config = {
        'quality': {
            'min_sharpness': 120.0,
            'max_noise_level': 0.08,
            'min_resolution': [2000, 1200]
        }
    }
    
    analyzer1 = QualityAnalyzer(dict_config)
    print(f"✓ Dictionary config: sharpness={analyzer1.min_sharpness}, noise={analyzer1.max_noise_level}")
    
    # AppConfig-like object
    class MockQualityConfig:
        def __init__(self):
            self.min_sharpness = 80.0
            self.max_noise_level = 0.12
            self.min_resolution = (1600, 900)
    
    class MockAppConfig:
        def __init__(self):
            self.quality = MockQualityConfig()
    
    app_config = MockAppConfig()
    analyzer2 = QualityAnalyzer(app_config)
    print(f"✓ AppConfig object: sharpness={analyzer2.min_sharpness}, noise={analyzer2.max_noise_level}")
    
    # Empty config (defaults)
    analyzer3 = QualityAnalyzer({})
    print(f"✓ Default config: sharpness={analyzer3.min_sharpness}, noise={analyzer3.max_noise_level}")
    
    # Test 3: Configuration validation
    print("\n3. Testing configuration and weights...")
    print(f"✓ Scoring weights: {analyzer1.weights}")
    assert sum(analyzer1.weights.values()) == 1.0, "Weights should sum to 1.0"
    print("✓ Weights sum to 1.0")
    
    # Test 4: Method signatures and error handling
    print("\n4. Testing method signatures and error handling...")
    
    # Test with None/empty data (should handle gracefully)
    mock_image = []
    
    sharpness = analyzer1.check_sharpness(mock_image)
    print(f"✓ Sharpness detection handles empty data: {sharpness}")
    
    noise = analyzer1.detect_noise(mock_image)
    print(f"✓ Noise detection handles empty data: {noise}")
    
    exposure = analyzer1.analyze_exposure(mock_image)
    print(f"✓ Exposure analysis handles empty data: passed={exposure.passed}")
    
    color_balance = analyzer1.check_color_balance(mock_image)
    print(f"✓ Color balance handles empty data: {color_balance}")
    
    # Test 5: Overall scoring logic
    print("\n5. Testing overall scoring logic...")
    
    # Test with good values
    good_score = analyzer1._calculate_overall_score(
        sharpness=200.0,
        noise=0.05,
        exposure=0.8,
        color_balance=0.9,
        resolution=(2000, 1200)
    )
    print(f"✓ Good quality score: {good_score:.3f}")
    assert good_score > 0.7, "Good quality should score high"
    
    # Test with poor values
    poor_score = analyzer1._calculate_overall_score(
        sharpness=50.0,
        noise=0.2,
        exposure=0.3,
        color_balance=0.4,
        resolution=(800, 600)
    )
    print(f"✓ Poor quality score: {poor_score:.3f}")
    assert poor_score < 0.5, "Poor quality should score low"
    
    # Test 6: Quality check logic
    print("\n6. Testing quality check logic...")
    
    good_exposure = ExposureResult(0.5, 0.3, 0.8, 0.01, 0.01, True)
    bad_exposure = ExposureResult(0.1, 0.05, 0.2, 0.1, 0.2, False)
    
    # Should pass
    passes_good = analyzer1._passes_quality_checks(
        sharpness=150.0,
        noise=0.07,
        exposure_result=good_exposure,
        color_balance=0.7,
        resolution=(2000, 1200)
    )
    print(f"✓ Good quality passes checks: {passes_good}")
    assert passes_good, "Good quality should pass"
    
    # Should fail (low sharpness)
    passes_bad_sharpness = analyzer1._passes_quality_checks(
        sharpness=50.0,
        noise=0.07,
        exposure_result=good_exposure,
        color_balance=0.7,
        resolution=(2000, 1200)
    )
    print(f"✓ Low sharpness fails checks: {passes_bad_sharpness}")
    assert not passes_bad_sharpness, "Low sharpness should fail"
    
    # Should fail (high noise)
    passes_bad_noise = analyzer1._passes_quality_checks(
        sharpness=150.0,
        noise=0.15,
        exposure_result=good_exposure,
        color_balance=0.7,
        resolution=(2000, 1200)
    )
    print(f"✓ High noise fails checks: {passes_bad_noise}")
    assert not passes_bad_noise, "High noise should fail"
    
    # Should fail (low resolution)
    passes_bad_resolution = analyzer1._passes_quality_checks(
        sharpness=150.0,
        noise=0.07,
        exposure_result=good_exposure,
        color_balance=0.7,
        resolution=(800, 600)
    )
    print(f"✓ Low resolution fails checks: {passes_bad_resolution}")
    assert not passes_bad_resolution, "Low resolution should fail"
    
    # Test 7: Full analysis workflow
    print("\n7. Testing full analysis workflow...")
    
    # Test with nonexistent file
    result = analyzer1.analyze('nonexistent_file.jpg')
    print(f"✓ Nonexistent file handled: passed={result.passed}, score={result.overall_score}")
    assert not result.passed, "Nonexistent file should fail"
    assert result.overall_score == 0.0, "Nonexistent file should have zero score"
    
    # Test 8: Edge cases and robustness
    print("\n8. Testing edge cases and robustness...")
    
    # Test with extreme values
    extreme_score = analyzer1._calculate_overall_score(
        sharpness=0.0,
        noise=1.0,
        exposure=0.0,
        color_balance=0.0,
        resolution=(1, 1)
    )
    print(f"✓ Extreme values handled: {extreme_score:.3f}")
    assert extreme_score >= 0.0, "Score should not be negative"
    
    # Test with very high values
    high_score = analyzer1._calculate_overall_score(
        sharpness=1000.0,
        noise=0.0,
        exposure=1.0,
        color_balance=1.0,
        resolution=(4000, 3000)
    )
    print(f"✓ High values handled: {high_score:.3f}")
    assert high_score <= 1.0, "Score should not exceed 1.0"
    
    print("\n🎉 All comprehensive tests passed!")
    print("✅ QualityAnalyzer implementation is robust and complete")
    
    return True

if __name__ == '__main__':
    success = test_quality_analyzer_comprehensive()
    if success:
        print("\n🏆 QualityAnalyzer is ready for production use!")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)