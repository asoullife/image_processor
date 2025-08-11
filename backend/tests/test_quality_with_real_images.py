"""
Test QualityAnalyzer with real test images
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sys
import os
sys.path.append('.')

try:
    from backend.analyzers.quality_analyzer import QualityAnalyzer, QualityResult
    from backend.config.config_loader import ConfigLoader
    
    print("Testing QualityAnalyzer with real images...")
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    
    # Initialize analyzer
    analyzer = QualityAnalyzer(config)
    print(f"✓ QualityAnalyzer initialized")
    print(f"  - min_sharpness: {analyzer.min_sharpness}")
    print(f"  - max_noise_level: {analyzer.max_noise_level}")
    print(f"  - min_resolution: {analyzer.min_resolution}")
    
    # Test with a few real images
    test_images = [
        'test_input/main_image_01.jpg',
        'test_input/main_image_02.png',
        'test_input/category_1/category_1_image_1.jpg'
    ]
    
    print("\nAnalyzing test images:")
    print("-" * 60)
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"\nAnalyzing: {image_path}")
            
            result = analyzer.analyze(image_path)
            
            print(f"  Resolution: {result.resolution}")
            print(f"  File size: {result.file_size:,} bytes")
            print(f"  Sharpness: {result.sharpness_score:.2f}")
            print(f"  Noise level: {result.noise_level:.4f}")
            print(f"  Exposure score: {result.exposure_score:.3f}")
            print(f"  Color balance: {result.color_balance_score:.3f}")
            print(f"  Overall score: {result.overall_score:.3f}")
            print(f"  Quality passed: {result.passed}")
            
            if result.exposure_result:
                print(f"  Exposure details:")
                print(f"    - Brightness: {result.exposure_result.brightness_score:.3f}")
                print(f"    - Contrast: {result.exposure_result.contrast_score:.3f}")
                print(f"    - Overexposed: {result.exposure_result.overexposed_pixels:.3f}")
                print(f"    - Underexposed: {result.exposure_result.underexposed_pixels:.3f}")
        else:
            print(f"❌ Image not found: {image_path}")
    
    # Test error handling
    print(f"\nTesting error handling with nonexistent file:")
    result = analyzer.analyze('nonexistent_image.jpg')
    print(f"  Result for nonexistent file: passed={result.passed}, score={result.overall_score}")
    
    print("\n🎉 QualityAnalyzer testing completed successfully!")
    
except ImportError as e:
    print(f"❌ Import error (expected if dependencies not installed): {e}")
    print("This is normal if OpenCV/NumPy are not installed.")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()