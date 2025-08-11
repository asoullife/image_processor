"""
Demo script showing QualityAnalyzer integration with the Adobe Stock Image Processor

This demonstrates how the QualityAnalyzer integrates with:
- Configuration system
- File management
- Logging system
- Overall processing pipeline
"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config.config_loader import ConfigLoader
from backend.utils.logger import initialize_logging
from backend.analyzers.quality_analyzer import QualityAnalyzer
from backend.utils.file_manager import FileManager

def demo_quality_analyzer():
    """Demonstrate QualityAnalyzer integration"""
    
    print("🎯 Adobe Stock Image Processor - Quality Analyzer Demo")
    print("=" * 60)
    
    # 1. Setup logging
    print("\n1. Setting up logging system...")
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    initialize_logging(config.logging)
    print("✓ Logging system initialized")
    
    # 2. Initialize QualityAnalyzer with loaded config
    print("\n2. Initializing QualityAnalyzer...")
    analyzer = QualityAnalyzer(config)
    print(f"✓ QualityAnalyzer initialized with:")
    print(f"  - Minimum sharpness: {analyzer.min_sharpness}")
    print(f"  - Maximum noise level: {analyzer.max_noise_level}")
    print(f"  - Minimum resolution: {analyzer.min_resolution}")
    print(f"  - Scoring weights: {analyzer.weights}")
    
    # 3. Scan for images using FileManager
    print("\n3. Scanning for test images...")
    file_manager = FileManager()
    test_images = file_manager.scan_images('backend/data/input')
    print(f"✓ Found {len(test_images)} images to analyze")
    
    # 4. Analyze a sample of images
    print("\n4. Analyzing sample images...")
    print("-" * 60)
    
    sample_images = test_images[:3]  # Analyze first 3 images
    results = []
    
    for i, image_path in enumerate(sample_images, 1):
        print(f"\n📸 Image {i}: {os.path.basename(image_path)}")
        
        # Analyze image quality
        result = analyzer.analyze(image_path)
        results.append(result)
        
        # Display results
        print(f"   Resolution: {result.resolution[0]}x{result.resolution[1]}")
        print(f"   File size: {result.file_size:,} bytes")
        print(f"   Sharpness score: {result.sharpness_score:.2f}")
        print(f"   Noise level: {result.noise_level:.4f}")
        print(f"   Exposure score: {result.exposure_score:.3f}")
        print(f"   Color balance: {result.color_balance_score:.3f}")
        print(f"   Overall score: {result.overall_score:.3f}")
        
        # Quality assessment
        if result.passed:
            print("   ✅ PASSED - Suitable for Adobe Stock")
        else:
            print("   ❌ FAILED - Quality issues detected")
            
        # Show detailed exposure info if available
        if result.exposure_result:
            exp = result.exposure_result
            print(f"   Exposure details:")
            print(f"     - Brightness: {exp.brightness_score:.3f}")
            print(f"     - Contrast: {exp.contrast_score:.3f}")
            print(f"     - Overexposed pixels: {exp.overexposed_pixels:.3f}")
            print(f"     - Underexposed pixels: {exp.underexposed_pixels:.3f}")
    
    # 5. Summary statistics
    print(f"\n5. Analysis Summary")
    print("-" * 60)
    
    if results:
        total_images = len(results)
        passed_images = sum(1 for r in results if r.passed)
        failed_images = total_images - passed_images
        
        avg_sharpness = sum(r.sharpness_score for r in results) / total_images
        avg_noise = sum(r.noise_level for r in results) / total_images
        avg_overall = sum(r.overall_score for r in results) / total_images
        
        print(f"📊 Quality Analysis Results:")
        print(f"   Total images analyzed: {total_images}")
        print(f"   Passed quality checks: {passed_images} ({passed_images/total_images*100:.1f}%)")
        print(f"   Failed quality checks: {failed_images} ({failed_images/total_images*100:.1f}%)")
        print(f"   Average sharpness: {avg_sharpness:.2f}")
        print(f"   Average noise level: {avg_noise:.4f}")
        print(f"   Average overall score: {avg_overall:.3f}")
        
        # Quality recommendations
        print(f"\n💡 Recommendations:")
        if avg_sharpness < analyzer.min_sharpness:
            print(f"   - Consider improving image sharpness (current: {avg_sharpness:.1f}, required: {analyzer.min_sharpness})")
        if avg_noise > analyzer.max_noise_level:
            print(f"   - Reduce image noise (current: {avg_noise:.3f}, max allowed: {analyzer.max_noise_level})")
        if avg_overall < 0.7:
            print(f"   - Overall quality could be improved (current: {avg_overall:.3f})")
        if passed_images == total_images:
            print(f"   - All images meet Adobe Stock quality standards! 🎉")
    
    # 6. Integration with batch processing
    print(f"\n6. Integration Notes")
    print("-" * 60)
    print("🔗 QualityAnalyzer integrates with:")
    print("   - BatchProcessor: Analyzes images in configurable batches")
    print("   - ProgressTracker: Saves quality scores to SQLite database")
    print("   - FileManager: Copies approved images to organized output folders")
    print("   - ReportGenerator: Includes quality metrics in Excel/HTML reports")
    print("   - ConfigLoader: Uses centralized configuration management")
    print("   - Logger: Provides detailed logging of analysis process")
    
    print(f"\n🎯 QualityAnalyzer Demo Complete!")
    print("✅ Ready for integration into the full processing pipeline")

if __name__ == '__main__':
    try:
        demo_quality_analyzer()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()