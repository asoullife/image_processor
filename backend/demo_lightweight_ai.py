#!/usr/bin/env python3
"""
Demo script for Lightweight AI/ML Enhancement System

This script demonstrates the lightweight AI/ML system capabilities:
- OpenCV-based computer vision
- ONNX model inference (CPU)
- Traditional image processing
- Robust fallback mechanisms
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def demo_lightweight_ai_model_manager():
    """Demo Lightweight AI Model Manager"""
    logger.info("🚀 Demonstrating Lightweight AI Model Manager")
    
    try:
        from analyzers.ai_model_manager import AIModelManager
        
        # Initialize with sample config
        config = {
            'processing': {'batch_size': 32},
            'quality': {'min_sharpness': 100.0}
        }
        
        model_manager = AIModelManager(config)
        
        # Show system status
        status = model_manager.get_system_status()
        logger.info("📊 System Status:")
        logger.info(f"  - CPU Cores: {status.get('cpu_cores', 'unknown')}")
        logger.info(f"  - OpenCV Available: {status.get('opencv_available', False)}")
        logger.info(f"  - ONNX Available: {status.get('onnx_available', False)}")
        logger.info(f"  - scikit-image Available: {status.get('skimage_available', False)}")
        logger.info(f"  - Processing Type: {status.get('processing_type', 'unknown')}")
        
        if 'opencv_info' in status:
            opencv_info = status['opencv_info']
            logger.info(f"  - OpenCV Version: {opencv_info.get('version', 'unknown')}")
            logger.info(f"  - OpenCV Threads: {opencv_info.get('threads', 'unknown')}")
            logger.info(f"  - OpenCL Available: {opencv_info.get('opencl_available', False)}")
        
        if 'onnx_info' in status:
            onnx_info = status['onnx_info']
            logger.info(f"  - ONNX Version: {onnx_info.get('version', 'unknown')}")
            logger.info(f"  - ONNX Providers: {onnx_info.get('providers', [])}")
        
        # Test performance modes
        logger.info("\n⚡ Testing Performance Modes:")
        for mode in ['speed', 'balanced', 'smart']:
            model_manager.set_performance_mode(mode)
            logger.info(f"  - {mode.capitalize()} mode configured")
        
        # Show available models
        available_models = status.get('available_models', [])
        logger.info(f"\n🤖 Available Models ({len(available_models)}):")
        for model in available_models:
            available = model_manager.is_model_available(model)
            status_icon = "✅" if available else "❌"
            logger.info(f"  {status_icon} {model}")
        
        # Get recommendations
        recommended = model_manager.get_recommended_models()
        logger.info(f"\n💡 Recommended Models: {recommended}")
        
        # Test model preloading
        logger.info("\n📥 Preloading recommended models...")
        model_manager.preload_models()
        
        logger.info("✅ Lightweight AI Model Manager demo completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return False

def demo_opencv_capabilities():
    """Demo OpenCV capabilities"""
    logger.info("\n🔍 Demonstrating OpenCV Capabilities")
    
    try:
        import cv2
        import numpy as np
        
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Basic operations
        logger.info("📸 Testing basic OpenCV operations:")
        
        # Convert to grayscale
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        logger.info(f"  ✅ Grayscale conversion: {gray.shape}")
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        logger.info(f"  ✅ Canny edge detection: {edges.shape}")
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        logger.info(f"  ✅ Gaussian blur: {blurred.shape}")
        
        # Feature detection
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        logger.info(f"  ✅ ORB feature detection: {len(keypoints)} keypoints")
        
        # Face detection (Haar cascade)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        logger.info(f"  ✅ Face detection: {len(faces)} faces detected")
        
        # Quality metrics
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        logger.info(f"  ✅ Sharpness (Laplacian variance): {laplacian_var:.2f}")
        
        logger.info("✅ OpenCV capabilities demo completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ OpenCV demo failed: {e}")
        return False

def demo_onnx_capabilities():
    """Demo ONNX Runtime capabilities"""
    logger.info("\n🧠 Demonstrating ONNX Runtime Capabilities")
    
    try:
        import onnxruntime as ort
        import numpy as np
        
        # Show available providers
        providers = ort.get_available_providers()
        logger.info(f"📋 Available ONNX providers: {providers}")
        
        # Create a simple test session (dummy model)
        logger.info("🔧 Testing ONNX Runtime setup:")
        logger.info(f"  ✅ ONNX Runtime version: {ort.__version__}")
        logger.info(f"  ✅ CPU provider available: {'CPUExecutionProvider' in providers}")
        
        # Test tensor operations
        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
        logger.info(f"  ✅ Test tensor created: {test_input.shape}")
        
        logger.info("✅ ONNX Runtime capabilities demo completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ ONNX demo failed: {e}")
        return False

def demo_scikit_image_capabilities():
    """Demo scikit-image capabilities"""
    logger.info("\n🔬 Demonstrating scikit-image Capabilities")
    
    try:
        from skimage import feature, filters, measure
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
        
        # Create test images
        image1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        image2 = image1 + np.random.randint(-10, 10, (100, 100), dtype=np.int16)
        image2 = np.clip(image2, 0, 255).astype(np.uint8)
        
        logger.info("🔍 Testing scikit-image operations:")
        
        # SSIM (Structural Similarity Index)
        ssim_score = ssim(image1, image2)
        logger.info(f"  ✅ SSIM calculation: {ssim_score:.3f}")
        
        # Local Binary Pattern
        lbp = feature.local_binary_pattern(image1, 8, 1, method='uniform')
        logger.info(f"  ✅ Local Binary Pattern: {lbp.shape}")
        
        # HOG features
        hog_features = feature.hog(image1, orientations=8, pixels_per_cell=(16, 16),
                                  cells_per_block=(1, 1))
        logger.info(f"  ✅ HOG features: {len(hog_features)} features")
        
        # Gaussian filter
        filtered = filters.gaussian(image1, sigma=1)
        logger.info(f"  ✅ Gaussian filter: {filtered.shape}")
        
        # Sobel edge detection
        edges = filters.sobel(image1)
        logger.info(f"  ✅ Sobel edge detection: {edges.shape}")
        
        # Region properties
        binary = image1 > 128
        labeled = measure.label(binary)
        props = measure.regionprops(labeled)
        logger.info(f"  ✅ Region analysis: {len(props)} regions")
        
        logger.info("✅ scikit-image capabilities demo completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ scikit-image demo failed: {e}")
        return False

def demo_performance_comparison():
    """Demo performance comparison between modes"""
    logger.info("\n⚡ Performance Comparison Demo")
    
    try:
        from analyzers.ai_model_manager import AIModelManager
        import numpy as np
        
        config = {'processing': {'batch_size': 16}}
        model_manager = AIModelManager(config)
        
        # Test different performance modes
        modes = ['speed', 'balanced', 'smart']
        results = {}
        
        for mode in modes:
            logger.info(f"\n🔧 Testing {mode} mode:")
            model_manager.set_performance_mode(mode)
            
            # Simulate batch processing
            start_time = time.time()
            
            # Simulate image processing workload
            for i in range(10):
                # Simulate lightweight processing
                test_data = np.random.randn(224, 224, 3)
                processed = np.mean(test_data, axis=2)  # Simple operation
            
            elapsed = time.time() - start_time
            results[mode] = elapsed
            
            logger.info(f"  ⏱️  Processing time: {elapsed:.3f}s")
            
            # Get performance recommendations
            if mode == 'smart':
                recommendations = model_manager.get_performance_recommendations()
                logger.info(f"  💡 Recommendations: {recommendations['recommended_mode']}")
                logger.info(f"     Batch size: {recommendations['recommended_batch_size']}")
                logger.info(f"     Reasons: {recommendations['reasons']}")
        
        # Show comparison
        logger.info("\n📊 Performance Summary:")
        fastest = min(results, key=results.get)
        for mode, time_taken in results.items():
            icon = "🏆" if mode == fastest else "⏱️"
            logger.info(f"  {icon} {mode.capitalize()}: {time_taken:.3f}s")
        
        logger.info("✅ Performance comparison demo completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance demo failed: {e}")
        return False

def main():
    """Run all lightweight AI demos"""
    logger.info("🚀 Starting Lightweight AI/ML Enhancement System Demo")
    logger.info("=" * 70)
    
    demos = [
        ("Lightweight AI Model Manager", demo_lightweight_ai_model_manager),
        ("OpenCV Capabilities", demo_opencv_capabilities),
        ("ONNX Runtime Capabilities", demo_onnx_capabilities),
        ("scikit-image Capabilities", demo_scikit_image_capabilities),
        ("Performance Comparison", demo_performance_comparison)
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        logger.info(f"\n{'='*20} {demo_name} {'='*20}")
        start_time = time.time()
        
        try:
            result = demo_func()
            results.append((demo_name, result))
            
            elapsed = time.time() - start_time
            logger.info(f"⏱️  {demo_name} completed in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"💥 {demo_name} crashed: {e}")
            results.append((demo_name, False))
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 DEMO SUMMARY")
    logger.info("=" * 70)
    
    passed = 0
    failed = 0
    
    for demo_name, result in results:
        status = "✅ SUCCESS" if result else "❌ FAILED"
        logger.info(f"{demo_name}: {status}")
        
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nTotal: {len(results)} demos")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("\n🎉 All Lightweight AI demos completed successfully!")
        logger.info("💡 The system is ready for lightweight AI/ML processing!")
        return 0
    else:
        logger.error(f"\n💥 {failed} demo(s) failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)