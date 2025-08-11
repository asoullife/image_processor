#!/usr/bin/env python3
"""
Demo script for DefectDetector functionality

This script demonstrates the defect detection capabilities including:
- Object defect detection
- Edge-based defect detection (cracks, breaks)
- Shape anomaly detection through template matching
- Confidence scoring and result aggregation
"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys
from backend.analyzers.defect_detector import DefectDetector
from backend.config.config_loader import AppConfig

def main():
    """Main demo function"""
    print("=== Adobe Stock Image Processor - Defect Detection Demo ===\n")
    
    # Load configuration
    try:
        config = AppConfig()
        print(f"✓ Configuration loaded successfully")
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        # Use default config for demo
        config = {
            'defect_detection': {
                'confidence_threshold': 0.5,
                'edge_threshold': 50,
                'model_path': None
            }
        }
        print("✓ Using default configuration for demo")
    
    # Initialize DefectDetector
    try:
        detector = DefectDetector(config)
        print(f"✓ DefectDetector initialized successfully")
        print(f"  - Confidence threshold: {detector.confidence_threshold}")
        print(f"  - Edge threshold: {detector.edge_threshold}")
    except Exception as e:
        print(f"✗ Error initializing DefectDetector: {e}")
        return
    
    # Test with sample images
    test_images = [
        'test_input/main_image_01.jpg',
        'test_input/main_image_02.png',
        'test_input/main_image_03.jpeg',
        'test_input/fake_image.jpg'
    ]
    
    print(f"\n=== Testing Defect Detection ===")
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"⚠ Skipping {image_path} (file not found)")
            continue
            
        print(f"\n--- Analyzing: {image_path} ---")
        
        try:
            # Perform defect detection
            result = detector.analyze(image_path)
            
            # Display results
            print(f"✓ Analysis completed")
            print(f"  - Defects detected: {result.defect_count}")
            print(f"  - Anomaly score: {result.anomaly_score:.3f}")
            print(f"  - Defect types: {result.defect_types}")
            print(f"  - Confidence scores: {[f'{score:.3f}' for score in result.confidence_scores]}")
            print(f"  - Passed quality check: {result.passed}")
            
            if result.detected_objects:
                print(f"  - Detected objects:")
                for i, obj in enumerate(result.detected_objects[:3]):  # Show first 3
                    print(f"    {i+1}. {obj.object_type}: {obj.defect_type} (confidence: {obj.confidence:.3f})")
                if len(result.detected_objects) > 3:
                    print(f"    ... and {len(result.detected_objects) - 3} more")
            
        except Exception as e:
            print(f"✗ Error analyzing {image_path}: {e}")
    
    # Test edge detection specifically
    print(f"\n=== Testing Edge Detection ===")
    
    try:
        from backend.analyzers.defect_detector import EdgeDetector
        edge_detector = EdgeDetector(edge_threshold=50)
        print(f"✓ EdgeDetector initialized")
        
        # Test severity determination
        severities = [
            (500, 2.0, 0.8),    # Low severity
            (2000, 6.0, 0.6),   # Medium severity
            (6000, 15.0, 0.3)   # High severity
        ]
        
        for area, aspect_ratio, solidity in severities:
            severity = edge_detector._determine_severity(area, aspect_ratio, solidity)
            print(f"  - Area: {area}, Aspect: {aspect_ratio}, Solidity: {solidity} → Severity: {severity}")
            
    except Exception as e:
        print(f"✗ Error testing edge detection: {e}")
    
    # Test shape matching
    print(f"\n=== Testing Shape Matching ===")
    
    try:
        from backend.analyzers.defect_detector import ShapeMatcher
        shape_matcher = ShapeMatcher()
        print(f"✓ ShapeMatcher initialized")
        
        # Test similarity calculation
        print(f"  - Shape similarity calculation available")
        
    except Exception as e:
        print(f"✗ Error testing shape matching: {e}")
    
    # Test anomaly detection
    print(f"\n=== Testing Anomaly Detection ===")
    
    try:
        from backend.analyzers.defect_detector import AnomalyDetector
        anomaly_detector = AnomalyDetector(confidence_threshold=0.5)
        print(f"✓ AnomalyDetector initialized")
        print(f"  - Confidence threshold: {anomaly_detector.confidence_threshold}")
        
    except Exception as e:
        print(f"✗ Error testing anomaly detection: {e}")
    
    print(f"\n=== Demo Complete ===")
    print(f"DefectDetector is ready for integration with the main processing pipeline.")


if __name__ == '__main__':
    main()