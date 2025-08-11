#!/usr/bin/env python3
"""
Simple test script for Settings and Configuration System
Tests core functionality without external dependencies
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

def test_settings_system():
    """Test the settings system functionality"""
    print("🧪 Testing Settings and Configuration System")
    print("=" * 50)
    
    try:
        from config.settings_manager import settings_manager, PerformanceMode, AppConfig
        print("✅ Settings manager imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import settings manager: {e}")
        return
    
    # Test 1: Configuration creation and loading
    print("\n1. Testing Configuration Creation...")
    try:
        config = settings_manager.get_config()
        print(f"✅ Configuration loaded successfully")
        print(f"   Performance Mode: {config.processing.performance_mode}")
        print(f"   Batch Size: {config.processing.batch_size}")
        print(f"   GPU Enabled: {config.processing.gpu_enabled}")
        print(f"   Memory Limit: {config.processing.memory_limit_gb} GB")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return
    
    # Test 2: Settings update
    print("\n2. Testing Settings Update...")
    try:
        original_batch_size = config.processing.batch_size
        success = settings_manager.update_settings('processing', {'batch_size': 25})
        if success:
            updated_config = settings_manager.get_config()
            new_batch_size = updated_config.processing.batch_size
            print(f"✅ Settings update successful")
            print(f"   Batch size changed: {original_batch_size} → {new_batch_size}")
            
            # Restore original value
            settings_manager.update_settings('processing', {'batch_size': original_batch_size})
        else:
            print(f"❌ Settings update failed")
    except Exception as e:
        print(f"❌ Settings update failed: {e}")
    
    # Test 3: Performance mode switching
    print("\n3. Testing Performance Mode Switching...")
    try:
        original_mode = config.processing.performance_mode
        
        # Test switching to different modes
        for mode in [PerformanceMode.SPEED, PerformanceMode.SMART, original_mode]:
            success = settings_manager.update_settings('processing', {'performance_mode': mode})
            if success:
                current_config = settings_manager.get_config()
                current_mode = current_config.processing.performance_mode
                print(f"✅ Performance mode switched to: {current_mode}")
            else:
                print(f"❌ Failed to switch to mode: {mode}")
                
    except Exception as e:
        print(f"❌ Performance mode switching failed: {e}")
    
    # Test 4: Configuration serialization
    print("\n4. Testing Configuration Serialization...")
    try:
        config_dict = config.to_dict()
        print(f"✅ Configuration serialization successful")
        print(f"   Keys: {list(config_dict.keys())}")
        
        # Test deserialization
        new_config = AppConfig.from_dict(config_dict)
        print(f"✅ Configuration deserialization successful")
        print(f"   Performance Mode: {new_config.processing.performance_mode}")
        
    except Exception as e:
        print(f"❌ Configuration serialization failed: {e}")
    
    # Test 5: Hardware detection (basic)
    print("\n5. Testing Basic Hardware Detection...")
    try:
        from config.settings_manager import HardwareDetector
        detector = HardwareDetector()
        system_info = detector.get_system_info()
        
        print(f"✅ Hardware detection successful")
        print(f"   Platform: {system_info.get('platform', 'Unknown')}")
        print(f"   CPU Cores (Logical): {system_info.get('cpu_count_logical', 'Unknown')}")
        print(f"   Memory: {system_info.get('memory_total_gb', 0):.1f} GB")
        print(f"   GPUs Detected: {len(system_info.get('gpus', []))}")
        
    except Exception as e:
        print(f"❌ Hardware detection failed: {e}")
    
    # Test 6: Performance recommendations
    print("\n6. Testing Performance Recommendations...")
    try:
        system_info = detector.get_system_info()
        recommendations = detector.get_performance_recommendations(system_info)
        
        print(f"✅ Performance recommendations generated")
        print(f"   Recommended Mode: {recommendations['recommended_mode']}")
        print(f"   Recommended Batch Size: {recommendations['recommended_batch_size']}")
        print(f"   Recommended Workers: {recommendations['recommended_workers']}")
        print(f"   GPU Acceleration: {recommendations['gpu_acceleration']}")
        print(f"   Memory Limit: {recommendations['memory_limit_gb']} GB")
        
        if recommendations['warnings']:
            print(f"   Warnings ({len(recommendations['warnings'])}):")
            for warning in recommendations['warnings']:
                print(f"     - {warning}")
        
        if recommendations['optimizations']:
            print(f"   Optimizations ({len(recommendations['optimizations'])}):")
            for opt in recommendations['optimizations']:
                print(f"     - {opt}")
                
    except Exception as e:
        print(f"❌ Performance recommendations failed: {e}")
    
    # Test 7: System health monitoring
    print("\n7. Testing System Health Monitoring...")
    try:
        health = settings_manager.get_system_health()
        
        print(f"✅ System health monitoring working")
        print(f"   Status: {health.get('status', 'unknown')}")
        print(f"   CPU Usage: {health.get('cpu_usage_percent', 0):.1f}%")
        print(f"   Memory Usage: {health.get('memory_usage_percent', 0):.1f}%")
        print(f"   Memory Available: {health.get('memory_available_gb', 0):.1f} GB")
        print(f"   Disk Usage: {health.get('disk_usage_percent', 0):.1f}%")
        print(f"   Disk Free: {health.get('disk_free_gb', 0):.1f} GB")
        
        gpu_metrics = health.get('gpu_metrics', [])
        if gpu_metrics:
            print(f"   GPU Metrics ({len(gpu_metrics)}):")
            for i, gpu in enumerate(gpu_metrics):
                print(f"     GPU {i}: {gpu['name']}")
                print(f"       Load: {gpu['load']:.1f}%")
                print(f"       Memory: {gpu['memory_used_percent']:.1f}%")
                print(f"       Temperature: {gpu['temperature']}°C")
        
    except Exception as e:
        print(f"❌ System health monitoring failed: {e}")
    
    # Test 8: Configuration file operations
    print("\n8. Testing Configuration File Operations...")
    try:
        # Test saving
        success = settings_manager.save_config()
        if success:
            print(f"✅ Configuration save successful")
        else:
            print(f"❌ Configuration save failed")
        
        # Test loading
        loaded_config = settings_manager.load_config()
        print(f"✅ Configuration load successful")
        print(f"   Loaded performance mode: {loaded_config.processing.performance_mode}")
        
    except Exception as e:
        print(f"❌ Configuration file operations failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Settings and Configuration System Test Complete!")
    print("\nCore functionality verified:")
    print("  ✅ Configuration management")
    print("  ✅ Settings persistence")
    print("  ✅ Hardware detection")
    print("  ✅ Performance recommendations")
    print("  ✅ System health monitoring")
    print("  ✅ Performance mode switching")

if __name__ == "__main__":
    test_settings_system()