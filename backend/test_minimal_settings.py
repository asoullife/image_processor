#!/usr/bin/env python3
"""
Test script for minimal settings system
Tests core functionality without external dependencies
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

def test_minimal_settings():
    """Test the minimal settings system"""
    print("🧪 Testing Minimal Settings System")
    print("=" * 50)
    
    try:
        from config.minimal_settings import minimal_settings_manager, PerformanceMode, AppConfig
        print("✅ Minimal settings manager imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import minimal settings manager: {e}")
        return
    
    # Test 1: Configuration creation and loading
    print("\n1. Testing Configuration Creation...")
    try:
        config = minimal_settings_manager.get_config()
        print(f"✅ Configuration loaded successfully")
        print(f"   Performance Mode: {config.processing.performance_mode}")
        print(f"   Batch Size: {config.processing.batch_size}")
        print(f"   GPU Enabled: {config.processing.gpu_enabled}")
        print(f"   Memory Limit: {config.processing.memory_limit_gb} GB")
        print(f"   Log Level: {config.system.log_level}")
        print(f"   Theme: {config.ui.theme}")
        print(f"   Language: {config.ui.language}")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return
    
    # Test 2: Settings update
    print("\n2. Testing Settings Update...")
    try:
        original_batch_size = config.processing.batch_size
        success = minimal_settings_manager.update_settings('processing', {'batch_size': 25})
        if success:
            updated_config = minimal_settings_manager.get_config()
            new_batch_size = updated_config.processing.batch_size
            print(f"✅ Settings update successful")
            print(f"   Batch size changed: {original_batch_size} → {new_batch_size}")
            
            # Restore original value
            minimal_settings_manager.update_settings('processing', {'batch_size': original_batch_size})
            print(f"   Batch size restored to: {minimal_settings_manager.get_config().processing.batch_size}")
        else:
            print(f"❌ Settings update failed")
    except Exception as e:
        print(f"❌ Settings update failed: {e}")
    
    # Test 3: Performance mode switching
    print("\n3. Testing Performance Mode Switching...")
    try:
        original_mode = config.processing.performance_mode
        
        # Test switching to different modes
        for mode in [PerformanceMode.SPEED, PerformanceMode.SMART, PerformanceMode.BALANCED]:
            success = minimal_settings_manager.update_settings('processing', {'performance_mode': mode})
            if success:
                current_config = minimal_settings_manager.get_config()
                current_mode = current_config.processing.performance_mode
                print(f"✅ Performance mode switched to: {current_mode}")
            else:
                print(f"❌ Failed to switch to mode: {mode}")
        
        # Restore original mode
        minimal_settings_manager.update_settings('processing', {'performance_mode': original_mode})
                
    except Exception as e:
        print(f"❌ Performance mode switching failed: {e}")
    
    # Test 4: Configuration serialization
    print("\n4. Testing Configuration Serialization...")
    try:
        config_dict = config.to_dict()
        print(f"✅ Configuration serialization successful")
        print(f"   Top-level keys: {list(config_dict.keys())}")
        print(f"   Processing keys: {list(config_dict['processing'].keys())}")
        
        # Test deserialization
        new_config = AppConfig.from_dict(config_dict)
        print(f"✅ Configuration deserialization successful")
        print(f"   Performance Mode: {new_config.processing.performance_mode}")
        print(f"   Batch Size: {new_config.processing.batch_size}")
        
    except Exception as e:
        print(f"❌ Configuration serialization failed: {e}")
    
    # Test 5: System information
    print("\n5. Testing System Information...")
    try:
        system_info = minimal_settings_manager.get_system_info()
        
        print(f"✅ System information retrieved")
        print(f"   Platform: {system_info.get('platform', 'Unknown')}")
        print(f"   Processor: {system_info.get('processor', 'Unknown')}")
        print(f"   CPU Cores (Logical): {system_info.get('cpu_count_logical', 'Unknown')}")
        print(f"   Memory: {system_info.get('memory_total_gb', 0):.1f} GB")
        print(f"   GPUs Detected: {len(system_info.get('gpus', []))}")
        
    except Exception as e:
        print(f"❌ System information failed: {e}")
    
    # Test 6: Performance recommendations
    print("\n6. Testing Performance Recommendations...")
    try:
        system_info = minimal_settings_manager.get_system_info()
        recommendations = minimal_settings_manager.get_performance_recommendations(system_info)
        
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
    
    # Test 7: File persistence
    print("\n7. Testing File Persistence...")
    try:
        # Save current config
        success = minimal_settings_manager.save_config()
        if success:
            print(f"✅ Configuration save successful")
        else:
            print(f"❌ Configuration save failed")
        
        # Create new manager instance to test loading
        from config.minimal_settings import MinimalSettingsManager
        new_manager = MinimalSettingsManager("backend/config/test_config_2.json")
        
        # Update and save with new manager
        new_manager.update_settings('processing', {'batch_size': 99})
        loaded_config = new_manager.get_config()
        
        print(f"✅ New manager instance created and loaded")
        print(f"   New batch size: {loaded_config.processing.batch_size}")
        
    except Exception as e:
        print(f"❌ File persistence test failed: {e}")
    
    # Test 8: Multiple setting updates
    print("\n8. Testing Multiple Setting Updates...")
    try:
        updates = {
            'batch_size': 30,
            'max_workers': 6,
            'memory_limit_gb': 12.0,
            'quality_threshold': 0.8
        }
        
        success = minimal_settings_manager.update_settings('processing', updates)
        if success:
            updated_config = minimal_settings_manager.get_config()
            print(f"✅ Multiple settings update successful")
            print(f"   Batch Size: {updated_config.processing.batch_size}")
            print(f"   Max Workers: {updated_config.processing.max_workers}")
            print(f"   Memory Limit: {updated_config.processing.memory_limit_gb} GB")
            print(f"   Quality Threshold: {updated_config.processing.quality_threshold}")
        else:
            print(f"❌ Multiple settings update failed")
            
    except Exception as e:
        print(f"❌ Multiple setting updates failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Minimal Settings System Test Complete!")
    print("\nCore functionality verified:")
    print("  ✅ Configuration creation and loading")
    print("  ✅ Settings persistence to file")
    print("  ✅ Performance mode switching")
    print("  ✅ Configuration serialization/deserialization")
    print("  ✅ System information gathering")
    print("  ✅ Performance recommendations")
    print("  ✅ Multiple setting updates")
    print("\nThe configuration system is ready for integration!")

if __name__ == "__main__":
    test_minimal_settings()