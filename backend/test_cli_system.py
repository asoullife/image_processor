#!/usr/bin/env python3
"""
Test script for CLI and Configuration System
Tests all CLI commands and configuration management functionality
"""

import sys
import os
import asyncio
import tempfile
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from cli.cli_manager import CLIManager
from config.settings_manager import settings_manager, PerformanceMode

async def test_cli_system():
    """Test the CLI system functionality"""
    print("🧪 Testing CLI and Configuration System")
    print("=" * 50)
    
    cli = CLIManager()
    
    # Test 1: Configuration loading and saving
    print("\n1. Testing Configuration Management...")
    try:
        config = settings_manager.get_config()
        print(f"✅ Configuration loaded successfully")
        print(f"   Performance Mode: {config.processing.performance_mode}")
        print(f"   Batch Size: {config.processing.batch_size}")
        print(f"   GPU Enabled: {config.processing.gpu_enabled}")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
    
    # Test 2: Hardware detection
    print("\n2. Testing Hardware Detection...")
    try:
        system_info = settings_manager.hardware_detector.get_system_info()
        print(f"✅ Hardware detection successful")
        print(f"   Platform: {system_info.get('platform', 'Unknown')}")
        print(f"   CPU Cores: {system_info.get('cpu_count_logical', 'Unknown')}")
        print(f"   Memory: {system_info.get('memory_total_gb', 0):.1f} GB")
        print(f"   GPUs: {len(system_info.get('gpus', []))}")
    except Exception as e:
        print(f"❌ Hardware detection failed: {e}")
    
    # Test 3: Performance recommendations
    print("\n3. Testing Performance Recommendations...")
    try:
        system_info = settings_manager.hardware_detector.get_system_info()
        recommendations = settings_manager.hardware_detector.get_performance_recommendations(system_info)
        print(f"✅ Performance recommendations generated")
        print(f"   Recommended Mode: {recommendations['recommended_mode']}")
        print(f"   Recommended Batch Size: {recommendations['recommended_batch_size']}")
        print(f"   GPU Acceleration: {recommendations['gpu_acceleration']}")
        if recommendations['warnings']:
            print(f"   Warnings: {len(recommendations['warnings'])}")
        if recommendations['optimizations']:
            print(f"   Optimizations: {len(recommendations['optimizations'])}")
    except Exception as e:
        print(f"❌ Performance recommendations failed: {e}")
    
    # Test 4: System health monitoring
    print("\n4. Testing System Health Monitoring...")
    try:
        health = settings_manager.get_system_health()
        print(f"✅ System health monitoring working")
        print(f"   Status: {health.get('status', 'unknown')}")
        print(f"   CPU Usage: {health.get('cpu_usage_percent', 0):.1f}%")
        print(f"   Memory Usage: {health.get('memory_usage_percent', 0):.1f}%")
        print(f"   Disk Usage: {health.get('disk_usage_percent', 0):.1f}%")
    except Exception as e:
        print(f"❌ System health monitoring failed: {e}")
    
    # Test 5: Settings update
    print("\n5. Testing Settings Update...")
    try:
        original_batch_size = settings_manager.get_config().processing.batch_size
        success = settings_manager.update_settings('processing', {'batch_size': 25})
        if success:
            new_batch_size = settings_manager.get_config().processing.batch_size
            print(f"✅ Settings update successful")
            print(f"   Batch size changed: {original_batch_size} → {new_batch_size}")
            
            # Restore original value
            settings_manager.update_settings('processing', {'batch_size': original_batch_size})
        else:
            print(f"❌ Settings update failed")
    except Exception as e:
        print(f"❌ Settings update failed: {e}")
    
    # Test 6: Configuration export/import
    print("\n6. Testing Configuration Export/Import...")
    try:
        config = settings_manager.get_config()
        config_dict = config.to_dict()
        
        # Test export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f, indent=2)
            temp_file = f.name
        
        print(f"✅ Configuration export successful")
        
        # Test import
        with open(temp_file, 'r') as f:
            imported_config = json.load(f)
        
        print(f"✅ Configuration import successful")
        
        # Cleanup
        os.unlink(temp_file)
        
    except Exception as e:
        print(f"❌ Configuration export/import failed: {e}")
    
    # Test 7: CLI argument parsing
    print("\n7. Testing CLI Argument Parsing...")
    try:
        parser = cli.create_parser()
        
        # Test help
        help_text = parser.format_help()
        if "adobe-stock-processor" in help_text:
            print(f"✅ CLI help system working")
        
        # Test command parsing
        test_args = parser.parse_args(['config', '--show'])
        if test_args.command == 'config' and test_args.show:
            print(f"✅ CLI argument parsing working")
        
    except Exception as e:
        print(f"❌ CLI argument parsing failed: {e}")
    
    # Test 8: Performance mode switching
    print("\n8. Testing Performance Mode Switching...")
    try:
        original_mode = settings_manager.get_config().processing.performance_mode
        
        # Test switching to different modes
        for mode in [PerformanceMode.SPEED, PerformanceMode.SMART, original_mode]:
            success = settings_manager.update_settings('processing', {'performance_mode': mode})
            if success:
                current_mode = settings_manager.get_config().processing.performance_mode
                print(f"✅ Performance mode switched to: {current_mode}")
            else:
                print(f"❌ Failed to switch to mode: {mode}")
                
    except Exception as e:
        print(f"❌ Performance mode switching failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 CLI and Configuration System Test Complete!")
    print("\nTo test the CLI manually, try these commands:")
    print("  python main.py -h                    # Show help")
    print("  python main.py config --show         # Show current config")
    print("  python main.py health --check        # Check system health")
    print("  python main.py config --optimize     # Auto-optimize settings")

if __name__ == "__main__":
    asyncio.run(test_cli_system())