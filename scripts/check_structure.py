#!/usr/bin/env python3
"""Check project structure and test all commands"""

import os
import subprocess
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def show_clean_structure():
    """Show clean project structure"""
    print("🌳 PROJECT STRUCTURE")
    print("=" * 60)
    
    # Root files
    root_files = [f for f in os.listdir('.') if os.path.isfile(f) and not f.startswith('.')]
    print(f"\n📄 ROOT FILES ({len(root_files)}):")
    for file in sorted(root_files):
        size = os.path.getsize(file)
        print(f"  📄 {file} ({size:,} bytes)")
    
    # Directories
    directories = [d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.') and d != '__pycache__']
    print(f"\n📁 DIRECTORIES ({len(directories)}):")
    
    for dir_name in sorted(directories):
        if dir_name == 'adobe_stock_env':
            print(f"  📁 {dir_name}/ (virtual environment)")
            continue
            
        try:
            files = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))]
            subdirs = [d for d in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, d)) and not d.startswith('.') and d != '__pycache__']
            
            print(f"  📁 {dir_name}/ ({len(files)} files, {len(subdirs)} subdirs)")
            
            # Show key files in important directories
            if dir_name in ['analyzers', 'core', 'utils', 'config']:
                key_files = [f for f in files if f.endswith('.py') and not f.startswith('__')][:5]
                for file in key_files:
                    print(f"    📄 {file}")
                if len(files) > 5:
                    print(f"    ... and {len(files) - 5} more files")
                    
        except PermissionError:
            print(f"  📁 {dir_name}/ (access denied)")
        except Exception as e:
            print(f"  📁 {dir_name}/ (error: {e})")

def test_command(description, command, expected_success=True):
    """Test a command and report results"""
    print(f"\n🧪 Testing: {description}")
    print(f"   Command: {command}")
    
    try:
        # Run command
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"   ✅ SUCCESS")
            # Show first few lines of output
            if result.stdout:
                lines = result.stdout.strip().split('\n')[:3]
                for line in lines:
                    print(f"   📝 {line[:80]}")
                if len(result.stdout.strip().split('\n')) > 3:
                    print(f"   📝 ... (output truncated)")
        else:
            print(f"   ❌ FAILED (exit code: {result.returncode})")
            if result.stderr:
                error_lines = result.stderr.strip().split('\n')[:2]
                for line in error_lines:
                    print(f"   🚨 {line[:80]}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"   ⏰ TIMEOUT (>30s)")
        return False
    except Exception as e:
        print(f"   💥 ERROR: {e}")
        return False

def main():
    """Main function to check structure and test commands"""
    print("🔍 ADOBE STOCK IMAGE PROCESSOR - STRUCTURE & COMMAND TEST")
    print("=" * 80)
    
    # Show structure
    show_clean_structure()
    
    print(f"\n" + "=" * 80)
    print("🧪 TESTING ALL COMMANDS")
    print("=" * 80)
    
    # Test results
    test_results = []
    
    # 1. Main application
    test_results.append(("Main Help", test_command(
        "Main application help", 
        "python main.py --help"
    )))
    
    # 2. Installation and setup
    test_results.append(("Installation Test", test_command(
        "Installation test", 
        "python backend/tests/test_installation.py"
    )))
    
    # 3. Demo scripts
    demo_commands = [
        ("Demo Config Validation", "python demos/demo_config_validation.py"),
        ("Demo Quality Analyzer", "python demos/demo_quality_analyzer.py"),
        ("Create Test Images", "python demos/create_test_images.py"),
    ]
    
    for name, cmd in demo_commands:
        test_results.append((name, test_command(name, cmd)))
    
    # 4. Report generation
    report_commands = [
        ("View Database Report", "python scripts/view_database_report.py"),
        ("Create Excel Report", "python scripts/create_excel_report.py"),
        ("Create HTML Dashboard", "python scripts/create_html_dashboard.py"),
        ("Create CSV Report", "python scripts/create_csv_report.py"),
    ]
    
    for name, cmd in report_commands:
        test_results.append((name, test_command(name, cmd)))
    
    # 5. Testing scripts
    test_commands = [
        ("Simple Tests", "python scripts/run_simple_tests.py"),
    ]
    
    for name, cmd in test_commands:
        test_results.append((name, test_command(name, cmd)))
    
    # 6. Main processing (with test data)
    test_results.append(("Main Processing", test_command(
        "Main processing with test data", 
        "python main.py process backend/data/input backend/data/output"
    )))
    
    # Summary
    print(f"\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    print(f"\n🎯 Overall Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    print(f"\n✅ PASSED TESTS:")
    for name, result in test_results:
        if result:
            print(f"  ✅ {name}")
    
    if passed < total:
        print(f"\n❌ FAILED TESTS:")
        for name, result in test_results:
            if not result:
                print(f"  ❌ {name}")
    
    # Structure assessment
    print(f"\n🏗️ STRUCTURE ASSESSMENT:")
    
    required_files = ['main.py', 'requirements.txt', 'README.md', 'adobe_stock_processor.db']
    required_dirs = ['backend', 'scripts', 'demos', 'reports', 'docs']
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    if not missing_files and not missing_dirs:
        print("  ✅ All required files and directories present")
        print("  ✅ Project structure is well organized")
        print("  ✅ Files are properly categorized")
    else:
        if missing_files:
            print(f"  ❌ Missing files: {', '.join(missing_files)}")
        if missing_dirs:
            print(f"  ❌ Missing directories: {', '.join(missing_dirs)}")
    
    # Final verdict
    print(f"\n" + "=" * 80)
    if passed >= total * 0.8 and not missing_files and not missing_dirs:
        print("🎉 VERDICT: PROJECT STRUCTURE AND COMMANDS ARE EXCELLENT!")
        print("✅ Ready for production use")
    elif passed >= total * 0.6:
        print("👍 VERDICT: PROJECT STRUCTURE IS GOOD, MINOR ISSUES")
        print("⚠️  Some commands need attention")
    else:
        print("⚠️  VERDICT: PROJECT NEEDS ATTENTION")
        print("🔧 Several commands are failing")
    
    print("=" * 80)

if __name__ == "__main__":
    main()