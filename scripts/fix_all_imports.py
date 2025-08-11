#!/usr/bin/env python3
"""
Comprehensive import fixer for the reorganized project structure.
This script fixes all import paths after the backend reorganization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import re
import glob
import sys

def fix_imports_in_file(file_path):
    """Fix imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Define import mappings
        import_mappings = [
            # Core modules
            (r'from core\.', 'from backend.core.'),
            (r'import core\.', 'import backend.core.'),
            
            # Analyzers
            (r'from analyzers\.', 'from backend.analyzers.'),
            (r'import analyzers\.', 'import backend.analyzers.'),
            
            # Utils
            (r'from utils\.', 'from backend.utils.'),
            (r'import utils\.', 'import backend.utils.'),
            
            # Config
            (r'from config\.', 'from backend.config.'),
            (r'import config\.', 'import backend.config.'),
        ]
        
        # Apply import mappings
        for old_pattern, new_pattern in import_mappings:
            content = re.sub(old_pattern, new_pattern, content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def fix_all_imports():
    """Fix imports in all Python files"""
    print("🔧 FIXING ALL IMPORT PATHS")
    print("=" * 60)
    
    # Directories to process
    directories = ['demos', 'scripts', 'tests']
    
    fixed_files = []
    
    for directory in directories:
        if not os.path.exists(directory):
            continue
            
        print(f"\n📁 Processing {directory}/")
        
        python_files = glob.glob(f"{directory}/*.py")
        
        for file_path in python_files:
            if fix_imports_in_file(file_path):
                fixed_files.append(file_path)
                print(f"  ✅ Fixed {file_path}")
            else:
                print(f"  ⚪ No changes needed: {file_path}")
    
    print(f"\n📊 SUMMARY")
    print(f"Fixed {len(fixed_files)} files")
    
    if fixed_files:
        print("\n✅ Fixed files:")
        for file_path in fixed_files:
            print(f"  - {file_path}")
    
    print("\n🧪 Testing key files...")
    
    # Test some key files
    test_files = [
        'demos/demo_config_validation.py',
        'demos/demo_quality_analyzer.py',
        'scripts/view_database_report.py'
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            try:
                import subprocess
                result = subprocess.run(
                    f'python {test_file} --help', 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                if result.returncode == 0 or 'usage:' in result.stdout or 'Adobe Stock' in result.stdout:
                    print(f"✅ {test_file} - OK")
                else:
                    print(f"⚠️  {test_file} - May have issues")
            except Exception:
                print(f"❓ {test_file} - Could not test")

if __name__ == "__main__":
    fix_all_imports()