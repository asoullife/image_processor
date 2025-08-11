#!/usr/bin/env python3
"""
Comprehensive import fixer for the reorganized project structure.
This script fixes all import paths after the backend reorganization.
"""

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
        
        # Define import mappings for backend modules
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
            
            # Database
            (r'from database\.', 'from backend.database.'),
            (r'import database\.', 'import backend.database.'),
            
            # API
            (r'from api\.', 'from backend.api.'),
            (r'import api\.', 'import backend.api.'),
        ]
        
        # Apply import mappings
        for old_pattern, new_pattern in import_mappings:
            content = re.sub(old_pattern, new_pattern, content)
        
        # Fix path references to test data
        path_mappings = [
            (r"'backend/data/input'", "'backend/data/input'"),
            (r'"backend/data/input"', '"backend/data/input"'),
            (r"'backend/data/output'", "'backend/data/output'"),
            (r'"backend/data/output"', '"backend/data/output"'),
        ]
        
        for old_path, new_path in path_mappings:
            content = re.sub(old_path, new_path, content)
        
        # Fix database path references
        db_path_mappings = [
            (r"'adobe_stock_processor\.db'", "'adobe_stock_processor.db'"),
            (r'"adobe_stock_processor\.db"', '"adobe_stock_processor.db"'),
        ]
        
        for old_db, new_db in db_path_mappings:
            content = re.sub(old_db, new_db, content)
        
        # Only write if content changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def add_sys_path_to_scripts():
    """Add sys.path.insert to script files that need it"""
    script_dirs = ['scripts', 'demos']
    
    for script_dir in script_dirs:
        if not os.path.exists(script_dir):
            continue
            
        python_files = glob.glob(f"{script_dir}/*.py")
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if sys.path.insert already exists
                if 'sys.path.insert' in content:
                    continue
                
                # Find the position to insert sys.path
                lines = content.split('\n')
                insert_pos = 0
                
                # Skip shebang and docstring, find first import
                in_docstring = False
                docstring_char = None
                
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    
                    # Handle shebang
                    if stripped.startswith('#!'):
                        continue
                    
                    # Handle docstrings
                    if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                        docstring_char = stripped[:3]
                        in_docstring = True
                        if stripped.count(docstring_char) >= 2:  # Single line docstring
                            in_docstring = False
                        continue
                    elif in_docstring and docstring_char in stripped:
                        in_docstring = False
                        continue
                    elif in_docstring:
                        continue
                    
                    # Skip empty lines and comments
                    if stripped == '' or stripped.startswith('#'):
                        continue
                    
                    # Found first non-docstring, non-comment line
                    if stripped.startswith('import ') or stripped.startswith('from '):
                        insert_pos = i
                        break
                
                # Insert sys.path.insert
                sys_path_lines = [
                    'import sys',
                    'import os',
                    'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))',
                    ''
                ]
                
                # Insert at the right position
                new_lines = lines[:insert_pos] + sys_path_lines + lines[insert_pos:]
                new_content = '\n'.join(new_lines)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print(f"✅ Added sys.path to {file_path}")
                
            except Exception as e:
                print(f"❌ Error fixing {file_path}: {e}")

def fix_backend_tests():
    """Fix imports in backend test files"""
    test_dir = 'backend/tests'
    if not os.path.exists(test_dir):
        return
    
    python_files = glob.glob(f"{test_dir}/*.py")
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix sys.path.insert for backend tests (different path)
            if 'sys.path.insert' in content:
                # Update existing sys.path.insert to point to project root
                content = re.sub(
                    r'sys\.path\.insert\(0, os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)\)',
                    'sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))',
                    content
                )
            
            # Only write if content changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ Fixed backend test {file_path}")
                
        except Exception as e:
            print(f"❌ Error fixing backend test {file_path}: {e}")

def fix_all_imports():
    """Fix imports in all Python files"""
    print("🔧 COMPREHENSIVE IMPORT PATH FIXING")
    print("=" * 60)
    
    # 1. Add sys.path.insert to scripts and demos
    print("\n📝 Adding sys.path.insert to script files...")
    add_sys_path_to_scripts()
    
    # 2. Fix backend test imports
    print("\n🧪 Fixing backend test imports...")
    fix_backend_tests()
    
    # 3. Fix imports in all directories
    directories = ['demos', 'scripts', 'backend/tests']
    
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
                    if result.stderr:
                        print(f"    Error: {result.stderr[:100]}...")
            except Exception as e:
                print(f"❓ {test_file} - Could not test: {e}")

if __name__ == "__main__":
    fix_all_imports()