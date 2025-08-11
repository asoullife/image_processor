#!/usr/bin/env python3
"""Fix import paths after file reorganization"""

import os
import re
import glob
import sys

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def fix_imports_in_file(file_path):
    """Fix imports in a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix common import issues
        fixes = [
            # Add sys.path.insert for files that need it
            (r'^(#!/usr/bin/env python3\n""".*?"""\n\n)', 
             r'\1import sys\nimport os\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n\n'),
            
            # Fix relative imports that are now broken
            (r'from create_test_images import', 'from demos.create_test_images import'),
            (r'from demo_', 'from demos.demo_'),
            (r'from test_', 'from tests.test_'),
            (r'from view_database_report import', 'from scripts.view_database_report import'),
            (r'from create_.*_report import', lambda m: f'from scripts.{m.group(0).split()[1]} import'),
        ]
        
        for pattern, replacement in fixes:
            if callable(replacement):
                content = re.sub(pattern, replacement, content)
            else:
                content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
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
    """Add sys.path.insert to script files"""
    script_dirs = ['scripts', 'demos', 'tests']
    
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
                
                # Skip shebang and docstring
                for i, line in enumerate(lines):
                    if line.startswith('#!') or line.startswith('"""') or line.startswith("'''"):
                        continue
                    if line.strip() == '' or line.startswith('#'):
                        continue
                    if line.startswith('import ') or line.startswith('from '):
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
                
                print(f"✅ Fixed imports in {file_path}")
                
            except Exception as e:
                print(f"❌ Error fixing {file_path}: {e}")

def main():
    """Main function to fix all import issues"""
    print("🔧 FIXING IMPORT PATHS AFTER FILE REORGANIZATION")
    print("=" * 60)
    
    # Add sys.path.insert to all script files
    print("\n📝 Adding sys.path.insert to script files...")
    add_sys_path_to_scripts()
    
    print("\n✅ Import fixes completed!")
    print("\n🧪 Testing a few key files...")
    
    # Test some key files
    test_files = [
        'demos/demo_config_validation.py',
        'scripts/view_database_report.py',
        'demos/create_test_images.py'
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            try:
                # Try to import the file
                import subprocess
                result = subprocess.run(f'python {test_file} --help', shell=True, capture_output=True, text=True, timeout=5)
                if result.returncode == 0 or 'usage:' in result.stdout or 'Adobe Stock' in result.stdout:
                    print(f"✅ {test_file} - OK")
                else:
                    print(f"⚠️  {test_file} - May have issues")
            except:
                print(f"❓ {test_file} - Could not test")

if __name__ == "__main__":
    main()