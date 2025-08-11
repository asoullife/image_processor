#!/usr/bin/env python3
"""
Fix imports within backend modules to use proper relative imports.
"""

import os
import re
import glob

def fix_backend_imports():
    """Fix imports within backend modules"""
    print("🔧 FIXING BACKEND INTERNAL IMPORTS")
    print("=" * 60)
    
    # Get all Python files in backend directory
    backend_files = []
    for root, dirs, files in os.walk('backend'):
        for file in files:
            if file.endswith('.py'):
                backend_files.append(os.path.join(root, file))
    
    fixed_files = []
    
    for file_path in backend_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix imports within backend modules
            import_mappings = [
                # Config imports
                (r'from config\.', 'from backend.config.'),
                (r'import config\.', 'import backend.config.'),
                
                # Core imports
                (r'from core\.', 'from backend.core.'),
                (r'import core\.', 'import backend.core.'),
                
                # Analyzers imports
                (r'from analyzers\.', 'from backend.analyzers.'),
                (r'import analyzers\.', 'import backend.analyzers.'),
                
                # Utils imports
                (r'from utils\.', 'from backend.utils.'),
                (r'import utils\.', 'import backend.utils.'),
                
                # Database imports
                (r'from database\.', 'from backend.database.'),
                (r'import database\.', 'import backend.database.'),
                
                # API imports
                (r'from api\.', 'from backend.api.'),
                (r'import api\.', 'import backend.api.'),
            ]
            
            # Apply import mappings
            for old_pattern, new_pattern in import_mappings:
                content = re.sub(old_pattern, new_pattern, content)
            
            # Only write if content changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_files.append(file_path)
                print(f"✅ Fixed {file_path}")
        
        except Exception as e:
            print(f"❌ Error fixing {file_path}: {e}")
    
    print(f"\n📊 SUMMARY")
    print(f"Fixed {len(fixed_files)} backend files")
    
    if fixed_files:
        print("\n✅ Fixed files:")
        for file_path in fixed_files:
            print(f"  - {file_path}")

if __name__ == "__main__":
    fix_backend_imports()