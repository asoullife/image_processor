#!/usr/bin/env python3
"""Organize files in root directory"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import shutil
from datetime import datetime

def organize_files():
    """Organize files into appropriate directories"""
    
    # Create directories if they don't exist
    directories = {
        'reports': 'Generated reports and outputs',
        'demos': 'Demo and example scripts', 
        'tests': 'Test files (if not already in tests/)',
        'docs': 'Documentation files',
        'temp': 'Temporary and generated files',
        'scripts': 'Utility scripts'
    }
    
    for dir_name in directories:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"📁 Created directory: {dir_name}")
    
    # File organization rules
    file_moves = {
        # Reports
        'reports': [
            'adobe_stock_dashboard_*.html',
            'adobe_stock_report_*.xlsx', 
            'adobe_stock_results_*.csv',
            'adobe_stock_summary_*.txt',
            'comprehensive_test_report.*',
            'final_integration_test_results.json',
            'test_results.json',
            'working_comprehensive_test_report.json'
        ],
        
        # Demo scripts
        'demos': [
            'demo_*.py',
            'create_test_images.py'
        ],
        
        # Test files (additional ones not in tests/)
        'tests': [
            'test_*.py'
        ],
        
        # Documentation
        'docs': [
            '*.md',
            'SETUP_GUIDE.md',
            'FINAL_INTEGRATION_SUMMARY.md',
            'COMPREHENSIVE_TEST_SUITE_IMPLEMENTATION.md'
        ],
        
        # Utility scripts
        'scripts': [
            'create_*.py',
            'view_*.py',
            'organize_files.py',
            'run_*.py',
            'optimize_performance.py',
            'install.py',
            'deploy.py'
        ],
        
        # Temporary files
        'temp': [
            '*.log',
            'final_integration_test.log'
        ]
    }
    
    print("🗂️ Organizing files...")
    print("=" * 50)
    
    # Keep these files in root (important main files)
    keep_in_root = {
        'main.py': 'Main application entry point',
        'requirements.txt': 'Python dependencies',
        'requirements-minimal.txt': 'Minimal dependencies',
        'README.md': 'Main documentation',
        'pytest.ini': 'Test configuration',
        'adobe_stock_processor.db': 'Main database file'
    }
    
    print("📌 Files to keep in root:")
    for file, desc in keep_in_root.items():
        if os.path.exists(file):
            print(f"  ✅ {file} - {desc}")
        else:
            print(f"  ❌ {file} - {desc} (missing)")
    
    print(f"\n📦 Moving files to organized directories:")
    
    # Move files according to rules
    import glob
    moved_count = 0
    
    for target_dir, patterns in file_moves.items():
        print(f"\n📁 {target_dir.upper()}:")
        
        for pattern in patterns:
            files = glob.glob(pattern)
            for file in files:
                if file not in keep_in_root and os.path.isfile(file):
                    try:
                        target_path = os.path.join(target_dir, file)
                        
                        # Handle duplicates
                        if os.path.exists(target_path):
                            base, ext = os.path.splitext(file)
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            target_path = os.path.join(target_dir, f"{base}_{timestamp}{ext}")
                        
                        shutil.move(file, target_path)
                        print(f"  📄 {file} → {target_path}")
                        moved_count += 1
                        
                    except Exception as e:
                        print(f"  ❌ Failed to move {file}: {e}")
    
    print(f"\n✅ Organization complete!")
    print(f"📊 Moved {moved_count} files")
    
    # Show final root directory structure
    print(f"\n📁 Final root directory structure:")
    root_files = [f for f in os.listdir('.') if os.path.isfile(f)]
    root_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    
    print(f"\n📂 Directories ({len(root_dirs)}):")
    for dir_name in sorted(root_dirs):
        if not dir_name.startswith('.') and dir_name != '__pycache__':
            file_count = len([f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name, f))])
            print(f"  📁 {dir_name}/ ({file_count} files)")
    
    print(f"\n📄 Root files ({len(root_files)}):")
    for file_name in sorted(root_files):
        if not file_name.startswith('.'):
            size = os.path.getsize(file_name)
            print(f"  📄 {file_name} ({size:,} bytes)")
    
    return moved_count

if __name__ == "__main__":
    organize_files()