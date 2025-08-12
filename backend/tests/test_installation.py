#!/usr/bin/env python3
"""
Installation Testing Script

This script tests the installation process and verifies that all components
are working correctly after installation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import sys
import os
import subprocess
import tempfile
import shutil
import json
from pathlib import Path
from typing import List, Dict, Any

class InstallationTester:
    """Test the installation process and verify functionality"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_results = []
        
    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test result"""
        status = "PASS" if passed else "FAIL"
        self.test_results.append({
            "test": test_name,
            "status": status,
            "message": message
        })
        print(f"[{status}] {test_name}: {message}")
        
    def test_python_imports(self) -> bool:
        """Test that all required Python packages can be imported"""
        required_packages = [
            ('fastapi', 'fastapi'),
            ('uvicorn', 'uvicorn'),
            ('sqlalchemy', 'sqlalchemy'),
            ('psycopg', 'psycopg[binary]'),
            ('socketio', 'python-socketio'),
            ('redis', 'redis'),
            ('cv2', 'opencv-python'),
            ('PIL', 'Pillow'),
            ('numpy', 'numpy'),
            ('pandas', 'pandas'),
            ('openpyxl', 'openpyxl'),
            ('imagehash', 'imagehash'),
            ('tqdm', 'tqdm'),
            ('psutil', 'psutil'),
            ('requests', 'requests')
        ]
        
        all_passed = True
        for package, pip_name in required_packages:
            try:
                __import__(package)
                self.log_test(f"Import {package}", True, f"Successfully imported {pip_name}")
            except ImportError as e:
                self.log_test(f"Import {package}", False, f"Failed to import {pip_name}: {e}")
                all_passed = False
                
        return all_passed
        
    def test_core_modules(self) -> bool:
        """Test that core application modules can be imported"""
        core_modules = [
            'core.database',
            'core.batch_processor',
            'core.progress_tracker',
            'core.decision_engine',
            'analyzers.quality_analyzer',
            'analyzers.defect_detector',
            'analyzers.similarity_finder',
            'analyzers.compliance_checker',
            'utils.file_manager',
            'utils.report_generator',
            'config.config_loader'
        ]
        
        all_passed = True
        for module in core_modules:
            try:
                __import__(module)
                self.log_test(f"Core module {module}", True, "Module imported successfully")
            except ImportError as e:
                self.log_test(f"Core module {module}", False, f"Import failed: {e}")
                all_passed = False
                
        return all_passed
        
    def test_configuration_files(self) -> bool:
        """Test that configuration files exist and are valid"""
        config_files = [
            'config/settings.json',
            'config/settings.example.json',
            'config/settings.development.json',
            'config/settings.production.json'
        ]
        
        all_passed = True
        for config_file in config_files:
            config_path = self.project_root / config_file
            
            if not config_path.exists():
                self.log_test(f"Config file {config_file}", False, "File does not exist")
                all_passed = False
                continue
                
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    
                # Check for required sections
                required_sections = ['processing', 'quality', 'similarity', 'compliance', 'output']
                missing_sections = [s for s in required_sections if s not in config_data]
                
                if missing_sections:
                    self.log_test(f"Config file {config_file}", False, 
                                f"Missing sections: {missing_sections}")
                    all_passed = False
                else:
                    self.log_test(f"Config file {config_file}", True, "Valid JSON with required sections")
                    
            except json.JSONDecodeError as e:
                self.log_test(f"Config file {config_file}", False, f"Invalid JSON: {e}")
                all_passed = False
            except Exception as e:
                self.log_test(f"Config file {config_file}", False, f"Error reading file: {e}")
                all_passed = False
                
        return all_passed
        
    def test_database_creation(self) -> bool:
        """Test database creation and basic operations"""
        try:
            from backend.core.database import DatabaseManager
            
            # Create temporary database
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
                db_path = tmp_db.name
                
            try:
                # Database is initialized automatically in constructor
                db_manager = DatabaseManager(db_path)
                
                # Test basic operations
                session_id = db_manager.create_session("backend/data/input", "backend/data/output", 100)
                if session_id:
                    self.log_test("Database creation", True, "Database created and session added")
                    return True
                else:
                    self.log_test("Database creation", False, "Failed to create session")
                    return False
                    
            finally:
                # Clean up
                if os.path.exists(db_path):
                    os.unlink(db_path)
                    
        except Exception as e:
            self.log_test("Database creation", False, f"Database test failed: {e}")
            return False
            
    def test_file_operations(self) -> bool:
        """Test file scanning and management operations"""
        try:
            from backend.utils.file_manager import FileManager
            
            # Create temporary test structure
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create test files
                test_files = [
                    temp_path / "test1.jpg",
                    temp_path / "test2.png",
                    temp_path / "subdir" / "test3.jpeg",
                    temp_path / "test.txt"  # Should be ignored
                ]
                
                for file_path in test_files:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_bytes(b"fake image data")
                    
                file_manager = FileManager()
                found_images = file_manager.scan_images(str(temp_path))
                
                # Should find 3 image files, ignore .txt
                if len(found_images) == 3:
                    self.log_test("File scanning", True, f"Found {len(found_images)} image files")
                    return True
                else:
                    self.log_test("File scanning", False, 
                                f"Expected 3 files, found {len(found_images)}")
                    return False
                    
        except Exception as e:
            self.log_test("File operations", False, f"File operations test failed: {e}")
            return False
            
    def test_analyzer_initialization(self) -> bool:
        """Test that all analyzer modules can be initialized"""
        # Create basic config for analyzers that need it
        basic_config = {
            'quality': {
                'min_sharpness': 100.0,
                'max_noise_level': 0.1,
                'min_resolution': [1920, 1080]
            },
            'defect_detection': {
                'confidence_threshold': 0.7
            }
        }
        
        analyzers = [
            ('analyzers.quality_analyzer', 'QualityAnalyzer', basic_config),
            ('analyzers.defect_detector', 'DefectDetector', basic_config),
            ('analyzers.similarity_finder', 'SimilarityFinder', None),
            ('analyzers.compliance_checker', 'ComplianceChecker', None)
        ]
        
        all_passed = True
        for module_name, class_name, config in analyzers:
            try:
                module = __import__(module_name, fromlist=[class_name])
                analyzer_class = getattr(module, class_name)
                
                # Try to initialize with appropriate config
                if config:
                    analyzer = analyzer_class(config)
                else:
                    analyzer = analyzer_class()
                self.log_test(f"Analyzer {class_name}", True, "Initialized successfully")
                
            except Exception as e:
                self.log_test(f"Analyzer {class_name}", False, f"Initialization failed: {e}")
                all_passed = False
                
        return all_passed
        
    def test_main_application(self) -> bool:
        """Test that the main application can be imported and basic help works"""
        try:
            # Test importing main module
            import backend.main as main
            self.log_test("Main application import", True, "Main module imported successfully")

            # Test command line help
            result = subprocess.run([sys.executable, "backend/main.py", "--help"],
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and "usage:" in result.stdout.lower():
                self.log_test("Main application help", True, "Help command works")
                return True
            else:
                self.log_test("Main application help", False, "Help command failed")
                return False
                
        except Exception as e:
            self.log_test("Main application", False, f"Main application test failed: {e}")
            return False
            
    def test_demo_scripts(self) -> bool:
        """Test that demo scripts can run without errors"""
        demo_scripts = [
            "demo_config_validation.py",
            "demo_file_manager.py"
        ]
        
        all_passed = True
        for script in demo_scripts:
            script_path = self.project_root / script
            if not script_path.exists():
                self.log_test(f"Demo script {script}", False, "Script file not found")
                all_passed = False
                continue
                
            try:
                result = subprocess.run([sys.executable, str(script_path)], 
                                      capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    self.log_test(f"Demo script {script}", True, "Script ran successfully")
                else:
                    self.log_test(f"Demo script {script}", False, 
                                f"Script failed with code {result.returncode}")
                    all_passed = False
                    
            except subprocess.TimeoutExpired:
                self.log_test(f"Demo script {script}", False, "Script timed out")
                all_passed = False
            except Exception as e:
                self.log_test(f"Demo script {script}", False, f"Script error: {e}")
                all_passed = False
                
        return all_passed
        
    def run_all_tests(self) -> bool:
        """Run all installation tests"""
        print("=" * 60)
        print("Adobe Stock Image Processor - Installation Tests")
        print("=" * 60)
        
        tests = [
            ("Python Package Imports", self.test_python_imports),
            ("Core Module Imports", self.test_core_modules),
            ("Configuration Files", self.test_configuration_files),
            ("Database Operations", self.test_database_creation),
            ("File Operations", self.test_file_operations),
            ("Analyzer Initialization", self.test_analyzer_initialization),
            ("Main Application", self.test_main_application),
            ("Demo Scripts", self.test_demo_scripts)
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            print(f"\nRunning {test_name}...")
            try:
                passed = test_func()
                if not passed:
                    all_passed = False
            except Exception as e:
                self.log_test(test_name, False, f"Test crashed: {e}")
                all_passed = False
                
        # Print summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        
        passed_count = sum(1 for result in self.test_results if result["status"] == "PASS")
        total_count = len(self.test_results)
        
        print(f"Total tests: {total_count}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {total_count - passed_count}")
        
        if all_passed:
            print("\n✓ All tests passed! Installation is successful.")
        else:
            print("\n✗ Some tests failed. Please check the errors above.")
            print("\nFailed tests:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"  - {result['test']}: {result['message']}")
                    
        return all_passed

def main():
    """Main entry point"""
    tester = InstallationTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()