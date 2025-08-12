#!/usr/bin/env python3
"""
Core Integration Test - Tests basic functionality without external dependencies
"""

import os
import sys
import logging
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime

# Add backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoreIntegrationTest:
    """Test core functionality without external dependencies."""
    
    def __init__(self):
        """Initialize the test."""
        self.test_results = {}
        self.project_root = Path(__file__).parent
        self.backend_path = self.project_root / "backend"
    
    def run_all_tests(self):
        """Run all core integration tests."""
        logger.info("🚀 Starting Core Integration Tests")
        logger.info("=" * 60)
        
        try:
            # Test 1: Project Structure Validation
            self.test_project_structure()
            
            # Test 2: Import Tests
            self.test_imports()
            
            # Test 3: Configuration System
            self.test_configuration_system()
            
            # Test 4: Database Models
            self.test_database_models()
            
            # Test 5: API Structure
            self.test_api_structure()
            
            # Test 6: Frontend Structure
            self.test_frontend_structure()
            
            # Generate test report
            self.generate_test_report()
            
            # Check overall success
            failed_tests = [name for name, result in self.test_results.items() if not result["passed"]]
            
            if not failed_tests:
                logger.info("✅ All core integration tests passed!")
                return True
            else:
                logger.error(f"❌ {len(failed_tests)} tests failed")
                return False
                
        except Exception as e:
            logger.error(f"Core integration tests failed: {e}")
            return False
    
    def test_project_structure(self):
        """Test project structure is correct."""
        logger.info("Testing project structure...")
        
        required_dirs = [
            "backend",
            "backend/api",
            "backend/core", 
            "backend/analyzers",
            "backend/database",
            "backend/utils",
            "backend/realtime",
            "frontend",
            "frontend/src",
            "scripts",
            "docs"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                missing_dirs.append(dir_path)
        
        passed = len(missing_dirs) == 0
        
        self.test_results["project_structure"] = {
            "passed": passed,
            "details": f"Missing directories: {missing_dirs}" if missing_dirs else "All required directories exist",
            "missing_dirs": missing_dirs
        }
        
        if passed:
            logger.info("✅ Project structure validation passed")
        else:
            logger.error(f"❌ Project structure validation failed: {missing_dirs}")
    
    def test_imports(self):
        """Test that core modules can be imported."""
        logger.info("Testing core module imports...")
        
        import_tests = [
            ("backend.config.config_loader", "Configuration loader"),
            ("backend.database.models", "Database models"),
            ("backend.api.main", "API main module"),
            ("backend.core.base", "Core base classes"),
            ("backend.utils.logger", "Logger utilities")
        ]
        
        failed_imports = []
        
        for module_name, description in import_tests:
            try:
                __import__(module_name)
                logger.info(f"✅ {description} import successful")
            except ImportError as e:
                failed_imports.append((module_name, str(e)))
                logger.error(f"❌ {description} import failed: {e}")
            except Exception as e:
                failed_imports.append((module_name, str(e)))
                logger.error(f"❌ {description} import error: {e}")
        
        passed = len(failed_imports) == 0
        
        self.test_results["imports"] = {
            "passed": passed,
            "details": f"Failed imports: {failed_imports}" if failed_imports else "All core imports successful",
            "failed_imports": failed_imports
        }
    
    def test_configuration_system(self):
        """Test configuration system."""
        logger.info("Testing configuration system...")
        
        try:
            # Test config file exists
            config_file = self.backend_path / "config" / "config_loader.py"
            if not config_file.exists():
                raise Exception("Configuration loader not found")
            
            # Test default config can be loaded
            from backend.config.config_loader import load_config, AppConfig
            
            # Try to load default config
            config = load_config()
            
            if not isinstance(config, AppConfig):
                raise Exception("Configuration not loaded as AppConfig instance")
            
            passed = True
            details = "Configuration system working correctly"
            
        except Exception as e:
            passed = False
            details = f"Configuration system error: {e}"
            logger.error(f"❌ Configuration system test failed: {e}")
        
        self.test_results["configuration"] = {
            "passed": passed,
            "details": details
        }
        
        if passed:
            logger.info("✅ Configuration system test passed")
    
    def test_database_models(self):
        """Test database models."""
        logger.info("Testing database models...")
        
        try:
            from backend.database.models import Base, User, Project, ProcessingSession
            
            # Check that models have required attributes
            required_models = [User, Project, ProcessingSession]
            
            for model in required_models:
                if not hasattr(model, '__tablename__'):
                    raise Exception(f"Model {model.__name__} missing __tablename__")
            
            passed = True
            details = "Database models structure correct"
            
        except Exception as e:
            passed = False
            details = f"Database models error: {e}"
            logger.error(f"❌ Database models test failed: {e}")
        
        self.test_results["database_models"] = {
            "passed": passed,
            "details": details
        }
        
        if passed:
            logger.info("✅ Database models test passed")
    
    def test_api_structure(self):
        """Test API structure."""
        logger.info("Testing API structure...")
        
        try:
            # Check API routes exist
            api_routes_dir = self.backend_path / "api" / "routes"
            if not api_routes_dir.exists():
                raise Exception("API routes directory not found")
            
            required_routes = [
                "health.py",
                "projects.py", 
                "sessions.py",
                "auth.py",
                "review.py"
            ]
            
            missing_routes = []
            for route_file in required_routes:
                route_path = api_routes_dir / route_file
                if not route_path.exists():
                    missing_routes.append(route_file)
            
            if missing_routes:
                raise Exception(f"Missing API routes: {missing_routes}")
            
            # Test main API module
            from backend.api.main import app
            
            if not hasattr(app, 'include_router'):
                raise Exception("FastAPI app not properly configured")
            
            passed = True
            details = "API structure correct"
            
        except Exception as e:
            passed = False
            details = f"API structure error: {e}"
            logger.error(f"❌ API structure test failed: {e}")
        
        self.test_results["api_structure"] = {
            "passed": passed,
            "details": details
        }
        
        if passed:
            logger.info("✅ API structure test passed")
    
    def test_frontend_structure(self):
        """Test frontend structure."""
        logger.info("Testing frontend structure...")
        
        try:
            frontend_dir = self.project_root / "frontend"
            if not frontend_dir.exists():
                raise Exception("Frontend directory not found")
            
            # Check package.json exists
            package_json = frontend_dir / "package.json"
            if not package_json.exists():
                raise Exception("package.json not found")
            
            # Check src directory exists
            src_dir = frontend_dir / "src"
            if not src_dir.exists():
                raise Exception("src directory not found")
            
            # Check for key directories
            required_dirs = ["pages", "components"]
            missing_dirs = []
            
            for dir_name in required_dirs:
                dir_path = src_dir / dir_name
                if not dir_path.exists():
                    missing_dirs.append(dir_name)
            
            if missing_dirs:
                raise Exception(f"Missing frontend directories: {missing_dirs}")
            
            passed = True
            details = "Frontend structure correct"
            
        except Exception as e:
            passed = False
            details = f"Frontend structure error: {e}"
            logger.error(f"❌ Frontend structure test failed: {e}")
        
        self.test_results["frontend_structure"] = {
            "passed": passed,
            "details": details
        }
        
        if passed:
            logger.info("✅ Frontend structure test passed")
    
    def generate_test_report(self):
        """Generate test report."""
        logger.info("📊 Generating Core Integration Test Report")
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r["passed"])
        failed_tests = total_tests - passed_tests
        
        # Create report
        report = {
            "test_suite": "Core Integration Tests",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "test_results": self.test_results,
            "failed_tests": [
                name for name, result in self.test_results.items() 
                if not result["passed"]
            ]
        }
        
        # Save report
        report_dir = Path("test_results")
        report_dir.mkdir(exist_ok=True)
        
        report_path = report_dir / f"core_integration_report_{int(datetime.now().timestamp())}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("🎯 CORE INTEGRATION TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {failed_tests}")
        logger.info(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        
        if report["failed_tests"]:
            logger.error("Failed Tests:")
            for test_name in report["failed_tests"]:
                logger.error(f"  - {test_name}: {self.test_results[test_name]['details']}")
        
        logger.info(f"📊 Report saved: {report_path}")
        logger.info("=" * 60)

def main():
    """Main entry point."""
    test = CoreIntegrationTest()
    success = test.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())