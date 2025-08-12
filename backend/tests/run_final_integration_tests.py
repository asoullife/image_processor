#!/usr/bin/env python3
"""
Final Integration Test Runner
Executes comprehensive integration tests for Adobe Stock Image Processor
"""

import os
import sys
import asyncio
import logging
import subprocess
import time
import signal
from pathlib import Path
from typing import List, Dict, Any
import json
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntegrationTestRunner:
    """Orchestrates and runs all integration tests."""
    
    def __init__(self):
        """Initialize the test runner."""
        self.project_root = Path(__file__).parent
        self.backend_path = self.project_root / "backend"
        self.frontend_path = self.project_root / "frontend"
        
        # Process tracking
        self.processes = []
        self.test_results = {}
        
        # Test configuration
        self.tests_to_run = [
            "test_final_integration.py",
            "test_backend_structure.py",
            "test_api_structure.py", 
            "test_core_functionality.py",
            "test_ai_enhancement.py",
            "test_multi_session.py",
            "test_realtime_monitoring.py",
            "test_reports_implementation.py"
        ]
    
    async def run_all_tests(self):
        """Run all integration tests in sequence."""
        logger.info("🚀 Starting Comprehensive Integration Test Suite")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Phase 1: Environment Preparation
            await self.prepare_test_environment()
            
            # Phase 2: Infrastructure Tests
            await self.run_infrastructure_tests()
            
            # Phase 3: Component Tests
            await self.run_component_tests()
            
            # Phase 4: Integration Tests
            await self.run_integration_tests()
            
            # Phase 5: Performance Tests
            await self.run_performance_tests()
            
            # Phase 6: End-to-End Tests
            await self.run_end_to_end_tests()
            
            # Generate final report
            await self.generate_comprehensive_report(start_time)
            
            logger.info("✅ All integration tests completed!")
            
        except Exception as e:
            logger.error(f"❌ Integration test suite failed: {e}")
            raise
        finally:
            await self.cleanup_test_environment()
    
    async def prepare_test_environment(self):
        """Prepare the test environment."""
        logger.info("🔧 Phase 1: Preparing Test Environment")
        
        # Check system requirements
        await self.check_system_requirements()
        
        # Setup test databases
        await self.setup_test_databases()
        
        # Install dependencies
        await self.verify_dependencies()
        
        # Create test data directories
        await self.create_test_directories()
        
        logger.info("✅ Test environment prepared")
    
    async def check_system_requirements(self):
        """Check system requirements for testing."""
        logger.info("Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise Exception(f"Python 3.8+ required, found {sys.version}")
        
        # Check available memory
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 4:
            logger.warning(f"Low memory detected: {memory_gb:.1f}GB (recommended: 8GB+)")
        
        # Check disk space
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        if free_gb < 5:
            logger.warning(f"Low disk space: {free_gb:.1f}GB free (recommended: 10GB+)")
        
        logger.info(f"✅ System check passed - Python {sys.version_info.major}.{sys.version_info.minor}, {memory_gb:.1f}GB RAM, {free_gb:.1f}GB free")
    
    async def setup_test_databases(self):
        """Setup test databases."""
        logger.info("Setting up test databases...")
        
        try:
            # Check if PostgreSQL is available
            result = subprocess.run(
                ["psql", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info("✅ PostgreSQL available")
            else:
                logger.error("❌ PostgreSQL not available")
                raise RuntimeError("PostgreSQL not available")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error("❌ PostgreSQL not found")
            raise
        
        # Initialize database
        try:
            from backend.database.init_db import init_database
            await init_database()
            logger.info("✅ Test database initialized")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def verify_dependencies(self):
        """Verify all required dependencies are installed."""
        logger.info("Verifying dependencies...")
        
        # Backend dependencies
        backend_deps = [
            'fastapi', 'uvicorn', 'sqlalchemy', 'psycopg',
            'cv2', 'PIL', 'numpy', 'pandas',
            'socketio', 'redis', 'pytest'
        ]
        
        missing_deps = []
        for dep in backend_deps:
            try:
                __import__(dep.replace('-', '_'))
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            logger.error(f"Missing dependencies: {missing_deps}")
            logger.info("Installing missing dependencies...")
            
            # Try to install missing dependencies
            for dep in missing_deps:
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                 check=True, capture_output=True)
                    logger.info(f"✅ Installed {dep}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install {dep}: {e}")
                    raise
        
        logger.info("✅ All dependencies verified")
    
    async def create_test_directories(self):
        """Create necessary test directories."""
        logger.info("Creating test directories...")
        
        test_dirs = [
            "integration_test_data",
            "integration_test_data/input",
            "integration_test_data/output", 
            "integration_test_data/reports",
            "integration_test_data/temp",
            "test_results"
        ]
        
        for dir_path in test_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Test directories created")
    
    async def run_infrastructure_tests(self):
        """Run infrastructure and setup tests."""
        logger.info("🏗️ Phase 2: Infrastructure Tests")
        
        infrastructure_tests = [
            "backend/test_backend_structure.py",
            "backend/test_api_structure.py"
        ]
        
        for test_file in infrastructure_tests:
            if Path(test_file).exists():
                result = await self.run_single_test(test_file)
                self.test_results[f"infrastructure_{Path(test_file).stem}"] = result
            else:
                logger.warning(f"Test file not found: {test_file}")
    
    async def run_component_tests(self):
        """Run individual component tests."""
        logger.info("🧩 Phase 3: Component Tests")
        
        component_tests = [
            "backend/test_core_functionality.py",
            "backend/test_ai_enhancement.py",
            "backend/test_ai_compliance.py",
            "backend/test_file_protection_simple.py"
        ]
        
        for test_file in component_tests:
            if Path(test_file).exists():
                result = await self.run_single_test(test_file)
                self.test_results[f"component_{Path(test_file).stem}"] = result
            else:
                logger.warning(f"Test file not found: {test_file}")
    
    async def run_integration_tests(self):
        """Run integration tests."""
        logger.info("🔗 Phase 4: Integration Tests")
        
        integration_tests = [
            "backend/test_multi_session.py",
            "backend/test_multi_session_api.py",
            "backend/test_compliance_integration.py"
        ]
        
        for test_file in integration_tests:
            if Path(test_file).exists():
                result = await self.run_single_test(test_file)
                self.test_results[f"integration_{Path(test_file).stem}"] = result
            else:
                logger.warning(f"Test file not found: {test_file}")
    
    async def run_performance_tests(self):
        """Run performance and monitoring tests."""
        logger.info("⚡ Phase 5: Performance Tests")
        
        performance_tests = [
            "backend/test_realtime_monitoring.py",
            "backend/test_monitoring_basic.py",
            "test_reports_implementation.py"
        ]
        
        for test_file in performance_tests:
            if Path(test_file).exists():
                result = await self.run_single_test(test_file)
                self.test_results[f"performance_{Path(test_file).stem}"] = result
            else:
                logger.warning(f"Test file not found: {test_file}")
    
    async def run_end_to_end_tests(self):
        """Run comprehensive end-to-end tests."""
        logger.info("🎯 Phase 6: End-to-End Tests")
        
        # Run the main integration test
        result = await self.run_single_test("test_final_integration.py")
        self.test_results["end_to_end_final_integration"] = result
    
    async def run_single_test(self, test_file: str) -> Dict[str, Any]:
        """Run a single test file and return results."""
        logger.info(f"Running test: {test_file}")
        
        start_time = time.time()
        
        try:
            # Determine if it's a pytest test or a standalone script
            if test_file.startswith("backend/test_") or test_file.endswith("_test.py"):
                # Run with pytest
                cmd = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"]
            else:
                # Run as standalone script
                cmd = [sys.executable, test_file]
            
            # Run the test
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.project_root
            )
            
            stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout
            
            duration = time.time() - start_time
            
            result = {
                "test_file": test_file,
                "status": "PASSED" if process.returncode == 0 else "FAILED",
                "return_code": process.returncode,
                "duration": duration,
                "stdout": stdout,
                "stderr": stderr
            }
            
            if process.returncode == 0:
                logger.info(f"✅ {test_file} - PASSED ({duration:.2f}s)")
            else:
                logger.error(f"❌ {test_file} - FAILED ({duration:.2f}s)")
                logger.error(f"Error output: {stderr}")
            
            return result
            
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {test_file} - TIMEOUT")
            return {
                "test_file": test_file,
                "status": "TIMEOUT",
                "return_code": -1,
                "duration": time.time() - start_time,
                "stdout": "",
                "stderr": "Test timed out after 5 minutes"
            }
        except Exception as e:
            logger.error(f"❌ {test_file} - ERROR: {e}")
            return {
                "test_file": test_file,
                "status": "ERROR",
                "return_code": -1,
                "duration": time.time() - start_time,
                "stdout": "",
                "stderr": str(e)
            }
    
    async def generate_comprehensive_report(self, start_time: float):
        """Generate comprehensive test report."""
        logger.info("📊 Generating Comprehensive Test Report")
        
        total_duration = time.time() - start_time
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r["status"] == "PASSED")
        failed_tests = sum(1 for r in self.test_results.values() if r["status"] == "FAILED")
        error_tests = sum(1 for r in self.test_results.values() if r["status"] in ["ERROR", "TIMEOUT"])
        
        # Create comprehensive report
        report = {
            "test_suite": "Comprehensive Integration Test Suite",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_duration": total_duration,
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_count": os.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "disk_free_gb": psutil.disk_usage('.').free / (1024**3)
            },
            "test_results": self.test_results,
            "failed_tests": [
                name for name, result in self.test_results.items() 
                if result["status"] in ["FAILED", "ERROR", "TIMEOUT"]
            ]
        }
        
        # Save detailed report
        report_path = Path("test_results") / f"comprehensive_integration_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("🎯 COMPREHENSIVE INTEGRATION TEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total Duration: {total_duration:.2f} seconds")
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"✅ Passed: {passed_tests}")
        logger.info(f"❌ Failed: {failed_tests}")
        logger.info(f"⚠️ Errors: {error_tests}")
        logger.info(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        
        if report["failed_tests"]:
            logger.error("Failed Tests:")
            for test_name in report["failed_tests"]:
                logger.error(f"  - {test_name}")
        
        logger.info(f"📊 Detailed report: {report_path}")
        logger.info("=" * 80)
        
        # Return overall success status
        return failed_tests == 0 and error_tests == 0
    
    async def cleanup_test_environment(self):
        """Cleanup test environment."""
        logger.info("🧹 Cleaning up test environment...")
        
        # Terminate any running processes
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error terminating process: {e}")
        
        logger.info("✅ Test environment cleaned up")

async def main():
    """Main entry point."""
    runner = IntegrationTestRunner()
    
    try:
        success = await runner.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("Test suite interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))