#!/usr/bin/env python3
"""
Comprehensive test suite runner for Adobe Stock Image Processor
Executes all unit tests, integration tests, performance tests, and stress tests
with detailed reporting and optimization recommendations
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys
import subprocess
import time
import json
import psutil
from pathlib import Path
import argparse
from typing import Dict, List, Any
import concurrent.futures


class ComprehensiveTestRunner:
    """Comprehensive test runner with advanced reporting and analysis"""
    
    def __init__(self, verbose=False, parallel=False):
        self.verbose = verbose
        self.parallel = parallel
        self.results = {
            'unit_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'stress_tests': {},
            'coverage': {},
            'optimization_recommendations': [],
            'summary': {},
            'system_info': self._get_system_info()
        }
        self.start_time = time.time()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for test context"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'python_version': sys.version,
            'platform': sys.platform
        }
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run comprehensive unit tests"""
        print("=" * 70)
        print("RUNNING COMPREHENSIVE UNIT TESTS")
        print("=" * 70)
        
        unit_test_files = [
            'tests/test_comprehensive_working.py',
            'tests/test_comprehensive_unit.py',
            'tests/test_config_validator.py',
            'tests/test_database.py',
            'tests/test_progress_tracker.py',
            'tests/test_file_manager.py',
            'tests/test_error_handler.py',
            'tests/test_error_integration.py',
            'tests/test_batch_processor.py',
            'tests/test_quality_analyzer.py',
            'tests/test_defect_detector.py',
            'tests/test_similarity_finder.py',
            'tests/test_compliance_checker.py',
            'tests/test_decision_engine.py',
            'tests/test_report_generator.py',
            'tests/test_resume_functionality.py',
            'tests/test_crash_recovery.py'
        ]
        
        return self._run_test_suite("Unit Tests", unit_test_files, timeout=600)
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration tests"""
        print("\n" + "=" * 70)
        print("RUNNING COMPREHENSIVE INTEGRATION TESTS")
        print("=" * 70)
        
        integration_test_files = [
            'tests/test_integration_comprehensive.py',
            'tests/test_integration_database_progress.py',
            'tests/test_integration_batch_processor.py',
            'tests/test_integration_main_app.py'
        ]
        
        return self._run_test_suite("Integration Tests", integration_test_files, timeout=1200)
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance and benchmark tests"""
        print("\n" + "=" * 70)
        print("RUNNING PERFORMANCE AND BENCHMARK TESTS")
        print("=" * 70)
        
        performance_test_files = [
            'tests/test_performance.py',
            'tests/test_performance_enhanced.py',
            'tests/test_benchmark_1000_images.py'
        ]
        
        # Run performance tests with special handling
        performance_results = self._run_test_suite("Performance Tests", performance_test_files, timeout=1800)
        
        # Run memory profiling
        print("\nRunning memory profiling...")
        try:
            result = subprocess.run([
                sys.executable, 'tests/memory_profiler.py'
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("  Memory profiling completed successfully")
                performance_results['memory_profiling'] = {
                    'status': 'success',
                    'output': result.stdout
                }
            else:
                print(f"  Memory profiling failed: {result.stderr}")
                performance_results['memory_profiling'] = {
                    'status': 'failed',
                    'error': result.stderr
                }
        
        except subprocess.TimeoutExpired:
            print("  Memory profiling timed out")
            performance_results['memory_profiling'] = {
                'status': 'timeout',
                'error': 'Memory profiling exceeded 10 minutes'
            }
        
        return performance_results
    
    def run_stress_tests(self) -> Dict[str, Any]:
        """Run stress and stability tests"""
        print("\n" + "=" * 70)
        print("RUNNING STRESS AND STABILITY TESTS")
        print("=" * 70)
        
        stress_test_files = [
            'tests/test_stress.py'
        ]
        
        return self._run_test_suite("Stress Tests", stress_test_files, timeout=2400)
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run comprehensive code coverage analysis"""
        print("\n" + "=" * 70)
        print("RUNNING CODE COVERAGE ANALYSIS")
        print("=" * 70)
        
        try:
            print("Generating comprehensive coverage report...")
            
            # Try pytest-cov first
            try:
                result = subprocess.run([
                    sys.executable, '-m', 'pytest', 
                    '--cov=.', 
                    '--cov-report=html:htmlcov',
                    '--cov-report=term-missing',
                    '--cov-report=json:coverage.json',
                    'tests/',
                    '--cov-exclude=tests/*',
                    '--cov-exclude=demo_*',
                    '--cov-exclude=test_*',
                    '--cov-exclude=*/__pycache__/*',
                    '--cov-exclude=*/venv/*',
                    '--cov-exclude=*/env/*'
                ], capture_output=True, text=True, timeout=1200)
                
            except FileNotFoundError:
                # Fallback to coverage.py directly
                print("Pytest not available, using coverage.py directly...")
                
                # Run coverage with unittest
                result = subprocess.run([
                    sys.executable, '-m', 'coverage', 'run', '--source=.', 
                    '--omit=tests/*,demo_*,test_*,*/__pycache__/*',
                    '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py'
                ], capture_output=True, text=True, timeout=1200)
                
                if result.returncode == 0:
                    # Generate coverage report
                    report_result = subprocess.run([
                        sys.executable, '-m', 'coverage', 'report'
                    ], capture_output=True, text=True, timeout=300)
                    
                    result.stdout = report_result.stdout
                    result.stderr = report_result.stderr
                    
                    # Generate JSON report
                    subprocess.run([
                        sys.executable, '-m', 'coverage', 'json'
                    ], capture_output=True, text=True, timeout=300)
            
            # Parse coverage results
            coverage_data = self._parse_coverage_output(result.stdout)
            
            # Load JSON coverage data if available
            json_coverage = {}
            if os.path.exists('coverage.json'):
                try:
                    with open('coverage.json', 'r') as f:
                        json_coverage = json.load(f)
                except:
                    pass
            
            coverage_results = {
                'overall_coverage': coverage_data.get('overall_coverage', 0),
                'file_coverage': coverage_data.get('file_coverage', {}),
                'missing_lines': coverage_data.get('missing_lines', {}),
                'json_data': json_coverage,
                'return_code': result.returncode,
                'output': result.stdout if self.verbose else '',
                'stderr': result.stderr if result.stderr else ''
            }
            
            print(f"Overall coverage: {coverage_results['overall_coverage']:.1f}%")
            
            if self.verbose and coverage_results['file_coverage']:
                print("\nFile-by-file coverage:")
                for filename, coverage in sorted(coverage_results['file_coverage'].items()):
                    print(f"  {filename}: {coverage}%")
            
            return coverage_results
        
        except subprocess.TimeoutExpired:
            return {
                'overall_coverage': 0,
                'file_coverage': {},
                'return_code': -1,
                'error': 'Coverage analysis timed out'
            }
        
        except Exception as e:
            return {
                'overall_coverage': 0,
                'file_coverage': {},
                'return_code': -1,
                'error': str(e)
            }
    
    def _run_test_suite(self, suite_name: str, test_files: List[str], timeout: int) -> Dict[str, Any]:
        """Run a test suite with comprehensive reporting"""
        suite_results = {
            'total_passed': 0,
            'total_failed': 0,
            'total_errors': 0,
            'total_skipped': 0,
            'details': {},
            'duration': 0,
            'suite_name': suite_name
        }
        
        start_time = time.time()
        
        if self.parallel and len(test_files) > 1:
            # Run tests in parallel
            suite_results = self._run_tests_parallel(suite_name, test_files, timeout)
        else:
            # Run tests sequentially
            for test_file in test_files:
                if os.path.exists(test_file):
                    print(f"\nRunning {test_file}...")
                    file_result = self._run_single_test_file(test_file, timeout)
                    
                    suite_results['details'][test_file] = file_result
                    suite_results['total_passed'] += file_result.get('passed', 0)
                    suite_results['total_failed'] += file_result.get('failed', 0)
                    suite_results['total_errors'] += file_result.get('errors', 0)
                    suite_results['total_skipped'] += file_result.get('skipped', 0)
                    
                    # Print summary
                    status = "PASS" if file_result.get('return_code', 1) == 0 else "FAIL"
                    print(f"  {status}: {file_result.get('passed', 0)} passed, "
                          f"{file_result.get('failed', 0)} failed, "
                          f"{file_result.get('errors', 0)} errors, "
                          f"{file_result.get('skipped', 0)} skipped")
                    
                    if file_result.get('return_code', 1) != 0 and self.verbose:
                        print(f"  STDERR: {file_result.get('stderr', '')}")
                
                else:
                    print(f"  SKIP: {test_file} not found")
        
        suite_results['duration'] = time.time() - start_time
        
        print(f"\n{suite_name} Summary: "
              f"{suite_results['total_passed']} passed, "
              f"{suite_results['total_failed']} failed, "
              f"{suite_results['total_errors']} errors, "
              f"{suite_results['total_skipped']} skipped "
              f"({suite_results['duration']:.1f}s)")
        
        return suite_results
    
    def _run_tests_parallel(self, suite_name: str, test_files: List[str], timeout: int) -> Dict[str, Any]:
        """Run tests in parallel using ThreadPoolExecutor"""
        print(f"Running {len(test_files)} test files in parallel...")
        
        suite_results = {
            'total_passed': 0,
            'total_failed': 0,
            'total_errors': 0,
            'total_skipped': 0,
            'details': {},
            'duration': 0,
            'suite_name': suite_name
        }
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(test_files))) as executor:
            # Submit all test files
            future_to_file = {
                executor.submit(self._run_single_test_file, test_file, timeout): test_file
                for test_file in test_files if os.path.exists(test_file)
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_file):
                test_file = future_to_file[future]
                try:
                    file_result = future.result()
                    suite_results['details'][test_file] = file_result
                    suite_results['total_passed'] += file_result.get('passed', 0)
                    suite_results['total_failed'] += file_result.get('failed', 0)
                    suite_results['total_errors'] += file_result.get('errors', 0)
                    suite_results['total_skipped'] += file_result.get('skipped', 0)
                    
                    status = "PASS" if file_result.get('return_code', 1) == 0 else "FAIL"
                    print(f"  {test_file}: {status}")
                
                except Exception as e:
                    print(f"  {test_file}: ERROR - {e}")
                    suite_results['details'][test_file] = {
                        'passed': 0, 'failed': 1, 'errors': 0, 'skipped': 0,
                        'return_code': -1, 'error': str(e)
                    }
                    suite_results['total_failed'] += 1
        
        suite_results['duration'] = time.time() - start_time
        return suite_results
    
    def _run_single_test_file(self, test_file: str, timeout: int) -> Dict[str, Any]:
        """Run a single test file and parse results"""
        try:
            # Try pytest first, fall back to unittest
            try:
                result = subprocess.run([
                    sys.executable, '-m', 'pytest', test_file, '-v', '--tb=short'
                ], capture_output=True, text=True, timeout=timeout)
                
                # Parse pytest output
                passed = result.stdout.count(' PASSED')
                failed = result.stdout.count(' FAILED')
                errors = result.stdout.count(' ERROR')
                skipped = result.stdout.count(' SKIPPED')
                
            except FileNotFoundError:
                # Pytest not available, use unittest
                result = subprocess.run([
                    sys.executable, test_file
                ], capture_output=True, text=True, timeout=timeout)
                
                # Parse unittest output
                output = result.stdout + result.stderr
                
                # Look for unittest patterns
                if 'OK' in output and result.returncode == 0:
                    # Count test methods run
                    import re
                    test_count_match = re.search(r'Ran (\d+) test', output)
                    passed = int(test_count_match.group(1)) if test_count_match else 1
                    failed = 0
                    errors = 0
                    skipped = 0
                else:
                    # Parse failure information
                    passed = 0
                    failed = output.count('FAIL:') + output.count('AssertionError')
                    errors = output.count('ERROR:') + output.count('Exception:')
                    skipped = output.count('SKIP:')
                    
                    # If no specific counts found but test failed, assume 1 failure
                    if failed == 0 and errors == 0 and result.returncode != 0:
                        failed = 1
            
            return {
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'skipped': skipped,
                'return_code': result.returncode,
                'output': result.stdout if self.verbose else '',
                'stderr': result.stderr if result.stderr else ''
            }
        
        except subprocess.TimeoutExpired:
            return {
                'passed': 0, 'failed': 1, 'errors': 0, 'skipped': 0,
                'return_code': -1, 'error': f'Test timeout after {timeout}s'
            }
        
        except Exception as e:
            return {
                'passed': 0, 'failed': 1, 'errors': 0, 'skipped': 0,
                'return_code': -1, 'error': str(e)
            }
    
    def _parse_coverage_output(self, output: str) -> Dict[str, Any]:
        """Parse coverage output to extract metrics"""
        coverage_data = {
            'overall_coverage': 0,
            'file_coverage': {},
            'missing_lines': {}
        }
        
        lines = output.split('\n')
        for line in lines:
            # Look for overall coverage line
            if 'TOTAL' in line and '%' in line:
                parts = line.split()
                for part in parts:
                    if '%' in part:
                        try:
                            coverage_data['overall_coverage'] = float(part.replace('%', ''))
                            break
                        except:
                            pass
            
            # Look for individual file coverage
            elif '.py' in line and '%' in line and 'TOTAL' not in line:
                parts = line.split()
                if len(parts) >= 4:
                    filename = parts[0]
                    for part in parts:
                        if '%' in part:
                            try:
                                coverage_pct = float(part.replace('%', ''))
                                coverage_data['file_coverage'][filename] = coverage_pct
                                break
                            except:
                                pass
        
        return coverage_data
    
    def generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on test results"""
        recommendations = []
        
        # System resource recommendations
        system_info = self.results['system_info']
        
        if system_info['memory_available_gb'] < 4:
            recommendations.append(
                "CRITICAL: Low available memory (<4GB). Consider upgrading RAM for better test performance."
            )
        elif system_info['memory_available_gb'] < 8:
            recommendations.append(
                "WARNING: Limited available memory (<8GB). Consider reducing test parallelism."
            )
        
        if system_info['cpu_count'] < 4:
            recommendations.append(
                "Consider upgrading to a multi-core processor for faster test execution."
            )
        
        # Test performance recommendations
        unit_tests = self.results.get('unit_tests', {})
        if unit_tests.get('duration', 0) > 300:  # 5 minutes
            recommendations.append(
                "Unit tests are taking longer than expected. Consider optimizing slow tests."
            )
        
        integration_tests = self.results.get('integration_tests', {})
        if integration_tests.get('duration', 0) > 600:  # 10 minutes
            recommendations.append(
                "Integration tests are slow. Consider using more mocks or smaller test datasets."
            )
        
        # Coverage recommendations
        coverage = self.results.get('coverage', {})
        overall_coverage = coverage.get('overall_coverage', 0)
        
        if overall_coverage < 60:
            recommendations.append(
                "CRITICAL: Code coverage is below 60%. Add more unit tests to improve coverage."
            )
        elif overall_coverage < 80:
            recommendations.append(
                "WARNING: Code coverage is below 80%. Consider adding more comprehensive tests."
            )
        
        # Test failure recommendations
        total_failed = (unit_tests.get('total_failed', 0) + 
                       integration_tests.get('total_failed', 0) +
                       self.results.get('performance_tests', {}).get('total_failed', 0) +
                       self.results.get('stress_tests', {}).get('total_failed', 0))
        
        if total_failed > 0:
            recommendations.append(
                f"CRITICAL: {total_failed} tests are failing. Fix failing tests before deployment."
            )
        
        # Performance recommendations
        performance_tests = self.results.get('performance_tests', {})
        if performance_tests.get('total_failed', 0) > 0:
            recommendations.append(
                "Performance tests are failing. Investigate performance regressions."
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append("All tests are passing with good coverage. System is ready for deployment.")
        
        recommendations.extend([
            "Run tests regularly in CI/CD pipeline to catch regressions early.",
            "Consider adding more edge case tests for better robustness.",
            "Monitor test execution time and optimize slow tests periodically.",
            "Keep test data and fixtures up to date with production scenarios."
        ])
        
        return recommendations
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report with analysis"""
        total_time = time.time() - self.start_time
        
        # Calculate totals
        unit_passed = self.results.get('unit_tests', {}).get('total_passed', 0)
        unit_failed = self.results.get('unit_tests', {}).get('total_failed', 0)
        unit_errors = self.results.get('unit_tests', {}).get('total_errors', 0)
        
        integration_passed = self.results.get('integration_tests', {}).get('total_passed', 0)
        integration_failed = self.results.get('integration_tests', {}).get('total_failed', 0)
        integration_errors = self.results.get('integration_tests', {}).get('total_errors', 0)
        
        performance_passed = self.results.get('performance_tests', {}).get('total_passed', 0)
        performance_failed = self.results.get('performance_tests', {}).get('total_failed', 0)
        performance_errors = self.results.get('performance_tests', {}).get('total_errors', 0)
        
        stress_passed = self.results.get('stress_tests', {}).get('total_passed', 0)
        stress_failed = self.results.get('stress_tests', {}).get('total_failed', 0)
        stress_errors = self.results.get('stress_tests', {}).get('total_errors', 0)
        
        total_passed = unit_passed + integration_passed + performance_passed + stress_passed
        total_failed = unit_failed + integration_failed + performance_failed + stress_failed
        total_errors = unit_errors + integration_errors + performance_errors + stress_errors
        total_tests = total_passed + total_failed + total_errors
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        coverage = self.results.get('coverage', {}).get('overall_coverage', 0)
        
        # Generate optimization recommendations
        self.results['optimization_recommendations'] = self.generate_optimization_recommendations()
        
        # Determine overall status
        if total_failed + total_errors == 0 and coverage >= 80:
            overall_status = "EXCELLENT"
        elif total_failed + total_errors == 0 and coverage >= 60:
            overall_status = "GOOD"
        elif success_rate >= 90:
            overall_status = "ACCEPTABLE"
        else:
            overall_status = "NEEDS IMPROVEMENT"
        
        # Create summary
        summary = {
            'total_time': total_time,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_errors': total_errors,
            'success_rate': success_rate,
            'coverage': coverage,
            'overall_status': overall_status,
            'system_info': self.results['system_info']
        }
        
        self.results['summary'] = summary
        
        return self.results
    
    def print_comprehensive_report(self):
        """Print comprehensive test report to console"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TEST SUITE REPORT")
        print("=" * 80)
        
        summary = self.results['summary']
        
        print(f"Total execution time: {summary['total_time']:.1f} seconds")
        print(f"System: {summary['system_info']['cpu_count']} CPUs, "
              f"{summary['system_info']['memory_total_gb']:.1f}GB RAM")
        print()
        
        # Test results breakdown
        print("TEST RESULTS BREAKDOWN:")
        print("-" * 40)
        
        for test_type in ['unit_tests', 'integration_tests', 'performance_tests', 'stress_tests']:
            test_data = self.results.get(test_type, {})
            if test_data:
                passed = test_data.get('total_passed', 0)
                failed = test_data.get('total_failed', 0)
                errors = test_data.get('total_errors', 0)
                duration = test_data.get('duration', 0)
                
                print(f"{test_type.replace('_', ' ').title():<20}: "
                      f"{passed} passed, {failed} failed, {errors} errors "
                      f"({duration:.1f}s)")
        
        print()
        print(f"TOTAL: {summary['total_passed']} passed, "
              f"{summary['total_failed']} failed, "
              f"{summary['total_errors']} errors")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Code Coverage: {summary['coverage']:.1f}%")
        print(f"Overall Status: {summary['overall_status']}")
        
        # Optimization recommendations
        print("\nOPTIMIZATION RECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(self.results['optimization_recommendations'], 1):
            print(f"{i}. {rec}")
        
        # Coverage details
        coverage_data = self.results.get('coverage', {})
        if coverage_data.get('file_coverage') and self.verbose:
            print("\nCOVERAGE BY FILE:")
            print("-" * 40)
            for filename, coverage in sorted(coverage_data['file_coverage'].items()):
                print(f"{filename:<40}: {coverage:>6.1f}%")
    
    def save_report(self, filename: str = 'comprehensive_test_report.json'):
        """Save comprehensive report to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to: {filename}")
        
        # Also save human-readable report
        txt_filename = filename.replace('.json', '.txt')
        with open(txt_filename, 'w') as f:
            f.write("Adobe Stock Image Processor - Comprehensive Test Report\n")
            f.write("=" * 70 + "\n\n")
            
            summary = self.results['summary']
            f.write(f"Execution Time: {summary['total_time']:.1f} seconds\n")
            f.write(f"System: {summary['system_info']['cpu_count']} CPUs, "
                   f"{summary['system_info']['memory_total_gb']:.1f}GB RAM\n\n")
            
            f.write("TEST RESULTS:\n")
            f.write(f"  Total Tests: {summary['total_tests']}\n")
            f.write(f"  Passed: {summary['total_passed']}\n")
            f.write(f"  Failed: {summary['total_failed']}\n")
            f.write(f"  Errors: {summary['total_errors']}\n")
            f.write(f"  Success Rate: {summary['success_rate']:.1f}%\n")
            f.write(f"  Code Coverage: {summary['coverage']:.1f}%\n")
            f.write(f"  Overall Status: {summary['overall_status']}\n\n")
            
            f.write("OPTIMIZATION RECOMMENDATIONS:\n")
            for i, rec in enumerate(self.results['optimization_recommendations'], 1):
                f.write(f"  {i}. {rec}\n")
        
        print(f"Human-readable report saved to: {txt_filename}")


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive test suite for Adobe Stock Image Processor')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--parallel', '-p', action='store_true', help='Run tests in parallel')
    parser.add_argument('--unit-only', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true', help='Run only integration tests')
    parser.add_argument('--performance-only', action='store_true', help='Run only performance tests')
    parser.add_argument('--stress-only', action='store_true', help='Run only stress tests')
    parser.add_argument('--no-coverage', action='store_true', help='Skip coverage analysis')
    parser.add_argument('--output', '-o', default='comprehensive_test_report.json', 
                       help='Output report filename')
    
    args = parser.parse_args()
    
    runner = ComprehensiveTestRunner(verbose=args.verbose, parallel=args.parallel)
    
    print("Adobe Stock Image Processor - Comprehensive Test Suite")
    print("=" * 70)
    print(f"System: {runner.results['system_info']['cpu_count']} CPUs, "
          f"{runner.results['system_info']['memory_total_gb']:.1f}GB RAM, "
          f"{runner.results['system_info']['disk_free_gb']:.1f}GB free")
    print()
    
    success = True
    
    # Run selected test suites
    if not any([args.integration_only, args.performance_only, args.stress_only]):
        runner.results['unit_tests'] = runner.run_unit_tests()
        if runner.results['unit_tests']['total_failed'] > 0:
            success = False
    
    if not any([args.unit_only, args.performance_only, args.stress_only]):
        runner.results['integration_tests'] = runner.run_integration_tests()
        if runner.results['integration_tests']['total_failed'] > 0:
            success = False
    
    if not any([args.unit_only, args.integration_only, args.stress_only]):
        runner.results['performance_tests'] = runner.run_performance_tests()
        if runner.results['performance_tests']['total_failed'] > 0:
            success = False
    
    if not any([args.unit_only, args.integration_only, args.performance_only]):
        runner.results['stress_tests'] = runner.run_stress_tests()
        if runner.results['stress_tests']['total_failed'] > 0:
            success = False
    
    # Run coverage analysis
    if not args.no_coverage:
        runner.results['coverage'] = runner.run_coverage_analysis()
    
    # Generate comprehensive report
    runner.generate_comprehensive_report()
    runner.print_comprehensive_report()
    runner.save_report(args.output)
    
    # Exit with appropriate code
    overall_success = runner.results['summary']['overall_status'] in ['EXCELLENT', 'GOOD']
    sys.exit(0 if success and overall_success else 1)


if __name__ == '__main__':
    main()