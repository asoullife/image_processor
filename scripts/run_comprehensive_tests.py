#!/usr/bin/env python3
"""
Comprehensive test runner for Adobe Stock Image Processor
Runs all unit tests, integration tests, and performance tests with coverage reporting
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys
import subprocess
import time
import json
from pathlib import Path
import argparse


class TestRunner:
    """Comprehensive test runner with reporting"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.results = {
            'unit_tests': {},
            'integration_tests': {},
            'performance_tests': {},
            'coverage': {},
            'summary': {}
        }
        self.start_time = time.time()
    
    def run_unit_tests(self):
        """Run all unit tests"""
        print("=" * 60)
        print("RUNNING UNIT TESTS")
        print("=" * 60)
        
        unit_test_files = [
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
        
        unit_results = {}
        total_passed = 0
        total_failed = 0
        
        for test_file in unit_test_files:
            if os.path.exists(test_file):
                print(f"\nRunning {test_file}...")
                try:
                    result = subprocess.run([
                        sys.executable, '-m', 'pytest', test_file, '-v'
                    ], capture_output=True, text=True, timeout=300)
                    
                    # Parse pytest output
                    passed = result.stdout.count(' PASSED')
                    failed = result.stdout.count(' FAILED')
                    errors = result.stdout.count(' ERROR')
                    
                    unit_results[test_file] = {
                        'passed': passed,
                        'failed': failed,
                        'errors': errors,
                        'return_code': result.returncode,
                        'output': result.stdout if self.verbose else '',
                        'stderr': result.stderr if result.stderr else ''
                    }
                    
                    total_passed += passed
                    total_failed += failed + errors
                    
                    status = "PASS" if result.returncode == 0 else "FAIL"
                    print(f"  {status}: {passed} passed, {failed} failed, {errors} errors")
                    
                    if result.returncode != 0 and self.verbose:
                        print(f"  STDERR: {result.stderr}")
                
                except subprocess.TimeoutExpired:
                    unit_results[test_file] = {
                        'passed': 0, 'failed': 1, 'errors': 0,
                        'return_code': -1, 'output': '', 'stderr': 'Test timeout'
                    }
                    total_failed += 1
                    print(f"  TIMEOUT: Test exceeded 5 minutes")
                
                except Exception as e:
                    unit_results[test_file] = {
                        'passed': 0, 'failed': 1, 'errors': 0,
                        'return_code': -1, 'output': '', 'stderr': str(e)
                    }
                    total_failed += 1
                    print(f"  ERROR: {str(e)}")
            else:
                print(f"  SKIP: {test_file} not found")
        
        self.results['unit_tests'] = {
            'total_passed': total_passed,
            'total_failed': total_failed,
            'details': unit_results
        }
        
        print(f"\nUnit Tests Summary: {total_passed} passed, {total_failed} failed")
    
    def run_integration_tests(self):
        """Run integration tests"""
        print("\n" + "=" * 60)
        print("RUNNING INTEGRATION TESTS")
        print("=" * 60)
        
        integration_test_files = [
            'tests/test_integration_database_progress.py',
            'tests/test_integration_batch_processor.py',
            'tests/test_integration_main_app.py'
        ]
        
        integration_results = {}
        total_passed = 0
        total_failed = 0
        
        for test_file in integration_test_files:
            if os.path.exists(test_file):
                print(f"\nRunning {test_file}...")
                try:
                    result = subprocess.run([
                        sys.executable, '-m', 'pytest', test_file, '-v'
                    ], capture_output=True, text=True, timeout=600)  # Longer timeout for integration
                    
                    passed = result.stdout.count(' PASSED')
                    failed = result.stdout.count(' FAILED')
                    errors = result.stdout.count(' ERROR')
                    
                    integration_results[test_file] = {
                        'passed': passed,
                        'failed': failed,
                        'errors': errors,
                        'return_code': result.returncode,
                        'output': result.stdout if self.verbose else '',
                        'stderr': result.stderr if result.stderr else ''
                    }
                    
                    total_passed += passed
                    total_failed += failed + errors
                    
                    status = "PASS" if result.returncode == 0 else "FAIL"
                    print(f"  {status}: {passed} passed, {failed} failed, {errors} errors")
                
                except subprocess.TimeoutExpired:
                    integration_results[test_file] = {
                        'passed': 0, 'failed': 1, 'errors': 0,
                        'return_code': -1, 'output': '', 'stderr': 'Test timeout'
                    }
                    total_failed += 1
                    print(f"  TIMEOUT: Test exceeded 10 minutes")
                
                except Exception as e:
                    integration_results[test_file] = {
                        'passed': 0, 'failed': 1, 'errors': 0,
                        'return_code': -1, 'output': '', 'stderr': str(e)
                    }
                    total_failed += 1
                    print(f"  ERROR: {str(e)}")
            else:
                print(f"  SKIP: {test_file} not found")
        
        self.results['integration_tests'] = {
            'total_passed': total_passed,
            'total_failed': total_failed,
            'details': integration_results
        }
        
        print(f"\nIntegration Tests Summary: {total_passed} passed, {total_failed} failed")
    
    def run_performance_tests(self):
        """Run performance tests"""
        print("\n" + "=" * 60)
        print("RUNNING PERFORMANCE TESTS")
        print("=" * 60)
        
        print("\nRunning performance tests...")
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pytest', 'tests/test_performance.py', '-v', '-s'
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            passed = result.stdout.count(' PASSED')
            failed = result.stdout.count(' FAILED')
            errors = result.stdout.count(' ERROR')
            
            self.results['performance_tests'] = {
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'return_code': result.returncode,
                'output': result.stdout,
                'stderr': result.stderr if result.stderr else ''
            }
            
            status = "PASS" if result.returncode == 0 else "FAIL"
            print(f"  {status}: {passed} passed, {failed} failed, {errors} errors")
            
            # Print performance output
            if self.verbose or result.returncode != 0:
                print("\nPerformance Test Output:")
                print(result.stdout)
        
        except subprocess.TimeoutExpired:
            self.results['performance_tests'] = {
                'passed': 0, 'failed': 1, 'errors': 0,
                'return_code': -1, 'output': '', 'stderr': 'Performance tests timeout'
            }
            print("  TIMEOUT: Performance tests exceeded 30 minutes")
        
        except Exception as e:
            self.results['performance_tests'] = {
                'passed': 0, 'failed': 1, 'errors': 0,
                'return_code': -1, 'output': '', 'stderr': str(e)
            }
            print(f"  ERROR: {str(e)}")
    
    def run_coverage_analysis(self):
        """Run code coverage analysis"""
        print("\n" + "=" * 60)
        print("RUNNING COVERAGE ANALYSIS")
        print("=" * 60)
        
        try:
            # Run coverage on all tests
            print("Generating coverage report...")
            result = subprocess.run([
                sys.executable, '-m', 'pytest', '--cov=.', '--cov-report=html', '--cov-report=term',
                'tests/', '--cov-exclude=tests/*', '--cov-exclude=demo_*', '--cov-exclude=test_*'
            ], capture_output=True, text=True, timeout=900)
            
            # Parse coverage output
            coverage_lines = result.stdout.split('\n')
            coverage_data = {}
            
            for line in coverage_lines:
                if '.py' in line and '%' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        filename = parts[0]
                        try:
                            coverage_pct = int(parts[-1].replace('%', ''))
                            coverage_data[filename] = coverage_pct
                        except:
                            pass
            
            # Calculate overall coverage
            if coverage_data:
                overall_coverage = sum(coverage_data.values()) / len(coverage_data)
            else:
                overall_coverage = 0
            
            self.results['coverage'] = {
                'overall_coverage': overall_coverage,
                'file_coverage': coverage_data,
                'return_code': result.returncode,
                'output': result.stdout if self.verbose else '',
                'stderr': result.stderr if result.stderr else ''
            }
            
            print(f"Overall coverage: {overall_coverage:.1f}%")
            
            if self.verbose:
                print("\nFile-by-file coverage:")
                for filename, coverage in sorted(coverage_data.items()):
                    print(f"  {filename}: {coverage}%")
        
        except Exception as e:
            self.results['coverage'] = {
                'overall_coverage': 0,
                'file_coverage': {},
                'return_code': -1,
                'output': '',
                'stderr': str(e)
            }
            print(f"Coverage analysis failed: {str(e)}")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY REPORT")
        print("=" * 60)
        
        total_time = time.time() - self.start_time
        
        # Calculate totals
        unit_passed = self.results['unit_tests'].get('total_passed', 0)
        unit_failed = self.results['unit_tests'].get('total_failed', 0)
        
        integration_passed = self.results['integration_tests'].get('total_passed', 0)
        integration_failed = self.results['integration_tests'].get('total_failed', 0)
        
        perf_passed = self.results['performance_tests'].get('passed', 0)
        perf_failed = self.results['performance_tests'].get('failed', 0) + \
                     self.results['performance_tests'].get('errors', 0)
        
        total_passed = unit_passed + integration_passed + perf_passed
        total_failed = unit_failed + integration_failed + perf_failed
        total_tests = total_passed + total_failed
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        coverage = self.results['coverage'].get('overall_coverage', 0)
        
        print(f"Total execution time: {total_time:.1f} seconds")
        print(f"")
        print(f"Unit Tests:        {unit_passed} passed, {unit_failed} failed")
        print(f"Integration Tests: {integration_passed} passed, {integration_failed} failed")
        print(f"Performance Tests: {perf_passed} passed, {perf_failed} failed")
        print(f"")
        print(f"TOTAL:            {total_passed} passed, {total_failed} failed")
        print(f"Success Rate:     {success_rate:.1f}%")
        print(f"Code Coverage:    {coverage:.1f}%")
        
        # Determine overall status
        if total_failed == 0 and coverage >= 80:
            overall_status = "EXCELLENT"
        elif total_failed == 0 and coverage >= 60:
            overall_status = "GOOD"
        elif success_rate >= 90:
            overall_status = "ACCEPTABLE"
        else:
            overall_status = "NEEDS IMPROVEMENT"
        
        print(f"Overall Status:   {overall_status}")
        
        # Save detailed results
        self.results['summary'] = {
            'total_time': total_time,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'success_rate': success_rate,
            'coverage': coverage,
            'overall_status': overall_status
        }
        
        # Save results to JSON file
        results_file = 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        if coverage > 0:
            print(f"HTML coverage report available at: htmlcov/index.html")
        
        return overall_status == "EXCELLENT" or overall_status == "GOOD"


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive tests for Adobe Stock Image Processor')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--unit-only', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration-only', action='store_true', help='Run only integration tests')
    parser.add_argument('--performance-only', action='store_true', help='Run only performance tests')
    parser.add_argument('--no-coverage', action='store_true', help='Skip coverage analysis')
    
    args = parser.parse_args()
    
    runner = TestRunner(verbose=args.verbose)
    
    print("Adobe Stock Image Processor - Comprehensive Test Suite")
    print("=" * 60)
    
    success = True
    
    if not args.integration_only and not args.performance_only:
        runner.run_unit_tests()
        if runner.results['unit_tests']['total_failed'] > 0:
            success = False
    
    if not args.unit_only and not args.performance_only:
        runner.run_integration_tests()
        if runner.results['integration_tests']['total_failed'] > 0:
            success = False
    
    if not args.unit_only and not args.integration_only:
        runner.run_performance_tests()
        if runner.results['performance_tests']['failed'] > 0:
            success = False
    
    if not args.no_coverage and not args.performance_only:
        runner.run_coverage_analysis()
    
    overall_success = runner.generate_summary_report()
    
    # Exit with appropriate code
    sys.exit(0 if success and overall_success else 1)


if __name__ == '__main__':
    main()