#!/usr/bin/env python3
"""
Simple test runner for Adobe Stock Image Processor using unittest
Runs tests without requiring pytest installation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys
import unittest
import time
import json
from pathlib import Path
import importlib.util


class SimpleTestRunner:
    """Simple test runner using unittest"""
    
    def __init__(self):
        self.results = {
            'unit_tests': {},
            'performance_tests': {},
            'summary': {}
        }
        self.start_time = time.time()
    
    def discover_and_run_tests(self, test_directory='tests'):
        """Discover and run all tests in the directory"""
        print("Adobe Stock Image Processor - Simple Test Runner")
        print("=" * 60)
        
        # Find all test files
        test_files = []
        test_dir = Path(test_directory)
        
        if test_dir.exists():
            test_files = list(test_dir.glob('test_*.py'))
        
        if not test_files:
            print(f"No test files found in {test_directory}")
            return False
        
        print(f"Found {len(test_files)} test files")
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        
        # Run each test file
        for test_file in test_files:
            print(f"\nRunning {test_file.name}...")
            
            try:
                # Import the test module
                spec = importlib.util.spec_from_file_location(
                    test_file.stem, test_file
                )
                test_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(test_module)
                
                # Create test suite
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromModule(test_module)
                
                # Run tests
                runner = unittest.TextTestRunner(
                    verbosity=1, 
                    stream=sys.stdout,
                    buffer=True
                )
                result = runner.run(suite)
                
                # Collect results
                tests_run = result.testsRun
                failures = len(result.failures)
                errors = len(result.errors)
                
                total_tests += tests_run
                total_failures += failures
                total_errors += errors
                
                status = "PASS" if (failures + errors) == 0 else "FAIL"
                print(f"  {status}: {tests_run} tests, {failures} failures, {errors} errors")
                
                # Store detailed results
                self.results['unit_tests'][test_file.name] = {
                    'tests_run': tests_run,
                    'failures': failures,
                    'errors': errors,
                    'success': (failures + errors) == 0
                }
                
            except Exception as e:
                print(f"  ERROR: Failed to run {test_file.name}: {e}")
                total_errors += 1
                self.results['unit_tests'][test_file.name] = {
                    'tests_run': 0,
                    'failures': 0,
                    'errors': 1,
                    'success': False,
                    'error_message': str(e)
                }
        
        # Generate summary
        self.generate_summary(total_tests, total_failures, total_errors)
        
        return (total_failures + total_errors) == 0
    
    def run_performance_tests(self):
        """Run performance tests specifically"""
        print("\n" + "=" * 60)
        print("RUNNING PERFORMANCE TESTS")
        print("=" * 60)
        
        try:
            # Import and run performance tests
            from tests.test_performance import (
                TestProcessingSpeed, TestMemoryUsage, 
                TestStressAndStability, TestBenchmarks
            )
            
            # Create test suite
            suite = unittest.TestSuite()
            
            # Add specific performance tests
            suite.addTest(TestProcessingSpeed('test_quality_analyzer_speed'))
            suite.addTest(TestMemoryUsage('test_batch_processor_memory_management'))
            
            # Run performance tests
            runner = unittest.TextTestRunner(verbosity=2, buffer=False)
            result = runner.run(suite)
            
            # Store results
            self.results['performance_tests'] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'success': (len(result.failures) + len(result.errors)) == 0
            }
            
            print(f"\nPerformance Tests: {result.testsRun} run, "
                  f"{len(result.failures)} failures, {len(result.errors)} errors")
            
        except Exception as e:
            print(f"Performance tests failed: {e}")
            self.results['performance_tests'] = {
                'tests_run': 0,
                'failures': 0,
                'errors': 1,
                'success': False,
                'error_message': str(e)
            }
    
    def run_memory_benchmark(self):
        """Run memory benchmark"""
        print("\n" + "=" * 60)
        print("RUNNING MEMORY BENCHMARK")
        print("=" * 60)
        
        try:
            from tests.memory_profiler import MemoryBenchmark
            
            benchmark = MemoryBenchmark()
            benchmark.run_comprehensive_benchmark()
            
            print("Memory benchmark completed successfully")
            
        except Exception as e:
            print(f"Memory benchmark failed: {e}")
    
    def generate_summary(self, total_tests, total_failures, total_errors):
        """Generate test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        total_time = time.time() - self.start_time
        success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total execution time: {total_time:.1f} seconds")
        print(f"Total tests run: {total_tests}")
        print(f"Failures: {total_failures}")
        print(f"Errors: {total_errors}")
        print(f"Success rate: {success_rate:.1f}%")
        
        # Determine status
        if total_failures + total_errors == 0:
            status = "EXCELLENT"
        elif success_rate >= 80:
            status = "GOOD"
        elif success_rate >= 60:
            status = "ACCEPTABLE"
        else:
            status = "NEEDS IMPROVEMENT"
        
        print(f"Overall status: {status}")
        
        # Save results
        self.results['summary'] = {
            'total_time': total_time,
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'success_rate': success_rate,
            'status': status
        }
        
        # Save to JSON
        with open('simple_test_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nDetailed results saved to: simple_test_results.json")


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple test runner for Adobe Stock Image Processor')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--memory', action='store_true', help='Run memory benchmark')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    runner = SimpleTestRunner()
    
    success = True
    
    if args.all or (not args.performance and not args.memory):
        # Run unit tests
        success = runner.discover_and_run_tests()
    
    if args.performance or args.all:
        # Run performance tests
        runner.run_performance_tests()
    
    if args.memory or args.all:
        # Run memory benchmark
        runner.run_memory_benchmark()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())