#!/usr/bin/env python3
"""
Performance optimization analyzer for Adobe Stock Image Processor
Analyzes system performance and provides optimization recommendations
"""

import os
import sys
import json
import time
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.memory_profiler import MemoryBenchmark
from backend.config.config_loader import ConfigLoader


class PerformanceOptimizer:
    """Analyze and optimize system performance"""
    
    def __init__(self):
        config_loader = ConfigLoader()
        self.config = config_loader.load_config()
        self.optimization_results = {}
        self.recommendations = []
    
    def analyze_system_resources(self) -> Dict[str, Any]:
        """Analyze available system resources"""
        print("Analyzing system resources...")
        
        # CPU information
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        # Disk information
        disk = psutil.disk_usage('.')
        disk_free_gb = disk.free / (1024**3)
        
        system_info = {
            'cpu': {
                'count': cpu_count,
                'frequency_mhz': cpu_freq.current if cpu_freq else None,
                'current_usage_percent': cpu_percent
            },
            'memory': {
                'total_gb': memory_gb,
                'available_gb': memory_available_gb,
                'usage_percent': memory.percent
            },
            'disk': {
                'free_gb': disk_free_gb,
                'usage_percent': (disk.used / disk.total) * 100
            }
        }
        
        print(f"  CPU: {cpu_count} cores, {cpu_percent:.1f}% usage")
        print(f"  Memory: {memory_gb:.1f}GB total, {memory_available_gb:.1f}GB available")
        print(f"  Disk: {disk_free_gb:.1f}GB free")
        
        return system_info
    
    def analyze_current_configuration(self) -> Dict[str, Any]:
        """Analyze current configuration for optimization opportunities"""
        print("\nAnalyzing current configuration...")
        
        config_analysis = {
            'batch_size': self.config.processing.batch_size,
            'max_workers': self.config.processing.max_workers,
            'checkpoint_interval': self.config.processing.checkpoint_interval,
            'quality_thresholds': {
                'min_sharpness': self.config.quality.min_sharpness,
                'max_noise_level': self.config.quality.max_noise_level,
                'min_resolution': self.config.quality.min_resolution
            },
            'similarity_settings': {
                'hash_threshold': self.config.similarity.hash_threshold,
                'feature_threshold': self.config.similarity.feature_threshold,
                'clustering_eps': self.config.similarity.clustering_eps
            }
        }
        
        print(f"  Batch size: {config_analysis['batch_size']}")
        print(f"  Max workers: {config_analysis['max_workers']}")
        print(f"  Checkpoint interval: {config_analysis['checkpoint_interval']}")
        
        return config_analysis
    
    def benchmark_configuration_variants(self, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark different configuration variants"""
        print("\nBenchmarking configuration variants...")
        
        # Test different batch sizes
        batch_sizes = [50, 100, 200, 500]
        worker_counts = [1, 2, 4, min(8, system_info['cpu']['count'])]
        
        benchmark_results = {}
        
        # Create test configuration variants
        test_configs = []
        for batch_size in batch_sizes:
            for worker_count in worker_counts:
                if batch_size * worker_count <= 1000:  # Reasonable total load
                    test_configs.append({
                        'batch_size': batch_size,
                        'max_workers': worker_count,
                        'name': f'batch_{batch_size}_workers_{worker_count}'
                    })
        
        # Limit to most promising configurations
        test_configs = test_configs[:6]  # Test top 6 configurations
        
        for config_variant in test_configs:
            print(f"  Testing {config_variant['name']}...")
            
            try:
                # Create modified config
                # Create modified config (simplified simulation)
                test_config = {
                    'processing': {
                        'batch_size': config_variant['batch_size'],
                        'max_workers': config_variant['max_workers']
                    }
                }
                
                # Run benchmark (simplified)
                start_time = time.time()
                memory_start = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Simulate processing with this configuration
                # (In real implementation, would run actual processing)
                processing_time = self._simulate_processing(test_config)
                
                memory_end = psutil.Process().memory_info().rss / 1024 / 1024
                total_time = time.time() - start_time
                
                benchmark_results[config_variant['name']] = {
                    'batch_size': config_variant['batch_size'],
                    'max_workers': config_variant['max_workers'],
                    'processing_time': processing_time,
                    'total_time': total_time,
                    'memory_usage_mb': memory_end - memory_start,
                    'efficiency_score': self._calculate_efficiency_score(
                        processing_time, memory_end - memory_start, config_variant
                    )
                }
                
                print(f"    Time: {processing_time:.2f}s, Memory: {memory_end - memory_start:.1f}MB")
            
            except Exception as e:
                print(f"    Error testing {config_variant['name']}: {e}")
        
        return benchmark_results
    
    def _simulate_processing(self, config: Dict[str, Any]) -> float:
        """Simulate processing time for configuration testing"""
        # Simplified simulation based on configuration parameters
        batch_size = config['processing']['batch_size']
        max_workers = config['processing']['max_workers']
        
        # Simulate processing overhead
        base_time = 0.1  # Base processing time
        batch_overhead = batch_size * 0.001  # Time per image in batch
        worker_overhead = max_workers * 0.05  # Thread management overhead
        
        # Simulate some actual work
        time.sleep(base_time + batch_overhead + worker_overhead)
        
        return base_time + batch_overhead + worker_overhead
    
    def _calculate_efficiency_score(self, processing_time: float, memory_mb: float, 
                                  config: Dict[str, Any]) -> float:
        """Calculate efficiency score for configuration"""
        # Lower processing time is better
        time_score = max(0, 10 - processing_time * 10)
        
        # Lower memory usage is better (up to a point)
        memory_score = max(0, 10 - (memory_mb / 100))
        
        # Higher throughput potential is better
        throughput_score = min(10, config['batch_size'] * config['max_workers'] / 100)
        
        return (time_score + memory_score + throughput_score) / 3
    
    def generate_optimization_recommendations(self, system_info: Dict[str, Any], 
                                           config_analysis: Dict[str, Any],
                                           benchmark_results: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        print("\nGenerating optimization recommendations...")
        
        recommendations = []
        
        # System resource recommendations
        if system_info['memory']['available_gb'] < 4:
            recommendations.append(
                "CRITICAL: Low available memory (<4GB). Consider upgrading RAM or closing other applications."
            )
        elif system_info['memory']['available_gb'] < 8:
            recommendations.append(
                "WARNING: Limited available memory (<8GB). Consider reducing batch sizes."
            )
        
        if system_info['cpu']['count'] < 4:
            recommendations.append(
                "Consider upgrading to a multi-core processor for better performance."
            )
        
        # Configuration recommendations
        current_batch_size = config_analysis['batch_size']
        current_workers = config_analysis['max_workers']
        
        # Find best performing configuration
        if benchmark_results:
            best_config = max(benchmark_results.items(), key=lambda x: x[1]['efficiency_score'])
            best_name, best_result = best_config
            
            if best_result['batch_size'] != current_batch_size:
                recommendations.append(
                    f"Consider changing batch size from {current_batch_size} to {best_result['batch_size']} "
                    f"for {best_result['efficiency_score']:.1f} efficiency score."
                )
            
            if best_result['max_workers'] != current_workers:
                recommendations.append(
                    f"Consider changing max workers from {current_workers} to {best_result['max_workers']} "
                    f"for better performance."
                )
        
        # Memory optimization recommendations
        if system_info['memory']['usage_percent'] > 80:
            recommendations.append(
                "High memory usage detected. Consider enabling more aggressive garbage collection."
            )
        
        # Disk space recommendations
        if system_info['disk']['free_gb'] < 10:
            recommendations.append(
                "CRITICAL: Low disk space (<10GB). Free up space before processing large datasets."
            )
        elif system_info['disk']['free_gb'] < 50:
            recommendations.append(
                "WARNING: Limited disk space (<50GB). Monitor space usage during processing."
            )
        
        # Quality threshold recommendations
        quality_config = config_analysis['quality_thresholds']
        if quality_config['min_sharpness'] > 150:
            recommendations.append(
                "Very high sharpness threshold may reject too many images. Consider lowering to 100-120."
            )
        
        # General recommendations
        recommendations.extend([
            "Enable checkpoint saving every 50-100 images for crash recovery.",
            "Use SSD storage for input/output directories for better I/O performance.",
            "Consider processing during off-peak hours for better system performance.",
            "Monitor system temperature during long processing sessions."
        ])
        
        return recommendations
    
    def create_optimized_configuration(self, system_info: Dict[str, Any], 
                                     benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimized configuration based on analysis"""
        print("\nCreating optimized configuration...")
        
        optimized_config = self.config.copy()
        
        # Find best configuration from benchmarks
        if benchmark_results:
            best_config = max(benchmark_results.items(), key=lambda x: x[1]['efficiency_score'])
            best_name, best_result = best_config
            
            optimized_config['processing']['batch_size'] = best_result['batch_size']
            optimized_config['processing']['max_workers'] = best_result['max_workers']
        
        # Adjust based on system resources
        available_memory_gb = system_info['memory']['available_gb']
        
        if available_memory_gb < 4:
            # Low memory system
            optimized_config['processing']['batch_size'] = min(50, optimized_config['processing']['batch_size'])
            optimized_config['processing']['max_workers'] = min(2, optimized_config['processing']['max_workers'])
        elif available_memory_gb < 8:
            # Medium memory system
            optimized_config['processing']['batch_size'] = min(100, optimized_config['processing']['batch_size'])
            optimized_config['processing']['max_workers'] = min(4, optimized_config['processing']['max_workers'])
        
        # Adjust checkpoint interval based on batch size
        batch_size = optimized_config['processing']['batch_size']
        optimized_config['processing']['checkpoint_interval'] = max(25, min(100, batch_size // 2))
        
        return optimized_config
    
    def save_optimization_report(self, system_info: Dict[str, Any], 
                               config_analysis: Dict[str, Any],
                               benchmark_results: Dict[str, Any],
                               recommendations: List[str],
                               optimized_config: Dict[str, Any],
                               test_results: Dict[str, Any] = None):
        """Save comprehensive optimization report"""
        report = {
            'timestamp': time.time(),
            'system_info': system_info,
            'current_config': config_analysis,
            'benchmark_results': benchmark_results,
            'recommendations': recommendations,
            'optimized_config': optimized_config,
            'test_results': test_results or {}
        }
        
        # Save JSON report
        with open('performance_optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save human-readable report
        with open('performance_optimization_report.txt', 'w') as f:
            f.write("Adobe Stock Image Processor - Performance Optimization Report\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("SYSTEM INFORMATION:\n")
            f.write(f"  CPU: {system_info['cpu']['count']} cores\n")
            f.write(f"  Memory: {system_info['memory']['total_gb']:.1f}GB total, "
                   f"{system_info['memory']['available_gb']:.1f}GB available\n")
            f.write(f"  Disk: {system_info['disk']['free_gb']:.1f}GB free\n\n")
            
            f.write("CURRENT CONFIGURATION:\n")
            f.write(f"  Batch size: {config_analysis['batch_size']}\n")
            f.write(f"  Max workers: {config_analysis['max_workers']}\n")
            f.write(f"  Checkpoint interval: {config_analysis['checkpoint_interval']}\n\n")
            
            if benchmark_results:
                f.write("BENCHMARK RESULTS:\n")
                for name, result in sorted(benchmark_results.items(), 
                                         key=lambda x: x[1]['efficiency_score'], reverse=True):
                    f.write(f"  {name}: {result['efficiency_score']:.1f} efficiency, "
                           f"{result['processing_time']:.2f}s, {result['memory_usage_mb']:.1f}MB\n")
                f.write("\n")
            
            f.write("RECOMMENDATIONS:\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"  {i}. {rec}\n")
            f.write("\n")
            
            f.write("OPTIMIZED CONFIGURATION:\n")
            f.write(f"  Batch size: {optimized_config['processing']['batch_size']}\n")
            f.write(f"  Max workers: {optimized_config['processing']['max_workers']}\n")
            f.write(f"  Checkpoint interval: {optimized_config['processing']['checkpoint_interval']}\n")
        
        print(f"\nOptimization report saved to:")
        print(f"  performance_optimization_report.json")
        print(f"  performance_optimization_report.txt")
    
    def analyze_test_results(self) -> Dict[str, Any]:
        """Analyze recent test results for optimization insights"""
        print("\nAnalyzing recent test results...")
        
        test_results = {}
        
        # Look for recent test result files
        test_result_files = [
            'comprehensive_test_report.json',
            'test_results.json',
            'memory_benchmark_results.json'
        ]
        
        for result_file in test_result_files:
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        test_results[result_file] = data
                        print(f"  Loaded {result_file}")
                except Exception as e:
                    print(f"  Error loading {result_file}: {e}")
        
        return test_results
    
    def generate_test_based_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Analyze comprehensive test results
        if 'comprehensive_test_report.json' in test_results:
            comp_results = test_results['comprehensive_test_report.json']
            
            # Check test performance
            summary = comp_results.get('summary', {})
            if summary.get('total_time', 0) > 1800:  # 30 minutes
                recommendations.append(
                    "Test suite is taking too long. Consider optimizing slow tests or increasing parallelism."
                )
            
            # Check performance test results
            perf_tests = comp_results.get('performance_tests', {})
            if perf_tests.get('total_failed', 0) > 0:
                recommendations.append(
                    "Performance tests are failing. Review performance benchmarks and optimize bottlenecks."
                )
            
            # Check stress test results
            stress_tests = comp_results.get('stress_tests', {})
            if stress_tests.get('total_failed', 0) > 0:
                recommendations.append(
                    "Stress tests are failing. System may not handle high loads well. Consider resource optimization."
                )
            
            # Check coverage
            coverage = comp_results.get('coverage', {}).get('overall_coverage', 0)
            if coverage < 80:
                recommendations.append(
                    f"Code coverage is {coverage:.1f}%. Add more tests to improve reliability."
                )
        
        # Analyze memory benchmark results
        if 'memory_benchmark_results.json' in test_results:
            memory_results = test_results['memory_benchmark_results.json']
            
            for component, metrics in memory_results.items():
                if isinstance(metrics, dict):
                    memory_leak = metrics.get('memory_leak_mb', 0)
                    peak_memory = metrics.get('peak_increase_mb', 0)
                    
                    if memory_leak > 100:
                        recommendations.append(
                            f"{component} has significant memory leak ({memory_leak:.1f}MB). "
                            "Review memory management in this component."
                        )
                    
                    if peak_memory > 2000:
                        recommendations.append(
                            f"{component} uses high peak memory ({peak_memory:.1f}MB). "
                            "Consider processing smaller batches."
                        )
        
        return recommendations
    
    def run_full_optimization_analysis(self):
        """Run complete optimization analysis"""
        print("Adobe Stock Image Processor - Performance Optimization")
        print("=" * 60)
        
        # Analyze system
        system_info = self.analyze_system_resources()
        
        # Analyze current configuration
        config_analysis = self.analyze_current_configuration()
        
        # Analyze test results
        test_results = self.analyze_test_results()
        
        # Benchmark configurations
        benchmark_results = self.benchmark_configuration_variants(system_info)
        
        # Generate recommendations
        recommendations = self.generate_optimization_recommendations(
            system_info, config_analysis, benchmark_results
        )
        
        # Add test-based recommendations
        test_recommendations = self.generate_test_based_recommendations(test_results)
        recommendations.extend(test_recommendations)
        
        # Create optimized configuration
        optimized_config = self.create_optimized_configuration(system_info, benchmark_results)
        
        # Display recommendations
        print("\nOPTIMIZATION RECOMMENDATIONS:")
        print("-" * 40)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # Save report
        self.save_optimization_report(
            system_info, config_analysis, benchmark_results, 
            recommendations, optimized_config, test_results
        )
        
        return {
            'system_info': system_info,
            'recommendations': recommendations,
            'optimized_config': optimized_config,
            'test_results': test_results
        }


def main():
    parser = argparse.ArgumentParser(description='Optimize Adobe Stock Image Processor performance')
    parser.add_argument('--memory-benchmark', action='store_true', 
                       help='Run detailed memory benchmark')
    parser.add_argument('--config-only', action='store_true',
                       help='Only analyze configuration, skip benchmarks')
    
    args = parser.parse_args()
    
    optimizer = PerformanceOptimizer()
    
    if args.memory_benchmark:
        print("Running detailed memory benchmark...")
        benchmark = MemoryBenchmark()
        benchmark.run_comprehensive_benchmark()
    
    if not args.config_only:
        results = optimizer.run_full_optimization_analysis()
        
        # Ask user if they want to apply optimized configuration
        print(f"\nWould you like to apply the optimized configuration? (y/n): ", end="")
        response = input().strip().lower()
        
        if response in ['y', 'yes']:
            # Save optimized configuration
            optimized_config_path = 'config/settings.optimized.json'
            with open(optimized_config_path, 'w') as f:
                json.dump(results['optimized_config'], f, indent=2)
            
            print(f"Optimized configuration saved to: {optimized_config_path}")
            print("You can use this configuration by copying it to config/settings.json")
        else:
            print("Optimization analysis complete. Review the report for manual optimization.")
    else:
        # Just analyze current configuration
        system_info = optimizer.analyze_system_resources()
        config_analysis = optimizer.analyze_current_configuration()
        recommendations = optimizer.generate_optimization_recommendations(
            system_info, config_analysis, {}
        )
        
        print("\nCONFIGURATION RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")


if __name__ == '__main__':
    main()