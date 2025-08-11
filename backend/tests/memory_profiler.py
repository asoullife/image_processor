#!/usr/bin/env python3
"""
Memory profiling utilities for Adobe Stock Image Processor
Provides detailed memory usage analysis and optimization recommendations
"""

import os
import sys
import time
import psutil
import gc
import tracemalloc
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import threading
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: float
    rss_mb: float
    vms_mb: float
    percent: float
    available_mb: float
    tracemalloc_current: Optional[int] = None
    tracemalloc_peak: Optional[int] = None


@dataclass
class MemoryProfile:
    """Complete memory profile for a test"""
    test_name: str
    start_snapshot: MemorySnapshot
    end_snapshot: MemorySnapshot
    peak_snapshot: MemorySnapshot
    snapshots: List[MemorySnapshot]
    duration: float
    memory_leak_mb: float
    peak_increase_mb: float
    avg_memory_mb: float
    recommendations: List[str]


class MemoryProfiler:
    """Advanced memory profiler for performance analysis"""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        self.snapshots = []
        self.start_time = None
        self.tracemalloc_enabled = False
    
    def start_profiling(self, enable_tracemalloc: bool = True):
        """Start memory profiling"""
        self.snapshots = []
        self.start_time = time.time()
        self.monitoring = True
        
        # Enable tracemalloc for detailed Python memory tracking
        if enable_tracemalloc:
            tracemalloc.start()
            self.tracemalloc_enabled = True
        
        # Take initial snapshot
        self._take_snapshot()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_profiling(self) -> List[MemorySnapshot]:
        """Stop profiling and return snapshots"""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # Take final snapshot
        self._take_snapshot()
        
        if self.tracemalloc_enabled:
            tracemalloc.stop()
            self.tracemalloc_enabled = False
        
        return self.snapshots
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot"""
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            virtual_memory = psutil.virtual_memory()
            
            tracemalloc_current = None
            tracemalloc_peak = None
            
            if self.tracemalloc_enabled:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc_current = current
                tracemalloc_peak = peak
            
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                rss_mb=memory_info.rss / 1024 / 1024,
                vms_mb=memory_info.vms / 1024 / 1024,
                percent=memory_percent,
                available_mb=virtual_memory.available / 1024 / 1024,
                tracemalloc_current=tracemalloc_current,
                tracemalloc_peak=tracemalloc_peak
            )
            
            self.snapshots.append(snapshot)
            return snapshot
            
        except Exception as e:
            print(f"Error taking memory snapshot: {e}")
            return None
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                self._take_snapshot()
                time.sleep(self.sample_interval)
            except Exception:
                break
    
    def analyze_profile(self, test_name: str) -> MemoryProfile:
        """Analyze memory profile and generate recommendations"""
        if len(self.snapshots) < 2:
            raise ValueError("Need at least 2 snapshots for analysis")
        
        start_snapshot = self.snapshots[0]
        end_snapshot = self.snapshots[-1]
        
        # Find peak memory usage
        peak_snapshot = max(self.snapshots, key=lambda s: s.rss_mb)
        
        # Calculate metrics
        duration = end_snapshot.timestamp - start_snapshot.timestamp
        memory_leak_mb = end_snapshot.rss_mb - start_snapshot.rss_mb
        peak_increase_mb = peak_snapshot.rss_mb - start_snapshot.rss_mb
        avg_memory_mb = sum(s.rss_mb for s in self.snapshots) / len(self.snapshots)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            start_snapshot, end_snapshot, peak_snapshot, memory_leak_mb, peak_increase_mb
        )
        
        return MemoryProfile(
            test_name=test_name,
            start_snapshot=start_snapshot,
            end_snapshot=end_snapshot,
            peak_snapshot=peak_snapshot,
            snapshots=self.snapshots,
            duration=duration,
            memory_leak_mb=memory_leak_mb,
            peak_increase_mb=peak_increase_mb,
            avg_memory_mb=avg_memory_mb,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, start: MemorySnapshot, end: MemorySnapshot, 
                                peak: MemorySnapshot, leak_mb: float, peak_increase_mb: float) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        # Memory leak detection
        if leak_mb > 50:
            recommendations.append(f"CRITICAL: Memory leak detected ({leak_mb:.1f}MB increase). Check for unclosed resources.")
        elif leak_mb > 20:
            recommendations.append(f"WARNING: Possible memory leak ({leak_mb:.1f}MB increase). Monitor for patterns.")
        
        # Peak memory usage
        if peak_increase_mb > 2000:
            recommendations.append("CRITICAL: Very high peak memory usage (>2GB). Consider processing smaller batches.")
        elif peak_increase_mb > 1000:
            recommendations.append("WARNING: High peak memory usage (>1GB). Consider memory optimization.")
        
        # Memory efficiency
        if peak.rss_mb > avg_memory_mb * 2:
            recommendations.append("Consider implementing memory cleanup between processing batches.")
        
        # System memory pressure
        if end.available_mb < 1000:
            recommendations.append("CRITICAL: Low system memory available (<1GB). Risk of system instability.")
        elif end.available_mb < 2000:
            recommendations.append("WARNING: Low system memory available (<2GB). Monitor system performance.")
        
        # Tracemalloc recommendations
        if start.tracemalloc_peak and end.tracemalloc_peak:
            python_leak = (end.tracemalloc_peak - start.tracemalloc_peak) / 1024 / 1024
            if python_leak > 100:
                recommendations.append(f"Python memory leak detected ({python_leak:.1f}MB). Check object references.")
        
        if not recommendations:
            recommendations.append("Memory usage appears optimal.")
        
        return recommendations


class MemoryBenchmark:
    """Memory benchmarking for different components"""
    
    def __init__(self):
        self.profiler = MemoryProfiler()
        self.results = {}
    
    def benchmark_component(self, component_name: str, test_function, *args, **kwargs) -> MemoryProfile:
        """Benchmark memory usage of a specific component"""
        print(f"Benchmarking {component_name}...")
        
        # Force garbage collection before test
        gc.collect()
        
        # Start profiling
        self.profiler.start_profiling()
        
        try:
            # Run the test function
            result = test_function(*args, **kwargs)
            
        except Exception as e:
            print(f"Error during {component_name} benchmark: {e}")
            result = None
        
        finally:
            # Stop profiling
            snapshots = self.profiler.stop_profiling()
        
        # Analyze profile
        profile = self.profiler.analyze_profile(component_name)
        self.results[component_name] = profile
        
        # Print summary
        print(f"  Duration: {profile.duration:.2f}s")
        print(f"  Memory leak: {profile.memory_leak_mb:.1f}MB")
        print(f"  Peak increase: {profile.peak_increase_mb:.1f}MB")
        print(f"  Average memory: {profile.avg_memory_mb:.1f}MB")
        
        if profile.recommendations:
            print("  Recommendations:")
            for rec in profile.recommendations:
                print(f"    - {rec}")
        
        return profile
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive memory benchmark of all components"""
        from backend.analyzers.quality_analyzer import QualityAnalyzer
        from backend.analyzers.defect_detector import DefectDetector
        from backend.analyzers.similarity_finder import SimilarityFinder
        from backend.analyzers.compliance_checker import ComplianceChecker
        from backend.core.batch_processor import BatchProcessor
        from backend.config.config_loader import ConfigLoader
        
        config_loader = ConfigLoader()
        config = config_loader.load_config()
        
        # Create test image for benchmarking
        test_image = self._create_test_image()
        
        print("Running comprehensive memory benchmark...")
        print("=" * 50)
        
        # Benchmark each component
        try:
            # Quality Analyzer
            def test_quality_analyzer():
                analyzer = QualityAnalyzer(config)
                for _ in range(10):
                    analyzer.analyze(test_image)
            
            self.benchmark_component("QualityAnalyzer", test_quality_analyzer)
            
            # Defect Detector
            def test_defect_detector():
                detector = DefectDetector(config)
                for _ in range(5):  # Fewer iterations for ML model
                    detector.detect_defects(test_image)
            
            self.benchmark_component("DefectDetector", test_defect_detector)
            
            # Similarity Finder
            def test_similarity_finder():
                finder = SimilarityFinder(config)
                test_images = [test_image] * 20
                finder.find_similar_groups(test_images)
            
            self.benchmark_component("SimilarityFinder", test_similarity_finder)
            
            # Compliance Checker
            def test_compliance_checker():
                checker = ComplianceChecker(config)
                for _ in range(10):
                    checker.check_compliance(test_image, {})
            
            self.benchmark_component("ComplianceChecker", test_compliance_checker)
            
            # Batch Processor
            def test_batch_processor():
                processor = BatchProcessor(config)
                test_images = [test_image] * 50
                processor.process_batch(test_images)
            
            self.benchmark_component("BatchProcessor", test_batch_processor)
            
        except Exception as e:
            print(f"Benchmark error: {e}")
        
        finally:
            # Clean up test image
            if os.path.exists(test_image):
                os.remove(test_image)
        
        # Generate summary report
        self._generate_benchmark_report()
    
    def _create_test_image(self) -> str:
        """Create a test image for benchmarking"""
        from PIL import Image
        import numpy as np
        import tempfile
        
        # Create a temporary test image
        img_array = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img.save(temp_file.name, 'JPEG', quality=85)
        temp_file.close()
        
        return temp_file.name
    
    def _generate_benchmark_report(self):
        """Generate comprehensive benchmark report"""
        print("\n" + "=" * 50)
        print("MEMORY BENCHMARK SUMMARY")
        print("=" * 50)
        
        if not self.results:
            print("No benchmark results available.")
            return
        
        # Summary table
        print(f"{'Component':<20} {'Duration':<10} {'Leak(MB)':<10} {'Peak(MB)':<10} {'Avg(MB)':<10}")
        print("-" * 70)
        
        total_duration = 0
        total_leak = 0
        max_peak = 0
        
        for name, profile in self.results.items():
            print(f"{name:<20} {profile.duration:<10.2f} {profile.memory_leak_mb:<10.1f} "
                  f"{profile.peak_increase_mb:<10.1f} {profile.avg_memory_mb:<10.1f}")
            
            total_duration += profile.duration
            total_leak += profile.memory_leak_mb
            max_peak = max(max_peak, profile.peak_increase_mb)
        
        print("-" * 70)
        print(f"{'TOTALS':<20} {total_duration:<10.2f} {total_leak:<10.1f} {max_peak:<10.1f}")
        
        # Critical recommendations
        print("\nCRITICAL RECOMMENDATIONS:")
        critical_found = False
        for name, profile in self.results.items():
            for rec in profile.recommendations:
                if rec.startswith("CRITICAL"):
                    print(f"  {name}: {rec}")
                    critical_found = True
        
        if not critical_found:
            print("  No critical issues found.")
        
        # Save detailed results
        self._save_benchmark_results()
    
    def _save_benchmark_results(self):
        """Save benchmark results to JSON file"""
        results_data = {}
        
        for name, profile in self.results.items():
            results_data[name] = {
                'duration': profile.duration,
                'memory_leak_mb': profile.memory_leak_mb,
                'peak_increase_mb': profile.peak_increase_mb,
                'avg_memory_mb': profile.avg_memory_mb,
                'recommendations': profile.recommendations,
                'snapshots_count': len(profile.snapshots)
            }
        
        with open('memory_benchmark_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\nDetailed benchmark results saved to: memory_benchmark_results.json")


def main():
    """Run memory profiling benchmark"""
    benchmark = MemoryBenchmark()
    benchmark.run_comprehensive_benchmark()


if __name__ == '__main__':
    main()