#!/usr/bin/env python3
"""
Enhanced performance tests for Adobe Stock Image Processor
Comprehensive benchmarking, optimization testing, and performance regression detection
"""

import unittest
import time
import psutil
import os
import tempfile
import shutil
import threading
import json
import gc
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.batch_processor import BatchProcessor
from backend.analyzers.quality_analyzer import QualityAnalyzer
from backend.analyzers.defect_detector import DefectDetector
from backend.analyzers.similarity_finder import SimilarityFinder
from backend.analyzers.compliance_checker import ComplianceChecker
from backend.utils.file_manager import FileManager
from backend.core.progress_tracker import ProgressTracker
from backend.core.database import DatabaseManager
from backend.config.config_loader import ConfigLoader


@dataclass
class PerformanceMetrics:
    """Performance metrics for a test"""
    test_name: str
    duration: float
    images_processed: int
    images_per_second: float
    peak_memory_mb: float
    avg_memory_mb: float
    memory_increase_mb: float
    cpu_avg_percent: float
    cpu_peak_percent: float
    success_rate: float
    error_count: int


class EnhancedPerformanceMonitor:
    """Enhanced performance monitoring with detailed metrics"""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = {
            'memory_samples': [],
            'cpu_samples': [],
            'start_time': None,
            'start_memory': None
        }
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.metrics = {
            'memory_samples': [],
            'cpu_samples': [],
            'start_time': time.time(),
            'start_memory': self.process.memory_info().rss / 1024 / 1024
        }
        self.monitoring = True
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        return {
            'duration': end_time - self.metrics['start_time'],
            'memory_samples': self.metrics['memory_samples'],
            'cpu_samples': self.metrics['cpu_samples'],
            'peak_memory_mb': max(self.metrics['memory_samples']) if self.metrics['memory_samples'] else 0,
            'avg_memory_mb': sum(self.metrics['memory_samples']) / len(self.metrics['memory_samples']) if self.metrics['memory_samples'] else 0,
            'memory_increase_mb': end_memory - self.metrics['start_memory'],
            'cpu_avg_percent': sum(self.metrics['cpu_samples']) / len(self.metrics['cpu_samples']) if self.metrics['cpu_samples'] else 0,
            'cpu_peak_percent': max(self.metrics['cpu_samples']) if self.metrics['cpu_samples'] else 0
        }
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()
                
                self.metrics['memory_samples'].append(memory_mb)
                self.metrics['cpu_samples'].append(cpu_percent)
                
                time.sleep(self.sample_interval)
            except:
                break


class EnhancedPerformanceTestCase(unittest.TestCase):
    """Enhanced performance test case with comprehensive benchmarking"""
    
    @classmethod
    def setUpClass(cls):
        """Set up enhanced performance test environment"""
        cls.test_dir = tempfile.mkdtemp(prefix='enhanced_perf_test_')
        config_loader = ConfigLoader()
        cls.config = config_loader.load_config()
        cls.monitor = EnhancedPerformanceMonitor()
        
        # Create comprehensive test dataset
        cls.test_images_dir = os.path.join(cls.test_dir, 'perf_images')
        os.makedirs(cls.test_images_dir)
        cls._create_performance_test_images()
        
        # Performance baselines (to be updated based on system)
        cls.performance_baselines = {
            'quality_analyzer_min_speed': 2.0,  # images per second
            'defect_detector_min_speed': 0.2,   # images per second
            'similarity_finder_min_speed': 1.0,  # images per second
            'compliance_checker_min_speed': 3.0, # images per second
            'batch_processor_min_speed': 1.0,    # images per second
            'max_memory_per_image': 50,          # MB per image
            'max_memory_leak': 100               # MB total leak
        }
        
        print(f"Enhanced performance test environment created at: {cls.test_dir}")
        print(f"Created {len(os.listdir(cls.test_images_dir))} test images")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _create_performance_test_images(cls):
        """Create comprehensive test images for performance testing"""
        from PIL import Image
        import numpy as np
        
        # Create various image configurations for comprehensive testing
        image_configs = [
            # (width, height, quality, count, description)
            (1920, 1080, 95, 20, 'hd_high_quality'),
            (2560, 1440, 85, 15, 'qhd_medium_quality'),
            (3840, 2160, 90, 10, '4k_high_quality'),
            (800, 600, 70, 25, 'small_medium_quality'),
            (1280, 720, 60, 20, 'hd_low_quality'),
            (640, 480, 50, 15, 'vga_low_quality')
        ]
        
        image_count = 0
        for width, height, quality, count, desc in image_configs:
            for i in range(count):
                # Create varied image content for realistic testing
                if 'high_quality' in desc:
                    # Sharp, detailed images
                    img_array = np.random.randint(50, 206, (height, width, 3), dtype=np.uint8)
                    # Add some structure
                    img_array[::20, :] = 255
                    img_array[:, ::20] = 0
                elif 'medium_quality' in desc:
                    # Medium detail images
                    img_array = np.random.randint(30, 226, (height, width, 3), dtype=np.uint8)
                else:
                    # Lower quality, noisier images
                    img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                    # Add noise
                    noise = np.random.normal(0, 20, img_array.shape)
                    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                
                img = Image.fromarray(img_array, 'RGB')
                filename = f'{desc}_{i:02d}_{image_count:03d}.jpg'
                img.save(os.path.join(cls.test_images_dir, filename), 'JPEG', quality=quality)
                image_count += 1
        
        print(f"Created {image_count} performance test images")
    
    def setUp(self):
        """Set up each test"""
        gc.collect()  # Clean memory before each test
        self.monitor = EnhancedPerformanceMonitor()
    
    def tearDown(self):
        """Clean up after each test"""
        gc.collect()  # Clean memory after each test
    
    def _run_performance_test(self, test_name: str, test_function, *args, **kwargs) -> PerformanceMetrics:
        """Run a performance test and collect comprehensive metrics"""
        print(f"\nRunning {test_name}...")
        
        self.monitor.start_monitoring()
        start_time = time.time()
        
        try:
            result = test_function(*args, **kwargs)
            success_rate = 1.0
            error_count = 0
            images_processed = result.get('images_processed', 0) if isinstance(result, dict) else len(result) if result else 0
        except Exception as e:
            print(f"  Error in {test_name}: {e}")
            success_rate = 0.0
            error_count = 1
            images_processed = 0
        
        duration = time.time() - start_time
        metrics_data = self.monitor.stop_monitoring()
        
        # Calculate performance metrics
        images_per_second = images_processed / duration if duration > 0 else 0
        
        metrics = PerformanceMetrics(
            test_name=test_name,
            duration=duration,
            images_processed=images_processed,
            images_per_second=images_per_second,
            peak_memory_mb=metrics_data['peak_memory_mb'],
            avg_memory_mb=metrics_data['avg_memory_mb'],
            memory_increase_mb=metrics_data['memory_increase_mb'],
            cpu_avg_percent=metrics_data['cpu_avg_percent'],
            cpu_peak_percent=metrics_data['cpu_peak_percent'],
            success_rate=success_rate,
            error_count=error_count
        )
        
        # Print summary
        print(f"  Duration: {duration:.2f}s")
        print(f"  Images processed: {images_processed}")
        print(f"  Speed: {images_per_second:.2f} images/sec")
        print(f"  Peak memory: {metrics.peak_memory_mb:.1f}MB")
        print(f"  Memory increase: {metrics.memory_increase_mb:.1f}MB")
        print(f"  CPU avg: {metrics.cpu_avg_percent:.1f}%")
        
        return metrics


class TestComponentPerformance(EnhancedPerformanceTestCase):
    """Test individual component performance"""
    
    def test_quality_analyzer_performance_benchmark(self):
        """Comprehensive quality analyzer performance benchmark"""
        def quality_analyzer_test():
            analyzer = QualityAnalyzer(self.config)
            test_images = list(Path(self.test_images_dir).glob('*.jpg'))[:50]
            
            results = []
            for image_path in test_images:
                result = analyzer.analyze(str(image_path))
                results.append(result)
            
            return {'images_processed': len(results)}
        
        metrics = self._run_performance_test("QualityAnalyzer Benchmark", quality_analyzer_test)
        
        # Performance assertions
        self.assertGreater(metrics.images_per_second, self.performance_baselines['quality_analyzer_min_speed'],
                          f"Quality analyzer too slow: {metrics.images_per_second:.2f} < {self.performance_baselines['quality_analyzer_min_speed']}")
        self.assertLess(metrics.memory_increase_mb, self.performance_baselines['max_memory_leak'],
                       f"Memory leak detected: {metrics.memory_increase_mb:.1f}MB")
        
        return metrics
    
    def test_defect_detector_performance_benchmark(self):
        """Comprehensive defect detector performance benchmark"""
        def defect_detector_test():
            detector = DefectDetector(self.config)
            test_images = list(Path(self.test_images_dir).glob('*.jpg'))[:20]  # Smaller set for ML model
            
            results = []
            for image_path in test_images:
                result = detector.detect_defects(str(image_path))
                results.append(result)
            
            return {'images_processed': len(results)}
        
        metrics = self._run_performance_test("DefectDetector Benchmark", defect_detector_test)
        
        # Performance assertions (ML models are slower)
        self.assertGreater(metrics.images_per_second, self.performance_baselines['defect_detector_min_speed'],
                          f"Defect detector too slow: {metrics.images_per_second:.2f} < {self.performance_baselines['defect_detector_min_speed']}")
        self.assertLess(metrics.memory_increase_mb, self.performance_baselines['max_memory_leak'] * 2,
                       f"Excessive memory usage: {metrics.memory_increase_mb:.1f}MB")
        
        return metrics
    
    def test_similarity_finder_performance_benchmark(self):
        """Comprehensive similarity finder performance benchmark"""
        def similarity_finder_test():
            finder = SimilarityFinder(self.config)
            test_images = list(Path(self.test_images_dir).glob('*.jpg'))[:30]
            
            # Test both hash computation and similarity grouping
            hashes = []
            for image_path in test_images:
                hash_result = finder.compute_hash(str(image_path))
                hashes.append(hash_result)
            
            # Test similarity grouping
            similar_groups = finder.find_similar_groups([str(p) for p in test_images])
            
            return {'images_processed': len(test_images)}
        
        metrics = self._run_performance_test("SimilarityFinder Benchmark", similarity_finder_test)
        
        # Performance assertions
        self.assertGreater(metrics.images_per_second, self.performance_baselines['similarity_finder_min_speed'],
                          f"Similarity finder too slow: {metrics.images_per_second:.2f} < {self.performance_baselines['similarity_finder_min_speed']}")
        
        return metrics
    
    def test_compliance_checker_performance_benchmark(self):
        """Comprehensive compliance checker performance benchmark"""
        def compliance_checker_test():
            checker = ComplianceChecker(self.config)
            test_images = list(Path(self.test_images_dir).glob('*.jpg'))[:40]
            
            results = []
            for image_path in test_images:
                metadata = {'keywords': ['test', 'image'], 'description': 'Test image'}
                result = checker.check_compliance(str(image_path), metadata)
                results.append(result)
            
            return {'images_processed': len(results)}
        
        metrics = self._run_performance_test("ComplianceChecker Benchmark", compliance_checker_test)
        
        # Performance assertions
        self.assertGreater(metrics.images_per_second, self.performance_baselines['compliance_checker_min_speed'],
                          f"Compliance checker too slow: {metrics.images_per_second:.2f} < {self.performance_baselines['compliance_checker_min_speed']}")
        
        return metrics
    
    def test_batch_processor_performance_benchmark(self):
        """Comprehensive batch processor performance benchmark"""
        def batch_processor_test():
            processor = BatchProcessor(self.config)
            test_images = list(Path(self.test_images_dir).glob('*.jpg'))[:25]
            
            results = processor.process_batch([str(p) for p in test_images])
            
            return {'images_processed': len(results)}
        
        metrics = self._run_performance_test("BatchProcessor Benchmark", batch_processor_test)
        
        # Performance assertions
        self.assertGreater(metrics.images_per_second, self.performance_baselines['batch_processor_min_speed'],
                          f"Batch processor too slow: {metrics.images_per_second:.2f} < {self.performance_baselines['batch_processor_min_speed']}")
        
        return metrics


class TestScalabilityPerformance(EnhancedPerformanceTestCase):
    """Test performance scalability with different loads"""
    
    def test_batch_size_scalability(self):
        """Test performance with different batch sizes"""
        processor = BatchProcessor(self.config)
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))
        
        batch_sizes = [10, 25, 50, 100]
        scalability_results = {}
        
        for batch_size in batch_sizes:
            if batch_size > len(test_images):
                continue
                
            def batch_size_test():
                processor.config['processing']['batch_size'] = batch_size
                batch = test_images[:batch_size]
                results = processor.process_batch([str(p) for p in batch])
                return {'images_processed': len(results)}
            
            metrics = self._run_performance_test(f"BatchSize_{batch_size}", batch_size_test)
            scalability_results[batch_size] = metrics
            
            # Cleanup between tests
            processor.cleanup_memory()
            gc.collect()
        
        # Analyze scalability
        print(f"\nBatch Size Scalability Analysis:")
        for batch_size, metrics in scalability_results.items():
            efficiency = metrics.images_per_second / batch_size  # Efficiency per image
            print(f"  Batch {batch_size}: {metrics.images_per_second:.2f} img/s, "
                  f"efficiency: {efficiency:.4f}, memory: {metrics.peak_memory_mb:.1f}MB")
        
        # Should maintain reasonable efficiency across batch sizes
        efficiencies = [m.images_per_second / bs for bs, m in scalability_results.items()]
        efficiency_variance = max(efficiencies) - min(efficiencies)
        self.assertLess(efficiency_variance, 0.01, "Batch size efficiency should be consistent")
    
    def test_concurrent_worker_scalability(self):
        """Test performance with different worker counts"""
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))[:20]
        worker_counts = [1, 2, 4]
        
        scalability_results = {}
        
        for worker_count in worker_counts:
            def worker_test():
                config = self.config.copy()
                config['processing']['max_workers'] = worker_count
                processor = BatchProcessor(config)
                
                results = processor.process_batch([str(p) for p in test_images])
                return {'images_processed': len(results)}
            
            metrics = self._run_performance_test(f"Workers_{worker_count}", worker_test)
            scalability_results[worker_count] = metrics
        
        # Analyze worker scalability
        print(f"\nWorker Scalability Analysis:")
        baseline_speed = scalability_results[1].images_per_second
        
        for worker_count, metrics in scalability_results.items():
            speedup = metrics.images_per_second / baseline_speed
            print(f"  {worker_count} workers: {metrics.images_per_second:.2f} img/s, "
                  f"speedup: {speedup:.2f}x, memory: {metrics.peak_memory_mb:.1f}MB")
        
        # Should see some improvement with more workers
        if len(scalability_results) > 1:
            max_speedup = max(m.images_per_second for m in scalability_results.values()) / baseline_speed
            self.assertGreater(max_speedup, 1.1, "Should see at least 10% improvement with more workers")


class TestMemoryPerformance(EnhancedPerformanceTestCase):
    """Test memory usage and optimization"""
    
    def test_memory_usage_per_image_size(self):
        """Test memory usage with different image sizes"""
        analyzer = QualityAnalyzer(self.config)
        
        # Group images by size
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))
        size_groups = {
            'small': [img for img in test_images if 'vga' in str(img) or 'small' in str(img)],
            'medium': [img for img in test_images if 'hd' in str(img) and '1920' in str(img)],
            'large': [img for img in test_images if '4k' in str(img) or '3840' in str(img)]
        }
        
        memory_results = {}
        
        for size_category, images in size_groups.items():
            if not images:
                continue
                
            def memory_test():
                results = []
                for image_path in images[:5]:  # Test 5 images per category
                    result = analyzer.analyze(str(image_path))
                    results.append(result)
                return {'images_processed': len(results)}
            
            metrics = self._run_performance_test(f"Memory_{size_category}", memory_test)
            memory_results[size_category] = metrics
        
        # Analyze memory usage by image size
        print(f"\nMemory Usage by Image Size:")
        for size_category, metrics in memory_results.items():
            memory_per_image = metrics.peak_memory_mb / metrics.images_processed if metrics.images_processed > 0 else 0
            print(f"  {size_category}: {memory_per_image:.1f}MB per image, "
                  f"peak: {metrics.peak_memory_mb:.1f}MB")
            
            # Memory usage should be reasonable
            self.assertLess(memory_per_image, self.performance_baselines['max_memory_per_image'],
                           f"Excessive memory per image for {size_category}: {memory_per_image:.1f}MB")
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated processing"""
        analyzer = QualityAnalyzer(self.config)
        test_image = str(list(Path(self.test_images_dir).glob('*.jpg'))[0])
        
        def leak_test():
            # Process same image multiple times
            for i in range(50):
                result = analyzer.analyze(test_image)
                if i % 10 == 0:
                    gc.collect()  # Force garbage collection
            
            return {'images_processed': 50}
        
        metrics = self._run_performance_test("MemoryLeak Detection", leak_test)
        
        # Should not leak significant memory
        self.assertLess(metrics.memory_increase_mb, self.performance_baselines['max_memory_leak'],
                       f"Memory leak detected: {metrics.memory_increase_mb:.1f}MB increase")
        
        print(f"  Memory leak test: {metrics.memory_increase_mb:.1f}MB increase over 50 iterations")
    
    def test_memory_cleanup_effectiveness(self):
        """Test effectiveness of memory cleanup mechanisms"""
        processor = BatchProcessor(self.config)
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))[:20]
        
        def cleanup_test():
            # Process batch
            results = processor.process_batch([str(p) for p in test_images])
            
            # Force cleanup
            processor.cleanup_memory()
            gc.collect()
            
            return {'images_processed': len(results)}
        
        metrics = self._run_performance_test("Memory Cleanup", cleanup_test)
        
        # Memory increase should be minimal after cleanup
        self.assertLess(metrics.memory_increase_mb, 200,
                       f"Memory cleanup ineffective: {metrics.memory_increase_mb:.1f}MB remaining")


class TestPerformanceRegression(EnhancedPerformanceTestCase):
    """Test for performance regressions"""
    
    def test_performance_regression_detection(self):
        """Test for performance regressions against baseline"""
        # Load previous performance results if available
        baseline_file = 'performance_baseline.json'
        baseline_metrics = {}
        
        if os.path.exists(baseline_file):
            try:
                with open(baseline_file, 'r') as f:
                    baseline_metrics = json.load(f)
                print("Loaded performance baseline for regression testing")
            except:
                print("Could not load performance baseline")
        
        # Run current performance tests
        current_metrics = {}
        
        # Test each component
        components = [
            ('quality_analyzer', self.test_quality_analyzer_performance_benchmark),
            ('batch_processor', self.test_batch_processor_performance_benchmark)
        ]
        
        for component_name, test_method in components:
            try:
                metrics = test_method()
                current_metrics[component_name] = {
                    'images_per_second': metrics.images_per_second,
                    'peak_memory_mb': metrics.peak_memory_mb,
                    'memory_increase_mb': metrics.memory_increase_mb
                }
            except Exception as e:
                print(f"Error testing {component_name}: {e}")
                continue
        
        # Compare with baseline
        regressions = []
        improvements = []
        
        for component, current in current_metrics.items():
            if component in baseline_metrics:
                baseline = baseline_metrics[component]
                
                # Check speed regression (>20% slower)
                speed_ratio = current['images_per_second'] / baseline['images_per_second']
                if speed_ratio < 0.8:
                    regressions.append(f"{component} speed regression: {speed_ratio:.2f}x")
                elif speed_ratio > 1.2:
                    improvements.append(f"{component} speed improvement: {speed_ratio:.2f}x")
                
                # Check memory regression (>50% more memory)
                memory_ratio = current['peak_memory_mb'] / baseline['peak_memory_mb']
                if memory_ratio > 1.5:
                    regressions.append(f"{component} memory regression: {memory_ratio:.2f}x")
        
        # Report results
        if regressions:
            print(f"\nPerformance Regressions Detected:")
            for regression in regressions:
                print(f"  - {regression}")
        
        if improvements:
            print(f"\nPerformance Improvements:")
            for improvement in improvements:
                print(f"  + {improvement}")
        
        # Save current metrics as new baseline
        with open(baseline_file, 'w') as f:
            json.dump(current_metrics, f, indent=2)
        
        # Fail test if significant regressions detected
        self.assertEqual(len(regressions), 0, f"Performance regressions detected: {regressions}")


class TestLargeDatasetPerformance(EnhancedPerformanceTestCase):
    """Test performance with large datasets"""
    
    def test_large_dataset_processing_performance(self):
        """Test processing performance with large dataset (500+ images)"""
        # Use all available test images multiple times to simulate large dataset
        base_images = list(Path(self.test_images_dir).glob('*.jpg'))
        multiplier = max(1, 500 // len(base_images) + 1)
        large_dataset = base_images * multiplier
        large_dataset = large_dataset[:500]  # Exactly 500 images
        
        def large_dataset_test():
            processor = BatchProcessor(self.config)
            batch_size = 50
            total_processed = 0
            
            for i in range(0, len(large_dataset), batch_size):
                batch = large_dataset[i:i + batch_size]
                results = processor.process_batch([str(p) for p in batch])
                total_processed += len(results)
                
                # Cleanup between batches
                if i % 200 == 0:
                    processor.cleanup_memory()
                    gc.collect()
            
            return {'images_processed': total_processed}
        
        metrics = self._run_performance_test("Large Dataset (500 images)", large_dataset_test)
        
        # Performance assertions for large dataset
        self.assertGreaterEqual(metrics.images_processed, 500, "Should process all 500 images")
        self.assertGreater(metrics.images_per_second, 0.5, "Should maintain at least 0.5 images/second")
        self.assertLess(metrics.peak_memory_mb, 8000, "Peak memory should stay under 8GB")
        
        # Calculate efficiency metrics
        memory_per_image = metrics.peak_memory_mb / metrics.images_processed
        self.assertLess(memory_per_image, 15, f"Memory per image too high: {memory_per_image:.1f}MB")
        
        print(f"  Large dataset efficiency: {memory_per_image:.2f}MB per image")
        
        return metrics


def create_performance_report(test_results: List[PerformanceMetrics]):
    """Create comprehensive performance report"""
    report = {
        'timestamp': time.time(),
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / 1024**3,
            'platform': sys.platform
        },
        'test_results': {},
        'summary': {}
    }
    
    # Process test results
    total_images = 0
    total_duration = 0
    
    for metrics in test_results:
        report['test_results'][metrics.test_name] = {
            'duration': metrics.duration,
            'images_processed': metrics.images_processed,
            'images_per_second': metrics.images_per_second,
            'peak_memory_mb': metrics.peak_memory_mb,
            'memory_increase_mb': metrics.memory_increase_mb,
            'success_rate': metrics.success_rate
        }
        
        total_images += metrics.images_processed
        total_duration += metrics.duration
    
    # Calculate summary
    report['summary'] = {
        'total_tests': len(test_results),
        'total_images_processed': total_images,
        'total_duration': total_duration,
        'overall_speed': total_images / total_duration if total_duration > 0 else 0,
        'successful_tests': sum(1 for m in test_results if m.success_rate > 0.9)
    }
    
    # Save report
    with open('enhanced_performance_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nEnhanced performance report saved to: enhanced_performance_report.json")
    return report


if __name__ == '__main__':
    # Run enhanced performance tests
    unittest.main(verbosity=2, buffer=False)