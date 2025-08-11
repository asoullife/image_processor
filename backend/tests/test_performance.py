#!/usr/bin/env python3
"""
Performance tests for Adobe Stock Image Processor
Tests processing speed, memory usage, and system stability under load
"""

import unittest
import time
import psutil
import os
import tempfile
import shutil
import threading
from pathlib import Path
from typing import List, Dict, Any
import gc
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


class PerformanceMonitor:
    """Monitor system performance during tests"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0
        self.monitoring = False
        self.monitor_thread = None
        self.memory_samples = []
        self.cpu_samples = []
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        self.monitoring = True
        self.memory_samples = []
        self.cpu_samples = []
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return performance metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'duration': end_time - self.start_time,
            'start_memory_mb': self.start_memory,
            'end_memory_mb': end_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': end_memory - self.start_memory,
            'avg_memory_mb': sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0,
            'avg_cpu_percent': sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0,
            'memory_samples': len(self.memory_samples),
            'cpu_samples': len(self.cpu_samples)
        }
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()
                
                self.memory_samples.append(memory_mb)
                self.cpu_samples.append(cpu_percent)
                self.peak_memory = max(self.peak_memory, memory_mb)
                
                time.sleep(0.1)  # Sample every 100ms
            except:
                break


class PerformanceTestCase(unittest.TestCase):
    """Base class for performance tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = tempfile.mkdtemp(prefix='perf_test_')
        config_loader = ConfigLoader()
        cls.config = config_loader.load_config()
        cls.monitor = PerformanceMonitor()
        
        # Create test images directory
        cls.test_images_dir = os.path.join(cls.test_dir, 'test_images')
        os.makedirs(cls.test_images_dir)
        
        # Generate test images
        cls._create_test_images()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _create_test_images(cls):
        """Create test images for performance testing"""
        from PIL import Image
        import numpy as np
        
        # Create various sizes of test images
        sizes = [(1920, 1080), (2560, 1440), (3840, 2160)]  # HD, QHD, 4K
        colors = ['RGB', 'RGBA']
        
        image_count = 0
        for size in sizes:
            for color in colors:
                for i in range(50):  # 50 images per size/color combination
                    # Create random image
                    if color == 'RGB':
                        img_array = np.random.randint(0, 256, (*size[::-1], 3), dtype=np.uint8)
                    else:
                        img_array = np.random.randint(0, 256, (*size[::-1], 4), dtype=np.uint8)
                    
                    img = Image.fromarray(img_array, color)
                    filename = f'test_image_{image_count:04d}_{size[0]}x{size[1]}_{color}.jpg'
                    img.save(os.path.join(cls.test_images_dir, filename), 'JPEG', quality=85)
                    image_count += 1
        
        print(f"Created {image_count} test images for performance testing")
    
    def setUp(self):
        """Set up each test"""
        gc.collect()  # Clean up memory before each test
    
    def tearDown(self):
        """Clean up after each test"""
        gc.collect()  # Clean up memory after each test


class TestProcessingSpeed(PerformanceTestCase):
    """Test processing speed benchmarks"""
    
    def test_quality_analyzer_speed(self):
        """Test quality analyzer processing speed"""
        analyzer = QualityAnalyzer(self.config)
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))[:100]
        
        self.monitor.start_monitoring()
        
        start_time = time.time()
        results = []
        for image_path in test_images:
            result = analyzer.analyze(str(image_path))
            results.append(result)
        
        metrics = self.monitor.stop_monitoring()
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        images_per_second = len(test_images) / total_time
        avg_time_per_image = total_time / len(test_images)
        
        print(f"\nQuality Analyzer Performance:")
        print(f"  Images processed: {len(test_images)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Images per second: {images_per_second:.2f}")
        print(f"  Average time per image: {avg_time_per_image:.3f}s")
        print(f"  Peak memory: {metrics['peak_memory_mb']:.1f}MB")
        print(f"  Memory increase: {metrics['memory_increase_mb']:.1f}MB")
        
        # Performance assertions
        self.assertGreater(images_per_second, 1.0, "Should process at least 1 image per second")
        self.assertLess(avg_time_per_image, 2.0, "Should process each image in under 2 seconds")
        self.assertLess(metrics['memory_increase_mb'], 500, "Memory increase should be under 500MB")
    
    def test_defect_detector_speed(self):
        """Test defect detector processing speed"""
        detector = DefectDetector(self.config)
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))[:50]  # Smaller set for ML model
        
        self.monitor.start_monitoring()
        
        start_time = time.time()
        results = []
        for image_path in test_images:
            result = detector.detect_defects(str(image_path))
            results.append(result)
        
        metrics = self.monitor.stop_monitoring()
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        images_per_second = len(test_images) / total_time
        avg_time_per_image = total_time / len(test_images)
        
        print(f"\nDefect Detector Performance:")
        print(f"  Images processed: {len(test_images)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Images per second: {images_per_second:.2f}")
        print(f"  Average time per image: {avg_time_per_image:.3f}s")
        print(f"  Peak memory: {metrics['peak_memory_mb']:.1f}MB")
        print(f"  Memory increase: {metrics['memory_increase_mb']:.1f}MB")
        
        # Performance assertions (ML models are slower)
        self.assertGreater(images_per_second, 0.1, "Should process at least 0.1 images per second")
        self.assertLess(avg_time_per_image, 30.0, "Should process each image in under 30 seconds")
        self.assertLess(metrics['memory_increase_mb'], 2000, "Memory increase should be under 2GB")
    
    def test_similarity_finder_speed(self):
        """Test similarity finder processing speed"""
        finder = SimilarityFinder(self.config)
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))[:100]
        
        self.monitor.start_monitoring()
        
        start_time = time.time()
        # Test hash computation
        hashes = []
        for image_path in test_images:
            hash_result = finder.compute_hash(str(image_path))
            hashes.append(hash_result)
        
        # Test similarity grouping
        similar_groups = finder.find_similar_groups([str(p) for p in test_images])
        
        metrics = self.monitor.stop_monitoring()
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        images_per_second = len(test_images) / total_time
        
        print(f"\nSimilarity Finder Performance:")
        print(f"  Images processed: {len(test_images)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Images per second: {images_per_second:.2f}")
        print(f"  Similar groups found: {len(similar_groups)}")
        print(f"  Peak memory: {metrics['peak_memory_mb']:.1f}MB")
        print(f"  Memory increase: {metrics['memory_increase_mb']:.1f}MB")
        
        # Performance assertions
        self.assertGreater(images_per_second, 0.5, "Should process at least 0.5 images per second")
        self.assertLess(metrics['memory_increase_mb'], 1000, "Memory increase should be under 1GB")


class TestMemoryUsage(PerformanceTestCase):
    """Test memory usage and optimization"""
    
    def test_batch_processor_memory_management(self):
        """Test batch processor memory management"""
        processor = BatchProcessor(self.config)
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))
        
        # Test different batch sizes
        batch_sizes = [10, 50, 100, 200]
        memory_results = {}
        
        for batch_size in batch_sizes:
            processor.config['processing']['batch_size'] = batch_size
            
            self.monitor.start_monitoring()
            
            # Process in batches
            for i in range(0, min(len(test_images), 200), batch_size):
                batch = test_images[i:i + batch_size]
                processor.process_batch([str(p) for p in batch])
                processor.cleanup_memory()  # Force cleanup
            
            metrics = self.monitor.stop_monitoring()
            memory_results[batch_size] = metrics
            
            print(f"\nBatch size {batch_size}:")
            print(f"  Peak memory: {metrics['peak_memory_mb']:.1f}MB")
            print(f"  Memory increase: {metrics['memory_increase_mb']:.1f}MB")
            print(f"  Average memory: {metrics['avg_memory_mb']:.1f}MB")
        
        # Verify memory management effectiveness
        # Larger batches should not use exponentially more memory
        small_batch_memory = memory_results[10]['peak_memory_mb']
        large_batch_memory = memory_results[200]['peak_memory_mb']
        memory_ratio = large_batch_memory / small_batch_memory
        
        self.assertLess(memory_ratio, 5.0, "Memory usage should not increase more than 5x with 20x batch size")
    
    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated processing"""
        analyzer = QualityAnalyzer(self.config)
        test_image = str(list(Path(self.test_images_dir).glob('*.jpg'))[0])
        
        # Record initial memory
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Process same image multiple times
        for i in range(100):
            result = analyzer.analyze(test_image)
            if i % 10 == 0:
                gc.collect()  # Force garbage collection
        
        # Record final memory
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory Leak Test:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Memory increase: {memory_increase:.1f}MB")
        
        # Should not leak significant memory
        self.assertLess(memory_increase, 100, "Memory increase should be under 100MB after 100 iterations")


class TestStressAndStability(PerformanceTestCase):
    """Test system stability under heavy loads"""
    
    def test_concurrent_processing(self):
        """Test concurrent processing stability"""
        import concurrent.futures
        
        analyzer = QualityAnalyzer(self.config)
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))[:50]
        
        self.monitor.start_monitoring()
        
        def process_image(image_path):
            try:
                return analyzer.analyze(str(image_path))
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Test with multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_image, img) for img in test_images]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        metrics = self.monitor.stop_monitoring()
        
        # Count successful vs failed results
        successful = sum(1 for r in results if not isinstance(r, str) or not r.startswith("Error"))
        failed = len(results) - successful
        
        print(f"\nConcurrent Processing Test:")
        print(f"  Total images: {len(test_images)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Success rate: {successful/len(results)*100:.1f}%")
        print(f"  Peak memory: {metrics['peak_memory_mb']:.1f}MB")
        print(f"  Average CPU: {metrics['avg_cpu_percent']:.1f}%")
        
        # Should have high success rate
        success_rate = successful / len(results)
        self.assertGreater(success_rate, 0.9, "Should have >90% success rate in concurrent processing")
    
    def test_large_dataset_processing(self):
        """Test processing large dataset (simulated)"""
        processor = BatchProcessor(self.config)
        
        # Use all available test images multiple times to simulate large dataset
        base_images = list(Path(self.test_images_dir).glob('*.jpg'))
        large_dataset = base_images * 5  # Simulate 1500+ images
        
        self.monitor.start_monitoring()
        
        start_time = time.time()
        processed_count = 0
        batch_size = 50
        
        try:
            for i in range(0, min(len(large_dataset), 1000), batch_size):  # Process up to 1000 images
                batch = large_dataset[i:i + batch_size]
                results = processor.process_batch([str(p) for p in batch])
                processed_count += len(results)
                
                # Simulate progress tracking
                if i % 200 == 0:
                    print(f"  Processed {processed_count} images...")
                    gc.collect()  # Force cleanup periodically
        
        except Exception as e:
            self.fail(f"Large dataset processing failed: {str(e)}")
        
        metrics = self.monitor.stop_monitoring()
        total_time = time.time() - start_time
        
        print(f"\nLarge Dataset Processing Test:")
        print(f"  Images processed: {processed_count}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Images per second: {processed_count/total_time:.2f}")
        print(f"  Peak memory: {metrics['peak_memory_mb']:.1f}MB")
        print(f"  Memory increase: {metrics['memory_increase_mb']:.1f}MB")
        
        # Should process reasonable number of images
        self.assertGreater(processed_count, 500, "Should process at least 500 images")
        self.assertLess(metrics['peak_memory_mb'], 4000, "Peak memory should stay under 4GB")
    
    def test_error_recovery_under_load(self):
        """Test error recovery mechanisms under load"""
        from backend.core.error_handler import ErrorHandler
        
        error_handler = ErrorHandler()
        analyzer = QualityAnalyzer(self.config)
        
        # Create mix of valid and invalid image paths
        valid_images = list(Path(self.test_images_dir).glob('*.jpg'))[:20]
        invalid_images = ['/nonexistent/path.jpg'] * 5
        mixed_images = list(valid_images) + invalid_images
        
        self.monitor.start_monitoring()
        
        successful = 0
        errors_handled = 0
        
        for image_path in mixed_images:
            try:
                result = analyzer.analyze(str(image_path))
                successful += 1
            except Exception as e:
                if error_handler.handle_processing_error(e, {'image_path': str(image_path)}):
                    errors_handled += 1
        
        metrics = self.monitor.stop_monitoring()
        
        print(f"\nError Recovery Test:")
        print(f"  Total images: {len(mixed_images)}")
        print(f"  Successful: {successful}")
        print(f"  Errors handled: {errors_handled}")
        print(f"  Peak memory: {metrics['peak_memory_mb']:.1f}MB")
        
        # Should handle errors gracefully
        self.assertEqual(successful, len(valid_images), "Should process all valid images")
        self.assertEqual(errors_handled, len(invalid_images), "Should handle all errors")


class TestBenchmarks(PerformanceTestCase):
    """Benchmark tests for performance comparison"""
    
    def test_processing_pipeline_benchmark(self):
        """Benchmark complete processing pipeline"""
        # Create temporary output directory
        output_dir = os.path.join(self.test_dir, 'benchmark_output')
        os.makedirs(output_dir)
        
        # Use subset of test images
        benchmark_images = list(Path(self.test_images_dir).glob('*.jpg'))[:100]
        
        self.monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            # Initialize analyzers with test config
            quality_analyzer = QualityAnalyzer(self.config)
            defect_detector = DefectDetector(self.config)
            compliance_checker = ComplianceChecker(self.config)
            
            # Process images (simulate main processing loop)
            results = []
            for image_path in benchmark_images:
                # Simulate full pipeline processing
                quality_result = quality_analyzer.analyze(str(image_path))
                defect_result = defect_detector.detect_defects(str(image_path))
                compliance_result = compliance_checker.check_compliance(str(image_path), {})
                
                # Simulate decision making
                final_decision = 'approved' if quality_result.passed else 'rejected'
                
                results.append({
                    'image_path': str(image_path),
                    'quality_passed': quality_result.passed,
                    'defect_passed': defect_result.passed,
                    'compliance_passed': compliance_result.overall_compliance,
                    'final_decision': final_decision
                })
        
        except Exception as e:
            self.fail(f"Pipeline benchmark failed: {str(e)}")
        
        metrics = self.monitor.stop_monitoring()
        total_time = time.time() - start_time
        
        # Calculate benchmark metrics
        images_per_second = len(benchmark_images) / total_time
        approved_count = sum(1 for r in results if r['final_decision'] == 'approved')
        approval_rate = approved_count / len(results)
        
        print(f"\nProcessing Pipeline Benchmark:")
        print(f"  Images processed: {len(benchmark_images)}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Images per second: {images_per_second:.2f}")
        print(f"  Approved images: {approved_count}")
        print(f"  Approval rate: {approval_rate:.1%}")
        print(f"  Peak memory: {metrics['peak_memory_mb']:.1f}MB")
        print(f"  Average CPU: {metrics['avg_cpu_percent']:.1f}%")
        
        # Benchmark assertions
        self.assertGreater(images_per_second, 0.1, "Should process at least 0.1 images per second")
        self.assertLess(metrics['peak_memory_mb'], 8000, "Peak memory should stay under 8GB")
        
        # Save benchmark results
        benchmark_file = os.path.join(self.test_dir, 'benchmark_results.txt')
        with open(benchmark_file, 'w') as f:
            f.write(f"Adobe Stock Image Processor - Performance Benchmark\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Images processed: {len(benchmark_images)}\n")
            f.write(f"Total time: {total_time:.2f}s\n")
            f.write(f"Images per second: {images_per_second:.2f}\n")
            f.write(f"Peak memory: {metrics['peak_memory_mb']:.1f}MB\n")
            f.write(f"Average CPU: {metrics['avg_cpu_percent']:.1f}%\n")
            f.write(f"Approval rate: {approval_rate:.1%}\n")
    
    def test_large_dataset_benchmark(self):
        """Benchmark processing with large dataset (1000+ images)"""
        print("\nRunning large dataset benchmark (1000+ images)...")
        
        # Use all available test images multiple times to reach 1000+
        base_images = list(Path(self.test_images_dir).glob('*.jpg'))
        multiplier = max(1, 1000 // len(base_images) + 1)
        large_dataset = base_images * multiplier
        large_dataset = large_dataset[:1000]  # Exactly 1000 images
        
        processor = BatchProcessor(self.config)
        
        self.monitor.start_monitoring()
        
        start_time = time.time()
        processed_count = 0
        batch_size = 100
        
        try:
            for i in range(0, len(large_dataset), batch_size):
                batch = large_dataset[i:i + batch_size]
                results = processor.process_batch([str(p) for p in batch])
                processed_count += len(results)
                
                # Progress update
                if i % 200 == 0:
                    elapsed = time.time() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    print(f"    Processed {processed_count}/{len(large_dataset)} images ({rate:.1f} img/s)")
                
                # Memory cleanup
                if i % 500 == 0:
                    processor.cleanup_memory()
                    gc.collect()
        
        except Exception as e:
            self.fail(f"Large dataset benchmark failed: {str(e)}")
        
        metrics = self.monitor.stop_monitoring()
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        images_per_second = processed_count / total_time
        memory_per_image = metrics['peak_memory_mb'] / processed_count
        
        print(f"\nLarge Dataset Benchmark Results:")
        print(f"  Images processed: {processed_count}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Images per second: {images_per_second:.2f}")
        print(f"  Peak memory: {metrics['peak_memory_mb']:.1f}MB")
        print(f"  Memory per image: {memory_per_image:.2f}MB")
        print(f"  Average CPU: {metrics['avg_cpu_percent']:.1f}%")
        
        # Performance assertions for large dataset
        self.assertGreaterEqual(processed_count, 1000, "Should process at least 1000 images")
        self.assertGreater(images_per_second, 0.5, "Should maintain at least 0.5 images/second")
        self.assertLess(metrics['peak_memory_mb'], 12000, "Peak memory should stay under 12GB")
        self.assertLess(memory_per_image, 10, "Should use less than 10MB per image on average")
    
    def test_processing_speed_per_component(self):
        """Benchmark processing speed for each component individually"""
        print("\nBenchmarking individual component speeds...")
        
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))[:50]
        component_benchmarks = {}
        
        # Quality Analyzer benchmark
        analyzer = QualityAnalyzer(self.config)
        start_time = time.time()
        for image_path in test_images:
            analyzer.analyze(str(image_path))
        quality_time = time.time() - start_time
        quality_speed = len(test_images) / quality_time
        
        component_benchmarks['QualityAnalyzer'] = {
            'total_time': quality_time,
            'images_per_second': quality_speed,
            'avg_time_per_image': quality_time / len(test_images)
        }
        
        # Defect Detector benchmark (smaller set due to ML model)
        detector = DefectDetector(self.config)
        test_images_small = test_images[:20]
        start_time = time.time()
        for image_path in test_images_small:
            detector.detect_defects(str(image_path))
        defect_time = time.time() - start_time
        defect_speed = len(test_images_small) / defect_time
        
        component_benchmarks['DefectDetector'] = {
            'total_time': defect_time,
            'images_per_second': defect_speed,
            'avg_time_per_image': defect_time / len(test_images_small)
        }
        
        # Similarity Finder benchmark
        finder = SimilarityFinder(self.config)
        start_time = time.time()
        finder.find_similar_groups([str(p) for p in test_images])
        similarity_time = time.time() - start_time
        similarity_speed = len(test_images) / similarity_time
        
        component_benchmarks['SimilarityFinder'] = {
            'total_time': similarity_time,
            'images_per_second': similarity_speed,
            'avg_time_per_image': similarity_time / len(test_images)
        }
        
        # Compliance Checker benchmark
        checker = ComplianceChecker(self.config)
        start_time = time.time()
        for image_path in test_images:
            checker.check_compliance(str(image_path), {})
        compliance_time = time.time() - start_time
        compliance_speed = len(test_images) / compliance_time
        
        component_benchmarks['ComplianceChecker'] = {
            'total_time': compliance_time,
            'images_per_second': compliance_speed,
            'avg_time_per_image': compliance_time / len(test_images)
        }
        
        # Print benchmark results
        print(f"\nComponent Speed Benchmarks:")
        print(f"{'Component':<20} {'Images/sec':<12} {'Avg Time/Image':<15}")
        print("-" * 50)
        
        for component, metrics in component_benchmarks.items():
            print(f"{component:<20} {metrics['images_per_second']:<12.2f} "
                  f"{metrics['avg_time_per_image']:<15.3f}s")
        
        # Performance assertions
        self.assertGreater(component_benchmarks['QualityAnalyzer']['images_per_second'], 1.0,
                          "Quality analyzer should process >1 image/second")
        self.assertGreater(component_benchmarks['DefectDetector']['images_per_second'], 0.1,
                          "Defect detector should process >0.1 image/second")
        self.assertGreater(component_benchmarks['SimilarityFinder']['images_per_second'], 0.5,
                          "Similarity finder should process >0.5 image/second")
        self.assertGreater(component_benchmarks['ComplianceChecker']['images_per_second'], 2.0,
                          "Compliance checker should process >2 images/second")
        
        return component_benchmarks


if __name__ == '__main__':
    # Run performance tests with verbose output
    unittest.main(verbosity=2, buffer=False)