#!/usr/bin/env python3
"""
Stress tests for Adobe Stock Image Processor
Tests system stability under extreme conditions and heavy loads
"""

import unittest
import os
import sys
import time
import threading
import tempfile
import shutil
import psutil
import gc
from pathlib import Path
from typing import List, Dict, Any
import concurrent.futures
import random

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.batch_processor import BatchProcessor
from backend.analyzers.quality_analyzer import QualityAnalyzer
from backend.analyzers.defect_detector import DefectDetector
from backend.analyzers.similarity_finder import SimilarityFinder
from backend.analyzers.compliance_checker import ComplianceChecker
from backend.core.progress_tracker import ProgressTracker
from backend.core.database import DatabaseManager
from backend.utils.file_manager import FileManager
from backend.config.config_loader import ConfigLoader
from backend.core.error_handler import ErrorHandler


class StressTestMonitor:
    """Monitor system resources during stress tests"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = {
            'cpu_samples': [],
            'memory_samples': [],
            'disk_io_samples': [],
            'errors': [],
            'start_time': None,
            'end_time': None
        }
    
    def start_monitoring(self):
        """Start system monitoring"""
        self.metrics = {
            'cpu_samples': [],
            'memory_samples': [],
            'disk_io_samples': [],
            'errors': [],
            'start_time': time.time(),
            'end_time': None
        }
        self.monitoring = True
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics"""
        self.monitoring = False
        self.metrics['end_time'] = time.time()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # Calculate summary statistics
        duration = self.metrics['end_time'] - self.metrics['start_time']
        
        cpu_avg = sum(self.metrics['cpu_samples']) / len(self.metrics['cpu_samples']) if self.metrics['cpu_samples'] else 0
        cpu_max = max(self.metrics['cpu_samples']) if self.metrics['cpu_samples'] else 0
        
        memory_avg = sum(self.metrics['memory_samples']) / len(self.metrics['memory_samples']) if self.metrics['memory_samples'] else 0
        memory_max = max(self.metrics['memory_samples']) if self.metrics['memory_samples'] else 0
        
        return {
            'duration': duration,
            'cpu_avg': cpu_avg,
            'cpu_max': cpu_max,
            'memory_avg_mb': memory_avg,
            'memory_max_mb': memory_max,
            'error_count': len(self.metrics['errors']),
            'sample_count': len(self.metrics['cpu_samples'])
        }
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = process.cpu_percent()
                self.metrics['cpu_samples'].append(cpu_percent)
                
                # Memory usage
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.metrics['memory_samples'].append(memory_mb)
                
                # Disk I/O (if available)
                try:
                    io_counters = process.io_counters()
                    self.metrics['disk_io_samples'].append({
                        'read_bytes': io_counters.read_bytes,
                        'write_bytes': io_counters.write_bytes
                    })
                except:
                    pass
                
                time.sleep(0.5)  # Sample every 500ms
                
            except Exception as e:
                self.metrics['errors'].append(str(e))
                time.sleep(1.0)


class StressTestCase(unittest.TestCase):
    """Base class for stress tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up stress test environment"""
        cls.test_dir = tempfile.mkdtemp(prefix='stress_test_')
        config_loader = ConfigLoader()
        cls.config = config_loader.load_config()
        cls.monitor = StressTestMonitor()
        
        # Create large test dataset
        cls.test_images_dir = os.path.join(cls.test_dir, 'stress_images')
        os.makedirs(cls.test_images_dir)
        cls._create_stress_test_images()
        
        print(f"Stress test environment created at: {cls.test_dir}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up stress test environment"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _create_stress_test_images(cls):
        """Create large dataset for stress testing"""
        from PIL import Image
        import numpy as np
        
        print("Creating stress test dataset...")
        
        # Create various image types and sizes
        configurations = [
            (1920, 1080, 'RGB'),   # HD
            (2560, 1440, 'RGB'),   # QHD
            (3840, 2160, 'RGB'),   # 4K
            (1920, 1080, 'RGBA'),  # HD with alpha
            (800, 600, 'RGB'),     # Small images
        ]
        
        images_per_config = 100  # 500 total images
        total_created = 0
        
        for width, height, mode in configurations:
            for i in range(images_per_config):
                # Create random image data
                if mode == 'RGB':
                    img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                else:  # RGBA
                    img_array = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)
                
                img = Image.fromarray(img_array, mode)
                
                # Add some variation to make images more realistic
                if random.random() < 0.3:  # 30% chance of adding noise
                    noise = np.random.randint(-20, 21, img_array.shape, dtype=np.int16)
                    img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img_array, mode)
                
                filename = f'stress_{total_created:04d}_{width}x{height}_{mode}.jpg'
                img.save(os.path.join(cls.test_images_dir, filename), 'JPEG', quality=85)
                total_created += 1
                
                if total_created % 50 == 0:
                    print(f"  Created {total_created} images...")
        
        print(f"Created {total_created} stress test images")
    
    def setUp(self):
        """Set up each stress test"""
        gc.collect()  # Clean memory before each test
        self.monitor = StressTestMonitor()
    
    def tearDown(self):
        """Clean up after each stress test"""
        gc.collect()  # Clean memory after each test


class TestHighVolumeProcessing(StressTestCase):
    """Test processing under high volume conditions"""
    
    def test_continuous_processing_stress(self):
        """Test continuous processing for extended period"""
        print("\nRunning continuous processing stress test...")
        
        analyzer = QualityAnalyzer(self.config)
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))
        
        self.monitor.start_monitoring()
        
        start_time = time.time()
        target_duration = 300  # 5 minutes of continuous processing
        processed_count = 0
        error_count = 0
        
        try:
            while time.time() - start_time < target_duration:
                # Process random image
                image_path = random.choice(test_images)
                
                try:
                    result = analyzer.analyze(str(image_path))
                    processed_count += 1
                    
                    # Periodic cleanup
                    if processed_count % 100 == 0:
                        gc.collect()
                        print(f"  Processed {processed_count} images...")
                
                except Exception as e:
                    error_count += 1
                    if error_count > 10:  # Too many errors
                        break
        
        except KeyboardInterrupt:
            print("  Test interrupted by user")
        
        metrics = self.monitor.stop_monitoring()
        
        # Calculate performance metrics
        actual_duration = time.time() - start_time
        images_per_second = processed_count / actual_duration
        error_rate = error_count / processed_count if processed_count > 0 else 1.0
        
        print(f"\nContinuous Processing Results:")
        print(f"  Duration: {actual_duration:.1f}s")
        print(f"  Images processed: {processed_count}")
        print(f"  Images per second: {images_per_second:.2f}")
        print(f"  Errors: {error_count}")
        print(f"  Error rate: {error_rate:.2%}")
        print(f"  Peak CPU: {metrics['cpu_max']:.1f}%")
        print(f"  Peak memory: {metrics['memory_max_mb']:.1f}MB")
        
        # Stress test assertions
        self.assertGreater(processed_count, 100, "Should process at least 100 images")
        self.assertLess(error_rate, 0.05, "Error rate should be under 5%")
        self.assertLess(metrics['memory_max_mb'], 8000, "Peak memory should stay under 8GB")
    
    def test_batch_processing_stress(self):
        """Test batch processing with large batches"""
        print("\nRunning batch processing stress test...")
        
        processor = BatchProcessor(self.config)
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))
        
        # Test with increasingly large batch sizes
        batch_sizes = [50, 100, 200, 300]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            processor.config['processing']['batch_size'] = batch_size
            
            self.monitor.start_monitoring()
            
            try:
                # Process multiple batches
                batch_count = 0
                total_processed = 0
                
                for i in range(0, min(len(test_images), 1000), batch_size):
                    batch = test_images[i:i + batch_size]
                    batch_results = processor.process_batch([str(p) for p in batch])
                    
                    total_processed += len(batch_results)
                    batch_count += 1
                    
                    # Force cleanup between batches
                    processor.cleanup_memory()
                    
                    if batch_count >= 3:  # Limit to 3 batches per size
                        break
            
            except Exception as e:
                print(f"    Error with batch size {batch_size}: {e}")
                total_processed = 0
            
            metrics = self.monitor.stop_monitoring()
            
            results[batch_size] = {
                'processed': total_processed,
                'peak_memory': metrics['memory_max_mb'],
                'avg_cpu': metrics['cpu_avg'],
                'duration': metrics['duration']
            }
            
            print(f"    Processed: {total_processed}, Peak memory: {metrics['memory_max_mb']:.1f}MB")
        
        # Analyze batch size efficiency
        print(f"\nBatch Size Analysis:")
        for batch_size, result in results.items():
            efficiency = result['processed'] / result['duration'] if result['duration'] > 0 else 0
            print(f"  Size {batch_size}: {efficiency:.1f} images/sec, {result['peak_memory']:.1f}MB peak")
        
        # Find optimal batch size (best efficiency with reasonable memory)
        optimal_size = max(results.keys(), 
                          key=lambda k: results[k]['processed'] / results[k]['duration'] 
                          if results[k]['duration'] > 0 and results[k]['peak_memory'] < 4000 else 0)
        
        print(f"  Optimal batch size: {optimal_size}")
        
        # Assertions
        self.assertGreater(len(results), 0, "Should complete at least one batch size test")
        self.assertTrue(any(r['processed'] > 0 for r in results.values()), "Should process some images")


class TestConcurrencyStress(StressTestCase):
    """Test system under concurrent load"""
    
    def test_multi_threaded_stress(self):
        """Test multi-threaded processing stress"""
        print("\nRunning multi-threaded stress test...")
        
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))[:100]  # Limit for stress test
        
        def worker_function(worker_id: int, image_batch: List[str]) -> Dict[str, Any]:
            """Worker function for concurrent processing"""
            analyzer = QualityAnalyzer(self.config)
            
            results = {
                'worker_id': worker_id,
                'processed': 0,
                'errors': 0,
                'start_time': time.time()
            }
            
            try:
                for image_path in image_batch:
                    try:
                        result = analyzer.analyze(image_path)
                        results['processed'] += 1
                    except Exception as e:
                        results['errors'] += 1
                        if results['errors'] > 5:  # Too many errors
                            break
            
            except Exception as e:
                results['errors'] += 1
            
            results['end_time'] = time.time()
            results['duration'] = results['end_time'] - results['start_time']
            
            return results
        
        # Test with different thread counts
        thread_counts = [2, 4, 8]
        stress_results = {}
        
        for thread_count in thread_counts:
            print(f"  Testing with {thread_count} threads...")
            
            self.monitor.start_monitoring()
            
            # Divide images among threads
            images_per_thread = len(test_images) // thread_count
            worker_batches = []
            
            for i in range(thread_count):
                start_idx = i * images_per_thread
                end_idx = start_idx + images_per_thread if i < thread_count - 1 else len(test_images)
                worker_batches.append([str(p) for p in test_images[start_idx:end_idx]])
            
            # Run concurrent workers
            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [
                    executor.submit(worker_function, i, batch) 
                    for i, batch in enumerate(worker_batches)
                ]
                
                worker_results = []
                for future in concurrent.futures.as_completed(futures, timeout=300):
                    try:
                        result = future.result()
                        worker_results.append(result)
                    except Exception as e:
                        print(f"    Worker failed: {e}")
            
            metrics = self.monitor.stop_monitoring()
            
            # Aggregate results
            total_processed = sum(r['processed'] for r in worker_results)
            total_errors = sum(r['errors'] for r in worker_results)
            avg_duration = sum(r['duration'] for r in worker_results) / len(worker_results)
            
            stress_results[thread_count] = {
                'total_processed': total_processed,
                'total_errors': total_errors,
                'avg_duration': avg_duration,
                'peak_memory': metrics['memory_max_mb'],
                'peak_cpu': metrics['cpu_max'],
                'throughput': total_processed / avg_duration if avg_duration > 0 else 0
            }
            
            print(f"    Processed: {total_processed}, Errors: {total_errors}")
            print(f"    Peak memory: {metrics['memory_max_mb']:.1f}MB, Peak CPU: {metrics['cpu_max']:.1f}%")
        
        # Analyze concurrency results
        print(f"\nConcurrency Analysis:")
        for thread_count, result in stress_results.items():
            print(f"  {thread_count} threads: {result['throughput']:.1f} images/sec, "
                  f"{result['peak_memory']:.1f}MB peak, {result['total_errors']} errors")
        
        # Find optimal thread count
        optimal_threads = max(stress_results.keys(), 
                            key=lambda k: stress_results[k]['throughput'])
        print(f"  Optimal thread count: {optimal_threads}")
        
        # Assertions
        self.assertTrue(any(r['total_processed'] > 0 for r in stress_results.values()), 
                       "Should process images in at least one configuration")
        self.assertTrue(all(r['total_errors'] < r['total_processed'] * 0.1 
                           for r in stress_results.values() if r['total_processed'] > 0),
                       "Error rate should be under 10% for all configurations")


class TestResourceExhaustionStress(StressTestCase):
    """Test system behavior under resource exhaustion"""
    
    def test_memory_pressure_stress(self):
        """Test behavior under memory pressure"""
        print("\nRunning memory pressure stress test...")
        
        analyzer = QualityAnalyzer(self.config)
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))
        
        self.monitor.start_monitoring()
        
        # Gradually increase memory pressure
        memory_hogs = []  # Keep references to prevent garbage collection
        processed_count = 0
        max_memory_mb = 0
        
        try:
            for i in range(100):  # Process 100 images while increasing memory pressure
                # Add memory pressure (simulate memory leak)
                if i % 10 == 0:
                    # Allocate 100MB of memory
                    memory_hog = bytearray(100 * 1024 * 1024)  # 100MB
                    memory_hogs.append(memory_hog)
                
                # Process image
                image_path = test_images[i % len(test_images)]
                
                try:
                    result = analyzer.analyze(str(image_path))
                    processed_count += 1
                    
                    # Monitor memory usage
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    max_memory_mb = max(max_memory_mb, current_memory)
                    
                    # Break if memory gets too high (prevent system crash)
                    if current_memory > 6000:  # 6GB limit
                        print(f"    Breaking at {current_memory:.1f}MB to prevent system crash")
                        break
                
                except MemoryError:
                    print(f"    MemoryError at image {i}")
                    break
                except Exception as e:
                    print(f"    Error at image {i}: {e}")
                    if "memory" in str(e).lower():
                        break
        
        except Exception as e:
            print(f"  Memory pressure test error: {e}")
        
        finally:
            # Clean up memory hogs
            memory_hogs.clear()
            gc.collect()
        
        metrics = self.monitor.stop_monitoring()
        
        print(f"\nMemory Pressure Results:")
        print(f"  Images processed: {processed_count}")
        print(f"  Max memory reached: {max_memory_mb:.1f}MB")
        print(f"  Peak system memory: {metrics['memory_max_mb']:.1f}MB")
        
        # Should handle some memory pressure gracefully
        self.assertGreater(processed_count, 10, "Should process at least 10 images under memory pressure")
    
    def test_disk_space_stress(self):
        """Test behavior when disk space is limited"""
        print("\nRunning disk space stress test...")
        
        # Create temporary directory with limited space simulation
        temp_output = os.path.join(self.test_dir, 'limited_output')
        os.makedirs(temp_output)
        
        file_manager = FileManager(self.config)
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))[:50]  # Limit for stress test
        
        self.monitor.start_monitoring()
        
        copied_count = 0
        disk_full_errors = 0
        
        try:
            for i, image_path in enumerate(test_images):
                try:
                    # Simulate disk space check
                    disk_usage = shutil.disk_usage(temp_output)
                    free_space_mb = disk_usage.free / 1024 / 1024
                    
                    # Simulate disk full condition after copying some files
                    if copied_count > 20 and free_space_mb < 1000:  # Less than 1GB
                        print(f"    Simulating disk full after {copied_count} files")
                        break
                    
                    # Copy file
                    dest_path = os.path.join(temp_output, f'copy_{i}_{os.path.basename(image_path)}')
                    shutil.copy2(str(image_path), dest_path)
                    copied_count += 1
                    
                    if copied_count % 10 == 0:
                        print(f"    Copied {copied_count} files...")
                
                except OSError as e:
                    if "No space left" in str(e) or "disk full" in str(e).lower():
                        disk_full_errors += 1
                        print(f"    Disk full error: {e}")
                        break
                    else:
                        raise
        
        except Exception as e:
            print(f"  Disk space test error: {e}")
        
        metrics = self.monitor.stop_monitoring()
        
        print(f"\nDisk Space Stress Results:")
        print(f"  Files copied: {copied_count}")
        print(f"  Disk full errors: {disk_full_errors}")
        print(f"  Peak memory: {metrics['memory_max_mb']:.1f}MB")
        
        # Should handle disk space issues gracefully
        self.assertGreater(copied_count, 5, "Should copy at least 5 files before disk issues")
    
    def test_cpu_intensive_stress(self):
        """Test behavior under high CPU load"""
        print("\nRunning CPU intensive stress test...")
        
        # Create CPU-intensive background task
        def cpu_intensive_task():
            """CPU-intensive background task"""
            end_time = time.time() + 30  # Run for 30 seconds
            while time.time() < end_time:
                # Perform CPU-intensive calculations
                sum(i * i for i in range(10000))
        
        # Start background CPU load
        cpu_threads = []
        for _ in range(psutil.cpu_count()):
            thread = threading.Thread(target=cpu_intensive_task)
            thread.daemon = True
            thread.start()
            cpu_threads.append(thread)
        
        # Process images under CPU stress
        analyzer = QualityAnalyzer(self.config)
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))[:20]
        
        self.monitor.start_monitoring()
        
        processed_count = 0
        error_count = 0
        start_time = time.time()
        
        try:
            for image_path in test_images:
                try:
                    result = analyzer.analyze(str(image_path))
                    processed_count += 1
                except Exception as e:
                    error_count += 1
                    print(f"    Error processing {image_path}: {e}")
        
        except Exception as e:
            print(f"  CPU stress test error: {e}")
        
        processing_time = time.time() - start_time
        metrics = self.monitor.stop_monitoring()
        
        # Wait for CPU threads to finish
        for thread in cpu_threads:
            thread.join(timeout=1.0)
        
        print(f"\nCPU Stress Results:")
        print(f"  Images processed: {processed_count}")
        print(f"  Errors: {error_count}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Peak CPU: {metrics['cpu_max']:.1f}%")
        print(f"  Peak memory: {metrics['memory_max_mb']:.1f}MB")
        
        # Should handle CPU stress reasonably
        self.assertGreater(processed_count, 10, "Should process at least 10 images under CPU stress")
        self.assertLess(error_count / len(test_images), 0.5, "Error rate should be under 50%")
    
    def test_io_intensive_stress(self):
        """Test behavior under heavy I/O load"""
        print("\nRunning I/O intensive stress test...")
        
        # Create I/O intensive background task
        def io_intensive_task():
            """I/O-intensive background task"""
            temp_file = os.path.join(self.test_dir, f'io_stress_{threading.current_thread().ident}.tmp')
            end_time = time.time() + 20  # Run for 20 seconds
            
            try:
                while time.time() < end_time:
                    # Write and read large amounts of data
                    with open(temp_file, 'wb') as f:
                        f.write(os.urandom(1024 * 1024))  # 1MB random data
                    
                    with open(temp_file, 'rb') as f:
                        f.read()
                    
                    os.remove(temp_file)
            except:
                pass  # Ignore I/O errors in background task
        
        # Start background I/O load
        io_threads = []
        for _ in range(4):  # 4 I/O threads
            thread = threading.Thread(target=io_intensive_task)
            thread.daemon = True
            thread.start()
            io_threads.append(thread)
        
        # Process images under I/O stress
        batch_processor = BatchProcessor(self.config)
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))[:15]
        
        self.monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            results = batch_processor.process_batch([str(p) for p in test_images])
            processed_count = len(results)
        except Exception as e:
            print(f"  I/O stress test error: {e}")
            processed_count = 0
        
        processing_time = time.time() - start_time
        metrics = self.monitor.stop_monitoring()
        
        # Wait for I/O threads to finish
        for thread in io_threads:
            thread.join(timeout=1.0)
        
        print(f"\nI/O Stress Results:")
        print(f"  Images processed: {processed_count}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Peak memory: {metrics['memory_max_mb']:.1f}MB")
        
        # Should handle I/O stress reasonably
        self.assertGreater(processed_count, 5, "Should process at least 5 images under I/O stress")
    
    def test_combined_resource_stress(self):
        """Test behavior under combined resource stress"""
        print("\nRunning combined resource stress test...")
        
        # Start multiple stress factors simultaneously
        stress_threads = []
        
        # CPU stress
        def cpu_stress():
            end_time = time.time() + 15
            while time.time() < end_time:
                sum(i * i for i in range(5000))
        
        # Memory stress
        memory_hogs = []
        def memory_stress():
            for _ in range(5):
                memory_hogs.append(bytearray(50 * 1024 * 1024))  # 50MB each
                time.sleep(1)
        
        # I/O stress
        def io_stress():
            temp_file = os.path.join(self.test_dir, 'combined_stress.tmp')
            end_time = time.time() + 15
            try:
                while time.time() < end_time:
                    with open(temp_file, 'wb') as f:
                        f.write(os.urandom(512 * 1024))  # 512KB
                    time.sleep(0.1)
            except:
                pass
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        # Start stress threads
        for stress_func in [cpu_stress, memory_stress, io_stress]:
            thread = threading.Thread(target=stress_func)
            thread.daemon = True
            thread.start()
            stress_threads.append(thread)
        
        # Process images under combined stress
        analyzer = QualityAnalyzer(self.config)
        test_images = list(Path(self.test_images_dir).glob('*.jpg'))[:10]
        
        self.monitor.start_monitoring()
        
        processed_count = 0
        error_count = 0
        start_time = time.time()
        
        try:
            for image_path in test_images:
                try:
                    result = analyzer.analyze(str(image_path))
                    processed_count += 1
                except Exception as e:
                    error_count += 1
                    print(f"    Error under stress: {e}")
        
        except Exception as e:
            print(f"  Combined stress test error: {e}")
        
        finally:
            # Clean up memory stress
            memory_hogs.clear()
        
        processing_time = time.time() - start_time
        metrics = self.monitor.stop_monitoring()
        
        # Wait for stress threads to finish
        for thread in stress_threads:
            thread.join(timeout=2.0)
        
        print(f"\nCombined Stress Results:")
        print(f"  Images processed: {processed_count}")
        print(f"  Errors: {error_count}")
        print(f"  Processing time: {processing_time:.2f}s")
        print(f"  Peak CPU: {metrics['cpu_max']:.1f}%")
        print(f"  Peak memory: {metrics['memory_max_mb']:.1f}MB")
        
        # Should survive combined stress
        self.assertGreater(processed_count, 3, "Should process at least 3 images under combined stress")
        
        # Clean up
        gc.collect()


if __name__ == '__main__':
    # Run stress tests with extended timeout
    unittest.main(verbosity=2, buffer=False)