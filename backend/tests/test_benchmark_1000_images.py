#!/usr/bin/env python3
"""
Benchmark test for processing 1000+ images
Specifically tests the requirement for handling large datasets efficiently
"""

import unittest
import time
import psutil
import os
import tempfile
import shutil
import gc
from pathlib import Path
from typing import List, Dict, Any
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.batch_processor import BatchProcessor
from backend.utils.file_manager import FileManager
from backend.core.progress_tracker import ProgressTracker
from backend.core.database import DatabaseManager
from backend.config.config_loader import ConfigLoader


class Benchmark1000ImagesTest(unittest.TestCase):
    """Benchmark test for processing 1000+ images efficiently"""
    
    @classmethod
    def setUpClass(cls):
        """Set up benchmark test environment with 1000+ images"""
        cls.test_dir = tempfile.mkdtemp(prefix='benchmark_1000_')
        cls.input_dir = os.path.join(cls.test_dir, 'input')
        cls.output_dir = os.path.join(cls.test_dir, 'output')
        cls.db_path = os.path.join(cls.test_dir, 'benchmark.db')
        
        os.makedirs(cls.input_dir)
        os.makedirs(cls.output_dir)
        
        # Load configuration
        config_loader = ConfigLoader()
        cls.config = config_loader.load_config()
        cls.config['database'] = {'path': cls.db_path}
        
        # Create exactly 1000 test images
        cls._create_1000_test_images()
        
        print(f"Benchmark environment created at: {cls.test_dir}")
        print(f"Created {len(os.listdir(cls.input_dir))} test images")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up benchmark test environment"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _create_1000_test_images(cls):
        """Create exactly 1000 test images with varied characteristics"""
        from PIL import Image
        import numpy as np
        
        print("Creating 1000 test images for benchmark...")
        
        # Define image variations to create realistic dataset
        variations = [
            # (width, height, quality, count, pattern_type)
            (1920, 1080, 95, 200, 'high_detail'),      # High quality HD
            (2560, 1440, 85, 150, 'medium_detail'),    # QHD medium quality
            (3840, 2160, 90, 100, 'ultra_detail'),     # 4K high quality
            (1280, 720, 75, 200, 'standard_detail'),   # HD standard quality
            (800, 600, 60, 150, 'low_detail'),         # Small medium quality
            (1600, 900, 80, 100, 'wide_detail'),       # Wide format
            (1024, 768, 70, 100, 'classic_detail')     # Classic 4:3 format
        ]
        
        image_count = 0
        
        for width, height, quality, count, pattern_type in variations:
            for i in range(count):
                # Create varied image content based on pattern type
                if pattern_type == 'high_detail':
                    # Sharp, high-contrast images
                    img_array = np.zeros((height, width, 3), dtype=np.uint8)
                    # Create checkerboard pattern
                    for y in range(0, height, 40):
                        for x in range(0, width, 40):
                            if (x // 40 + y // 40) % 2 == 0:
                                img_array[y:y+40, x:x+40] = [255, 255, 255]
                            else:
                                img_array[y:y+40, x:x+40] = [0, 0, 0]
                    # Add some color variation
                    img_array[:, :, 0] = np.clip(img_array[:, :, 0] + np.random.randint(-50, 51, (height, width)), 0, 255)
                    img_array[:, :, 1] = np.clip(img_array[:, :, 1] + np.random.randint(-30, 31, (height, width)), 0, 255)
                    img_array[:, :, 2] = np.clip(img_array[:, :, 2] + np.random.randint(-30, 31, (height, width)), 0, 255)
                
                elif pattern_type == 'medium_detail':
                    # Medium complexity images
                    img_array = np.random.randint(50, 206, (height, width, 3), dtype=np.uint8)
                    # Add some structure
                    for y in range(0, height, 60):
                        img_array[y:y+5, :] = [255, 255, 255]
                    for x in range(0, width, 80):
                        img_array[:, x:x+5] = [0, 0, 0]
                
                elif pattern_type == 'ultra_detail':
                    # Very detailed, high-resolution content
                    img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                    # Add fine details
                    for y in range(0, height, 20):
                        for x in range(0, width, 20):
                            # Add small detailed squares
                            detail = np.random.randint(0, 256, (min(20, height-y), min(20, width-x), 3), dtype=np.uint8)
                            img_array[y:y+20, x:x+20] = detail
                
                elif pattern_type == 'low_detail':
                    # Simple, low-detail images
                    base_color = np.random.randint(0, 256, 3)
                    img_array = np.full((height, width, 3), base_color, dtype=np.uint8)
                    # Add some noise
                    noise = np.random.randint(-50, 51, (height, width, 3))
                    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                
                else:
                    # Standard random images
                    img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
                
                # Create PIL image and save
                img = Image.fromarray(img_array, 'RGB')
                filename = f'benchmark_{pattern_type}_{i:03d}_{image_count:04d}.jpg'
                img.save(os.path.join(cls.input_dir, filename), 'JPEG', quality=quality)
                image_count += 1
                
                # Progress indicator
                if image_count % 100 == 0:
                    print(f"  Created {image_count}/1000 images...")
        
        print(f"Successfully created {image_count} benchmark images")
        assert image_count == 1000, f"Expected 1000 images, created {image_count}"
    
    def setUp(self):
        """Set up each test"""
        gc.collect()  # Clean memory before test
    
    def tearDown(self):
        """Clean up after each test"""
        gc.collect()  # Clean memory after test
    
    def test_1000_images_processing_benchmark(self):
        """Benchmark processing 1000 images with full pipeline"""
        print("\n" + "="*70)
        print("BENCHMARK: Processing 1000 Images")
        print("="*70)
        
        # Initialize components
        db_manager = DatabaseManager(self.db_path)
        progress_tracker = ProgressTracker(db_manager)
        file_manager = FileManager(self.config)
        batch_processor = BatchProcessor(self.config)
        
        # Monitor system resources
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()
        
        # Track detailed metrics
        metrics = {
            'memory_samples': [],
            'processing_times': [],
            'batch_sizes': [],
            'errors': []
        }
        
        try:
            # Step 1: Scan input images
            print("Step 1: Scanning input images...")
            input_images = file_manager.scan_images(self.input_dir)
            scan_time = time.time() - start_time
            
            self.assertEqual(len(input_images), 1000, f"Expected 1000 images, found {len(input_images)}")
            print(f"  Scanned {len(input_images)} images in {scan_time:.2f}s")
            
            # Step 2: Create processing session
            print("Step 2: Creating processing session...")
            session_id = progress_tracker.create_session(
                self.input_dir, self.output_dir, len(input_images)
            )
            self.assertIsNotNone(session_id)
            print(f"  Created session: {session_id}")
            
            # Step 3: Process images in batches
            print("Step 3: Processing images in batches...")
            batch_size = self.config['processing']['batch_size']
            processed_results = []
            batch_count = 0
            
            processing_start = time.time()
            
            for i in range(0, len(input_images), batch_size):
                batch = input_images[i:i + batch_size]
                batch_count += 1
                batch_start = time.time()
                
                try:
                    # Process batch
                    batch_results = batch_processor.process_batch(batch)
                    processed_results.extend(batch_results)
                    
                    # Update progress
                    progress_tracker.update_progress(session_id, len(processed_results))
                    
                    # Record metrics
                    batch_time = time.time() - batch_start
                    current_memory = process.memory_info().rss / 1024 / 1024
                    
                    metrics['processing_times'].append(batch_time)
                    metrics['batch_sizes'].append(len(batch))
                    metrics['memory_samples'].append(current_memory)
                    
                    # Progress reporting
                    if batch_count % 5 == 0:
                        progress_pct = len(processed_results) / len(input_images) * 100
                        rate = len(processed_results) / (time.time() - processing_start)
                        print(f"    Batch {batch_count}: {len(processed_results)}/{len(input_images)} "
                              f"({progress_pct:.1f}%) - {rate:.1f} img/s - {current_memory:.1f}MB")
                    
                    # Save checkpoint every 200 images
                    if len(processed_results) % 200 == 0:
                        checkpoint_data = {
                            'processed_count': len(processed_results),
                            'timestamp': time.time()
                        }
                        progress_tracker.save_checkpoint(session_id, checkpoint_data)
                    
                    # Memory cleanup every 10 batches
                    if batch_count % 10 == 0:
                        batch_processor.cleanup_memory()
                        gc.collect()
                
                except Exception as e:
                    error_msg = f"Error in batch {batch_count}: {str(e)}"
                    metrics['errors'].append(error_msg)
                    print(f"    {error_msg}")
                    continue
            
            processing_time = time.time() - processing_start
            total_time = time.time() - start_time
            end_memory = process.memory_info().rss / 1024 / 1024
            
            # Step 4: Analyze results
            print("Step 4: Analyzing results...")
            approved_count = sum(1 for r in processed_results if r.final_decision == 'approved')
            rejected_count = len(processed_results) - approved_count
            approval_rate = approved_count / len(processed_results) * 100 if processed_results else 0
            
            # Step 5: Complete session
            progress_tracker.complete_session(session_id)
            
            # Calculate performance metrics
            images_per_second = len(processed_results) / processing_time if processing_time > 0 else 0
            memory_increase = end_memory - start_memory
            peak_memory = max(metrics['memory_samples']) if metrics['memory_samples'] else end_memory
            avg_batch_time = sum(metrics['processing_times']) / len(metrics['processing_times']) if metrics['processing_times'] else 0
            
            # Print comprehensive results
            print("\n" + "="*70)
            print("BENCHMARK RESULTS")
            print("="*70)
            print(f"Images processed: {len(processed_results)}/1000")
            print(f"Total time: {total_time:.2f}s")
            print(f"Processing time: {processing_time:.2f}s")
            print(f"Images per second: {images_per_second:.2f}")
            print(f"Approved: {approved_count} ({approval_rate:.1f}%)")
            print(f"Rejected: {rejected_count}")
            print(f"Errors: {len(metrics['errors'])}")
            print(f"Batches processed: {batch_count}")
            print(f"Average batch time: {avg_batch_time:.2f}s")
            print(f"Start memory: {start_memory:.1f}MB")
            print(f"End memory: {end_memory:.1f}MB")
            print(f"Peak memory: {peak_memory:.1f}MB")
            print(f"Memory increase: {memory_increase:.1f}MB")
            
            # Performance assertions
            self.assertGreaterEqual(len(processed_results), 1000, "Should process all 1000 images")
            self.assertGreater(images_per_second, 0.5, f"Processing too slow: {images_per_second:.2f} img/s")
            self.assertLess(peak_memory, 12000, f"Peak memory too high: {peak_memory:.1f}MB")
            self.assertLess(memory_increase, 2000, f"Memory increase too high: {memory_increase:.1f}MB")
            self.assertLess(len(metrics['errors']), 50, f"Too many errors: {len(metrics['errors'])}")
            
            # Save benchmark results
            benchmark_results = {
                'timestamp': time.time(),
                'images_processed': len(processed_results),
                'total_time': total_time,
                'processing_time': processing_time,
                'images_per_second': images_per_second,
                'approval_rate': approval_rate,
                'memory_start_mb': start_memory,
                'memory_end_mb': end_memory,
                'memory_peak_mb': peak_memory,
                'memory_increase_mb': memory_increase,
                'batch_count': batch_count,
                'avg_batch_time': avg_batch_time,
                'error_count': len(metrics['errors']),
                'system_info': {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / 1024**3,
                    'platform': sys.platform
                }
            }
            
            with open('benchmark_1000_images_results.json', 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            
            print(f"\nBenchmark results saved to: benchmark_1000_images_results.json")
            
            return benchmark_results
        
        except Exception as e:
            self.fail(f"Benchmark failed with error: {str(e)}")
    
    def test_1000_images_memory_efficiency(self):
        """Test memory efficiency when processing 1000 images"""
        print("\n" + "="*70)
        print("MEMORY EFFICIENCY TEST: 1000 Images")
        print("="*70)
        
        batch_processor = BatchProcessor(self.config)
        input_images = list(Path(self.input_dir).glob('*.jpg'))[:1000]
        
        # Monitor memory usage throughout processing
        process = psutil.Process()
        memory_samples = []
        batch_size = 100  # Smaller batches for memory efficiency
        
        start_memory = process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        processed_count = 0
        
        try:
            for i in range(0, len(input_images), batch_size):
                batch = input_images[i:i + batch_size]
                
                # Process batch
                batch_results = batch_processor.process_batch([str(p) for p in batch])
                processed_count += len(batch_results)
                
                # Record memory usage
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory)
                
                # Force cleanup after each batch
                batch_processor.cleanup_memory()
                gc.collect()
                
                # Memory check
                if current_memory > start_memory + 4000:  # 4GB increase limit
                    print(f"  WARNING: High memory usage at batch {i//batch_size + 1}: {current_memory:.1f}MB")
            
            end_memory = process.memory_info().rss / 1024 / 1024
            processing_time = time.time() - start_time
            
            # Calculate memory efficiency metrics
            peak_memory = max(memory_samples)
            avg_memory = sum(memory_samples) / len(memory_samples)
            memory_increase = end_memory - start_memory
            memory_per_image = peak_memory / processed_count if processed_count > 0 else 0
            
            print(f"\nMemory Efficiency Results:")
            print(f"Images processed: {processed_count}")
            print(f"Processing time: {processing_time:.2f}s")
            print(f"Start memory: {start_memory:.1f}MB")
            print(f"End memory: {end_memory:.1f}MB")
            print(f"Peak memory: {peak_memory:.1f}MB")
            print(f"Average memory: {avg_memory:.1f}MB")
            print(f"Memory increase: {memory_increase:.1f}MB")
            print(f"Memory per image: {memory_per_image:.2f}MB")
            
            # Memory efficiency assertions
            self.assertEqual(processed_count, 1000, "Should process all 1000 images")
            self.assertLess(memory_per_image, 10, f"Memory per image too high: {memory_per_image:.2f}MB")
            self.assertLess(memory_increase, 1000, f"Memory leak detected: {memory_increase:.1f}MB")
            self.assertLess(peak_memory, 8000, f"Peak memory too high: {peak_memory:.1f}MB")
            
            return {
                'processed_count': processed_count,
                'memory_per_image': memory_per_image,
                'memory_increase': memory_increase,
                'peak_memory': peak_memory
            }
        
        except Exception as e:
            self.fail(f"Memory efficiency test failed: {str(e)}")
    
    def test_1000_images_processing_speed_benchmark(self):
        """Benchmark processing speed for 1000 images with different configurations"""
        print("\n" + "="*70)
        print("PROCESSING SPEED BENCHMARK: 1000 Images")
        print("="*70)
        
        input_images = list(Path(self.input_dir).glob('*.jpg'))[:1000]
        
        # Test different batch sizes
        batch_sizes = [50, 100, 200]
        speed_results = {}
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            # Configure batch processor
            config = self.config.copy()
            config['processing']['batch_size'] = batch_size
            batch_processor = BatchProcessor(config)
            
            start_time = time.time()
            processed_count = 0
            
            try:
                for i in range(0, len(input_images), batch_size):
                    batch = input_images[i:i + batch_size]
                    batch_results = batch_processor.process_batch([str(p) for p in batch])
                    processed_count += len(batch_results)
                    
                    # Cleanup between batches
                    if i % (batch_size * 5) == 0:
                        batch_processor.cleanup_memory()
                        gc.collect()
                
                processing_time = time.time() - start_time
                images_per_second = processed_count / processing_time
                
                speed_results[batch_size] = {
                    'processed_count': processed_count,
                    'processing_time': processing_time,
                    'images_per_second': images_per_second
                }
                
                print(f"  Processed: {processed_count} images")
                print(f"  Time: {processing_time:.2f}s")
                print(f"  Speed: {images_per_second:.2f} images/sec")
                
                # Speed assertion
                self.assertGreater(images_per_second, 0.5, 
                                 f"Batch size {batch_size} too slow: {images_per_second:.2f} img/s")
            
            except Exception as e:
                print(f"  Error with batch size {batch_size}: {e}")
                speed_results[batch_size] = {'error': str(e)}
        
        # Find optimal batch size
        valid_results = {k: v for k, v in speed_results.items() if 'error' not in v}
        if valid_results:
            optimal_batch_size = max(valid_results.keys(), 
                                   key=lambda k: valid_results[k]['images_per_second'])
            optimal_speed = valid_results[optimal_batch_size]['images_per_second']
            
            print(f"\nOptimal Configuration:")
            print(f"  Batch size: {optimal_batch_size}")
            print(f"  Speed: {optimal_speed:.2f} images/sec")
            print(f"  Time for 1000 images: {1000/optimal_speed:.1f}s")
        
        return speed_results
    
    def test_1000_images_stress_stability(self):
        """Test system stability when processing 1000 images under stress"""
        print("\n" + "="*70)
        print("STRESS STABILITY TEST: 1000 Images")
        print("="*70)
        
        # Create stress conditions
        def cpu_stress_task():
            """Background CPU stress"""
            import threading
            end_time = time.time() + 60  # 1 minute of stress
            while time.time() < end_time:
                sum(i * i for i in range(1000))
        
        # Start background stress
        stress_thread = threading.Thread(target=cpu_stress_task)
        stress_thread.daemon = True
        stress_thread.start()
        
        # Process images under stress
        batch_processor = BatchProcessor(self.config)
        input_images = list(Path(self.input_dir).glob('*.jpg'))[:1000]
        
        start_time = time.time()
        processed_count = 0
        error_count = 0
        batch_size = 50  # Smaller batches for stability
        
        try:
            for i in range(0, len(input_images), batch_size):
                batch = input_images[i:i + batch_size]
                
                try:
                    batch_results = batch_processor.process_batch([str(p) for p in batch])
                    processed_count += len(batch_results)
                except Exception as e:
                    error_count += 1
                    print(f"  Batch error at {i}: {e}")
                    if error_count > 10:  # Too many errors
                        break
                
                # Progress update
                if i % (batch_size * 10) == 0:
                    progress = processed_count / len(input_images) * 100
                    print(f"  Progress: {processed_count}/1000 ({progress:.1f}%) - {error_count} errors")
        
        except Exception as e:
            print(f"  Critical error: {e}")
        
        processing_time = time.time() - start_time
        success_rate = processed_count / len(input_images) * 100
        
        print(f"\nStress Test Results:")
        print(f"Images processed: {processed_count}/1000")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Errors: {error_count}")
        print(f"Processing time: {processing_time:.2f}s")
        
        # Stability assertions
        self.assertGreater(success_rate, 80, f"Success rate too low under stress: {success_rate:.1f}%")
        self.assertLess(error_count, 50, f"Too many errors under stress: {error_count}")
        
        # Wait for stress thread to finish
        stress_thread.join(timeout=5)
        
        return {
            'processed_count': processed_count,
            'success_rate': success_rate,
            'error_count': error_count,
            'processing_time': processing_time
        }


if __name__ == '__main__':
    # Run 1000 images benchmark
    unittest.main(verbosity=2, buffer=False)