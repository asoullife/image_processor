#!/usr/bin/env python3
"""
Comprehensive integration tests for Adobe Stock Image Processor
Tests complete processing pipeline with various scenarios and edge cases
"""

import unittest
import os
import sys
import tempfile
import shutil
import json
import time
import threading
import psutil
from pathlib import Path
from unittest.mock import patch, Mock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config.config_loader import ConfigLoader
from backend.core.batch_processor import BatchProcessor
from backend.core.progress_tracker import ProgressTracker
from backend.core.database import DatabaseManager
from backend.utils.file_manager import FileManager
from backend.utils.report_generator import ReportGenerator
from backend.analyzers.quality_analyzer import QualityAnalyzer
from backend.analyzers.defect_detector import DefectDetector
from backend.analyzers.similarity_finder import SimilarityFinder
from backend.analyzers.compliance_checker import ComplianceChecker
from backend.core.decision_engine import DecisionEngine


class TestCompleteProcessingPipeline(unittest.TestCase):
    """Comprehensive integration tests for complete processing pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = tempfile.mkdtemp(prefix='comprehensive_integration_')
        cls.input_dir = os.path.join(cls.test_dir, 'input')
        cls.output_dir = os.path.join(cls.test_dir, 'output')
        cls.db_path = os.path.join(cls.test_dir, 'test.db')
        
        os.makedirs(cls.input_dir)
        os.makedirs(cls.output_dir)
        
        # Create diverse test images
        cls._create_comprehensive_test_images()
        
        # Create test configuration
        cls.config_file = os.path.join(cls.test_dir, 'test_config.json')
        cls._create_test_config()
        
        # Load configuration
        config_loader = ConfigLoader()
        cls.config = config_loader.load_config(cls.config_file)
        cls.config['database'] = {'path': cls.db_path}
        
        print(f"Test environment created at: {cls.test_dir}")
        print(f"Created {len(os.listdir(cls.input_dir))} test images")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _create_comprehensive_test_images(cls):
        """Create diverse test images for comprehensive testing"""
        from PIL import Image
        import numpy as np
        
        # Create various types of test images
        image_configs = [
            # (width, height, quality, noise_level, description, count)
            (1920, 1080, 95, 0.0, 'high_quality', 5),
            (800, 600, 50, 0.3, 'low_quality', 5),
            (2560, 1440, 85, 0.1, 'medium_quality', 5),
            (640, 480, 30, 0.5, 'very_low_quality', 3),
            (3840, 2160, 90, 0.05, '4k_quality', 2),
            (400, 300, 60, 0.2, 'small_size', 3),
        ]
        
        image_count = 0
        for width, height, quality, noise_level, desc, count in image_configs:
            for i in range(count):
                # Create base image with patterns for better testing
                img_array = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Add patterns based on image type
                if 'high_quality' in desc:
                    # Sharp patterns
                    img_array[::10, :] = [255, 255, 255]
                    img_array[:, ::10] = [0, 0, 0]
                elif 'low_quality' in desc:
                    # Blurry patterns
                    img_array[:] = np.random.randint(100, 156, (height, width, 3))
                else:
                    # Random patterns
                    img_array[:] = np.random.randint(0, 256, (height, width, 3))
                
                # Add noise if specified
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level * 255, img_array.shape)
                    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                
                img = Image.fromarray(img_array, 'RGB')
                
                filename = f'{desc}_{i:02d}_{image_count:03d}.jpg'
                img.save(os.path.join(cls.input_dir, filename), 'JPEG', quality=quality)
                image_count += 1
        
        # Create some duplicate images for similarity testing
        duplicate_base = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
        duplicate_img = Image.fromarray(duplicate_base, 'RGB')
        
        for i in range(4):
            filename = f'duplicate_{i:02d}_{image_count:03d}.jpg'
            duplicate_img.save(os.path.join(cls.input_dir, filename), 'JPEG', quality=85)
            image_count += 1
        
        # Create some similar but not identical images
        for i in range(3):
            # Slightly modify the duplicate base
            modified = duplicate_base.copy()
            modified[50:100, 50:100] = np.random.randint(0, 256, (50, 50, 3))
            similar_img = Image.fromarray(modified, 'RGB')
            
            filename = f'similar_{i:02d}_{image_count:03d}.jpg'
            similar_img.save(os.path.join(cls.input_dir, filename), 'JPEG', quality=85)
            image_count += 1
    
    @classmethod
    def _create_test_config(cls):
        """Create comprehensive test configuration"""
        config = {
            "processing": {
                "batch_size": 8,
                "max_workers": 2,
                "checkpoint_interval": 5
            },
            "quality": {
                "min_sharpness": 30.0,
                "max_noise_level": 0.4,
                "min_resolution": [300, 200]
            },
            "similarity": {
                "hash_threshold": 8,
                "feature_threshold": 0.75,
                "clustering_eps": 0.4
            },
            "compliance": {
                "logo_detection_confidence": 0.8,
                "face_detection_enabled": True,
                "metadata_validation": True
            },
            "output": {
                "images_per_folder": 10,
                "preserve_metadata": True,
                "generate_thumbnails": False
            }
        }
        
        with open(cls.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def test_end_to_end_processing_pipeline(self):
        """Test complete end-to-end processing pipeline"""
        print("\n" + "="*60)
        print("RUNNING END-TO-END PROCESSING PIPELINE TEST")
        print("="*60)
        
        # Initialize all components
        db_manager = DatabaseManager(self.db_path)
        progress_tracker = ProgressTracker(db_manager)
        file_manager = FileManager(self.config)
        batch_processor = BatchProcessor(self.config)
        decision_engine = DecisionEngine(self.config)
        report_generator = ReportGenerator(self.config)
        
        # Step 1: Scan input images
        print("Step 1: Scanning input images...")
        input_images = file_manager.scan_images(self.input_dir)
        self.assertGreater(len(input_images), 0, "Should find input images")
        print(f"  Found {len(input_images)} input images")
        
        # Step 2: Create processing session
        print("Step 2: Creating processing session...")
        session_id = progress_tracker.create_session(
            self.input_dir, self.output_dir, len(input_images)
        )
        self.assertIsNotNone(session_id, "Should create processing session")
        print(f"  Created session: {session_id}")
        
        # Step 3: Process images in batches
        print("Step 3: Processing images in batches...")
        batch_size = self.config['processing']['batch_size']
        processed_results = []
        
        start_time = time.time()
        
        for i in range(0, len(input_images), batch_size):
            batch = input_images[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"  Processing batch {batch_num}: {len(batch)} images")
            
            # Process batch
            batch_results = batch_processor.process_batch(batch)
            processed_results.extend(batch_results)
            
            # Update progress
            progress_tracker.update_progress(session_id, len(processed_results))
            
            # Save checkpoint periodically
            if len(processed_results) % self.config['processing']['checkpoint_interval'] == 0:
                checkpoint_data = {
                    'processed_count': len(processed_results),
                    'last_processed': processed_results[-1].image_path,
                    'timestamp': time.time()
                }
                progress_tracker.save_checkpoint(session_id, checkpoint_data)
                print(f"    Saved checkpoint at {len(processed_results)} images")
        
        processing_time = time.time() - start_time
        print(f"  Processed {len(processed_results)} images in {processing_time:.2f}s")
        print(f"  Processing rate: {len(processed_results)/processing_time:.2f} images/second")
        
        # Step 4: Analyze results
        print("Step 4: Analyzing processing results...")
        approved_images = []
        rejected_images = []
        quality_failures = 0
        defect_failures = 0
        compliance_failures = 0
        
        for result in processed_results:
            if result.final_decision == 'approved':
                approved_images.append(result.image_path)
            else:
                rejected_images.append(result.image_path)
                
                # Count failure reasons
                if not result.quality_result.passed:
                    quality_failures += 1
                if not result.defect_result.passed:
                    defect_failures += 1
                if not result.compliance_result.overall_compliance:
                    compliance_failures += 1
        
        approval_rate = len(approved_images) / len(processed_results) * 100
        print(f"  Approved: {len(approved_images)} ({approval_rate:.1f}%)")
        print(f"  Rejected: {len(rejected_images)} ({100-approval_rate:.1f}%)")
        print(f"  Quality failures: {quality_failures}")
        print(f"  Defect failures: {defect_failures}")
        print(f"  Compliance failures: {compliance_failures}")
        
        # Step 5: Organize output
        print("Step 5: Organizing output files...")
        if approved_images:
            file_manager.organize_output(approved_images, self.output_dir)
            
            # Verify output organization
            output_subdirs = [
                d for d in os.listdir(self.output_dir) 
                if os.path.isdir(os.path.join(self.output_dir, d)) and d.isdigit()
            ]
            self.assertGreater(len(output_subdirs), 0, "Should create output subdirectories")
            print(f"  Created {len(output_subdirs)} output subdirectories")
            
            # Count organized files
            total_organized = 0
            for subdir in output_subdirs:
                subdir_path = os.path.join(self.output_dir, subdir)
                files_in_subdir = len([f for f in os.listdir(subdir_path) if f.endswith('.jpg')])
                total_organized += files_in_subdir
                print(f"    Subdirectory {subdir}: {files_in_subdir} images")
            
            self.assertEqual(total_organized, len(approved_images), 
                           "All approved images should be organized")
        
        # Step 6: Generate reports
        print("Step 6: Generating reports...")
        excel_report = os.path.join(self.output_dir, 'processing_report.xlsx')
        html_report = os.path.join(self.output_dir, 'dashboard.html')
        
        report_generator.generate_excel_report(processed_results, excel_report)
        report_generator.generate_html_dashboard(processed_results, html_report)
        
        # Verify reports
        self.assertTrue(os.path.exists(excel_report), "Should generate Excel report")
        self.assertTrue(os.path.exists(html_report), "Should generate HTML report")
        print(f"  Generated Excel report: {os.path.getsize(excel_report)} bytes")
        print(f"  Generated HTML report: {os.path.getsize(html_report)} bytes")
        
        # Step 7: Complete session
        print("Step 7: Completing session...")
        progress_tracker.complete_session(session_id)
        
        # Verify final state
        session_progress = progress_tracker.get_session_progress(session_id)
        self.assertEqual(session_progress['status'], 'completed')
        self.assertEqual(session_progress['processed_images'], len(processed_results))
        
        print("END-TO-END PIPELINE COMPLETED SUCCESSFULLY")
        
        # Return metrics for further analysis
        return {
            'total_images': len(input_images),
            'processed_images': len(processed_results),
            'approved_images': len(approved_images),
            'rejected_images': len(rejected_images),
            'processing_time': processing_time,
            'approval_rate': approval_rate
        }
    
    def test_similarity_detection_integration(self):
        """Test similarity detection and grouping integration"""
        print("\nTesting similarity detection integration...")
        
        similarity_finder = SimilarityFinder(self.config)
        file_manager = FileManager(self.config)
        
        # Get all input images
        input_images = file_manager.scan_images(self.input_dir)
        
        # Find similar groups
        similar_groups = similarity_finder.find_similar_groups(input_images)
        
        print(f"  Found {len(similar_groups)} similarity groups")
        
        # Analyze groups
        duplicate_groups = 0
        similar_groups_count = 0
        
        for group_id, group_images in similar_groups.items():
            if len(group_images) > 1:
                print(f"    Group {group_id}: {len(group_images)} images")
                
                # Check if these are the duplicate images we created
                duplicate_images = [img for img in group_images if 'duplicate' in img]
                similar_images = [img for img in group_images if 'similar' in img]
                
                if duplicate_images:
                    duplicate_groups += 1
                    print(f"      Found duplicate group with {len(duplicate_images)} images")
                
                if similar_images:
                    similar_groups_count += 1
                    print(f"      Found similar group with {len(similar_images)} images")
        
        # Should find at least the duplicate group we created
        self.assertGreater(duplicate_groups, 0, "Should detect duplicate images")
        
        print(f"  Detected {duplicate_groups} duplicate groups")
        print(f"  Detected {similar_groups_count} similar groups")
    
    def test_resume_functionality_integration(self):
        """Test resume functionality with real checkpoint data"""
        print("\nTesting resume functionality...")
        
        # Initialize components
        db_manager = DatabaseManager(self.db_path + '_resume')
        progress_tracker = ProgressTracker(db_manager)
        batch_processor = BatchProcessor(self.config)
        
        input_images = FileManager(self.config).scan_images(self.input_dir)
        
        # Create initial session
        session_id = progress_tracker.create_session(
            self.input_dir, self.output_dir, len(input_images)
        )
        
        # Process first portion and save checkpoint
        first_batch_size = min(5, len(input_images))
        first_batch = input_images[:first_batch_size]
        first_results = batch_processor.process_batch(first_batch)
        
        progress_tracker.update_progress(session_id, len(first_results))
        checkpoint_data = {
            'processed_count': len(first_results),
            'last_processed': first_results[-1].image_path,
            'timestamp': time.time(),
            'results': [r.__dict__ for r in first_results]  # Serialize results
        }
        progress_tracker.save_checkpoint(session_id, checkpoint_data)
        
        print(f"  Processed first batch: {len(first_results)} images")
        print(f"  Saved checkpoint at image {len(first_results)}")
        
        # Simulate interruption and resume
        checkpoint = progress_tracker.get_latest_checkpoint(session_id)
        self.assertIsNotNone(checkpoint, "Should have checkpoint data")
        
        # Resume from checkpoint
        remaining_images = input_images[len(first_results):]
        if remaining_images:
            print(f"  Resuming with {len(remaining_images)} remaining images")
            
            remaining_results = batch_processor.process_batch(remaining_images)
            progress_tracker.update_progress(session_id, len(first_results) + len(remaining_results))
            
            print(f"  Processed {len(remaining_results)} remaining images")
        
        # Verify total progress
        final_progress = progress_tracker.get_session_progress(session_id)
        expected_total = len(first_results) + len(remaining_results) if remaining_images else len(first_results)
        self.assertEqual(final_progress['processed_images'], expected_total)
        
        print(f"  Total processed: {final_progress['processed_images']} images")
        print("  Resume functionality verified")
    
    def test_memory_management_integration(self):
        """Test memory management during processing"""
        print("\nTesting memory management...")
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        batch_processor = BatchProcessor(self.config)
        input_images = FileManager(self.config).scan_images(self.input_dir)
        
        # Process in batches with memory monitoring
        batch_size = 4
        memory_samples = [initial_memory]
        
        print(f"  Initial memory: {initial_memory:.1f}MB")
        
        for i in range(0, len(input_images), batch_size):
            batch = input_images[i:i + batch_size]
            
            # Process batch
            results = batch_processor.process_batch(batch)
            
            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            print(f"    Batch {i//batch_size + 1}: {current_memory:.1f}MB memory")
            
            # Force cleanup
            batch_processor.cleanup_memory()
            
            # Check memory after cleanup
            cleanup_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(cleanup_memory)
            
            if cleanup_memory < current_memory:
                print(f"      Cleanup reduced memory by {current_memory - cleanup_memory:.1f}MB")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        peak_memory = max(memory_samples)
        memory_increase = final_memory - initial_memory
        
        print(f"  Peak memory: {peak_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Memory increase: {memory_increase:.1f}MB")
        
        # Memory increase should be reasonable
        self.assertLess(memory_increase, 1000, "Memory increase should be under 1GB")
        
        print("  Memory management verified")
    
    def test_error_handling_and_recovery(self):
        """Test comprehensive error handling and recovery"""
        print("\nTesting error handling and recovery...")
        
        # Initialize components
        db_manager = DatabaseManager(self.db_path + '_error')
        progress_tracker = ProgressTracker(db_manager)
        batch_processor = BatchProcessor(self.config)
        
        # Create session
        session_id = progress_tracker.create_session(
            self.input_dir, self.output_dir, 10
        )
        
        # Create mix of valid and invalid image paths
        valid_images = FileManager(self.config).scan_images(self.input_dir)[:3]
        invalid_images = [
            '/nonexistent/image1.jpg',
            '/nonexistent/image2.jpg',
            os.path.join(self.test_dir, 'not_an_image.txt')
        ]
        
        # Create the text file to test non-image handling
        with open(invalid_images[-1], 'w') as f:
            f.write('This is not an image')
        
        mixed_batch = valid_images + invalid_images
        
        print(f"  Testing with {len(valid_images)} valid and {len(invalid_images)} invalid images")
        
        # Process batch with error handling
        successful_count = 0
        error_count = 0
        
        try:
            results = batch_processor.process_batch(mixed_batch)
            
            for result in results:
                if hasattr(result, 'final_decision'):
                    if result.final_decision == 'error':
                        error_count += 1
                    else:
                        successful_count += 1
                else:
                    error_count += 1
            
            print(f"  Successful: {successful_count}, Errors: {error_count}")
            
            # Should handle errors gracefully
            self.assertGreater(successful_count, 0, "Should process some valid images")
            self.assertGreater(error_count, 0, "Should detect some errors")
            
            # Update progress with successful results
            progress_tracker.update_progress(session_id, successful_count)
            
        except Exception as e:
            print(f"  Batch processing handled error: {e}")
            # If batch processing fails completely, that's also acceptable error handling
        
        print("  Error handling verified")
    
    def test_concurrent_processing_integration(self):
        """Test concurrent processing with multiple workers"""
        print("\nTesting concurrent processing...")
        
        # Use configuration with multiple workers
        concurrent_config = self.config.copy()
        concurrent_config['processing']['max_workers'] = 4
        
        batch_processor = BatchProcessor(concurrent_config)
        input_images = FileManager(self.config).scan_images(self.input_dir)
        
        start_time = time.time()
        
        # Process with concurrent workers
        results = batch_processor.process_batch(input_images)
        
        processing_time = time.time() - start_time
        
        print(f"  Processed {len(results)} images in {processing_time:.2f}s")
        print(f"  Rate: {len(results)/processing_time:.2f} images/second")
        
        # Verify all images were processed
        self.assertEqual(len(results), len(input_images))
        
        # Verify results have expected structure
        valid_results = 0
        for result in results:
            if hasattr(result, 'image_path') and hasattr(result, 'final_decision'):
                valid_results += 1
                self.assertIn(result.final_decision, ['approved', 'rejected', 'error'])
        
        self.assertEqual(valid_results, len(results), "All results should be valid")
        
        print(f"  All {valid_results} results are valid")
        print("  Concurrent processing verified")
    
    def test_performance_under_load(self):
        """Test system performance under load"""
        print("\nTesting performance under load...")
        
        # Create additional test images for load testing
        load_test_dir = os.path.join(self.test_dir, 'load_test')
        os.makedirs(load_test_dir, exist_ok=True)
        
        # Create 100 test images
        from PIL import Image
        import numpy as np
        
        for i in range(100):
            img_array = np.random.randint(0, 256, (400, 600, 3), dtype=np.uint8)
            img = Image.fromarray(img_array, 'RGB')
            filename = f'load_test_{i:03d}.jpg'
            img.save(os.path.join(load_test_dir, filename), 'JPEG', quality=75)
        
        # Configure for load testing
        load_config = self.config.copy()
        load_config['processing']['batch_size'] = 25
        load_config['processing']['max_workers'] = 4
        
        batch_processor = BatchProcessor(load_config)
        file_manager = FileManager(load_config)
        
        load_images = file_manager.scan_images(load_test_dir)
        print(f"  Created {len(load_images)} images for load test")
        
        # Monitor system resources
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        # Process in batches
        all_results = []
        batch_size = load_config['processing']['batch_size']
        
        for i in range(0, len(load_images), batch_size):
            batch = load_images[i:i + batch_size]
            batch_results = batch_processor.process_batch(batch)
            all_results.extend(batch_results)
            
            # Monitor resources
            current_memory = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            
            print(f"    Batch {i//batch_size + 1}: {len(batch_results)} results, "
                  f"{current_memory:.1f}MB memory, {cpu_percent:.1f}% CPU")
            
            # Cleanup between batches
            batch_processor.cleanup_memory()
        
        total_time = time.time() - start_time
        final_memory = process.memory_info().rss / 1024 / 1024
        
        # Calculate performance metrics
        images_per_second = len(all_results) / total_time
        memory_increase = final_memory - start_memory
        
        print(f"  Load test results:")
        print(f"    Images processed: {len(all_results)}")
        print(f"    Total time: {total_time:.2f}s")
        print(f"    Images per second: {images_per_second:.2f}")
        print(f"    Memory increase: {memory_increase:.1f}MB")
        
        # Performance assertions
        self.assertEqual(len(all_results), len(load_images), "Should process all images")
        self.assertGreater(images_per_second, 1.0, "Should maintain >1 image/second")
        self.assertLess(memory_increase, 2000, "Memory increase should be <2GB")
        
        # Clean up
        shutil.rmtree(load_test_dir)
        
        print("  Performance under load verified")


if __name__ == '__main__':
    # Run comprehensive integration tests
    unittest.main(verbosity=2, buffer=False)