#!/usr/bin/env python3
"""End-to-end test for main application."""

import os
import sys
import tempfile
import shutil
from unittest.mock import patch

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import ImageProcessor


def create_test_images(input_dir, count=5):
    """Create test images using PIL."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        # Fallback: create simple files that will be skipped
        print("⚠️  PIL not available, creating dummy files")
        image_paths = []
        for i in range(count):
            image_path = os.path.join(input_dir, f'test_image_{i:03d}.jpg')
            with open(image_path, 'wb') as f:
                f.write(b'dummy_file')
            image_paths.append(image_path)
        return image_paths
    
    image_paths = []
    for i in range(count):
        image_path = os.path.join(input_dir, f'test_image_{i:03d}.jpg')
        
        # Create a simple RGB image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        img.save(image_path, 'JPEG')
        
        image_paths.append(image_path)
    
    return image_paths


def test_end_to_end_processing():
    """Test complete end-to-end processing."""
    print("🧪 End-to-End Processing Test")
    print("=" * 50)
    
    # Create test environment
    test_dir = tempfile.mkdtemp(prefix='e2e_test_')
    input_dir = os.path.join(test_dir, 'input')
    output_dir = os.path.join(test_dir, 'output')
    
    os.makedirs(input_dir)
    os.makedirs(output_dir)
    
    try:
        # Create test images
        print("📁 Creating test images...")
        test_images = create_test_images(input_dir, count=3)
        print(f"✅ Created {len(test_images)} test images")
        
        # Initialize processor
        print("\n🚀 Initializing processor...")
        processor = ImageProcessor()
        print("✅ Processor initialized")
        
        # Mock the individual analyzers to avoid complex dependencies
        with patch.object(processor, '_process_single_image') as mock_process:
            # Mock successful processing results
            def mock_process_image(image_path, similarity_groups):
                from backend.core.base import ProcessingResult, QualityResult, DefectResult, ComplianceResult
                from datetime import datetime
                
                return ProcessingResult(
                    image_path=image_path,
                    filename=os.path.basename(image_path),
                    quality_result=QualityResult(
                        sharpness_score=120.0,
                        noise_level=0.05,
                        exposure_score=0.9,
                        color_balance_score=0.85,
                        resolution=(1920, 1080),
                        file_size=1024000,
                        overall_score=0.9,
                        passed=True
                    ),
                    defect_result=DefectResult(
                        detected_objects=[],
                        anomaly_score=0.05,
                        defect_count=0,
                        defect_types=[],
                        confidence_scores=[],
                        passed=True
                    ),
                    similarity_group=0,
                    compliance_result=ComplianceResult(
                        logo_detections=[],
                        privacy_violations=[],
                        metadata_issues=[],
                        keyword_relevance=0.9,
                        overall_compliance=True
                    ),
                    final_decision='approved',
                    rejection_reasons=[],
                    processing_time=0.1,
                    timestamp=datetime.now()
                )
            
            mock_process.side_effect = mock_process_image
            
            # Mock file organization to avoid actual file operations
            with patch.object(processor.file_manager, 'organize_output') as mock_organize:
                mock_organize.return_value = None
                
                # Mock report generation
                with patch.object(processor, '_generate_final_reports') as mock_reports:
                    mock_reports.return_value = None
                    
                    # Run processing
                    print("\n⚡ Running processing pipeline...")
                    success = processor.run(input_dir, output_dir, resume=False)
                    
                    if success:
                        print("✅ Processing completed successfully!")
                        
                        # Verify mocks were called
                        assert mock_process.call_count == len(test_images), f"Expected {len(test_images)} calls, got {mock_process.call_count}"
                        mock_organize.assert_called_once()
                        mock_reports.assert_called_once()
                        
                        print(f"📊 Processed {processor.processed_count} images")
                        print(f"✅ Approved {processor.approved_count} images")
                        print(f"❌ Rejected {processor.rejected_count} images")
                        
                        return True
                    else:
                        print("❌ Processing failed")
                        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up test directory")


def test_resume_functionality():
    """Test resume functionality."""
    print("\n🔄 Resume Functionality Test")
    print("=" * 50)
    
    # Create test environment
    test_dir = tempfile.mkdtemp(prefix='resume_test_')
    input_dir = os.path.join(test_dir, 'input')
    output_dir = os.path.join(test_dir, 'output')
    
    os.makedirs(input_dir)
    os.makedirs(output_dir)
    
    try:
        # Create test images
        test_images = create_test_images(input_dir, count=5)
        print(f"📁 Created {len(test_images)} test images")
        
        # Initialize processor
        processor = ImageProcessor()
        
        # Create a session manually
        session_id = processor.progress_tracker.create_session(
            input_folder=input_dir,
            output_folder=output_dir,
            total_images=len(test_images),
            config=processor.config_dict
        )
        
        # Save some progress - create dummy results
        from backend.core.base import ProcessingResult, QualityResult, DefectResult, ComplianceResult
        from datetime import datetime
        
        dummy_results = []
        for i in range(2):
            result = ProcessingResult(
                image_path=test_images[i],
                filename=os.path.basename(test_images[i]),
                quality_result=QualityResult(
                    sharpness_score=100.0, noise_level=0.1, exposure_score=0.8,
                    color_balance_score=0.9, resolution=(1920, 1080), file_size=1024000,
                    overall_score=0.85, passed=True
                ),
                defect_result=DefectResult(
                    detected_objects=[], anomaly_score=0.1, defect_count=0,
                    defect_types=[], confidence_scores=[], passed=True
                ),
                similarity_group=0,
                compliance_result=ComplianceResult(
                    logo_detections=[], privacy_violations=[], metadata_issues=[],
                    keyword_relevance=0.8, overall_compliance=True
                ),
                final_decision='approved' if i == 0 else 'rejected',
                rejection_reasons=[] if i == 0 else ['demo_reason'],
                processing_time=0.1,
                timestamp=datetime.now()
            )
            dummy_results.append(result)
        
        processor.progress_tracker.save_checkpoint(
            session_id=session_id,
            processed_count=2,
            total_count=len(test_images),
            results=dummy_results
        )
        
        print(f"💾 Created session with checkpoint: {session_id}")
        
        # Test finding resumable sessions
        resumable_sessions = processor._find_resumable_sessions(input_dir, output_dir)
        
        if resumable_sessions:
            print(f"✅ Found {len(resumable_sessions)} resumable session(s)")
            
            # Test loading checkpoint data
            checkpoint_data = processor.progress_tracker.load_checkpoint(session_id)
            if checkpoint_data:
                print(f"✅ Checkpoint data loaded: {checkpoint_data['processed_count']} processed")
                return True
            else:
                print("❌ Failed to load checkpoint data")
                return False
        else:
            print("❌ No resumable sessions found")
            return False
        
    except Exception as e:
        print(f"❌ Resume test failed: {e}")
        return False
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir, ignore_errors=True)
        print(f"🧹 Cleaned up test directory")


def main():
    """Run all end-to-end tests."""
    print("🧪 Adobe Stock Image Processor - End-to-End Tests")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: End-to-end processing
    if test_end_to_end_processing():
        tests_passed += 1
    
    # Test 2: Resume functionality
    if test_resume_functionality():
        tests_passed += 1
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)