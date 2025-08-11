#!/usr/bin/env python3
"""Test the integration fixes for the final integration test."""

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
        # Fallback: create simple files
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


def test_basic_integration():
    """Test basic integration with fixes."""
    print("🧪 Testing Basic Integration with Fixes")
    print("=" * 50)
    
    # Create test environment
    test_dir = tempfile.mkdtemp(prefix='integration_fix_test_')
    input_dir = os.path.join(test_dir, 'input')
    output_dir = os.path.join(test_dir, 'output')
    
    os.makedirs(input_dir)
    os.makedirs(output_dir)
    
    try:
        # Create test images
        print("📁 Creating test images...")
        test_images = create_test_images(input_dir, count=5)
        print(f"✅ Created {len(test_images)} test images")
        
        # Initialize processor
        print("\n🚀 Initializing processor...")
        processor = ImageProcessor()
        print("✅ Processor initialized")
        
        # Test that the missing methods now exist
        print("\n🔍 Testing missing methods...")
        
        # Test save_image_result method exists
        assert hasattr(processor.progress_tracker, 'save_image_result'), "save_image_result method missing"
        print("✅ save_image_result method exists")
        
        # Test update_session_progress method exists
        assert hasattr(processor.progress_tracker, 'update_session_progress'), "update_session_progress method missing"
        print("✅ update_session_progress method exists")
        
        # Test get_session_results method exists
        assert hasattr(processor.progress_tracker, 'get_session_results'), "get_session_results method missing"
        print("✅ get_session_results method exists")
        
        # Test file manager initialization
        assert isinstance(processor.file_manager.images_per_folder, int), "FileManager images_per_folder should be int"
        print(f"✅ FileManager properly initialized with images_per_folder={processor.file_manager.images_per_folder}")
        
        # Test basic processing with mocked analyzers
        print("\n⚡ Testing basic processing...")
        
        with patch.object(processor, '_process_single_image') as mock_process:
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
            
            # Run processing
            success = processor.run(input_dir, output_dir, resume=False)
            
            if success:
                print("✅ Basic processing completed successfully!")
                print(f"📊 Processed {processor.processed_count} images")
                print(f"✅ Approved {processor.approved_count} images")
                print(f"❌ Rejected {processor.rejected_count} images")
                
                # Check output folder structure
                output_folders = [d for d in os.listdir(output_dir) 
                                if os.path.isdir(os.path.join(output_dir, d)) and d.isdigit()]
                print(f"📁 Created {len(output_folders)} output folders")
                
                return True
            else:
                print("❌ Basic processing failed")
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


def main():
    """Run integration fix test."""
    print("🧪 Integration Fixes Test")
    print("=" * 40)
    
    if test_basic_integration():
        print("\n✅ All integration fixes working!")
        return True
    else:
        print("\n❌ Integration fixes failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)