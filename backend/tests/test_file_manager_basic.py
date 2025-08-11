#!/usr/bin/env python3
"""Basic test script for FileManager functionality."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import os
import tempfile
import shutil
from PIL import Image
from backend.utils.file_manager import FileManager

def test_basic_functionality():
    """Test basic FileManager functionality."""
    print("Testing FileManager basic functionality...")
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    input_dir = os.path.join(temp_dir, 'input')
    output_dir = os.path.join(temp_dir, 'output')
    os.makedirs(input_dir)
    
    try:
        # Create test images
        print("Creating test images...")
        test_images = []
        for i in range(5):
            img = Image.new('RGB', (100, 100), color=(i*50, i*50, i*50))
            img_path = os.path.join(input_dir, f'test_image_{i}.jpg')
            img.save(img_path, 'JPEG')
            test_images.append(img_path)
        
        # Create subdirectory with image
        sub_dir = os.path.join(input_dir, 'subdir')
        os.makedirs(sub_dir)
        sub_img = Image.new('RGB', (150, 150), color='green')
        sub_img_path = os.path.join(sub_dir, 'sub_image.png')
        sub_img.save(sub_img_path, 'PNG')
        test_images.append(sub_img_path)
        
        # Initialize FileManager
        file_manager = FileManager(images_per_folder=3)
        
        # Test scanning
        print("Testing image scanning...")
        found_images = file_manager.scan_images(input_dir)
        print(f"Found {len(found_images)} images")
        assert len(found_images) == 6, f"Expected 6 images, found {len(found_images)}"
        
        # Test organization
        print("Testing image organization...")
        results = file_manager.organize_output(found_images, output_dir)
        print(f"Organization results: {results}")
        
        assert results['total_images'] == 6
        assert results['successful_copies'] == 6
        assert results['failed_copies'] == 0
        assert results['folders_created'] == 2  # 3 + 3 images
        
        # Verify folder structure
        print("Verifying folder structure...")
        folder1_path = os.path.join(output_dir, '1')
        folder2_path = os.path.join(output_dir, '2')
        
        assert os.path.exists(folder1_path), "Folder 1 should exist"
        assert os.path.exists(folder2_path), "Folder 2 should exist"
        
        folder1_count = len(os.listdir(folder1_path))
        folder2_count = len(os.listdir(folder2_path))
        
        assert folder1_count == 3, f"Folder 1 should have 3 images, has {folder1_count}"
        assert folder2_count == 3, f"Folder 2 should have 3 images, has {folder2_count}"
        
        # Test file integrity
        print("Testing file integrity...")
        for original_path in found_images[:2]:  # Test first 2 files
            filename = os.path.basename(original_path)
            copy_path = os.path.join(folder1_path, filename)
            if not os.path.exists(copy_path):
                copy_path = os.path.join(folder2_path, filename)
            
            assert os.path.exists(copy_path), f"Copy should exist: {copy_path}"
            
            # Compare file sizes
            original_size = os.path.getsize(original_path)
            copy_size = os.path.getsize(copy_path)
            assert original_size == copy_size, f"File sizes should match: {original_size} vs {copy_size}"
        
        # Test validation
        print("Testing output validation...")
        validation = file_manager.validate_output_structure(output_dir)
        assert validation['valid'], "Output structure should be valid"
        assert validation['total_images'] == 6, f"Should validate 6 images, got {validation['total_images']}"
        
        print("✅ All basic tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Cleanup completed")

if __name__ == '__main__':
    test_basic_functionality()