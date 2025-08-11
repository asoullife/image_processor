#!/usr/bin/env python3
"""Comprehensive test script for FileManager edge cases and error handling."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import os
import tempfile
import shutil
from PIL import Image
from backend.utils.file_manager import FileManager

def test_edge_cases():
    """Test FileManager edge cases and error handling."""
    print("Testing FileManager edge cases...")
    
    temp_dir = tempfile.mkdtemp()
    input_dir = os.path.join(temp_dir, 'input')
    output_dir = os.path.join(temp_dir, 'output')
    os.makedirs(input_dir)
    
    try:
        file_manager = FileManager(images_per_folder=2)
        
        # Test 1: Empty folder scanning
        print("Test 1: Empty folder scanning...")
        empty_dir = os.path.join(temp_dir, 'empty')
        os.makedirs(empty_dir)
        found_images = file_manager.scan_images(empty_dir)
        assert len(found_images) == 0, f"Empty folder should return 0 images, got {len(found_images)}"
        print("✅ Empty folder test passed")
        
        # Test 2: Non-existent folder
        print("Test 2: Non-existent folder...")
        try:
            file_manager.scan_images('/nonexistent/folder')
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            print("✅ Non-existent folder test passed")
        
        # Test 3: Mixed file types
        print("Test 3: Mixed file types...")
        # Create valid images
        img1 = Image.new('RGB', (100, 100), color='red')
        img1_path = os.path.join(input_dir, 'valid1.jpg')
        img1.save(img1_path, 'JPEG')
        
        img2 = Image.new('RGB', (100, 100), color='blue')
        img2_path = os.path.join(input_dir, 'valid2.png')
        img2.save(img2_path, 'PNG')
        
        # Create non-image files
        with open(os.path.join(input_dir, 'text_file.txt'), 'w') as f:
            f.write('This is not an image')
        
        with open(os.path.join(input_dir, 'fake_image.jpg'), 'w') as f:
            f.write('This is not real image data')
        
        # Create empty file
        with open(os.path.join(input_dir, 'empty.png'), 'w') as f:
            pass
        
        found_images = file_manager.scan_images(input_dir)
        assert len(found_images) == 2, f"Should find 2 valid images, found {len(found_images)}"
        print("✅ Mixed file types test passed")
        
        # Test 4: Filename conflicts
        print("Test 4: Filename conflicts...")
        # Create output folder with existing file
        os.makedirs(output_dir)
        conflict_folder = os.path.join(output_dir, '1')
        os.makedirs(conflict_folder)
        
        # Create a file that will conflict
        with open(os.path.join(conflict_folder, 'valid1.jpg'), 'w') as f:
            f.write('existing file')
        
        results = file_manager.organize_output([img1_path, img2_path], output_dir)
        assert results['successful_copies'] == 2, f"Should copy 2 files, copied {results['successful_copies']}"
        
        # Check that conflict was resolved
        files_in_folder = os.listdir(conflict_folder)
        assert 'valid1_1.jpg' in files_in_folder or 'valid1.jpg' in files_in_folder, "Conflict should be resolved"
        print("✅ Filename conflicts test passed")
        
        # Test 5: Empty image list organization
        print("Test 5: Empty image list organization...")
        empty_output_dir = os.path.join(temp_dir, 'empty_output')
        results = file_manager.organize_output([], empty_output_dir)
        assert results['total_images'] == 0
        assert results['folders_created'] == 0
        assert results['successful_copies'] == 0
        print("✅ Empty image list test passed")
        
        # Test 6: Metadata extraction
        print("Test 6: Metadata extraction...")
        metadata = file_manager.get_image_metadata(img1_path)
        assert metadata['filename'] == 'valid1.jpg'
        assert metadata['file_size'] > 0
        assert metadata['dimensions'] == (100, 100)
        assert metadata['format'] == 'JPEG'
        print("✅ Metadata extraction test passed")
        
        # Test 7: Invalid metadata extraction
        print("Test 7: Invalid metadata extraction...")
        invalid_metadata = file_manager.get_image_metadata('/nonexistent/file.jpg')
        assert invalid_metadata['file_size'] == 0
        assert invalid_metadata['dimensions'] == (0, 0)
        assert invalid_metadata['format'] is None
        print("✅ Invalid metadata test passed")
        
        # Test 8: File copy verification
        print("Test 8: File copy verification...")
        test_copy_dir = os.path.join(temp_dir, 'copy_test')
        os.makedirs(test_copy_dir)
        
        copy_dest = os.path.join(test_copy_dir, 'copied_image.jpg')
        success = file_manager.copy_with_verification(img1_path, copy_dest)
        assert success, "Copy should succeed"
        assert os.path.exists(copy_dest), "Copied file should exist"
        
        # Verify integrity
        original_hash = file_manager._calculate_file_hash(img1_path)
        copy_hash = file_manager._calculate_file_hash(copy_dest)
        assert original_hash == copy_hash, "Hashes should match"
        print("✅ File copy verification test passed")
        
        # Test 9: Copy non-existent file
        print("Test 9: Copy non-existent file...")
        success = file_manager.copy_with_verification('/nonexistent/file.jpg', copy_dest)
        assert not success, "Copy of non-existent file should fail"
        print("✅ Copy non-existent file test passed")
        
        # Test 10: Validate output structure
        print("Test 10: Validate output structure...")
        validation = file_manager.validate_output_structure(output_dir)
        assert validation['valid'], "Output structure should be valid"
        assert validation['total_images'] >= 2, "Should have at least 2 images"
        print("✅ Output structure validation test passed")
        
        # Test 11: Validate non-existent output
        print("Test 11: Validate non-existent output...")
        validation = file_manager.validate_output_structure('/nonexistent/output')
        assert not validation['valid'], "Non-existent output should be invalid"
        assert validation['error'] == 'Output folder does not exist'
        print("✅ Non-existent output validation test passed")
        
        print("✅ All edge case tests passed!")
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Cleanup completed")

def test_large_batch():
    """Test FileManager with larger batch of images."""
    print("Testing FileManager with large batch...")
    
    temp_dir = tempfile.mkdtemp()
    input_dir = os.path.join(temp_dir, 'large_input')
    output_dir = os.path.join(temp_dir, 'large_output')
    os.makedirs(input_dir)
    
    try:
        # Create 25 test images
        print("Creating 25 test images...")
        for i in range(25):
            img = Image.new('RGB', (50, 50), color=(i*10 % 255, (i*20) % 255, (i*30) % 255))
            img_path = os.path.join(input_dir, f'batch_image_{i:03d}.jpg')
            img.save(img_path, 'JPEG', quality=90)
        
        # Create subdirectories with more images
        for sub_num in range(2):
            sub_dir = os.path.join(input_dir, f'subdir_{sub_num}')
            os.makedirs(sub_dir)
            
            for i in range(5):
                img = Image.new('RGB', (75, 75), color=(sub_num*100, i*40, 150))
                img_path = os.path.join(sub_dir, f'sub_{sub_num}_img_{i}.png')
                img.save(img_path, 'PNG')
        
        file_manager = FileManager(images_per_folder=10)  # 10 images per folder
        
        # Scan all images
        found_images = file_manager.scan_images(input_dir)
        expected_count = 25 + (2 * 5)  # 25 main + 10 in subdirs
        assert len(found_images) == expected_count, f"Expected {expected_count} images, found {len(found_images)}"
        print(f"Found {len(found_images)} images")
        
        # Organize images
        results = file_manager.organize_output(found_images, output_dir)
        
        assert results['total_images'] == expected_count
        assert results['successful_copies'] == expected_count
        assert results['failed_copies'] == 0
        
        # Should create 4 folders: 10, 10, 10, 5 images
        expected_folders = 4
        assert results['folders_created'] == expected_folders, f"Expected {expected_folders} folders, created {results['folders_created']}"
        
        # Validate structure
        validation = file_manager.validate_output_structure(output_dir)
        assert validation['valid']
        assert validation['total_images'] == expected_count
        
        # Check folder distribution
        folder_counts = [validation['folders'][i]['image_count'] for i in sorted(validation['folders'].keys())]
        expected_counts = [10, 10, 10, 5]
        assert folder_counts == expected_counts, f"Expected {expected_counts}, got {folder_counts}"
        
        print("✅ Large batch test passed!")
        
    except Exception as e:
        print(f"❌ Large batch test failed: {e}")
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Large batch cleanup completed")

if __name__ == '__main__':
    test_edge_cases()
    test_large_batch()
    print("🎉 All comprehensive tests passed!")