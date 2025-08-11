"""Unit tests for FileManager class."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import os
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from PIL import Image
import hashlib

from backend.utils.file_manager import FileManager
from backend.config.config_loader import get_config


class TestFileManager(unittest.TestCase):
    """Test cases for FileManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.input_dir)
        os.makedirs(self.output_dir)
        
        # Create test images
        self.test_images = []
        self._create_test_images()
        
        self.file_manager = FileManager(images_per_folder=3)  # Small number for testing
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_images(self):
        """Create test image files."""
        # Create valid test images
        for i in range(5):
            img = Image.new('RGB', (100, 100), color=(i*50, i*50, i*50))
            img_path = os.path.join(self.input_dir, f'test_image_{i}.jpg')
            img.save(img_path, 'JPEG')
            self.test_images.append(img_path)
        
        # Create PNG image
        png_img = Image.new('RGB', (200, 200), color='blue')
        png_path = os.path.join(self.input_dir, 'test_image.png')
        png_img.save(png_path, 'PNG')
        self.test_images.append(png_path)
        
        # Create subdirectory with images
        sub_dir = os.path.join(self.input_dir, 'subdir')
        os.makedirs(sub_dir)
        sub_img = Image.new('RGB', (150, 150), color='green')
        sub_img_path = os.path.join(sub_dir, 'sub_image.jpeg')
        sub_img.save(sub_img_path, 'JPEG')
        self.test_images.append(sub_img_path)
        
        # Create non-image file
        text_file = os.path.join(self.input_dir, 'not_image.txt')
        with open(text_file, 'w') as f:
            f.write('This is not an image')
        
        # Create empty file
        empty_file = os.path.join(self.input_dir, 'empty.jpg')
        with open(empty_file, 'w') as f:
            pass  # Create empty file
    
    def test_init(self):
        """Test FileManager initialization."""
        fm = FileManager(images_per_folder=100)
        self.assertEqual(fm.images_per_folder, 100)
        self.assertIsNotNone(fm.logger)
    
    def test_scan_images_success(self):
        """Test successful image scanning."""
        found_images = self.file_manager.scan_images(self.input_dir)
        
        # Should find all valid images (excluding empty file and text file)
        self.assertEqual(len(found_images), 7)  # 5 JPG + 1 PNG + 1 JPEG in subdir
        
        # Check that all found files are absolute paths
        for img_path in found_images:
            self.assertTrue(os.path.isabs(img_path))
            self.assertTrue(os.path.exists(img_path))
        
        # Check that results are sorted
        self.assertEqual(found_images, sorted(found_images))
    
    def test_scan_images_nonexistent_folder(self):
        """Test scanning non-existent folder."""
        with self.assertRaises(FileNotFoundError):
            self.file_manager.scan_images('/nonexistent/folder')
    
    def test_scan_images_not_directory(self):
        """Test scanning a file instead of directory."""
        file_path = os.path.join(self.temp_dir, 'test_file.txt')
        with open(file_path, 'w') as f:
            f.write('test')
        
        with self.assertRaises(ValueError):
            self.file_manager.scan_images(file_path)
    
    @patch('os.listdir')
    def test_scan_images_permission_error(self, mock_listdir):
        """Test scanning folder with permission error."""
        mock_listdir.side_effect = PermissionError("Access denied")
        
        with self.assertRaises(PermissionError):
            self.file_manager.scan_images(self.input_dir)
    
    def test_is_supported_image(self):
        """Test supported image extension checking."""
        # Test supported extensions
        self.assertTrue(self.file_manager._is_supported_image('test.jpg'))
        self.assertTrue(self.file_manager._is_supported_image('test.jpeg'))
        self.assertTrue(self.file_manager._is_supported_image('test.png'))
        self.assertTrue(self.file_manager._is_supported_image('test.JPG'))
        self.assertTrue(self.file_manager._is_supported_image('test.JPEG'))
        self.assertTrue(self.file_manager._is_supported_image('test.PNG'))
        
        # Test unsupported extensions
        self.assertFalse(self.file_manager._is_supported_image('test.gif'))
        self.assertFalse(self.file_manager._is_supported_image('test.bmp'))
        self.assertFalse(self.file_manager._is_supported_image('test.txt'))
        self.assertFalse(self.file_manager._is_supported_image('test'))
    
    def test_validate_image_file(self):
        """Test image file validation."""
        # Test valid image
        valid_img_path = self.test_images[0]
        self.assertTrue(self.file_manager._validate_image_file(valid_img_path))
        
        # Test empty file
        empty_file = os.path.join(self.temp_dir, 'empty.jpg')
        with open(empty_file, 'w') as f:
            pass
        self.assertFalse(self.file_manager._validate_image_file(empty_file))
        
        # Test non-existent file
        self.assertFalse(self.file_manager._validate_image_file('/nonexistent/file.jpg'))
        
        # Test corrupted image file
        corrupted_file = os.path.join(self.temp_dir, 'corrupted.jpg')
        with open(corrupted_file, 'w') as f:
            f.write('This is not image data')
        self.assertFalse(self.file_manager._validate_image_file(corrupted_file))
    
    def test_organize_output_success(self):
        """Test successful output organization."""
        # Use 7 images with 3 per folder
        approved_images = self.test_images[:7]
        
        results = self.file_manager.organize_output(approved_images, self.output_dir)
        
        # Check results
        self.assertEqual(results['total_images'], 7)
        self.assertEqual(results['folders_created'], 3)  # 3, 3, 1 images per folder
        self.assertEqual(results['successful_copies'], 7)
        self.assertEqual(results['failed_copies'], 0)
        
        # Check folder structure
        self.assertIn('1', results['folder_structure'])
        self.assertIn('2', results['folder_structure'])
        self.assertIn('3', results['folder_structure'])
        
        # Check actual folder creation
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, '1')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, '2')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, '3')))
        
        # Check image counts per folder
        folder1_images = len(os.listdir(os.path.join(self.output_dir, '1')))
        folder2_images = len(os.listdir(os.path.join(self.output_dir, '2')))
        folder3_images = len(os.listdir(os.path.join(self.output_dir, '3')))
        
        self.assertEqual(folder1_images, 3)
        self.assertEqual(folder2_images, 3)
        self.assertEqual(folder3_images, 1)
    
    def test_organize_output_empty_list(self):
        """Test organizing empty image list."""
        results = self.file_manager.organize_output([], self.output_dir)
        
        self.assertEqual(results['total_images'], 0)
        self.assertEqual(results['folders_created'], 0)
        self.assertEqual(results['successful_copies'], 0)
        self.assertEqual(results['failed_copies'], 0)
    
    def test_organize_output_invalid_folder(self):
        """Test organizing with invalid output folder."""
        # Try to create folder in non-existent parent
        invalid_output = '/nonexistent/parent/output'
        
        with self.assertRaises(ValueError):
            self.file_manager.organize_output(self.test_images[:2], invalid_output)
    
    def test_resolve_filename_conflict(self):
        """Test filename conflict resolution."""
        # Create a file that will cause conflict
        conflict_file = os.path.join(self.output_dir, 'test.jpg')
        with open(conflict_file, 'w') as f:
            f.write('existing file')
        
        # Test conflict resolution
        resolved_path = self.file_manager._resolve_filename_conflict(conflict_file)
        expected_path = os.path.join(self.output_dir, 'test_1.jpg')
        self.assertEqual(resolved_path, expected_path)
        
        # Create the resolved file and test again
        with open(resolved_path, 'w') as f:
            f.write('another file')
        
        resolved_path2 = self.file_manager._resolve_filename_conflict(conflict_file)
        expected_path2 = os.path.join(self.output_dir, 'test_2.jpg')
        self.assertEqual(resolved_path2, expected_path2)
    
    def test_copy_with_verification_success(self):
        """Test successful file copy with verification."""
        source = self.test_images[0]
        destination = os.path.join(self.output_dir, 'copied_image.jpg')
        
        result = self.file_manager.copy_with_verification(source, destination)
        
        self.assertTrue(result)
        self.assertTrue(os.path.exists(destination))
        
        # Verify file sizes match
        source_size = os.path.getsize(source)
        dest_size = os.path.getsize(destination)
        self.assertEqual(source_size, dest_size)
        
        # Verify file hashes match
        source_hash = self.file_manager._calculate_file_hash(source)
        dest_hash = self.file_manager._calculate_file_hash(destination)
        self.assertEqual(source_hash, dest_hash)
    
    def test_copy_with_verification_nonexistent_source(self):
        """Test copy with non-existent source file."""
        source = '/nonexistent/file.jpg'
        destination = os.path.join(self.output_dir, 'copied_image.jpg')
        
        result = self.file_manager.copy_with_verification(source, destination)
        
        self.assertFalse(result)
        self.assertFalse(os.path.exists(destination))
    
    @patch('shutil.copy2')
    def test_copy_with_verification_copy_failure(self, mock_copy):
        """Test copy failure handling."""
        mock_copy.side_effect = OSError("Copy failed")
        
        source = self.test_images[0]
        destination = os.path.join(self.output_dir, 'copied_image.jpg')
        
        result = self.file_manager.copy_with_verification(source, destination)
        
        self.assertFalse(result)
    
    def test_calculate_file_hash(self):
        """Test file hash calculation."""
        test_file = self.test_images[0]
        
        # Calculate hash using FileManager
        fm_hash = self.file_manager._calculate_file_hash(test_file)
        
        # Calculate hash manually for comparison
        hash_md5 = hashlib.md5()
        with open(test_file, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        expected_hash = hash_md5.hexdigest()
        
        self.assertEqual(fm_hash, expected_hash)
    
    def test_calculate_file_hash_nonexistent(self):
        """Test hash calculation for non-existent file."""
        result = self.file_manager._calculate_file_hash('/nonexistent/file.jpg')
        self.assertIsNone(result)
    
    def test_get_image_metadata(self):
        """Test image metadata extraction."""
        test_image = self.test_images[0]
        metadata = self.file_manager.get_image_metadata(test_image)
        
        # Check required fields
        self.assertIn('filename', metadata)
        self.assertIn('file_path', metadata)
        self.assertIn('file_size', metadata)
        self.assertIn('dimensions', metadata)
        self.assertIn('format', metadata)
        self.assertIn('mode', metadata)
        self.assertIn('exif', metadata)
        
        # Check values
        self.assertEqual(metadata['filename'], os.path.basename(test_image))
        self.assertEqual(metadata['file_path'], test_image)
        self.assertGreater(metadata['file_size'], 0)
        self.assertEqual(metadata['dimensions'], (100, 100))  # From test image creation
        self.assertEqual(metadata['format'], 'JPEG')
    
    def test_get_image_metadata_invalid_file(self):
        """Test metadata extraction from invalid file."""
        metadata = self.file_manager.get_image_metadata('/nonexistent/file.jpg')
        
        # Should return default metadata structure
        self.assertEqual(metadata['file_size'], 0)
        self.assertEqual(metadata['dimensions'], (0, 0))
        self.assertIsNone(metadata['format'])
    
    def test_validate_output_structure(self):
        """Test output structure validation."""
        # First organize some images
        approved_images = self.test_images[:5]
        self.file_manager.organize_output(approved_images, self.output_dir)
        
        # Validate the structure
        results = self.file_manager.validate_output_structure(self.output_dir)
        
        self.assertTrue(results['valid'])
        self.assertIsNone(results['error'])
        self.assertEqual(results['total_images'], 5)
        self.assertIn(1, results['folders'])
        self.assertIn(2, results['folders'])
    
    def test_validate_output_structure_nonexistent(self):
        """Test validation of non-existent output folder."""
        results = self.file_manager.validate_output_structure('/nonexistent/folder')
        
        self.assertFalse(results['valid'])
        self.assertEqual(results['error'], 'Output folder does not exist')
        self.assertEqual(results['total_images'], 0)
    
    def test_cleanup_empty_folders(self):
        """Test empty folder cleanup."""
        # Create some empty folders
        empty_folder1 = os.path.join(self.temp_dir, 'empty1')
        empty_folder2 = os.path.join(self.temp_dir, 'empty2')
        non_empty_folder = os.path.join(self.temp_dir, 'nonempty')
        
        os.makedirs(empty_folder1)
        os.makedirs(empty_folder2)
        os.makedirs(non_empty_folder)
        
        # Add file to non-empty folder
        with open(os.path.join(non_empty_folder, 'file.txt'), 'w') as f:
            f.write('content')
        
        # Run cleanup
        removed_count = self.file_manager.cleanup_empty_folders(self.temp_dir)
        
        # Check results
        self.assertEqual(removed_count, 2)
        self.assertFalse(os.path.exists(empty_folder1))
        self.assertFalse(os.path.exists(empty_folder2))
        self.assertTrue(os.path.exists(non_empty_folder))


class TestFileManagerIntegration(unittest.TestCase):
    """Integration tests for FileManager with real file operations."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.input_dir)
        
        # Create a larger set of test images
        self.test_images = []
        self._create_large_test_set()
        
        # Use configuration from settings
        try:
            config = get_config()
            images_per_folder = config.output.images_per_folder
        except:
            images_per_folder = 200  # Default fallback
        
        self.file_manager = FileManager(images_per_folder=images_per_folder)
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_large_test_set(self):
        """Create a larger set of test images for integration testing."""
        # Create images in main directory
        for i in range(10):
            img = Image.new('RGB', (200, 200), color=(i*25, i*25, i*25))
            img_path = os.path.join(self.input_dir, f'main_image_{i:03d}.jpg')
            img.save(img_path, 'JPEG', quality=95)
            self.test_images.append(img_path)
        
        # Create subdirectories with images
        for sub_num in range(3):
            sub_dir = os.path.join(self.input_dir, f'subdir_{sub_num}')
            os.makedirs(sub_dir)
            
            for i in range(5):
                img = Image.new('RGB', (150, 150), color=(sub_num*80, i*50, 100))
                img_path = os.path.join(sub_dir, f'sub_{sub_num}_image_{i}.png')
                img.save(img_path, 'PNG')
                self.test_images.append(img_path)
        
        # Create nested subdirectories
        nested_dir = os.path.join(self.input_dir, 'level1', 'level2')
        os.makedirs(nested_dir)
        nested_img = Image.new('RGB', (100, 100), color='red')
        nested_path = os.path.join(nested_dir, 'nested_image.jpeg')
        nested_img.save(nested_path, 'JPEG')
        self.test_images.append(nested_path)
    
    def test_full_workflow(self):
        """Test complete workflow from scan to organize."""
        # Step 1: Scan for images
        found_images = self.file_manager.scan_images(self.input_dir)
        
        # Should find all created images
        expected_count = 10 + (3 * 5) + 1  # main + subdirs + nested
        self.assertEqual(len(found_images), expected_count)
        
        # Step 2: Organize images
        results = self.file_manager.organize_output(found_images, self.output_dir)
        
        # Check organization results
        self.assertEqual(results['total_images'], expected_count)
        self.assertEqual(results['successful_copies'], expected_count)
        self.assertEqual(results['failed_copies'], 0)
        self.assertGreater(results['folders_created'], 0)
        
        # Step 3: Validate output structure
        validation = self.file_manager.validate_output_structure(self.output_dir)
        
        self.assertTrue(validation['valid'])
        self.assertEqual(validation['total_images'], expected_count)
        
        # Step 4: Verify all files were copied correctly
        for original_path in found_images:
            # Find the copied file
            found_copy = False
            for folder_num, folder_info in validation['folders'].items():
                if os.path.basename(original_path) in folder_info['images']:
                    copy_path = os.path.join(self.output_dir, str(folder_num), os.path.basename(original_path))
                    self.assertTrue(os.path.exists(copy_path))
                    
                    # Verify file integrity
                    original_hash = self.file_manager._calculate_file_hash(original_path)
                    copy_hash = self.file_manager._calculate_file_hash(copy_path)
                    self.assertEqual(original_hash, copy_hash)
                    
                    found_copy = True
                    break
            
            self.assertTrue(found_copy, f"Copy not found for {original_path}")
    
    def test_large_batch_processing(self):
        """Test processing with larger number of images."""
        # Create more images to test folder organization
        large_input_dir = os.path.join(self.temp_dir, 'large_input')
        os.makedirs(large_input_dir)
        
        # Create 250 images to test multiple folder creation
        large_images = []
        for i in range(250):
            img = Image.new('RGB', (50, 50), color=(i % 255, (i*2) % 255, (i*3) % 255))
            img_path = os.path.join(large_input_dir, f'large_image_{i:04d}.jpg')
            img.save(img_path, 'JPEG', quality=85)
            large_images.append(img_path)
        
        # Scan and organize
        found_images = self.file_manager.scan_images(large_input_dir)
        self.assertEqual(len(found_images), 250)
        
        large_output_dir = os.path.join(self.temp_dir, 'large_output')
        results = self.file_manager.organize_output(found_images, large_output_dir)
        
        # With default 200 images per folder, should create 2 folders
        # Folder 1: 200 images, Folder 2: 50 images
        expected_folders = 2
        self.assertEqual(results['folders_created'], expected_folders)
        self.assertEqual(results['successful_copies'], 250)
        
        # Validate folder contents
        validation = self.file_manager.validate_output_structure(large_output_dir)
        self.assertTrue(validation['valid'])
        self.assertEqual(validation['total_images'], 250)
        
        # Check specific folder counts
        self.assertEqual(validation['folders'][1]['image_count'], 200)
        self.assertEqual(validation['folders'][2]['image_count'], 50)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    unittest.main()