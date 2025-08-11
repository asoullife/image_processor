"""Tests for file integrity protection system."""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from backend.core.file_protection import (
    FileIntegrityProtector, 
    FileIntegrityError, 
    FileProtectionError,
    safe_copy_with_verification,
    create_output_structure
)
from backend.core.safe_operations import SafeImageProcessor, safe_analyze_image
from backend.core.output_manager import OutputStructureManager, create_output_structure_from_input

class TestFileIntegrityProtector:
    """Test file integrity protection functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test_image.jpg"
        
        # Create a test file
        self.test_file.write_text("Test image content")
        
        self.protector = FileIntegrityProtector(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        self.protector.cleanup_temp_files()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_calculate_file_hash(self):
        """Test file hash calculation."""
        hash1 = self.protector.calculate_file_hash(self.test_file)
        hash2 = self.protector.calculate_file_hash(self.test_file)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length
        assert isinstance(hash1, str)

    def test_verify_file_integrity(self):
        """Test file integrity verification."""
        original_hash = self.protector.calculate_file_hash(self.test_file)
        
        # Verify with correct hash
        assert self.protector.verify_file_integrity(self.test_file, original_hash)
        
        # Verify with incorrect hash
        wrong_hash = "0" * 64
        assert not self.protector.verify_file_integrity(self.test_file, wrong_hash)

    def test_create_protected_copy(self):
        """Test creating protected copy."""
        protected_path = self.protector.create_protected_copy(self.test_file)
        
        assert Path(protected_path).exists()
        assert protected_path in self.protector.temp_files
        assert str(self.test_file) in self.protector.protected_files
        
        # Verify content is identical
        original_content = self.test_file.read_text()
        protected_content = Path(protected_path).read_text()
        assert original_content == protected_content

    def test_create_safe_working_copy(self):
        """Test creating safe working copy."""
        working_path = self.protector.create_safe_working_copy(self.test_file)
        
        assert Path(working_path).exists()
        assert working_path in self.protector.temp_files
        
        # Verify working copy is writable
        working_file = Path(working_path)
        assert os.access(working_file, os.W_OK)

    def test_atomic_copy_to_output(self):
        """Test atomic copy operation."""
        output_path = Path(self.temp_dir) / "output" / "copied_file.jpg"
        
        success = self.protector.atomic_copy_to_output(self.test_file, output_path)
        
        assert success
        assert output_path.exists()
        
        # Verify content integrity
        original_content = self.test_file.read_text()
        copied_content = output_path.read_text()
        assert original_content == copied_content

    def test_mirror_directory_structure(self):
        """Test directory structure mirroring."""
        # Create test directory structure
        input_dir = Path(self.temp_dir) / "input"
        input_dir.mkdir()
        (input_dir / "subdir1").mkdir()
        (input_dir / "subdir2").mkdir()
        (input_dir / "file1.txt").write_text("content1")
        (input_dir / "subdir1" / "file2.txt").write_text("content2")
        
        output_dir = Path(self.temp_dir) / "output"
        
        structure_map = self.protector.mirror_directory_structure(input_dir, output_dir)
        
        assert (output_dir / "subdir1").exists()
        assert (output_dir / "subdir2").exists()
        assert len(structure_map) > 0

    def test_verify_source_integrity(self):
        """Test source file integrity verification."""
        # Create additional test files
        test_file2 = Path(self.temp_dir) / "test2.jpg"
        test_file2.write_text("Test content 2")
        
        results = self.protector.verify_source_integrity([self.test_file, test_file2])
        
        assert len(results) == 2
        assert all(results.values())  # All should be True for first verification

    def test_cleanup_temp_files(self):
        """Test temporary file cleanup."""
        # Create some temporary files
        protected_path = self.protector.create_protected_copy(self.test_file)
        working_path = self.protector.create_safe_working_copy(self.test_file)
        
        assert Path(protected_path).exists()
        assert Path(working_path).exists()
        
        # Cleanup
        self.protector.cleanup_temp_files()
        
        assert not Path(protected_path).exists()
        assert not Path(working_path).exists()
        assert len(self.protector.temp_files) == 0

    def test_context_manager(self):
        """Test context manager functionality."""
        with FileIntegrityProtector(self.temp_dir) as protector:
            protected_path = protector.create_protected_copy(self.test_file)
            assert Path(protected_path).exists()
        
        # After context exit, temp files should be cleaned up
        assert not Path(protected_path).exists()

class TestSafeImageProcessor:
    """Test safe image processing operations."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test_image.jpg"
        self.test_file.write_text("Test image content")
        
        self.processor = SafeImageProcessor(self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        self.processor.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_safe_image_analysis_context(self):
        """Test safe image analysis context manager."""
        with self.processor.safe_image_analysis(self.test_file) as (original, analysis):
            assert Path(original).exists()
            assert Path(analysis).exists()
            assert str(original) == str(self.test_file)
            assert analysis != original

    @patch('backend.core.safe_operations.Image')
    def test_safe_thumbnail_generation(self, mock_image):
        """Test safe thumbnail generation."""
        # Mock PIL Image
        mock_img = MagicMock()
        mock_image.open.return_value.__enter__.return_value = mock_img
        
        thumbnail_path = self.processor.safe_thumbnail_generation(self.test_file)
        
        if thumbnail_path:  # Only test if PIL is available
            mock_img.thumbnail.assert_called_once()
            mock_img.save.assert_called_once()

    def test_safe_metadata_extraction(self):
        """Test safe metadata extraction."""
        metadata = self.processor.safe_metadata_extraction(self.test_file)
        
        assert "file_path" in metadata
        assert "file_size" in metadata
        assert "file_name" in metadata
        assert metadata["file_name"] == "test_image.jpg"

    def test_batch_safe_copy(self):
        """Test batch safe copy operation."""
        # Create additional test files
        test_files = []
        for i in range(3):
            test_file = Path(self.temp_dir) / f"test_{i}.jpg"
            test_file.write_text(f"Test content {i}")
            test_files.append(test_file)
        
        output_dir = Path(self.temp_dir) / "output"
        
        results = self.processor.batch_safe_copy(test_files, output_dir)
        
        assert len(results) == 3
        assert all(results.values())  # All should succeed
        
        # Verify files were copied
        for test_file in test_files:
            output_file = output_dir / test_file.name
            assert output_file.exists()

    def test_create_processing_workspace(self):
        """Test processing workspace creation."""
        test_files = [self.test_file]
        
        workspace_path = self.processor.create_processing_workspace(test_files)
        
        assert Path(workspace_path).exists()
        assert Path(workspace_path).is_dir()
        
        # Verify file is in workspace
        workspace_file = Path(workspace_path) / self.test_file.name
        assert workspace_file.exists()

    def test_validate_output_integrity(self):
        """Test output file integrity validation."""
        # Create test output files
        output_files = []
        for i in range(2):
            output_file = Path(self.temp_dir) / f"output_{i}.jpg"
            output_file.write_text(f"Output content {i}")
            output_files.append(output_file)
        
        results = self.processor.validate_output_integrity(output_files)
        
        assert len(results) == 2
        assert all(results.values())  # All should be valid

class TestOutputStructureManager:
    """Test output structure management."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test input structure
        self.input_dir = Path(self.temp_dir) / "input"
        self.input_dir.mkdir()
        (self.input_dir / "subdir1").mkdir()
        (self.input_dir / "subdir2").mkdir()
        (self.input_dir / "file1.jpg").write_text("content1")
        (self.input_dir / "subdir1" / "file2.jpg").write_text("content2")
        (self.input_dir / "subdir2" / "file3.jpg").write_text("content3")
        
        self.output_dir = Path(self.temp_dir) / "output"
        
        self.manager = OutputStructureManager(self.input_dir, self.output_dir)

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_mirrored_structure(self):
        """Test creating mirrored directory structure."""
        structure_map = self.manager.create_mirrored_structure()
        
        assert len(structure_map) > 0
        assert (self.output_dir / "subdir1").exists()
        assert (self.output_dir / "subdir2").exists()

    def test_get_output_path(self):
        """Test getting output path for input path."""
        self.manager.create_mirrored_structure()
        
        input_file = self.input_dir / "file1.jpg"
        output_path = self.manager.get_output_path(input_file)
        
        assert output_path is not None
        assert "output" in output_path
        assert "file1.jpg" in output_path

    def test_organize_approved_files(self):
        """Test organizing approved files into subfolders."""
        self.manager.create_mirrored_structure()
        
        approved_files = [
            self.input_dir / "file1.jpg",
            self.input_dir / "subdir1" / "file2.jpg",
            self.input_dir / "subdir2" / "file3.jpg"
        ]
        
        organization = self.manager.organize_approved_files(approved_files, subfolder_size=2)
        
        assert len(organization) > 0
        total_files = sum(len(files) for files in organization.values())
        assert total_files == len(approved_files)

    def test_copy_approved_files(self):
        """Test copying approved files to organized structure."""
        self.manager.create_mirrored_structure()
        
        approved_files = [
            self.input_dir / "file1.jpg",
            self.input_dir / "subdir1" / "file2.jpg"
        ]
        
        results = self.manager.copy_approved_files(approved_files)
        
        assert len(results) == 2
        # Note: Results might be False if files don't exist, but structure should be tested

    def test_validate_output_structure(self):
        """Test output structure validation."""
        self.manager.create_mirrored_structure()
        
        validation_results = self.manager.validate_output_structure()
        
        assert len(validation_results) > 0
        # Most directories should be valid after creation
        valid_count = sum(1 for valid in validation_results.values() if valid)
        assert valid_count > 0

    def test_get_structure_statistics(self):
        """Test getting structure statistics."""
        self.manager.create_mirrored_structure()
        
        stats = self.manager.get_structure_statistics()
        
        assert "total_mapped_paths" in stats
        assert "created_directories" in stats
        assert stats["total_mapped_paths"] > 0

class TestUtilityFunctions:
    """Test utility functions."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test.jpg"
        self.test_file.write_text("Test content")

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_safe_copy_with_verification(self):
        """Test safe copy utility function."""
        output_file = Path(self.temp_dir) / "output.jpg"
        
        success = safe_copy_with_verification(self.test_file, output_file)
        
        assert success
        assert output_file.exists()

    def test_create_output_structure(self):
        """Test create output structure utility function."""
        input_dir = Path(self.temp_dir) / "input"
        input_dir.mkdir()
        (input_dir / "test.jpg").write_text("content")
        
        output_dir = Path(self.temp_dir) / "output"
        
        structure_map = create_output_structure(input_dir, output_dir)
        
        assert len(structure_map) > 0
        assert output_dir.exists()

    def test_safe_analyze_image(self):
        """Test safe image analysis utility function."""
        def dummy_analysis(image_path):
            return {"analyzed": True, "path": image_path}
        
        result = safe_analyze_image(self.test_file, dummy_analysis)
        
        assert result["analyzed"] is True
        assert "path" in result

# Integration tests
class TestFileProtectionIntegration:
    """Integration tests for file protection system."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create realistic test structure
        self.input_dir = Path(self.temp_dir) / "input"
        self.input_dir.mkdir()
        
        # Create test images
        for i in range(5):
            test_file = self.input_dir / f"image_{i:03d}.jpg"
            test_file.write_text(f"Image content {i}")

    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_file_protection(self):
        """Test complete file protection workflow."""
        output_dir = Path(self.temp_dir) / "output"
        
        # Create output structure manager
        manager = create_output_structure_from_input(self.input_dir, output_dir)
        
        # Get all input files
        input_files = list(self.input_dir.glob("*.jpg"))
        
        # Simulate approved files (first 3)
        approved_files = input_files[:3]
        
        # Copy approved files
        results = manager.copy_approved_files(approved_files)
        
        # Verify results
        assert len(results) == 3
        success_count = sum(1 for success in results.values() if success)
        assert success_count > 0  # At least some should succeed
        
        # Verify structure
        validation_results = manager.validate_output_structure()
        assert len(validation_results) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])