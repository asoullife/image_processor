"""File management and folder scanning utilities for Adobe Stock Image Processor."""

import os
import shutil
import hashlib
from typing import List, Set, Optional, Tuple
from pathlib import Path
import logging
from PIL import Image
from PIL.ExifTags import TAGS
import mimetypes

from backend.utils.logger import get_logger


class FileManager:
    """File manager for scanning, organizing, and copying image files."""
    
    # Supported image extensions
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    
    def __init__(self, images_per_folder: int = 200):
        """Initialize FileManager.
        
        Args:
            images_per_folder: Number of images per output subfolder.
        """
        self.images_per_folder = images_per_folder
        self.logger = get_logger('FileManager')
        
    def scan_images(self, input_folder: str) -> List[str]:
        """Recursively scan folder for supported image files.
        
        Args:
            input_folder: Path to input folder to scan.
            
        Returns:
            List[str]: List of absolute paths to image files.
            
        Raises:
            FileNotFoundError: If input folder doesn't exist.
            PermissionError: If folder is not accessible.
        """
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        if not os.path.isdir(input_folder):
            raise ValueError(f"Path is not a directory: {input_folder}")
        
        try:
            os.listdir(input_folder)
        except PermissionError as e:
            raise PermissionError(f"Cannot access input folder: {input_folder}") from e
        
        self.logger.info(f"Scanning for images in: {input_folder}")
        image_files = []
        
        try:
            for root, dirs, files in os.walk(input_folder):
                for file in files:
                    if self._is_supported_image(file):
                        full_path = os.path.abspath(os.path.join(root, file))
                        if self._validate_image_file(full_path):
                            image_files.append(full_path)
                        else:
                            self.logger.warning(f"Skipping invalid image file: {full_path}")
        
        except Exception as e:
            self.logger.error(f"Error scanning folder {input_folder}: {e}")
            raise
        
        self.logger.info(f"Found {len(image_files)} valid image files")
        return sorted(image_files)
    
    def _is_supported_image(self, filename: str) -> bool:
        """Check if file has supported image extension.
        
        Args:
            filename: Name of the file.
            
        Returns:
            bool: True if file has supported extension.
        """
        return Path(filename).suffix in self.SUPPORTED_EXTENSIONS
    
    def _validate_image_file(self, file_path: str) -> bool:
        """Validate that file is a readable image.
        
        Args:
            file_path: Path to image file.
            
        Returns:
            bool: True if file is valid image.
        """
        try:
            # Check file size (skip empty files)
            if os.path.getsize(file_path) == 0:
                return False
            
            # Try to open with PIL to validate
            with Image.open(file_path) as img:
                img.verify()
            
            # Check MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and not mime_type.startswith('image/'):
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Invalid image file {file_path}: {e}")
            return False
    
    def organize_output(self, approved_images: List[str], output_folder: str) -> dict:
        """Organize approved images into numbered subfolders.
        
        Args:
            approved_images: List of paths to approved images.
            output_folder: Base output folder path.
            
        Returns:
            dict: Organization results with statistics.
            
        Raises:
            ValueError: If output folder creation fails.
            PermissionError: If insufficient permissions.
        """
        if not approved_images:
            self.logger.warning("No approved images to organize")
            return {
                'total_images': 0,
                'folders_created': 0,
                'successful_copies': 0,
                'failed_copies': 0,
                'folder_structure': {}
            }
        
        self.logger.info(f"Organizing {len(approved_images)} images into: {output_folder}")
        
        # Create base output folder
        try:
            os.makedirs(output_folder, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create output folder {output_folder}: {e}")
        
        results = {
            'total_images': len(approved_images),
            'folders_created': 0,
            'successful_copies': 0,
            'failed_copies': 0,
            'folder_structure': {}
        }
        
        current_folder_num = 1
        current_folder_count = 0
        current_folder_path = None
        
        for i, image_path in enumerate(approved_images):
            # Create new subfolder if needed
            if current_folder_count == 0:
                current_folder_path = os.path.join(output_folder, str(current_folder_num))
                try:
                    os.makedirs(current_folder_path, exist_ok=True)
                    results['folders_created'] += 1
                    results['folder_structure'][str(current_folder_num)] = []
                    self.logger.debug(f"Created folder: {current_folder_path}")
                except Exception as e:
                    self.logger.error(f"Failed to create folder {current_folder_path}: {e}")
                    results['failed_copies'] += 1
                    continue
            
            # Copy image to current folder
            filename = os.path.basename(image_path)
            destination_path = os.path.join(current_folder_path, filename)
            
            # Handle filename conflicts
            destination_path = self._resolve_filename_conflict(destination_path)
            
            if self.copy_with_verification(image_path, destination_path):
                results['successful_copies'] += 1
                results['folder_structure'][str(current_folder_num)].append(filename)
                current_folder_count += 1
            else:
                results['failed_copies'] += 1
                continue
            
            # Move to next folder if current is full
            if current_folder_count >= self.images_per_folder:
                current_folder_num += 1
                current_folder_count = 0
        
        self.logger.info(f"Organization complete: {results['successful_copies']} images copied to {results['folders_created']} folders")
        
        if results['failed_copies'] > 0:
            self.logger.warning(f"{results['failed_copies']} images failed to copy")
        
        return results
    
    def _resolve_filename_conflict(self, destination_path: str) -> str:
        """Resolve filename conflicts by adding counter.
        
        Args:
            destination_path: Original destination path.
            
        Returns:
            str: Resolved path without conflicts.
        """
        if not os.path.exists(destination_path):
            return destination_path
        
        base_path = Path(destination_path)
        base_name = base_path.stem
        extension = base_path.suffix
        parent_dir = base_path.parent
        
        counter = 1
        while True:
            new_name = f"{base_name}_{counter}{extension}"
            new_path = parent_dir / new_name
            if not os.path.exists(new_path):
                return str(new_path)
            counter += 1
    
    def copy_with_verification(self, source: str, destination: str) -> bool:
        """Copy file with integrity verification.
        
        Args:
            source: Source file path.
            destination: Destination file path.
            
        Returns:
            bool: True if copy was successful and verified.
        """
        try:
            # Check source file exists and is readable
            if not os.path.exists(source):
                self.logger.error(f"Source file not found: {source}")
                return False
            
            if not os.access(source, os.R_OK):
                self.logger.error(f"Source file not readable: {source}")
                return False
            
            # Get source file hash before copying
            source_hash = self._calculate_file_hash(source)
            if not source_hash:
                return False
            
            # Create destination directory if needed
            dest_dir = os.path.dirname(destination)
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)
            
            # Copy file
            shutil.copy2(source, destination)
            
            # Verify copy integrity
            dest_hash = self._calculate_file_hash(destination)
            if source_hash != dest_hash:
                self.logger.error(f"Copy verification failed for {source} -> {destination}")
                # Clean up failed copy
                try:
                    os.remove(destination)
                except:
                    pass
                return False
            
            # Verify file sizes match
            source_size = os.path.getsize(source)
            dest_size = os.path.getsize(destination)
            if source_size != dest_size:
                self.logger.error(f"File size mismatch for {source} -> {destination}")
                try:
                    os.remove(destination)
                except:
                    pass
                return False
            
            self.logger.debug(f"Successfully copied and verified: {os.path.basename(source)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to copy {source} -> {destination}: {e}")
            # Clean up partial copy
            try:
                if os.path.exists(destination):
                    os.remove(destination)
            except:
                pass
            return False
    
    def _calculate_file_hash(self, file_path: str) -> Optional[str]:
        """Calculate MD5 hash of file.
        
        Args:
            file_path: Path to file.
            
        Returns:
            str or None: MD5 hash or None if error.
        """
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return None
    
    def get_image_metadata(self, image_path: str) -> dict:
        """Extract metadata from image file.
        
        Args:
            image_path: Path to image file.
            
        Returns:
            dict: Image metadata including EXIF data.
        """
        metadata = {
            'filename': os.path.basename(image_path),
            'file_path': image_path,
            'file_size': 0,
            'dimensions': (0, 0),
            'format': None,
            'mode': None,
            'exif': {}
        }
        
        try:
            # Get file size
            metadata['file_size'] = os.path.getsize(image_path)
            
            # Open image and get basic info
            with Image.open(image_path) as img:
                metadata['dimensions'] = img.size
                metadata['format'] = img.format
                metadata['mode'] = img.mode
                
                # Extract EXIF data
                if hasattr(img, '_getexif') and img._getexif() is not None:
                    exif_data = img._getexif()
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        metadata['exif'][tag] = value
                
        except Exception as e:
            self.logger.warning(f"Failed to extract metadata from {image_path}: {e}")
        
        return metadata
    
    def validate_output_structure(self, output_folder: str) -> dict:
        """Validate output folder structure and count images.
        
        Args:
            output_folder: Path to output folder.
            
        Returns:
            dict: Validation results.
        """
        if not os.path.exists(output_folder):
            return {
                'valid': False,
                'error': 'Output folder does not exist',
                'total_images': 0,
                'folders': {}
            }
        
        results = {
            'valid': True,
            'error': None,
            'total_images': 0,
            'folders': {}
        }
        
        try:
            for item in os.listdir(output_folder):
                item_path = os.path.join(output_folder, item)
                if os.path.isdir(item_path) and item.isdigit():
                    folder_num = int(item)
                    images = self.scan_images(item_path)
                    results['folders'][folder_num] = {
                        'image_count': len(images),
                        'images': [os.path.basename(img) for img in images]
                    }
                    results['total_images'] += len(images)
        
        except Exception as e:
            results['valid'] = False
            results['error'] = str(e)
        
        return results
    
    def cleanup_empty_folders(self, base_folder: str) -> int:
        """Remove empty folders from directory tree.
        
        Args:
            base_folder: Base folder to clean up.
            
        Returns:
            int: Number of folders removed.
        """
        removed_count = 0
        
        try:
            for root, dirs, files in os.walk(base_folder, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        if not os.listdir(dir_path):  # Empty directory
                            os.rmdir(dir_path)
                            removed_count += 1
                            self.logger.debug(f"Removed empty folder: {dir_path}")
                    except OSError:
                        # Folder not empty or permission issue
                        pass
        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} empty folders")
        
        return removed_count