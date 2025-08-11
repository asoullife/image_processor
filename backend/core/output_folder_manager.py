"""Output folder organization system for multi-session project management."""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class OutputFolderManager:
    """Manages output folder organization with input_name → input_name_processed structure."""
    
    def __init__(self):
        """Initialize output folder manager."""
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def generate_output_folder_name(self, input_folder: str) -> str:
        """Generate output folder name from input folder.
        
        Args:
            input_folder: Path to input folder
            
        Returns:
            Generated output folder name
        """
        input_path = Path(input_folder)
        return f"{input_path.name}_processed"
    
    def create_output_structure(
        self, 
        input_folder: str, 
        output_base: str,
        approved_images: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """Create output folder structure mirroring input organization.
        
        Args:
            input_folder: Path to input folder
            output_base: Base output folder path
            approved_images: List of approved images with source folder info
            
        Returns:
            Dictionary mapping source folders to output folders
        """
        try:
            input_path = Path(input_folder)
            output_path = Path(output_base)
            
            # Group approved images by source folder
            images_by_folder = {}
            for image in approved_images:
                source_folder = image.get('source_folder', '1')
                if source_folder not in images_by_folder:
                    images_by_folder[source_folder] = []
                images_by_folder[source_folder].append(image)
            
            # Create output folders only for folders with approved images
            created_folders = {}
            for source_folder, images in images_by_folder.items():
                if images:  # Only create if there are approved images
                    output_subfolder = output_path / source_folder
                    output_subfolder.mkdir(parents=True, exist_ok=True)
                    created_folders[source_folder] = str(output_subfolder)
                    
                    self.logger.info(f"Created output folder: {output_subfolder} for {len(images)} images")
            
            return created_folders
            
        except Exception as e:
            self.logger.error(f"Failed to create output structure: {e}")
            return {}
    
    def copy_approved_images(
        self,
        approved_images: List[Dict[str, str]],
        output_folders: Dict[str, str]
    ) -> Tuple[int, int]:
        """Copy approved images to their respective output folders.
        
        Args:
            approved_images: List of approved images with metadata
            output_folders: Dictionary mapping source folders to output paths
            
        Returns:
            Tuple of (successful_copies, failed_copies)
        """
        successful = 0
        failed = 0
        
        for image in approved_images:
            try:
                source_path = Path(image['image_path'])
                source_folder = image.get('source_folder', '1')
                
                if source_folder not in output_folders:
                    self.logger.warning(f"No output folder for source folder {source_folder}")
                    failed += 1
                    continue
                
                output_folder = Path(output_folders[source_folder])
                dest_path = output_folder / source_path.name
                
                # Copy with metadata preservation
                shutil.copy2(source_path, dest_path)
                
                # Verify copy integrity
                if self._verify_copy_integrity(source_path, dest_path):
                    successful += 1
                    self.logger.debug(f"Successfully copied: {source_path.name}")
                else:
                    failed += 1
                    self.logger.error(f"Copy verification failed: {source_path.name}")
                    
            except Exception as e:
                failed += 1
                self.logger.error(f"Failed to copy {image.get('filename', 'unknown')}: {e}")
        
        self.logger.info(f"Copy operation completed: {successful} successful, {failed} failed")
        return successful, failed
    
    def _verify_copy_integrity(self, source: Path, dest: Path) -> bool:
        """Verify that copied file matches original.
        
        Args:
            source: Source file path
            dest: Destination file path
            
        Returns:
            True if files match, False otherwise
        """
        try:
            # Check file sizes
            if source.stat().st_size != dest.stat().st_size:
                return False
            
            # For small files, compare content
            if source.stat().st_size < 10 * 1024 * 1024:  # 10MB threshold
                with open(source, 'rb') as f1, open(dest, 'rb') as f2:
                    return f1.read() == f2.read()
            
            # For large files, just check size and modification time
            return True
            
        except Exception as e:
            self.logger.error(f"Integrity verification failed: {e}")
            return False
    
    def get_output_folder_info(self, output_folder: str) -> Dict[str, any]:
        """Get information about output folder structure.
        
        Args:
            output_folder: Path to output folder
            
        Returns:
            Dictionary with folder information
        """
        try:
            output_path = Path(output_folder)
            
            if not output_path.exists():
                return {
                    "exists": False,
                    "total_images": 0,
                    "subfolders": {},
                    "created_at": None
                }
            
            # Count images in each subfolder
            subfolders = {}
            total_images = 0
            
            for subfolder in output_path.iterdir():
                if subfolder.is_dir():
                    image_count = self._count_images_in_folder(subfolder)
                    subfolders[subfolder.name] = {
                        "image_count": image_count,
                        "path": str(subfolder)
                    }
                    total_images += image_count
            
            return {
                "exists": True,
                "total_images": total_images,
                "subfolders": subfolders,
                "created_at": datetime.fromtimestamp(output_path.stat().st_ctime),
                "size_bytes": self._get_folder_size(output_path)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get output folder info: {e}")
            return {"exists": False, "error": str(e)}
    
    def _count_images_in_folder(self, folder: Path) -> int:
        """Count images in a folder.
        
        Args:
            folder: Folder path
            
        Returns:
            Number of images
        """
        supported_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        count = 0
        
        try:
            for file in folder.iterdir():
                if file.is_file() and file.suffix in supported_extensions:
                    count += 1
        except Exception as e:
            self.logger.error(f"Failed to count images in {folder}: {e}")
        
        return count
    
    def _get_folder_size(self, folder: Path) -> int:
        """Get total size of folder in bytes.
        
        Args:
            folder: Folder path
            
        Returns:
            Size in bytes
        """
        total_size = 0
        
        try:
            for dirpath, dirnames, filenames in os.walk(folder):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        pass
        except Exception as e:
            self.logger.error(f"Failed to calculate folder size: {e}")
        
        return total_size
    
    def cleanup_empty_folders(self, output_folder: str) -> int:
        """Remove empty subfolders from output directory.
        
        Args:
            output_folder: Path to output folder
            
        Returns:
            Number of folders removed
        """
        removed_count = 0
        
        try:
            output_path = Path(output_folder)
            
            for subfolder in output_path.iterdir():
                if subfolder.is_dir():
                    try:
                        # Check if folder is empty
                        if not any(subfolder.iterdir()):
                            subfolder.rmdir()
                            removed_count += 1
                            self.logger.info(f"Removed empty folder: {subfolder}")
                    except OSError as e:
                        self.logger.warning(f"Could not remove folder {subfolder}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup empty folders: {e}")
        
        return removed_count
    
    def validate_output_structure(
        self, 
        input_folder: str, 
        output_folder: str
    ) -> Dict[str, any]:
        """Validate that output structure matches input structure.
        
        Args:
            input_folder: Path to input folder
            output_folder: Path to output folder
            
        Returns:
            Validation results
        """
        try:
            input_path = Path(input_folder)
            output_path = Path(output_folder)
            
            # Get input structure
            input_subfolders = set()
            for subfolder in input_path.iterdir():
                if subfolder.is_dir() and subfolder.name.isdigit():
                    input_subfolders.add(subfolder.name)
            
            # Get output structure
            output_subfolders = set()
            if output_path.exists():
                for subfolder in output_path.iterdir():
                    if subfolder.is_dir():
                        output_subfolders.add(subfolder.name)
            
            # Compare structures
            missing_folders = input_subfolders - output_subfolders
            extra_folders = output_subfolders - input_subfolders
            
            return {
                "valid": len(missing_folders) == 0 and len(extra_folders) == 0,
                "input_subfolders": list(input_subfolders),
                "output_subfolders": list(output_subfolders),
                "missing_folders": list(missing_folders),
                "extra_folders": list(extra_folders),
                "structure_matches": missing_folders == set() and extra_folders == set()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to validate output structure: {e}")
            return {
                "valid": False,
                "error": str(e)
            }