"""Output folder structure management for Adobe Stock Image Processor."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json
from datetime import datetime

from .file_protection import FileIntegrityProtector, safe_copy_with_verification

logger = logging.getLogger(__name__)

class OutputStructureManager:
    """Manages output folder structure mirroring input organization."""

    def __init__(self, input_base_dir: Union[str, Path], 
                 output_base_dir: Union[str, Path]):
        """Initialize output structure manager.
        
        Args:
            input_base_dir: Base input directory
            output_base_dir: Base output directory
        """
        self.input_base_dir = Path(input_base_dir)
        self.output_base_dir = Path(output_base_dir)
        self.structure_map: Dict[str, str] = {}
        self.created_dirs: List[str] = []
        
        # Ensure base directories exist
        if not self.input_base_dir.exists():
            raise ValueError(f"Input directory does not exist: {self.input_base_dir}")
        
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def create_mirrored_structure(self) -> Dict[str, str]:
        """Create output directory structure mirroring input.
        
        Returns:
            Dictionary mapping input paths to output paths
        """
        try:
            logger.info(f"Creating mirrored structure: {self.input_base_dir} -> {self.output_base_dir}")
            
            # Walk through input directory
            for root, dirs, files in os.walk(self.input_base_dir):
                root_path = Path(root)
                
                # Calculate relative path from input base
                try:
                    relative_path = root_path.relative_to(self.input_base_dir)
                except ValueError:
                    # Skip if path is not relative to input base
                    continue
                
                # Create corresponding output directory
                output_dir = self.output_base_dir / relative_path
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Track created directory
                if str(output_dir) not in self.created_dirs:
                    self.created_dirs.append(str(output_dir))
                
                # Map directory
                self.structure_map[str(root_path)] = str(output_dir)
                
                # Map files
                for file in files:
                    input_file = root_path / file
                    output_file = output_dir / file
                    self.structure_map[str(input_file)] = str(output_file)
            
            logger.info(f"Created {len(self.created_dirs)} directories, mapped {len(self.structure_map)} paths")
            return self.structure_map.copy()
            
        except Exception as e:
            logger.error(f"Failed to create mirrored structure: {e}")
            raise

    def get_output_path(self, input_path: Union[str, Path]) -> Optional[str]:
        """Get corresponding output path for an input path.
        
        Args:
            input_path: Input file or directory path
            
        Returns:
            Corresponding output path or None if not found
        """
        input_path_str = str(Path(input_path))
        
        # Check direct mapping first
        if input_path_str in self.structure_map:
            return self.structure_map[input_path_str]
        
        # Try to calculate relative path
        try:
            input_path = Path(input_path)
            relative_path = input_path.relative_to(self.input_base_dir)
            output_path = self.output_base_dir / relative_path
            return str(output_path)
        except ValueError:
            return None

    def organize_approved_files(self, approved_files: List[Union[str, Path]], 
                              subfolder_size: int = 200) -> Dict[str, List[str]]:
        """Organize approved files into subfolders with specified size limit.
        
        Args:
            approved_files: List of approved file paths
            subfolder_size: Maximum number of files per subfolder
            
        Returns:
            Dictionary mapping subfolder names to file lists
        """
        try:
            organization = {}
            
            # Group files by their parent directory
            dir_groups: Dict[str, List[str]] = {}
            
            for file_path in approved_files:
                file_path = Path(file_path)
                parent_dir = str(file_path.parent)
                
                if parent_dir not in dir_groups:
                    dir_groups[parent_dir] = []
                dir_groups[parent_dir].append(str(file_path))
            
            # Organize each directory group
            for parent_dir, files in dir_groups.items():
                # Get corresponding output directory
                output_parent = self.get_output_path(parent_dir)
                if not output_parent:
                    logger.warning(f"No output path found for {parent_dir}")
                    continue
                
                # Create subfolders if needed
                if len(files) <= subfolder_size:
                    # All files fit in one folder
                    organization[output_parent] = files
                else:
                    # Split into multiple subfolders
                    for i in range(0, len(files), subfolder_size):
                        batch = files[i:i + subfolder_size]
                        subfolder_name = f"{output_parent}_batch_{i // subfolder_size + 1:03d}"
                        organization[subfolder_name] = batch
            
            logger.info(f"Organized {len(approved_files)} files into {len(organization)} folders")
            return organization
            
        except Exception as e:
            logger.error(f"Failed to organize approved files: {e}")
            raise

    def copy_approved_files(self, approved_files: List[Union[str, Path]], 
                          subfolder_size: int = 200,
                          verify_integrity: bool = True) -> Dict[str, bool]:
        """Copy approved files to organized output structure.
        
        Args:
            approved_files: List of approved file paths
            subfolder_size: Maximum files per subfolder
            verify_integrity: Whether to verify file integrity
            
        Returns:
            Dictionary mapping source files to copy success status
        """
        try:
            results = {}
            
            # Organize files into subfolders
            organization = self.organize_approved_files(approved_files, subfolder_size)
            
            # Copy files to their organized locations
            for output_folder, file_list in organization.items():
                # Ensure output folder exists
                output_folder_path = Path(output_folder)
                output_folder_path.mkdir(parents=True, exist_ok=True)
                
                for source_file in file_list:
                    try:
                        source_path = Path(source_file)
                        output_path = output_folder_path / source_path.name
                        
                        # Perform safe copy with verification
                        success = safe_copy_with_verification(source_path, output_path)
                        results[str(source_path)] = success
                        
                        if success:
                            logger.debug(f"Copied: {source_path} -> {output_path}")
                        else:
                            logger.error(f"Failed to copy: {source_path}")
                            
                    except Exception as e:
                        logger.error(f"Error copying {source_file}: {e}")
                        results[str(source_file)] = False
            
            success_count = sum(1 for success in results.values() if success)
            logger.info(f"Successfully copied {success_count}/{len(approved_files)} files")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to copy approved files: {e}")
            raise

    def create_processing_summary(self, processing_results: Dict[str, any],
                                output_file: Optional[Union[str, Path]] = None) -> str:
        """Create processing summary file.
        
        Args:
            processing_results: Dictionary of processing results
            output_file: Optional output file path
            
        Returns:
            Path to created summary file
        """
        try:
            if output_file:
                summary_path = Path(output_file)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_path = self.output_base_dir / f"processing_summary_{timestamp}.json"
            
            summary_data = {
                "timestamp": datetime.now().isoformat(),
                "input_directory": str(self.input_base_dir),
                "output_directory": str(self.output_base_dir),
                "structure_mapping": self.structure_map,
                "created_directories": self.created_dirs,
                "processing_results": processing_results,
                "statistics": {
                    "total_files": len(processing_results),
                    "successful_copies": sum(1 for result in processing_results.values() if result),
                    "failed_copies": sum(1 for result in processing_results.values() if not result),
                    "directories_created": len(self.created_dirs)
                }
            }
            
            # Ensure parent directory exists
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write summary file
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processing summary created: {summary_path}")
            return str(summary_path)
            
        except Exception as e:
            logger.error(f"Failed to create processing summary: {e}")
            raise

    def validate_output_structure(self) -> Dict[str, bool]:
        """Validate that output structure matches input structure.
        
        Returns:
            Dictionary mapping paths to validation status
        """
        validation_results = {}
        
        try:
            # Check that all mapped directories exist
            for input_path, output_path in self.structure_map.items():
                input_path_obj = Path(input_path)
                output_path_obj = Path(output_path)
                
                if input_path_obj.is_dir():
                    # Validate directory exists
                    validation_results[output_path] = output_path_obj.exists() and output_path_obj.is_dir()
                else:
                    # For files, just check if parent directory exists
                    validation_results[output_path] = output_path_obj.parent.exists()
            
            success_count = sum(1 for valid in validation_results.values() if valid)
            total_count = len(validation_results)
            
            logger.info(f"Structure validation: {success_count}/{total_count} paths valid")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Failed to validate output structure: {e}")
            return {}

    def cleanup_empty_directories(self) -> int:
        """Remove empty directories from output structure.
        
        Returns:
            Number of directories removed
        """
        removed_count = 0
        
        try:
            # Walk through output directory in reverse order (deepest first)
            for root, dirs, files in os.walk(self.output_base_dir, topdown=False):
                root_path = Path(root)
                
                # Skip base directory
                if root_path == self.output_base_dir:
                    continue
                
                # Check if directory is empty
                try:
                    if not any(root_path.iterdir()):
                        root_path.rmdir()
                        removed_count += 1
                        logger.debug(f"Removed empty directory: {root_path}")
                except OSError:
                    # Directory not empty or permission error
                    pass
            
            if removed_count > 0:
                logger.info(f"Removed {removed_count} empty directories")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup empty directories: {e}")
            return 0

    def get_structure_statistics(self) -> Dict[str, int]:
        """Get statistics about the output structure.
        
        Returns:
            Dictionary containing structure statistics
        """
        try:
            stats = {
                "total_mapped_paths": len(self.structure_map),
                "created_directories": len(self.created_dirs),
                "input_directories": 0,
                "input_files": 0,
                "output_directories": 0,
                "output_files": 0
            }
            
            # Count input paths
            for input_path in self.structure_map.keys():
                path_obj = Path(input_path)
                if path_obj.is_dir():
                    stats["input_directories"] += 1
                else:
                    stats["input_files"] += 1
            
            # Count existing output paths
            for output_path in self.structure_map.values():
                path_obj = Path(output_path)
                if path_obj.exists():
                    if path_obj.is_dir():
                        stats["output_directories"] += 1
                    else:
                        stats["output_files"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get structure statistics: {e}")
            return {}

    async def copy_approved_image(self, source_path: Union[str, Path], 
                                source_folder: str) -> bool:
        """Copy a single approved image to output folder (for human review).
        
        Args:
            source_path: Path to source image
            source_folder: Source folder identifier (e.g., "1", "2", "3")
            
        Returns:
            True if copy was successful, False otherwise
        """
        try:
            source_path = Path(source_path)
            
            # Determine output subfolder path
            subfolder_path = self.output_base_dir / str(source_folder)
            subfolder_path.mkdir(parents=True, exist_ok=True)
            
            # Copy file with integrity verification
            filename = source_path.name
            dest_path = subfolder_path / filename
            
            success = safe_copy_with_verification(source_path, dest_path)
            
            if success:
                logger.info(f"Human review: Copied approved image {filename} to {subfolder_path}")
            else:
                logger.error(f"Human review: Failed to copy {filename}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error copying approved image {source_path}: {e}")
            return False

    async def remove_rejected_image(self, source_path: Union[str, Path], 
                                  source_folder: str) -> bool:
        """Remove a rejected image from output folder (for human review).
        
        Args:
            source_path: Path to source image
            source_folder: Source folder identifier (e.g., "1", "2", "3")
            
        Returns:
            True if removal was successful, False otherwise
        """
        try:
            source_path = Path(source_path)
            
            # Determine output file path
            subfolder_path = self.output_base_dir / str(source_folder)
            filename = source_path.name
            dest_path = subfolder_path / filename
            
            if dest_path.exists():
                dest_path.unlink()
                logger.info(f"Human review: Removed rejected image {filename} from {subfolder_path}")
                
                # Remove empty subfolder if no images left
                if subfolder_path.exists() and not any(subfolder_path.iterdir()):
                    subfolder_path.rmdir()
                    logger.info(f"Human review: Removed empty subfolder {subfolder_path}")
                
                return True
            else:
                logger.warning(f"Human review: Image {filename} not found in output folder")
                return True  # Consider success if file doesn't exist
                
        except Exception as e:
            logger.error(f"Error removing rejected image {source_path}: {e}")
            return False

# Utility functions
def create_output_structure_from_input(input_dir: Union[str, Path], 
                                     output_dir: Union[str, Path]) -> OutputStructureManager:
    """Create output structure manager and initialize structure.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        
    Returns:
        Configured OutputStructureManager
    """
    manager = OutputStructureManager(input_dir, output_dir)
    manager.create_mirrored_structure()
    return manager

def organize_and_copy_files(input_dir: Union[str, Path], 
                          output_dir: Union[str, Path],
                          approved_files: List[Union[str, Path]],
                          subfolder_size: int = 200) -> Dict[str, bool]:
    """Organize and copy approved files with mirrored structure.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        approved_files: List of approved files to copy
        subfolder_size: Maximum files per subfolder
        
    Returns:
        Dictionary mapping source files to copy success status
    """
    manager = create_output_structure_from_input(input_dir, output_dir)
    return manager.copy_approved_files(approved_files, subfolder_size)