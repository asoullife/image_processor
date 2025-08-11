"""File integrity protection system for Adobe Stock Image Processor."""

import os
import shutil
import hashlib
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from contextlib import contextmanager
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class FileIntegrityError(Exception):
    """Exception raised when file integrity is compromised."""
    pass

class FileProtectionError(Exception):
    """Exception raised when file protection operations fail."""
    pass

class FileIntegrityProtector:
    """Provides strict file protection mechanisms preventing modification of source files."""

    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize file integrity protector.
        
        Args:
            temp_dir: Optional custom temporary directory path
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.temp_files: List[str] = []
        self.integrity_cache: Dict[str, str] = {}
        self.protected_files: Dict[str, str] = {}  # original_path -> temp_path
        
    def calculate_file_hash(self, file_path: Union[str, Path]) -> str:
        """Calculate SHA-256 hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            SHA-256 hash string
            
        Raises:
            FileIntegrityError: If file cannot be read
        """
        try:
            file_path = Path(file_path)
            hash_sha256 = hashlib.sha256()
            
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
                    
            return hash_sha256.hexdigest()
            
        except Exception as e:
            raise FileIntegrityError(f"Failed to calculate hash for {file_path}: {e}")

    def verify_file_integrity(self, file_path: Union[str, Path], expected_hash: str) -> bool:
        """Verify file integrity against expected hash.
        
        Args:
            file_path: Path to the file
            expected_hash: Expected SHA-256 hash
            
        Returns:
            True if integrity is maintained, False otherwise
        """
        try:
            current_hash = self.calculate_file_hash(file_path)
            return current_hash == expected_hash
        except FileIntegrityError:
            return False

    def create_protected_copy(self, source_path: Union[str, Path], 
                            preserve_structure: bool = True) -> str:
        """Create a protected copy of a file in temporary directory.
        
        Args:
            source_path: Path to source file
            preserve_structure: Whether to preserve directory structure in temp
            
        Returns:
            Path to protected copy
            
        Raises:
            FileProtectionError: If copy operation fails
        """
        try:
            source_path = Path(source_path)
            
            if not source_path.exists():
                raise FileProtectionError(f"Source file does not exist: {source_path}")
                
            # Calculate original hash for integrity verification
            original_hash = self.calculate_file_hash(source_path)
            self.integrity_cache[str(source_path)] = original_hash
            
            # Create temporary file path
            if preserve_structure:
                # Preserve directory structure in temp
                relative_path = source_path.name
                temp_subdir = Path(self.temp_dir) / "protected_files" / str(abs(hash(str(source_path.parent))))
                temp_subdir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_subdir / relative_path
            else:
                # Create unique temporary file
                temp_fd, temp_path = tempfile.mkstemp(
                    suffix=source_path.suffix,
                    prefix=f"protected_{source_path.stem}_",
                    dir=self.temp_dir
                )
                os.close(temp_fd)  # Close file descriptor, we'll use the path
                temp_path = Path(temp_path)
            
            # Copy file with integrity verification
            shutil.copy2(source_path, temp_path)
            
            # Verify copy integrity
            copy_hash = self.calculate_file_hash(temp_path)
            if copy_hash != original_hash:
                temp_path.unlink(missing_ok=True)
                raise FileProtectionError(f"Copy integrity verification failed for {source_path}")
            
            # Make copy read-only to prevent accidental modification
            temp_path.chmod(0o444)
            
            # Track temporary file for cleanup
            self.temp_files.append(str(temp_path))
            self.protected_files[str(source_path)] = str(temp_path)
            
            logger.debug(f"Created protected copy: {source_path} -> {temp_path}")
            return str(temp_path)
            
        except Exception as e:
            raise FileProtectionError(f"Failed to create protected copy of {source_path}: {e}")

    def create_safe_working_copy(self, source_path: Union[str, Path], 
                               working_suffix: str = "_working") -> str:
        """Create a safe working copy that can be modified for analysis.
        
        Args:
            source_path: Path to source file
            working_suffix: Suffix to add to working copy filename
            
        Returns:
            Path to working copy
            
        Raises:
            FileProtectionError: If copy operation fails
        """
        try:
            source_path = Path(source_path)
            
            # Get protected copy first
            if str(source_path) in self.protected_files:
                protected_path = self.protected_files[str(source_path)]
            else:
                protected_path = self.create_protected_copy(source_path)
            
            # Create working copy from protected copy
            protected_path = Path(protected_path)
            working_name = f"{protected_path.stem}{working_suffix}{protected_path.suffix}"
            working_path = protected_path.parent / working_name
            
            # Copy and make writable
            shutil.copy2(protected_path, working_path)
            working_path.chmod(0o644)  # Read-write for owner
            
            # Track for cleanup
            self.temp_files.append(str(working_path))
            
            logger.debug(f"Created working copy: {protected_path} -> {working_path}")
            return str(working_path)
            
        except Exception as e:
            raise FileProtectionError(f"Failed to create working copy of {source_path}: {e}")

    @contextmanager
    def temporary_resize(self, source_path: Union[str, Path], 
                        max_size: Tuple[int, int] = (1024, 1024)):
        """Context manager for temporary image resizing without affecting original.
        
        Args:
            source_path: Path to source image
            max_size: Maximum dimensions (width, height)
            
        Yields:
            Path to temporarily resized image
            
        Raises:
            FileProtectionError: If resize operation fails
        """
        temp_resized_path = None
        try:
            from PIL import Image
            
            source_path = Path(source_path)
            
            # Create working copy
            working_path = self.create_safe_working_copy(source_path, "_resize_temp")
            
            # Resize image
            with Image.open(working_path) as img:
                # Calculate new size maintaining aspect ratio
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Save resized version
                temp_resized_path = working_path.parent / f"{working_path.stem}_resized{working_path.suffix}"
                img.save(temp_resized_path, optimize=True, quality=85)
            
            # Track for cleanup
            self.temp_files.append(str(temp_resized_path))
            
            logger.debug(f"Created temporary resized image: {temp_resized_path}")
            yield str(temp_resized_path)
            
        except ImportError:
            logger.warning("PIL not available, using original image")
            yield str(source_path)
        except Exception as e:
            raise FileProtectionError(f"Failed to create temporary resize of {source_path}: {e}")
        finally:
            # Cleanup resized file
            if temp_resized_path and Path(temp_resized_path).exists():
                Path(temp_resized_path).unlink(missing_ok=True)
                if str(temp_resized_path) in self.temp_files:
                    self.temp_files.remove(str(temp_resized_path))

    def atomic_copy_to_output(self, source_path: Union[str, Path], 
                            output_path: Union[str, Path],
                            verify_integrity: bool = True) -> bool:
        """Perform atomic copy operation with rollback capability.
        
        Args:
            source_path: Path to source file
            output_path: Path to output destination
            verify_integrity: Whether to verify file integrity after copy
            
        Returns:
            True if copy successful, False otherwise
            
        Raises:
            FileProtectionError: If atomic copy fails
        """
        try:
            source_path = Path(source_path)
            output_path = Path(output_path)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create temporary output path for atomic operation
            temp_output = output_path.parent / f".{output_path.name}.tmp"
            
            try:
                # Get original hash if available
                original_hash = self.integrity_cache.get(str(source_path))
                if not original_hash:
                    original_hash = self.calculate_file_hash(source_path)
                
                # Copy to temporary location
                shutil.copy2(source_path, temp_output)
                
                # Verify integrity if requested
                if verify_integrity:
                    copy_hash = self.calculate_file_hash(temp_output)
                    if copy_hash != original_hash:
                        temp_output.unlink(missing_ok=True)
                        raise FileProtectionError("Integrity verification failed during atomic copy")
                
                # Atomic move to final location
                temp_output.replace(output_path)
                
                logger.debug(f"Atomic copy completed: {source_path} -> {output_path}")
                return True
                
            except Exception as e:
                # Rollback: remove temporary file
                temp_output.unlink(missing_ok=True)
                raise FileProtectionError(f"Atomic copy failed: {e}")
                
        except Exception as e:
            raise FileProtectionError(f"Failed atomic copy from {source_path} to {output_path}: {e}")

    def mirror_directory_structure(self, input_dir: Union[str, Path], 
                                 output_dir: Union[str, Path],
                                 create_only: bool = True) -> Dict[str, str]:
        """Mirror input directory structure in output directory.
        
        Args:
            input_dir: Source directory to mirror
            output_dir: Output directory to create structure in
            create_only: If True, only create directories without copying files
            
        Returns:
            Dictionary mapping input paths to output paths
            
        Raises:
            FileProtectionError: If directory mirroring fails
        """
        try:
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)
            
            if not input_dir.exists():
                raise FileProtectionError(f"Input directory does not exist: {input_dir}")
            
            structure_map = {}
            
            # Walk through input directory
            for root, dirs, files in os.walk(input_dir):
                root_path = Path(root)
                
                # Calculate relative path from input_dir
                relative_path = root_path.relative_to(input_dir)
                
                # Create corresponding output directory
                output_subdir = output_dir / relative_path
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                # Map directories
                structure_map[str(root_path)] = str(output_subdir)
                
                # Map files if not create_only
                if not create_only:
                    for file in files:
                        input_file = root_path / file
                        output_file = output_subdir / file
                        structure_map[str(input_file)] = str(output_file)
            
            logger.info(f"Mirrored directory structure: {input_dir} -> {output_dir}")
            return structure_map
            
        except Exception as e:
            raise FileProtectionError(f"Failed to mirror directory structure: {e}")

    def verify_source_integrity(self, source_paths: List[Union[str, Path]]) -> Dict[str, bool]:
        """Verify integrity of multiple source files.
        
        Args:
            source_paths: List of source file paths to verify
            
        Returns:
            Dictionary mapping file paths to integrity status
        """
        results = {}
        
        for source_path in source_paths:
            source_path = str(source_path)
            try:
                if source_path in self.integrity_cache:
                    expected_hash = self.integrity_cache[source_path]
                    results[source_path] = self.verify_file_integrity(source_path, expected_hash)
                else:
                    # First time seeing this file, calculate and cache hash
                    self.integrity_cache[source_path] = self.calculate_file_hash(source_path)
                    results[source_path] = True
            except Exception as e:
                logger.error(f"Failed to verify integrity of {source_path}: {e}")
                results[source_path] = False
        
        return results

    def save_integrity_manifest(self, manifest_path: Union[str, Path]) -> None:
        """Save integrity manifest to file for later verification.
        
        Args:
            manifest_path: Path to save manifest file
            
        Raises:
            FileProtectionError: If manifest save fails
        """
        try:
            manifest_path = Path(manifest_path)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            
            manifest_data = {
                "created_at": datetime.now().isoformat(),
                "file_hashes": self.integrity_cache.copy(),
                "protected_files": self.protected_files.copy()
            }
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest_data, f, indent=2)
            
            logger.info(f"Integrity manifest saved: {manifest_path}")
            
        except Exception as e:
            raise FileProtectionError(f"Failed to save integrity manifest: {e}")

    def load_integrity_manifest(self, manifest_path: Union[str, Path]) -> None:
        """Load integrity manifest from file.
        
        Args:
            manifest_path: Path to manifest file
            
        Raises:
            FileProtectionError: If manifest load fails
        """
        try:
            manifest_path = Path(manifest_path)
            
            if not manifest_path.exists():
                raise FileProtectionError(f"Manifest file does not exist: {manifest_path}")
            
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            
            self.integrity_cache.update(manifest_data.get("file_hashes", {}))
            self.protected_files.update(manifest_data.get("protected_files", {}))
            
            logger.info(f"Integrity manifest loaded: {manifest_path}")
            
        except Exception as e:
            raise FileProtectionError(f"Failed to load integrity manifest: {e}")

    def cleanup_temp_files(self) -> None:
        """Clean up all temporary files created by this protector."""
        cleaned_count = 0
        
        for temp_file in self.temp_files.copy():
            try:
                temp_path = Path(temp_file)
                if temp_path.exists():
                    # Remove read-only flag if present
                    temp_path.chmod(0o644)
                    temp_path.unlink()
                    cleaned_count += 1
                self.temp_files.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
        
        # Clean up empty temp directories
        try:
            temp_base = Path(self.temp_dir) / "protected_files"
            if temp_base.exists():
                # Remove empty subdirectories
                for subdir in temp_base.iterdir():
                    if subdir.is_dir() and not any(subdir.iterdir()):
                        subdir.rmdir()
                
                # Remove base directory if empty
                if not any(temp_base.iterdir()):
                    temp_base.rmdir()
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directories: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} temporary files")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_temp_files()

# Convenience functions for common operations
def safe_copy_with_verification(source_path: Union[str, Path], 
                              output_path: Union[str, Path]) -> bool:
    """Safely copy a file with integrity verification.
    
    Args:
        source_path: Source file path
        output_path: Output file path
        
    Returns:
        True if copy successful and verified
    """
    with FileIntegrityProtector() as protector:
        return protector.atomic_copy_to_output(source_path, output_path, verify_integrity=True)

def create_output_structure(input_dir: Union[str, Path], 
                          output_dir: Union[str, Path]) -> Dict[str, str]:
    """Create output directory structure mirroring input.
    
    Args:
        input_dir: Input directory to mirror
        output_dir: Output directory to create
        
    Returns:
        Dictionary mapping input paths to output paths
    """
    with FileIntegrityProtector() as protector:
        return protector.mirror_directory_structure(input_dir, output_dir, create_only=True)