"""Safe file operations for image processing without modifying originals."""

import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import contextmanager
import shutil

from .file_protection import FileIntegrityProtector, FileProtectionError

logger = logging.getLogger(__name__)

class SafeImageProcessor:
    """Provides safe image processing operations that never modify original files."""

    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize safe image processor.
        
        Args:
            temp_dir: Optional custom temporary directory
        """
        self.protector = FileIntegrityProtector(temp_dir)
        self.processing_cache: Dict[str, Any] = {}

    @contextmanager
    def safe_image_analysis(self, image_path: Union[str, Path], 
                          max_analysis_size: Tuple[int, int] = (2048, 2048)):
        """Context manager for safe image analysis without modifying original.
        
        Args:
            image_path: Path to original image
            max_analysis_size: Maximum size for analysis (width, height)
            
        Yields:
            Tuple of (original_path, analysis_path) where analysis_path is safe to modify
        """
        analysis_path = None
        try:
            image_path = Path(image_path)
            
            # Create protected copy first
            protected_path = self.protector.create_protected_copy(image_path)
            
            # Create analysis copy with size optimization
            with self.protector.temporary_resize(protected_path, max_analysis_size) as resized_path:
                # Create working copy for analysis
                analysis_path = self.protector.create_safe_working_copy(resized_path, "_analysis")
                
                logger.debug(f"Safe analysis setup: {image_path} -> {analysis_path}")
                yield str(image_path), analysis_path
                
        except Exception as e:
            logger.error(f"Failed to setup safe image analysis for {image_path}: {e}")
            raise FileProtectionError(f"Safe analysis setup failed: {e}")
        finally:
            # Cleanup is handled by protector context manager
            pass

    def safe_thumbnail_generation(self, image_path: Union[str, Path], 
                                thumbnail_size: Tuple[int, int] = (200, 200),
                                output_path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """Generate thumbnail without modifying original image.
        
        Args:
            image_path: Path to original image
            thumbnail_size: Thumbnail dimensions (width, height)
            output_path: Optional output path for thumbnail
            
        Returns:
            Path to generated thumbnail or None if failed
        """
        try:
            from PIL import Image
            
            image_path = Path(image_path)
            
            # Use safe analysis context
            with self.safe_image_analysis(image_path, max_analysis_size=(4096, 4096)) as (original, analysis):
                with Image.open(analysis) as img:
                    # Create thumbnail
                    img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                    
                    # Determine output path
                    if output_path:
                        thumb_path = Path(output_path)
                    else:
                        thumb_path = Path(analysis).parent / f"{Path(analysis).stem}_thumb.jpg"
                    
                    # Save thumbnail
                    thumb_path.parent.mkdir(parents=True, exist_ok=True)
                    img.save(thumb_path, "JPEG", quality=85, optimize=True)
                    
                    logger.debug(f"Thumbnail generated: {image_path} -> {thumb_path}")
                    return str(thumb_path)
                    
        except ImportError:
            logger.warning("PIL not available for thumbnail generation")
            return None
        except Exception as e:
            logger.error(f"Failed to generate thumbnail for {image_path}: {e}")
            return None

    def safe_metadata_extraction(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata without modifying original image.
        
        Args:
            image_path: Path to original image
            
        Returns:
            Dictionary containing extracted metadata
        """
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            
            image_path = Path(image_path)
            metadata = {
                "file_path": str(image_path),
                "file_size": image_path.stat().st_size,
                "file_name": image_path.name,
                "file_extension": image_path.suffix.lower()
            }
            
            # Use protected copy for metadata extraction
            protected_path = self.protector.create_protected_copy(image_path)
            
            with Image.open(protected_path) as img:
                # Basic image info
                metadata.update({
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "format": img.format,
                    "has_transparency": img.mode in ("RGBA", "LA") or "transparency" in img.info
                })
                
                # EXIF data
                exif_data = {}
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        exif_data[tag] = value
                
                metadata["exif"] = exif_data
                
            logger.debug(f"Metadata extracted from {image_path}")
            return metadata
            
        except ImportError:
            logger.warning("PIL not available for metadata extraction")
            return {"error": "PIL not available"}
        except Exception as e:
            logger.error(f"Failed to extract metadata from {image_path}: {e}")
            return {"error": str(e)}

    def safe_image_conversion(self, image_path: Union[str, Path], 
                            output_format: str = "JPEG",
                            quality: int = 95) -> Optional[str]:
        """Convert image format without modifying original.
        
        Args:
            image_path: Path to original image
            output_format: Target format (JPEG, PNG, etc.)
            quality: Quality setting for lossy formats
            
        Returns:
            Path to converted image or None if failed
        """
        try:
            from PIL import Image
            
            image_path = Path(image_path)
            
            # Create working copy for conversion
            working_path = self.protector.create_safe_working_copy(image_path, "_convert")
            
            with Image.open(working_path) as img:
                # Convert format if needed
                if output_format.upper() == "JPEG" and img.mode in ("RGBA", "LA"):
                    # Convert to RGB for JPEG
                    img = img.convert("RGB")
                
                # Generate output path
                extension = ".jpg" if output_format.upper() == "JPEG" else f".{output_format.lower()}"
                output_path = Path(working_path).parent / f"{Path(working_path).stem}_converted{extension}"
                
                # Save converted image
                save_kwargs = {"format": output_format.upper()}
                if output_format.upper() == "JPEG":
                    save_kwargs.update({"quality": quality, "optimize": True})
                
                img.save(output_path, **save_kwargs)
                
                # Track for cleanup
                self.protector.temp_files.append(str(output_path))
                
                logger.debug(f"Image converted: {image_path} -> {output_path}")
                return str(output_path)
                
        except ImportError:
            logger.warning("PIL not available for image conversion")
            return None
        except Exception as e:
            logger.error(f"Failed to convert image {image_path}: {e}")
            return None

    def batch_safe_copy(self, source_files: List[Union[str, Path]], 
                       output_dir: Union[str, Path],
                       preserve_structure: bool = True,
                       verify_integrity: bool = True) -> Dict[str, bool]:
        """Safely copy multiple files with integrity verification.
        
        Args:
            source_files: List of source file paths
            output_dir: Output directory
            preserve_structure: Whether to preserve directory structure
            verify_integrity: Whether to verify file integrity
            
        Returns:
            Dictionary mapping source paths to success status
        """
        results = {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for source_file in source_files:
            try:
                source_path = Path(source_file)
                
                if preserve_structure:
                    # Preserve relative structure
                    # This assumes all files have a common base directory
                    # For now, just use filename
                    output_path = output_dir / source_path.name
                else:
                    output_path = output_dir / source_path.name
                
                # Perform atomic copy
                success = self.protector.atomic_copy_to_output(
                    source_path, output_path, verify_integrity
                )
                results[str(source_path)] = success
                
            except Exception as e:
                logger.error(f"Failed to copy {source_file}: {e}")
                results[str(source_file)] = False
        
        return results

    def create_processing_workspace(self, source_files: List[Union[str, Path]], 
                                  workspace_dir: Optional[Union[str, Path]] = None) -> str:
        """Create a safe processing workspace with protected copies.
        
        Args:
            source_files: List of source files to include in workspace
            workspace_dir: Optional workspace directory path
            
        Returns:
            Path to created workspace directory
        """
        try:
            if workspace_dir:
                workspace_path = Path(workspace_dir)
            else:
                workspace_path = Path(self.protector.temp_dir) / f"workspace_{os.getpid()}"
            
            workspace_path.mkdir(parents=True, exist_ok=True)
            
            # Create protected copies in workspace
            for source_file in source_files:
                source_path = Path(source_file)
                protected_copy = self.protector.create_protected_copy(source_path, preserve_structure=False)
                
                # Create symlink or copy to workspace
                workspace_file = workspace_path / source_path.name
                if not workspace_file.exists():
                    shutil.copy2(protected_copy, workspace_file)
            
            logger.info(f"Processing workspace created: {workspace_path}")
            return str(workspace_path)
            
        except Exception as e:
            raise FileProtectionError(f"Failed to create processing workspace: {e}")

    def validate_output_integrity(self, output_files: List[Union[str, Path]]) -> Dict[str, bool]:
        """Validate integrity of output files.
        
        Args:
            output_files: List of output file paths to validate
            
        Returns:
            Dictionary mapping file paths to validation status
        """
        results = {}
        
        for output_file in output_files:
            try:
                output_path = Path(output_file)
                
                if not output_path.exists():
                    results[str(output_path)] = False
                    continue
                
                # Basic validation: file exists and is readable
                try:
                    with open(output_path, 'rb') as f:
                        f.read(1024)  # Try to read first 1KB
                    results[str(output_path)] = True
                except Exception:
                    results[str(output_path)] = False
                    
            except Exception as e:
                logger.error(f"Failed to validate {output_file}: {e}")
                results[str(output_file)] = False
        
        return results

    def cleanup(self):
        """Clean up all temporary files and resources."""
        self.protector.cleanup_temp_files()
        self.processing_cache.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

# Utility functions for common safe operations
def safe_analyze_image(image_path: Union[str, Path], 
                      analysis_function: callable,
                      max_size: Tuple[int, int] = (2048, 2048)) -> Any:
    """Safely analyze an image without modifying the original.
    
    Args:
        image_path: Path to image to analyze
        analysis_function: Function to call with analysis image path
        max_size: Maximum size for analysis
        
    Returns:
        Result of analysis function
    """
    with SafeImageProcessor() as processor:
        with processor.safe_image_analysis(image_path, max_size) as (original, analysis):
            return analysis_function(analysis)

def safe_batch_process(source_files: List[Union[str, Path]], 
                      output_dir: Union[str, Path],
                      processing_function: callable,
                      verify_integrity: bool = True) -> Dict[str, Any]:
    """Safely process a batch of files.
    
    Args:
        source_files: List of source files
        output_dir: Output directory
        processing_function: Function to process each file
        verify_integrity: Whether to verify integrity
        
    Returns:
        Dictionary mapping source files to processing results
    """
    results = {}
    
    with SafeImageProcessor() as processor:
        for source_file in source_files:
            try:
                with processor.safe_image_analysis(source_file) as (original, analysis):
                    result = processing_function(original, analysis)
                    results[str(source_file)] = result
            except Exception as e:
                logger.error(f"Failed to process {source_file}: {e}")
                results[str(source_file)] = {"error": str(e)}
    
    return results