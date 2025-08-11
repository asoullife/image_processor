"""
Thumbnail Generator
Generates and manages thumbnail images for web interface
"""

from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import hashlib
from PIL import Image, ImageOps
from uuid import UUID
import logging

from backend.database.models import ImageResult
from backend.api.schemas import ThumbnailData
from backend.utils.thai_translations import get_thai_rejection_reason

logger = logging.getLogger(__name__)

class ThumbnailGenerator:
    """Generate and manage thumbnail images for web interface"""
    
    def __init__(self, db: Session, thumbnail_dir: str = "backend/data/thumbnails"):
        self.db = db
        self.thumbnail_dir = Path(thumbnail_dir)
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        
        # Thumbnail settings
        self.thumbnail_size = (200, 200)
        self.thumbnail_quality = 85
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    async def get_session_thumbnails(
        self,
        session_id: str,
        page: int = 1,
        page_size: int = 100
    ) -> Dict[str, Any]:
        """Get thumbnail data for a session with pagination"""
        try:
            session_uuid = UUID(session_id)
            
            # Get paginated results
            offset = (page - 1) * page_size
            results = self.db.query(ImageResult).filter(
                ImageResult.session_id == session_uuid
            ).offset(offset).limit(page_size).all()
            
            total_count = self.db.query(ImageResult).filter(
                ImageResult.session_id == session_uuid
            ).count()
            
            thumbnails = []
            for result in results:
                thumbnail_data = await self._create_thumbnail_data(result)
                if thumbnail_data:
                    thumbnails.append(thumbnail_data)
            
            return {
                "thumbnails": thumbnails,
                "total_count": total_count,
                "page": page,
                "page_size": page_size,
                "total_pages": (total_count + page_size - 1) // page_size
            }
            
        except Exception as e:
            logger.error(f"Error getting session thumbnails: {str(e)}")
            return {
                "thumbnails": [],
                "total_count": 0,
                "page": page,
                "page_size": page_size,
                "total_pages": 0
            }
    
    async def get_thumbnail_url(self, image_id: str) -> Optional[str]:
        """Get thumbnail URL for a specific image"""
        try:
            image_uuid = UUID(image_id)
            
            result = self.db.query(ImageResult).filter(
                ImageResult.id == image_uuid
            ).first()
            
            if not result:
                return None
            
            # Generate thumbnail if it doesn't exist
            thumbnail_path = await self._ensure_thumbnail_exists(result)
            if thumbnail_path:
                return f"/api/thumbnails/{image_id}"
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting thumbnail URL: {str(e)}")
            return None
    
    async def get_thumbnail_file(self, image_id: str) -> Optional[Path]:
        """Get thumbnail file path for serving"""
        try:
            image_uuid = UUID(image_id)
            
            result = self.db.query(ImageResult).filter(
                ImageResult.id == image_uuid
            ).first()
            
            if not result:
                return None
            
            thumbnail_path = await self._ensure_thumbnail_exists(result)
            return thumbnail_path
            
        except Exception as e:
            logger.error(f"Error getting thumbnail file: {str(e)}")
            return None
    
    async def generate_thumbnails_batch(self, session_id: str, batch_size: int = 50) -> int:
        """Generate thumbnails for a session in batches"""
        try:
            session_uuid = UUID(session_id)
            
            # Get all results without thumbnails
            results = self.db.query(ImageResult).filter(
                ImageResult.session_id == session_uuid
            ).all()
            
            generated_count = 0
            for i in range(0, len(results), batch_size):
                batch = results[i:i + batch_size]
                
                for result in batch:
                    thumbnail_path = await self._ensure_thumbnail_exists(result)
                    if thumbnail_path:
                        generated_count += 1
                
                # Log progress
                logger.info(f"Generated thumbnails: {min(i + batch_size, len(results))}/{len(results)}")
            
            return generated_count
            
        except Exception as e:
            logger.error(f"Error generating thumbnails batch: {str(e)}")
            return 0
    
    async def cleanup_orphaned_thumbnails(self) -> int:
        """Clean up thumbnail files that no longer have corresponding database records"""
        try:
            cleaned_count = 0
            
            # Get all thumbnail files
            thumbnail_files = list(self.thumbnail_dir.glob("*.jpg"))
            
            for thumbnail_file in thumbnail_files:
                # Extract image ID from filename
                try:
                    image_id = thumbnail_file.stem
                    image_uuid = UUID(image_id)
                    
                    # Check if record exists
                    result = self.db.query(ImageResult).filter(
                        ImageResult.id == image_uuid
                    ).first()
                    
                    if not result:
                        # Remove orphaned thumbnail
                        thumbnail_file.unlink()
                        cleaned_count += 1
                        logger.info(f"Removed orphaned thumbnail: {thumbnail_file}")
                        
                except (ValueError, OSError) as e:
                    logger.warning(f"Error processing thumbnail file {thumbnail_file}: {str(e)}")
                    continue
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up thumbnails: {str(e)}")
            return 0
    
    async def _create_thumbnail_data(self, result: ImageResult) -> Optional[ThumbnailData]:
        """Create thumbnail data object for an image result"""
        try:
            # Ensure thumbnail exists
            thumbnail_path = await self._ensure_thumbnail_exists(result)
            if not thumbnail_path:
                return None
            
            # Convert rejection reasons to Thai
            thai_rejection_reasons = []
            if result.rejection_reasons:
                thai_rejection_reasons = [
                    get_thai_rejection_reason(reason) for reason in result.rejection_reasons
                ]
            
            # Extract confidence scores
            confidence_scores = {}
            if result.quality_scores:
                confidence_scores = {
                    "overall": result.quality_scores.get("overall_score", 0.0),
                    "sharpness": result.quality_scores.get("sharpness_score", 0.0),
                    "noise": result.quality_scores.get("noise_level", 0.0),
                    "exposure": result.quality_scores.get("exposure_score", 0.0)
                }
            
            return ThumbnailData(
                image_id=result.id,
                filename=result.filename,
                thumbnail_url=f"/api/thumbnails/{result.id}",
                decision=result.final_decision,
                rejection_reasons=thai_rejection_reasons,
                confidence_scores=confidence_scores,
                human_override=result.human_override,
                processing_time=result.processing_time,
                source_folder=result.source_folder or "Unknown",
                created_at=result.created_at
            )
            
        except Exception as e:
            logger.error(f"Error creating thumbnail data: {str(e)}")
            return None
    
    async def _ensure_thumbnail_exists(self, result: ImageResult) -> Optional[Path]:
        """Ensure thumbnail exists for an image result, create if necessary"""
        try:
            thumbnail_filename = f"{result.id}.jpg"
            thumbnail_path = self.thumbnail_dir / thumbnail_filename
            
            # Check if thumbnail already exists
            if thumbnail_path.exists():
                return thumbnail_path
            
            # Check if source image exists
            source_path = Path(result.image_path)
            if not source_path.exists():
                logger.warning(f"Source image not found: {source_path}")
                return None
            
            # Check if source is supported format
            if source_path.suffix.lower() not in self.supported_formats:
                logger.warning(f"Unsupported image format: {source_path.suffix}")
                return None
            
            # Generate thumbnail
            success = await self._generate_thumbnail(source_path, thumbnail_path)
            if success:
                return thumbnail_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error ensuring thumbnail exists: {str(e)}")
            return None
    
    async def _generate_thumbnail(self, source_path: Path, thumbnail_path: Path) -> bool:
        """Generate thumbnail from source image"""
        try:
            # Open and process image
            with Image.open(source_path) as img:
                # Convert to RGB if necessary (for PNG with transparency, etc.)
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create thumbnail with proper aspect ratio
                img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                
                # Create square thumbnail with padding if needed
                thumbnail = Image.new('RGB', self.thumbnail_size, (255, 255, 255))
                
                # Calculate position to center the image
                x = (self.thumbnail_size[0] - img.width) // 2
                y = (self.thumbnail_size[1] - img.height) // 2
                
                thumbnail.paste(img, (x, y))
                
                # Save thumbnail
                thumbnail.save(
                    thumbnail_path,
                    'JPEG',
                    quality=self.thumbnail_quality,
                    optimize=True
                )
                
                logger.debug(f"Generated thumbnail: {thumbnail_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error generating thumbnail for {source_path}: {str(e)}")
            return False
    
    def get_thumbnail_stats(self) -> Dict[str, Any]:
        """Get thumbnail generation statistics"""
        try:
            thumbnail_files = list(self.thumbnail_dir.glob("*.jpg"))
            total_size = sum(f.stat().st_size for f in thumbnail_files)
            
            return {
                "total_thumbnails": len(thumbnail_files),
                "total_size_mb": total_size / (1024 * 1024),
                "thumbnail_directory": str(self.thumbnail_dir),
                "thumbnail_size": self.thumbnail_size,
                "thumbnail_quality": self.thumbnail_quality
            }
            
        except Exception as e:
            logger.error(f"Error getting thumbnail stats: {str(e)}")
            return {
                "total_thumbnails": 0,
                "total_size_mb": 0,
                "thumbnail_directory": str(self.thumbnail_dir),
                "thumbnail_size": self.thumbnail_size,
                "thumbnail_quality": self.thumbnail_quality
            }