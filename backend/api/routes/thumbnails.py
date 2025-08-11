"""Thumbnail generation and serving API endpoints."""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session
from pathlib import Path
import hashlib
import os
from PIL import Image
import io
from typing import Optional
from uuid import UUID

from backend.database.connection import get_db
from backend.database.models import ImageResult
from backend.core.file_protection import FileIntegrityProtector

router = APIRouter(prefix="/api/thumbnails", tags=["thumbnails"])

# Thumbnail configuration
THUMBNAIL_SIZE = (300, 300)
THUMBNAIL_QUALITY = 85
THUMBNAIL_CACHE_DIR = Path("backend/data/thumbnails")

# Ensure thumbnail cache directory exists
THUMBNAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def generate_thumbnail_path(image_path: str) -> Path:
    """Generate thumbnail cache path based on image path hash."""
    path_hash = hashlib.md5(image_path.encode()).hexdigest()
    return THUMBNAIL_CACHE_DIR / f"{path_hash}.jpg"

def create_thumbnail(source_path: str, thumbnail_path: Path) -> bool:
    """Create thumbnail from source image."""
    try:
        with Image.open(source_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Create thumbnail maintaining aspect ratio
            img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            
            # Save thumbnail
            img.save(thumbnail_path, 'JPEG', quality=THUMBNAIL_QUALITY, optimize=True)
            return True
    except Exception as e:
        print(f"Error creating thumbnail for {source_path}: {e}")
        return False

@router.get("/{image_id}")
async def get_thumbnail(
    image_id: UUID,
    db: Session = Depends(get_db)
) -> FileResponse:
    """Get thumbnail for an image by ID."""
    
    # Get image result from database
    image_result = db.query(ImageResult).filter(ImageResult.id == image_id).first()
    if not image_result:
        raise HTTPException(status_code=404, detail="Image not found")
    
    source_path = image_result.image_path
    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail="Source image file not found")
    
    # Check if thumbnail exists in cache
    thumbnail_path = generate_thumbnail_path(source_path)
    
    # Create thumbnail if it doesn't exist or is older than source
    if (not thumbnail_path.exists() or 
        thumbnail_path.stat().st_mtime < os.path.getmtime(source_path)):
        
        success = create_thumbnail(source_path, thumbnail_path)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to generate thumbnail")
    
    # Return thumbnail file
    return FileResponse(
        path=str(thumbnail_path),
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
            "ETag": f'"{thumbnail_path.stat().st_mtime}"'
        }
    )

@router.get("/{image_id}/full")
async def get_full_image(
    image_id: UUID,
    db: Session = Depends(get_db)
) -> FileResponse:
    """Get full resolution image by ID (for detailed viewing)."""
    
    # Get image result from database
    image_result = db.query(ImageResult).filter(ImageResult.id == image_id).first()
    if not image_result:
        raise HTTPException(status_code=404, detail="Image not found")
    
    source_path = image_result.image_path
    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail="Source image file not found")
    
    # Determine media type based on file extension
    file_ext = Path(source_path).suffix.lower()
    media_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    media_type = media_type_map.get(file_ext, 'application/octet-stream')
    
    # Return full image file
    return FileResponse(
        path=source_path,
        media_type=media_type,
        headers={
            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
            "Content-Disposition": f'inline; filename="{image_result.filename}"'
        }
    )

@router.delete("/cache")
async def clear_thumbnail_cache() -> dict:
    """Clear thumbnail cache (admin endpoint)."""
    try:
        deleted_count = 0
        for thumbnail_file in THUMBNAIL_CACHE_DIR.glob("*.jpg"):
            thumbnail_file.unlink()
            deleted_count += 1
        
        return {
            "success": True,
            "message": f"Cleared {deleted_count} thumbnails from cache",
            "deleted_count": deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

@router.get("/data/{session_id}")
async def get_session_thumbnail_data(
    session_id: UUID,
    page: int = 1,
    page_size: int = 100,
    db: Session = Depends(get_db)
):
    """Get thumbnail data for a session with pagination"""
    try:
        from backend.utils.thumbnail_generator import ThumbnailGenerator
        
        thumbnail_generator = ThumbnailGenerator(db)
        thumbnail_data = await thumbnail_generator.get_session_thumbnails(
            str(session_id), page, page_size
        )
        
        return thumbnail_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting thumbnail data: {str(e)}")

@router.post("/generate/{session_id}")
async def generate_session_thumbnails(
    session_id: UUID,
    batch_size: int = 50,
    db: Session = Depends(get_db)
):
    """Generate thumbnails for all images in a session"""
    try:
        from backend.utils.thumbnail_generator import ThumbnailGenerator
        
        thumbnail_generator = ThumbnailGenerator(db)
        generated_count = await thumbnail_generator.generate_thumbnails_batch(
            str(session_id), batch_size
        )
        
        return {
            "message": f"Generated {generated_count} thumbnails",
            "session_id": str(session_id),
            "generated_count": generated_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating thumbnails: {str(e)}")

@router.get("/cache/stats")
async def get_cache_stats() -> dict:
    """Get thumbnail cache statistics."""
    try:
        thumbnail_files = list(THUMBNAIL_CACHE_DIR.glob("*.jpg"))
        total_size = sum(f.stat().st_size for f in thumbnail_files)
        
        return {
            "cache_directory": str(THUMBNAIL_CACHE_DIR),
            "thumbnail_count": len(thumbnail_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "thumbnail_size": THUMBNAIL_SIZE,
            "thumbnail_quality": THUMBNAIL_QUALITY
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")