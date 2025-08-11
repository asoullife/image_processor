"""Image analysis API routes."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import Dict, Any, List
import logging
import tempfile
import os

import sys
from pathlib import Path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from api.dependencies import get_analysis_service
from core.services import AnalysisService
from api.schemas import AnalysisRequest, AnalysisResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/single", response_model=AnalysisResponse)
async def analyze_single_image(
    request: AnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service)
) -> AnalysisResponse:
    """Analyze a single image by file path.
    
    Args:
        request: Analysis request with image path and options
        service: Analysis service
        
    Returns:
        Analysis results
    """
    try:
        if not os.path.exists(request.image_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        result = await service.analyze_image(
            image_path=request.image_path,
            analysis_types=request.analysis_types,
            options=request.options
        )
        
        return AnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze image {request.image_path}: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze image")

@router.post("/upload", response_model=AnalysisResponse)
async def analyze_uploaded_image(
    file: UploadFile = File(...),
    analysis_types: str = "quality,defect,compliance",
    service: AnalysisService = Depends(get_analysis_service)
) -> AnalysisResponse:
    """Analyze an uploaded image file.
    
    Args:
        file: Uploaded image file
        analysis_types: Comma-separated list of analysis types
        service: Analysis service
        
    Returns:
        Analysis results
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Analyze the temporary file
            result = await service.analyze_image(
                image_path=temp_path,
                analysis_types=analysis_types.split(','),
                options={}
            )
            
            return AnalysisResponse(**result)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze uploaded image: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze uploaded image")

@router.post("/batch", response_model=List[AnalysisResponse])
async def analyze_batch_images(
    request: Dict[str, Any],
    service: AnalysisService = Depends(get_analysis_service)
) -> List[AnalysisResponse]:
    """Analyze a batch of images.
    
    Args:
        request: Batch analysis request with image paths and options
        service: Analysis service
        
    Returns:
        List of analysis results
    """
    try:
        image_paths = request.get('image_paths', [])
        analysis_types = request.get('analysis_types', ['quality', 'defect', 'compliance'])
        options = request.get('options', {})
        
        if not image_paths:
            raise HTTPException(status_code=400, detail="No image paths provided")
        
        # Validate all paths exist
        missing_files = [path for path in image_paths if not os.path.exists(path)]
        if missing_files:
            raise HTTPException(
                status_code=404, 
                detail=f"Image files not found: {', '.join(missing_files)}"
            )
        
        results = await service.analyze_batch(
            image_paths=image_paths,
            analysis_types=analysis_types,
            options=options
        )
        
        return [AnalysisResponse(**result) for result in results]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze batch images: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze batch images")

@router.get("/types")
async def get_analysis_types() -> Dict[str, Any]:
    """Get available analysis types and their descriptions.
    
    Returns:
        Available analysis types
    """
    return {
        "analysis_types": {
            "quality": {
                "name": "Quality Analysis",
                "description": "Analyze image quality including sharpness, noise, exposure, and color balance",
                "metrics": ["sharpness_score", "noise_level", "exposure_score", "color_balance_score", "overall_score"]
            },
            "defect": {
                "name": "Defect Detection",
                "description": "Detect visual defects and anomalies in images",
                "metrics": ["defect_count", "anomaly_score", "defect_types", "confidence_scores"]
            },
            "compliance": {
                "name": "Compliance Checking",
                "description": "Check compliance with Adobe Stock guidelines",
                "metrics": ["logo_detections", "privacy_violations", "metadata_issues", "overall_compliance"]
            },
            "similarity": {
                "name": "Similarity Detection",
                "description": "Find similar or duplicate images",
                "metrics": ["similarity_hash", "feature_vector", "similar_images"]
            }
        }
    }

@router.get("/config")
async def get_analysis_config(
    service: AnalysisService = Depends(get_analysis_service)
) -> Dict[str, Any]:
    """Get current analysis configuration.
    
    Args:
        service: Analysis service
        
    Returns:
        Analysis configuration
    """
    try:
        config = await service.get_analysis_config()
        return config
        
    except Exception as e:
        logger.error(f"Failed to get analysis config: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analysis config")