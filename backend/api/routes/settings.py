"""
Settings API routes for configuration management
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
import logging
from pydantic import BaseModel, Field

from config.settings_manager import settings_manager, PerformanceMode, AppConfig
from api.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings", tags=["settings"])

# Pydantic models for API
class ProcessingSettingsUpdate(BaseModel):
    performance_mode: Optional[PerformanceMode] = None
    batch_size: Optional[int] = Field(None, ge=1, le=1000)
    max_workers: Optional[int] = Field(None, ge=1, le=32)
    gpu_enabled: Optional[bool] = None
    memory_limit_gb: Optional[float] = Field(None, ge=1.0, le=64.0)
    checkpoint_interval: Optional[int] = Field(None, ge=1, le=100)
    quality_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    defect_confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    use_ai_enhancement: Optional[bool] = None
    fallback_to_opencv: Optional[bool] = None
    model_precision: Optional[str] = Field(None, regex="^(fp32|fp16|int8)$")

class SystemSettingsUpdate(BaseModel):
    log_level: Optional[str] = Field(None, regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    temp_dir: Optional[str] = None
    max_file_size_mb: Optional[int] = Field(None, ge=1, le=1000)
    auto_cleanup: Optional[bool] = None
    backup_checkpoints: Optional[bool] = None

class UISettingsUpdate(BaseModel):
    theme: Optional[str] = Field(None, regex="^(light|dark)$")
    language: Optional[str] = Field(None, regex="^(en|th)$")
    auto_refresh_interval: Optional[int] = Field(None, ge=1000, le=60000)
    thumbnail_size: Optional[int] = Field(None, ge=100, le=500)
    items_per_page: Optional[int] = Field(None, ge=10, le=200)
    show_confidence_scores: Optional[bool] = None

class PerformanceModeRequest(BaseModel):
    mode: PerformanceMode

class HardwareOptimizationResponse(BaseModel):
    system_info: Dict[str, Any]
    recommendations: Dict[str, Any]
    applied: bool
    message: str

@router.get("/")
async def get_all_settings():
    """Get all current settings"""
    try:
        config = settings_manager.get_config()
        return {
            "success": True,
            "data": config.to_dict()
        }
    except Exception as e:
        logger.error(f"Error getting settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to get settings")

@router.get("/processing")
async def get_processing_settings():
    """Get processing settings"""
    try:
        config = settings_manager.get_config()
        return {
            "success": True,
            "data": config.processing.__dict__
        }
    except Exception as e:
        logger.error(f"Error getting processing settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to get processing settings")

@router.put("/processing")
async def update_processing_settings(settings: ProcessingSettingsUpdate):
    """Update processing settings"""
    try:
        # Convert to dict and filter None values
        updates = {k: v for k, v in settings.dict().items() if v is not None}
        
        success = settings_manager.update_settings('processing', updates)
        
        if success:
            config = settings_manager.get_config()
            return {
                "success": True,
                "message": "Processing settings updated successfully",
                "data": config.processing.__dict__
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to update processing settings")
            
    except Exception as e:
        logger.error(f"Error updating processing settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system")
async def get_system_settings():
    """Get system settings"""
    try:
        config = settings_manager.get_config()
        return {
            "success": True,
            "data": config.system.__dict__
        }
    except Exception as e:
        logger.error(f"Error getting system settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system settings")

@router.put("/system")
async def update_system_settings(settings: SystemSettingsUpdate):
    """Update system settings"""
    try:
        updates = {k: v for k, v in settings.dict().items() if v is not None}
        
        success = settings_manager.update_settings('system', updates)
        
        if success:
            config = settings_manager.get_config()
            return {
                "success": True,
                "message": "System settings updated successfully",
                "data": config.system.__dict__
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to update system settings")
            
    except Exception as e:
        logger.error(f"Error updating system settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ui")
async def get_ui_settings():
    """Get UI settings"""
    try:
        config = settings_manager.get_config()
        return {
            "success": True,
            "data": config.ui.__dict__
        }
    except Exception as e:
        logger.error(f"Error getting UI settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to get UI settings")

@router.put("/ui")
async def update_ui_settings(settings: UISettingsUpdate):
    """Update UI settings"""
    try:
        updates = {k: v for k, v in settings.dict().items() if v is not None}
        
        success = settings_manager.update_settings('ui', updates)
        
        if success:
            config = settings_manager.get_config()
            return {
                "success": True,
                "message": "UI settings updated successfully",
                "data": config.ui.__dict__
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to update UI settings")
            
    except Exception as e:
        logger.error(f"Error updating UI settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/performance-mode")
async def set_performance_mode(request: PerformanceModeRequest):
    """Set performance mode and apply optimized settings"""
    try:
        mode = request.mode
        
        # Get hardware recommendations for the selected mode
        system_info = settings_manager.hardware_detector.get_system_info()
        recommendations = settings_manager.hardware_detector.get_performance_recommendations(system_info)
        
        # Apply mode-specific settings
        mode_settings = {
            PerformanceMode.SPEED: {
                'batch_size': min(recommendations['recommended_batch_size'] // 2, 10),
                'max_workers': max(recommendations['recommended_workers'] // 2, 2),
                'use_ai_enhancement': False,
                'model_precision': 'int8'
            },
            PerformanceMode.BALANCED: {
                'batch_size': recommendations['recommended_batch_size'],
                'max_workers': recommendations['recommended_workers'],
                'use_ai_enhancement': True,
                'model_precision': 'fp16'
            },
            PerformanceMode.SMART: {
                'batch_size': min(recommendations['recommended_batch_size'] * 2, 100),
                'max_workers': recommendations['recommended_workers'],
                'use_ai_enhancement': True,
                'model_precision': 'fp32'
            }
        }
        
        # Update settings
        updates = mode_settings[mode]
        updates['performance_mode'] = mode
        
        success = settings_manager.update_settings('processing', updates)
        
        if success:
            config = settings_manager.get_config()
            return {
                "success": True,
                "message": f"Performance mode set to {mode.value}",
                "data": {
                    "mode": mode.value,
                    "settings": config.processing.__dict__,
                    "recommendations": recommendations
                }
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to set performance mode")
            
    except Exception as e:
        logger.error(f"Error setting performance mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/hardware")
async def get_hardware_info():
    """Get detailed hardware information"""
    try:
        system_info = settings_manager.hardware_detector.get_system_info()
        return {
            "success": True,
            "data": system_info
        }
    except Exception as e:
        logger.error(f"Error getting hardware info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get hardware information")

@router.get("/recommendations")
async def get_performance_recommendations():
    """Get performance recommendations based on current hardware"""
    try:
        system_info = settings_manager.hardware_detector.get_system_info()
        recommendations = settings_manager.hardware_detector.get_performance_recommendations(system_info)
        
        return {
            "success": True,
            "data": {
                "system_info": system_info,
                "recommendations": recommendations
            }
        }
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendations")

@router.post("/optimize")
async def optimize_for_hardware() -> HardwareOptimizationResponse:
    """Automatically optimize settings based on current hardware"""
    try:
        result = settings_manager.optimize_for_hardware()
        
        return HardwareOptimizationResponse(
            system_info=result['system_info'],
            recommendations=result['recommendations'],
            applied=result['applied'],
            message="Settings optimized for your hardware"
        )
    except Exception as e:
        logger.error(f"Error optimizing settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to optimize settings")

@router.get("/health")
async def get_system_health():
    """Get current system health and performance metrics"""
    try:
        health = settings_manager.get_system_health()
        return {
            "success": True,
            "data": health
        }
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system health")

@router.post("/reset")
async def reset_to_defaults():
    """Reset all settings to defaults"""
    try:
        # Create new default config
        settings_manager._config = settings_manager._create_default_config()
        success = settings_manager.save_config()
        
        if success:
            config = settings_manager.get_config()
            return {
                "success": True,
                "message": "Settings reset to defaults",
                "data": config.to_dict()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save default settings")
            
    except Exception as e:
        logger.error(f"Error resetting settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset settings")

@router.get("/export")
async def export_settings():
    """Export current settings as JSON"""
    try:
        config = settings_manager.get_config()
        return {
            "success": True,
            "data": config.to_dict(),
            "filename": "adobe_stock_processor_config.json"
        }
    except Exception as e:
        logger.error(f"Error exporting settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to export settings")

@router.post("/import")
async def import_settings(config_data: Dict[str, Any]):
    """Import settings from JSON data"""
    try:
        # Validate and create config from imported data
        imported_config = AppConfig.from_dict(config_data)
        
        # Update settings manager
        settings_manager._config = imported_config
        success = settings_manager.save_config()
        
        if success:
            return {
                "success": True,
                "message": "Settings imported successfully",
                "data": imported_config.to_dict()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save imported settings")
            
    except Exception as e:
        logger.error(f"Error importing settings: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid configuration data: {str(e)}")

@router.get("/performance-modes")
async def get_performance_modes():
    """Get available performance modes with descriptions"""
    return {
        "success": True,
        "data": {
            "modes": [
                {
                    "value": "speed",
                    "label": "Speed Mode",
                    "description": "Fast processing with basic analysis - uses smaller batches and CPU-only processing",
                    "recommended_for": "Quick previews, large datasets, limited hardware"
                },
                {
                    "value": "balanced", 
                    "label": "Balanced Mode",
                    "description": "Optimal balance of speed and accuracy - uses moderate batches with AI enhancement",
                    "recommended_for": "Most use cases, general processing"
                },
                {
                    "value": "smart",
                    "label": "Smart Mode", 
                    "description": "Thorough analysis with full AI models - uses larger batches and maximum accuracy",
                    "recommended_for": "High-quality analysis, sufficient hardware resources"
                }
            ]
        }
    }