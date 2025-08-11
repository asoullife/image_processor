"""
Settings Manager for Adobe Stock Image Processor
Handles configuration loading, validation, and management
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import GPUtil
import platform
import logging

logger = logging.getLogger(__name__)

class PerformanceMode(str, Enum):
    SPEED = "speed"
    BALANCED = "balanced"
    SMART = "smart"

@dataclass
class ProcessingSettings:
    """Processing configuration settings"""
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    batch_size: int = 20
    max_workers: int = 4
    gpu_enabled: bool = True
    memory_limit_gb: float = 8.0
    checkpoint_interval: int = 10
    
    # Quality thresholds
    quality_threshold: float = 0.7
    similarity_threshold: float = 0.85
    defect_confidence_threshold: float = 0.8
    
    # AI Model settings
    use_ai_enhancement: bool = True
    fallback_to_opencv: bool = True
    model_precision: str = "fp16"  # fp32, fp16, int8

@dataclass
class SystemSettings:
    """System configuration settings"""
    log_level: str = "INFO"
    temp_dir: str = "temp"
    max_file_size_mb: int = 100
    supported_formats: List[str] = None
    auto_cleanup: bool = True
    backup_checkpoints: bool = True
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.jpg', '.jpeg', '.png']

@dataclass
class UISettings:
    """UI configuration settings"""
    theme: str = "light"
    language: str = "th"
    auto_refresh_interval: int = 5000  # milliseconds
    thumbnail_size: int = 200
    items_per_page: int = 50
    show_confidence_scores: bool = True

@dataclass
class AppConfig:
    """Complete application configuration"""
    processing: ProcessingSettings
    system: SystemSettings
    ui: UISettings
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        return cls(
            processing=ProcessingSettings(**data.get('processing', {})),
            system=SystemSettings(**data.get('system', {})),
            ui=UISettings(**data.get('ui', {}))
        )

class HardwareDetector:
    """Detect and analyze system hardware capabilities"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            # CPU Information
            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()
            
            # Memory Information
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            
            # GPU Information
            gpus = []
            try:
                gpu_list = GPUtil.getGPUs()
                for gpu in gpu_list:
                    gpus.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'memory_total': gpu.memoryTotal,
                        'memory_free': gpu.memoryFree,
                        'memory_used': gpu.memoryUsed,
                        'temperature': gpu.temperature,
                        'load': gpu.load
                    })
            except Exception as e:
                logger.warning(f"Could not detect GPU: {e}")
            
            # Disk Information
            disk_usage = psutil.disk_usage('/')
            
            return {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'cpu_count_physical': cpu_count,
                'cpu_count_logical': cpu_count_logical,
                'cpu_frequency_mhz': cpu_freq.current if cpu_freq else None,
                'memory_total_gb': round(memory_gb, 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'memory_percent_used': memory.percent,
                'gpus': gpus,
                'disk_total_gb': round(disk_usage.total / (1024**3), 2),
                'disk_free_gb': round(disk_usage.free / (1024**3), 2),
                'disk_used_percent': round((disk_usage.used / disk_usage.total) * 100, 2)
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}
    
    @staticmethod
    def get_performance_recommendations(system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance recommendations based on hardware"""
        recommendations = {
            'recommended_mode': PerformanceMode.BALANCED,
            'recommended_batch_size': 20,
            'recommended_workers': 4,
            'gpu_acceleration': False,
            'memory_limit_gb': 4.0,
            'warnings': [],
            'optimizations': []
        }
        
        try:
            # Memory-based recommendations
            memory_gb = system_info.get('memory_total_gb', 8)
            if memory_gb >= 16:
                recommendations['recommended_mode'] = PerformanceMode.SMART
                recommendations['recommended_batch_size'] = 50
                recommendations['memory_limit_gb'] = min(memory_gb * 0.6, 12.0)
                recommendations['optimizations'].append("High memory detected - enabling Smart mode with larger batches")
            elif memory_gb >= 8:
                recommendations['recommended_mode'] = PerformanceMode.BALANCED
                recommendations['recommended_batch_size'] = 20
                recommendations['memory_limit_gb'] = memory_gb * 0.5
            else:
                recommendations['recommended_mode'] = PerformanceMode.SPEED
                recommendations['recommended_batch_size'] = 10
                recommendations['memory_limit_gb'] = memory_gb * 0.4
                recommendations['warnings'].append("Low memory detected - using Speed mode with smaller batches")
            
            # CPU-based recommendations
            cpu_count = system_info.get('cpu_count_logical', 4)
            recommendations['recommended_workers'] = min(max(cpu_count - 1, 2), 8)
            
            # GPU-based recommendations
            gpus = system_info.get('gpus', [])
            if gpus:
                for gpu in gpus:
                    if gpu['memory_total'] >= 4000:  # 4GB VRAM
                        recommendations['gpu_acceleration'] = True
                        recommendations['optimizations'].append(f"GPU detected: {gpu['name']} - enabling GPU acceleration")
                        break
                    else:
                        recommendations['warnings'].append(f"GPU {gpu['name']} has insufficient VRAM ({gpu['memory_total']}MB)")
            else:
                recommendations['warnings'].append("No GPU detected - using CPU-only processing")
            
            # Disk space warnings
            disk_free_gb = system_info.get('disk_free_gb', 0)
            if disk_free_gb < 10:
                recommendations['warnings'].append("Low disk space - ensure sufficient space for processing")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations['warnings'].append("Could not analyze system - using default settings")
        
        return recommendations

class SettingsManager:
    """Manage application settings and configuration"""
    
    def __init__(self, config_path: str = "backend/config/app_config.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config: Optional[AppConfig] = None
        self.hardware_detector = HardwareDetector()
    
    def load_config(self) -> AppConfig:
        """Load configuration from file or create default"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._config = AppConfig.from_dict(data)
                logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                self._config = self._create_default_config()
        else:
            self._config = self._create_default_config()
            self.save_config()
        
        return self._config
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        if not self._config:
            return False
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False
    
    def _create_default_config(self) -> AppConfig:
        """Create default configuration with hardware optimization"""
        system_info = self.hardware_detector.get_system_info()
        recommendations = self.hardware_detector.get_performance_recommendations(system_info)
        
        # Apply hardware recommendations to default settings
        processing_settings = ProcessingSettings(
            performance_mode=recommendations['recommended_mode'],
            batch_size=recommendations['recommended_batch_size'],
            max_workers=recommendations['recommended_workers'],
            gpu_enabled=recommendations['gpu_acceleration'],
            memory_limit_gb=recommendations['memory_limit_gb']
        )
        
        return AppConfig(
            processing=processing_settings,
            system=SystemSettings(),
            ui=UISettings()
        )
    
    def update_settings(self, section: str, settings: Dict[str, Any]) -> bool:
        """Update specific settings section"""
        if not self._config:
            self.load_config()
        
        try:
            if section == 'processing':
                for key, value in settings.items():
                    if hasattr(self._config.processing, key):
                        setattr(self._config.processing, key, value)
            elif section == 'system':
                for key, value in settings.items():
                    if hasattr(self._config.system, key):
                        setattr(self._config.system, key, value)
            elif section == 'ui':
                for key, value in settings.items():
                    if hasattr(self._config.ui, key):
                        setattr(self._config.ui, key, value)
            else:
                return False
            
            return self.save_config()
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return False
    
    def get_config(self) -> AppConfig:
        """Get current configuration"""
        if not self._config:
            self.load_config()
        return self._config
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health and performance metrics"""
        try:
            # Current system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')
            
            # GPU metrics
            gpu_metrics = []
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_metrics.append({
                        'name': gpu.name,
                        'load': gpu.load * 100,
                        'memory_used_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'temperature': gpu.temperature
                    })
            except:
                pass
            
            # Process metrics
            current_process = psutil.Process()
            process_memory = current_process.memory_info()
            
            return {
                'timestamp': psutil.boot_time(),
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_usage_percent': round((disk_usage.used / disk_usage.total) * 100, 2),
                'disk_free_gb': round(disk_usage.free / (1024**3), 2),
                'gpu_metrics': gpu_metrics,
                'process_memory_mb': round(process_memory.rss / (1024**2), 2),
                'process_cpu_percent': current_process.cpu_percent(),
                'status': self._get_health_status(cpu_percent, memory.percent, disk_usage)
            }
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _get_health_status(self, cpu_percent: float, memory_percent: float, disk_usage) -> str:
        """Determine overall system health status"""
        disk_percent = (disk_usage.used / disk_usage.total) * 100
        
        if cpu_percent > 90 or memory_percent > 90 or disk_percent > 95:
            return 'critical'
        elif cpu_percent > 70 or memory_percent > 70 or disk_percent > 85:
            return 'warning'
        else:
            return 'healthy'
    
    def optimize_for_hardware(self) -> Dict[str, Any]:
        """Automatically optimize settings based on current hardware"""
        system_info = self.hardware_detector.get_system_info()
        recommendations = self.hardware_detector.get_performance_recommendations(system_info)
        
        # Apply recommendations
        self.update_settings('processing', {
            'performance_mode': recommendations['recommended_mode'],
            'batch_size': recommendations['recommended_batch_size'],
            'max_workers': recommendations['recommended_workers'],
            'gpu_enabled': recommendations['gpu_acceleration'],
            'memory_limit_gb': recommendations['memory_limit_gb']
        })
        
        return {
            'system_info': system_info,
            'recommendations': recommendations,
            'applied': True
        }

# Global settings manager instance
settings_manager = SettingsManager()