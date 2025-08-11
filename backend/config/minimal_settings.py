"""
Minimal Settings Manager for testing without external dependencies
"""

import json
import os
import platform
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

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
    quality_threshold: float = 0.7
    similarity_threshold: float = 0.85

@dataclass
class SystemSettings:
    """System configuration settings"""
    log_level: str = "INFO"
    auto_cleanup: bool = True
    backup_checkpoints: bool = True

@dataclass
class UISettings:
    """UI configuration settings"""
    theme: str = "light"
    language: str = "th"
    auto_refresh_interval: int = 5000

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

class MinimalSettingsManager:
    """Minimal settings manager for testing"""
    
    def __init__(self, config_path: str = "backend/config/test_config.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config: Optional[AppConfig] = None
    
    def get_config(self) -> AppConfig:
        """Get current configuration"""
        if not self._config:
            self.load_config()
        return self._config
    
    def load_config(self) -> AppConfig:
        """Load configuration from file or create default"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._config = AppConfig.from_dict(data)
            except Exception:
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
            return True
        except Exception:
            return False
    
    def _create_default_config(self) -> AppConfig:
        """Create default configuration"""
        return AppConfig(
            processing=ProcessingSettings(),
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
        except Exception:
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get basic system information without external dependencies"""
        try:
            return {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'cpu_count_logical': os.cpu_count() or 4,
                'memory_total_gb': 8.0,  # Default estimate
                'gpus': []  # No GPU detection without external libs
            }
        except Exception:
            return {
                'platform': 'Unknown',
                'processor': 'Unknown',
                'cpu_count_logical': 4,
                'memory_total_gb': 8.0,
                'gpus': []
            }
    
    def get_performance_recommendations(self, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic performance recommendations"""
        cpu_count = system_info.get('cpu_count_logical', 4)
        memory_gb = system_info.get('memory_total_gb', 8.0)
        
        recommendations = {
            'recommended_mode': PerformanceMode.BALANCED,
            'recommended_batch_size': 20,
            'recommended_workers': max(cpu_count - 1, 2),
            'gpu_acceleration': False,
            'memory_limit_gb': memory_gb * 0.5,
            'warnings': [],
            'optimizations': []
        }
        
        if memory_gb >= 16:
            recommendations['recommended_mode'] = PerformanceMode.SMART
            recommendations['recommended_batch_size'] = 50
            recommendations['optimizations'].append("High memory detected - enabling Smart mode")
        elif memory_gb < 8:
            recommendations['recommended_mode'] = PerformanceMode.SPEED
            recommendations['recommended_batch_size'] = 10
            recommendations['warnings'].append("Low memory detected - using Speed mode")
        
        return recommendations

# Global instance for testing
minimal_settings_manager = MinimalSettingsManager()