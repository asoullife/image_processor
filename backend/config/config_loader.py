"""Configuration loading system with JSON validation."""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
from .config_validator import ConfigValidator, ValidationResult


@dataclass
class ProcessingConfig:
    """Processing configuration parameters."""
    batch_size: int
    max_workers: int
    checkpoint_interval: int


@dataclass
class QualityConfig:
    """Quality analysis configuration parameters."""
    min_sharpness: float
    max_noise_level: float
    min_resolution: tuple


@dataclass
class SimilarityConfig:
    """Similarity detection configuration parameters."""
    hash_threshold: int
    feature_threshold: float
    clustering_eps: float


@dataclass
class ComplianceConfig:
    """Compliance checking configuration parameters."""
    logo_detection_confidence: float
    face_detection_enabled: bool
    metadata_validation: bool


@dataclass
class OutputConfig:
    """Output configuration parameters."""
    images_per_folder: int
    preserve_metadata: bool
    generate_thumbnails: bool


@dataclass
class DecisionConfig:
    """Decision engine configuration parameters."""
    quality_weight: float
    defect_weight: float
    similarity_weight: float
    compliance_weight: float
    technical_weight: float
    approval_threshold: float
    rejection_threshold: float
    quality_min_threshold: float
    defect_max_threshold: float
    similarity_max_threshold: float
    compliance_min_threshold: float
    quality_critical_threshold: float
    defect_critical_threshold: float
    compliance_critical_threshold: float


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    level: str
    file: str
    max_file_size: str
    backup_count: int


@dataclass
class AppConfig:
    """Main application configuration."""
    processing: ProcessingConfig
    quality: QualityConfig
    similarity: SimilarityConfig
    compliance: ComplianceConfig
    decision: DecisionConfig
    output: OutputConfig
    logging: LoggingConfig


class ConfigLoader:
    """Configuration loader with validation."""
    
    def __init__(self, config_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """Initialize configuration loader.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
            logger: Optional logger instance.
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'settings.json')
        self.config_path = config_path
        self._config: Optional[AppConfig] = None
        self.logger = logger or logging.getLogger(__name__)
        self.validator = ConfigValidator(self.logger)
    
    def load_config(self, validate_runtime: bool = True) -> AppConfig:
        """Load and validate configuration from JSON file.
        
        Args:
            validate_runtime: Whether to perform runtime validation checks.
        
        Returns:
            AppConfig: Validated configuration object.
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            ValueError: If configuration is invalid.
            json.JSONDecodeError: If JSON is malformed.
        """
        # Use validator to load and validate configuration
        validation_result = self.validator.validate_config_file(self.config_path)
        
        if not validation_result.is_valid:
            error_messages = [f"{error.field}: {error.message}" for error in validation_result.errors]
            raise ValueError(f"Configuration validation failed: {'; '.join(error_messages)}")
        
        # Log warnings if any
        if validation_result.has_warnings():
            for warning in validation_result.warnings:
                self.logger.warning(f"Configuration warning - {warning.field}: {warning.message}")
        
        # Load the actual configuration data
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in configuration file: {e}")
        
        # Perform runtime validation if requested
        if validate_runtime:
            runtime_result = self.validator.validate_runtime_config(config_data)
            if not runtime_result.is_valid:
                error_messages = [f"{error.field}: {error.message}" for error in runtime_result.errors]
                raise ValueError(f"Runtime configuration validation failed: {'; '.join(error_messages)}")
            
            # Log runtime warnings
            for warning in runtime_result.warnings:
                self.logger.warning(f"Runtime configuration warning - {warning.field}: {warning.message}")
        
        # Validate and create configuration objects
        try:
            processing_config = ProcessingConfig(
                batch_size=config_data['processing']['batch_size'],
                max_workers=config_data['processing']['max_workers'],
                checkpoint_interval=config_data['processing']['checkpoint_interval']
            )
            
            quality_config = QualityConfig(
                min_sharpness=config_data['quality']['min_sharpness'],
                max_noise_level=config_data['quality']['max_noise_level'],
                min_resolution=tuple(config_data['quality']['min_resolution'])
            )
            
            similarity_config = SimilarityConfig(
                hash_threshold=config_data['similarity']['hash_threshold'],
                feature_threshold=config_data['similarity']['feature_threshold'],
                clustering_eps=config_data['similarity']['clustering_eps']
            )
            
            compliance_config = ComplianceConfig(
                logo_detection_confidence=config_data['compliance']['logo_detection_confidence'],
                face_detection_enabled=config_data['compliance']['face_detection_enabled'],
                metadata_validation=config_data['compliance']['metadata_validation']
            )
            
            decision_config = DecisionConfig(
                quality_weight=config_data['decision']['quality_weight'],
                defect_weight=config_data['decision']['defect_weight'],
                similarity_weight=config_data['decision']['similarity_weight'],
                compliance_weight=config_data['decision']['compliance_weight'],
                technical_weight=config_data['decision']['technical_weight'],
                approval_threshold=config_data['decision']['approval_threshold'],
                rejection_threshold=config_data['decision']['rejection_threshold'],
                quality_min_threshold=config_data['decision']['quality_min_threshold'],
                defect_max_threshold=config_data['decision']['defect_max_threshold'],
                similarity_max_threshold=config_data['decision']['similarity_max_threshold'],
                compliance_min_threshold=config_data['decision']['compliance_min_threshold'],
                quality_critical_threshold=config_data['decision']['quality_critical_threshold'],
                defect_critical_threshold=config_data['decision']['defect_critical_threshold'],
                compliance_critical_threshold=config_data['decision']['compliance_critical_threshold']
            )
            
            output_config = OutputConfig(
                images_per_folder=config_data['output']['images_per_folder'],
                preserve_metadata=config_data['output']['preserve_metadata'],
                generate_thumbnails=config_data['output']['generate_thumbnails']
            )
            
            logging_config = LoggingConfig(
                level=config_data['logging']['level'],
                file=config_data['logging']['file'],
                max_file_size=config_data['logging']['max_file_size'],
                backup_count=config_data['logging']['backup_count']
            )
            
            self._config = AppConfig(
                processing=processing_config,
                quality=quality_config,
                similarity=similarity_config,
                compliance=compliance_config,
                decision=decision_config,
                output=output_config,
                logging=logging_config
            )
            
        except KeyError as e:
            raise ValueError(f"Missing required configuration key: {e}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid configuration value: {e}")
        
        # Validate configuration values
        self._validate_config()
        
        return self._config
    
    def _validate_config(self):
        """Validate configuration values."""
        if not self._config:
            return
        
        # Validate processing config
        if self._config.processing.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._config.processing.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if self._config.processing.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        
        # Validate quality config
        if self._config.quality.min_sharpness < 0:
            raise ValueError("min_sharpness must be non-negative")
        if not (0 <= self._config.quality.max_noise_level <= 1):
            raise ValueError("max_noise_level must be between 0 and 1")
        if len(self._config.quality.min_resolution) != 2:
            raise ValueError("min_resolution must be a tuple of 2 integers")
        
        # Validate similarity config
        if self._config.similarity.hash_threshold < 0:
            raise ValueError("hash_threshold must be non-negative")
        if not (0 <= self._config.similarity.feature_threshold <= 1):
            raise ValueError("feature_threshold must be between 0 and 1")
        if self._config.similarity.clustering_eps <= 0:
            raise ValueError("clustering_eps must be positive")
        
        # Validate compliance config
        if not (0 <= self._config.compliance.logo_detection_confidence <= 1):
            raise ValueError("logo_detection_confidence must be between 0 and 1")
        
        # Validate decision config
        weights_sum = (self._config.decision.quality_weight + 
                      self._config.decision.defect_weight + 
                      self._config.decision.similarity_weight + 
                      self._config.decision.compliance_weight + 
                      self._config.decision.technical_weight)
        if abs(weights_sum - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Decision weights must sum to 1.0, got {weights_sum}")
        
        for threshold_name, threshold_value in [
            ('approval_threshold', self._config.decision.approval_threshold),
            ('rejection_threshold', self._config.decision.rejection_threshold),
            ('quality_min_threshold', self._config.decision.quality_min_threshold),
            ('defect_max_threshold', self._config.decision.defect_max_threshold),
            ('similarity_max_threshold', self._config.decision.similarity_max_threshold),
            ('compliance_min_threshold', self._config.decision.compliance_min_threshold),
            ('quality_critical_threshold', self._config.decision.quality_critical_threshold),
            ('defect_critical_threshold', self._config.decision.defect_critical_threshold),
            ('compliance_critical_threshold', self._config.decision.compliance_critical_threshold)
        ]:
            if not (0 <= threshold_value <= 1):
                raise ValueError(f"{threshold_name} must be between 0 and 1")
        
        # Validate output config
        if self._config.output.images_per_folder <= 0:
            raise ValueError("images_per_folder must be positive")
        
        # Validate logging config
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self._config.logging.level not in valid_levels:
            raise ValueError(f"logging level must be one of: {valid_levels}")
        if self._config.logging.backup_count < 0:
            raise ValueError("backup_count must be non-negative")
    
    def get_config(self) -> Optional[AppConfig]:
        """Get loaded configuration.
        
        Returns:
            AppConfig or None if not loaded yet.
        """
        return self._config
    
    def reload_config(self, validate_runtime: bool = True) -> AppConfig:
        """Reload configuration from file.
        
        Args:
            validate_runtime: Whether to perform runtime validation checks.
        
        Returns:
            AppConfig: Reloaded configuration.
        """
        return self.load_config(validate_runtime)
    
    def validate_config_file(self) -> ValidationResult:
        """Validate the configuration file without loading it.
        
        Returns:
            ValidationResult: Validation result with errors and warnings.
        """
        return self.validator.validate_config_file(self.config_path)
    
    def migrate_config(self, backup: bool = True) -> tuple[bool, list[str]]:
        """Migrate configuration file to current version.
        
        Args:
            backup: Whether to create backup before migration.
            
        Returns:
            Tuple of (success, migration_messages).
        """
        return self.validator.migrate_config_file(self.config_path, backup)
    
    def generate_default_config(self, overwrite: bool = False) -> bool:
        """Generate default configuration file.
        
        Args:
            overwrite: Whether to overwrite existing file.
            
        Returns:
            True if successful, False otherwise.
        """
        return self.validator.generate_default_config_file(self.config_path, overwrite)
    
    def get_validation_errors(self) -> list[str]:
        """Get current configuration validation errors.
        
        Returns:
            List of validation error messages.
        """
        validation_result = self.validate_config_file()
        return [f"{error.field}: {error.message}" for error in validation_result.errors]
    
    def get_validation_warnings(self) -> list[str]:
        """Get current configuration validation warnings.
        
        Returns:
            List of validation warning messages.
        """
        validation_result = self.validate_config_file()
        return [f"{warning.field}: {warning.message}" for warning in validation_result.warnings]


# Global configuration instance
_config_loader = ConfigLoader()


def get_config() -> AppConfig:
    """Get global configuration instance.
    
    Returns:
        AppConfig: Application configuration.
    """
    if _config_loader.get_config() is None:
        _config_loader.load_config()
    return _config_loader.get_config()


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from specified path.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        AppConfig: Loaded configuration.
    """
    global _config_loader
    _config_loader = ConfigLoader(config_path)
    return _config_loader.load_config()