"""Configuration validation and schema management system."""

import json
import os
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import re
from pathlib import Path


class ConfigVersion(Enum):
    """Configuration version enumeration."""
    V1_0 = "1.0"
    V1_1 = "1.1"
    CURRENT = V1_1


@dataclass
class ValidationError:
    """Configuration validation error."""
    field: str
    value: Any
    message: str
    severity: str = "error"  # error, warning


@dataclass
class ValidationResult:
    """Configuration validation result."""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0


class ConfigSchema:
    """Configuration schema definition and validation."""
    
    # Schema definition for configuration validation
    SCHEMA = {
        "version": {
            "type": str,
            "required": False,
            "default": ConfigVersion.CURRENT.value,
            "pattern": r"^\d+\.\d+$"
        },
        "processing": {
            "type": dict,
            "required": True,
            "fields": {
                "batch_size": {
                    "type": int,
                    "required": True,
                    "min": 1,
                    "max": 10000,
                    "default": 200
                },
                "max_workers": {
                    "type": int,
                    "required": True,
                    "min": 1,
                    "max": 32,
                    "default": 4
                },
                "checkpoint_interval": {
                    "type": int,
                    "required": True,
                    "min": 1,
                    "max": 1000,
                    "default": 50
                }
            }
        },
        "quality": {
            "type": dict,
            "required": True,
            "fields": {
                "min_sharpness": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "default": 100.0
                },
                "max_noise_level": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 1,
                    "default": 0.1
                },
                "min_resolution": {
                    "type": list,
                    "required": True,
                    "length": 2,
                    "item_type": int,
                    "item_min": 1,
                    "default": [1920, 1080]
                }
            }
        },
        "similarity": {
            "type": dict,
            "required": True,
            "fields": {
                "hash_threshold": {
                    "type": int,
                    "required": True,
                    "min": 0,
                    "max": 64,
                    "default": 5
                },
                "feature_threshold": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 1,
                    "default": 0.85
                },
                "clustering_eps": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 2,
                    "default": 0.3
                }
            }
        },
        "compliance": {
            "type": dict,
            "required": True,
            "fields": {
                "logo_detection_confidence": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 1,
                    "default": 0.7
                },
                "face_detection_enabled": {
                    "type": bool,
                    "required": True,
                    "default": True
                },
                "metadata_validation": {
                    "type": bool,
                    "required": True,
                    "default": True
                }
            }
        },
        "decision": {
            "type": dict,
            "required": True,
            "fields": {
                "quality_weight": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 1,
                    "default": 0.35
                },
                "defect_weight": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 1,
                    "default": 0.25
                },
                "similarity_weight": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 1,
                    "default": 0.20
                },
                "compliance_weight": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 1,
                    "default": 0.15
                },
                "technical_weight": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 1,
                    "default": 0.05
                },
                "approval_threshold": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 1,
                    "default": 0.75
                },
                "rejection_threshold": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 1,
                    "default": 0.40
                },
                "quality_min_threshold": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 1,
                    "default": 0.60
                },
                "defect_max_threshold": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 1,
                    "default": 0.30
                },
                "similarity_max_threshold": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 1,
                    "default": 0.85
                },
                "compliance_min_threshold": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 1,
                    "default": 0.80
                },
                "quality_critical_threshold": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 1,
                    "default": 0.30
                },
                "defect_critical_threshold": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 1,
                    "default": 0.70
                },
                "compliance_critical_threshold": {
                    "type": (int, float),
                    "required": True,
                    "min": 0,
                    "max": 1,
                    "default": 0.50
                }
            }
        },
        "output": {
            "type": dict,
            "required": True,
            "fields": {
                "images_per_folder": {
                    "type": int,
                    "required": True,
                    "min": 1,
                    "max": 10000,
                    "default": 200
                },
                "preserve_metadata": {
                    "type": bool,
                    "required": True,
                    "default": True
                },
                "generate_thumbnails": {
                    "type": bool,
                    "required": True,
                    "default": True
                }
            }
        },
        "logging": {
            "type": dict,
            "required": True,
            "fields": {
                "level": {
                    "type": str,
                    "required": True,
                    "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    "default": "INFO"
                },
                "file": {
                    "type": str,
                    "required": True,
                    "default": "adobe_stock_processor.log"
                },
                "max_file_size": {
                    "type": str,
                    "required": True,
                    "pattern": r"^\d+[KMGT]?B$",
                    "default": "10MB"
                },
                "backup_count": {
                    "type": int,
                    "required": True,
                    "min": 0,
                    "max": 100,
                    "default": 5
                }
            }
        }
    }
    
    @classmethod
    def validate_field(cls, field_name: str, value: Any, schema: Dict[str, Any]) -> List[ValidationError]:
        """Validate a single field against its schema.
        
        Args:
            field_name: Name of the field being validated
            value: Value to validate
            schema: Schema definition for the field
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check type
        expected_type = schema.get("type")
        if expected_type:
            if isinstance(expected_type, tuple):
                if not isinstance(value, expected_type):
                    errors.append(ValidationError(
                        field=field_name,
                        value=value,
                        message=f"Expected type {expected_type}, got {type(value).__name__}"
                    ))
            else:
                if not isinstance(value, expected_type):
                    errors.append(ValidationError(
                        field=field_name,
                        value=value,
                        message=f"Expected type {expected_type.__name__}, got {type(value).__name__}"
                    ))
        
        # Check numeric constraints
        if isinstance(value, (int, float)):
            if "min" in schema and value < schema["min"]:
                errors.append(ValidationError(
                    field=field_name,
                    value=value,
                    message=f"Value {value} is below minimum {schema['min']}"
                ))
            if "max" in schema and value > schema["max"]:
                errors.append(ValidationError(
                    field=field_name,
                    value=value,
                    message=f"Value {value} is above maximum {schema['max']}"
                ))
        
        # Check string constraints
        if isinstance(value, str):
            if "pattern" in schema:
                if not re.match(schema["pattern"], value):
                    errors.append(ValidationError(
                        field=field_name,
                        value=value,
                        message=f"Value '{value}' does not match pattern '{schema['pattern']}'"
                    ))
            if "choices" in schema:
                if value not in schema["choices"]:
                    errors.append(ValidationError(
                        field=field_name,
                        value=value,
                        message=f"Value '{value}' not in allowed choices: {schema['choices']}"
                    ))
        
        # Check list constraints
        if isinstance(value, list):
            if "length" in schema and len(value) != schema["length"]:
                errors.append(ValidationError(
                    field=field_name,
                    value=value,
                    message=f"List length {len(value)} does not match required length {schema['length']}"
                ))
            if "item_type" in schema:
                for i, item in enumerate(value):
                    if not isinstance(item, schema["item_type"]):
                        errors.append(ValidationError(
                            field=f"{field_name}[{i}]",
                            value=item,
                            message=f"List item type {type(item).__name__} does not match required type {schema['item_type'].__name__}"
                        ))
            if "item_min" in schema:
                for i, item in enumerate(value):
                    if isinstance(item, (int, float)) and item < schema["item_min"]:
                        errors.append(ValidationError(
                            field=f"{field_name}[{i}]",
                            value=item,
                            message=f"List item {item} is below minimum {schema['item_min']}"
                        ))
        
        return errors
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> ValidationResult:
        """Validate entire configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []
        
        # Check for required top-level fields
        for field_name, field_schema in cls.SCHEMA.items():
            if field_schema.get("required", False) and field_name not in config:
                errors.append(ValidationError(
                    field=field_name,
                    value=None,
                    message=f"Required field '{field_name}' is missing"
                ))
                continue
            
            if field_name not in config:
                continue
            
            field_value = config[field_name]
            
            # Validate field type and constraints
            field_errors = cls.validate_field(field_name, field_value, field_schema)
            errors.extend(field_errors)
            
            # Validate nested fields
            if field_schema.get("type") == dict and "fields" in field_schema:
                if isinstance(field_value, dict):
                    for sub_field_name, sub_field_schema in field_schema["fields"].items():
                        full_field_name = f"{field_name}.{sub_field_name}"
                        
                        if sub_field_schema.get("required", False) and sub_field_name not in field_value:
                            errors.append(ValidationError(
                                field=full_field_name,
                                value=None,
                                message=f"Required field '{full_field_name}' is missing"
                            ))
                            continue
                        
                        if sub_field_name not in field_value:
                            continue
                        
                        sub_field_value = field_value[sub_field_name]
                        sub_field_errors = cls.validate_field(full_field_name, sub_field_value, sub_field_schema)
                        errors.extend(sub_field_errors)
        
        # Special validation for decision weights
        if "decision" in config and isinstance(config["decision"], dict):
            decision_config = config["decision"]
            weight_fields = ["quality_weight", "defect_weight", "similarity_weight", "compliance_weight", "technical_weight"]
            
            if all(field in decision_config for field in weight_fields):
                total_weight = sum(decision_config[field] for field in weight_fields)
                if abs(total_weight - 1.0) > 0.01:
                    errors.append(ValidationError(
                        field="decision.weights",
                        value=total_weight,
                        message=f"Decision weights must sum to 1.0, got {total_weight:.3f}"
                    ))
        
        # Check for unknown fields (warnings)
        cls._check_unknown_fields(config, cls.SCHEMA, "", warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    @classmethod
    def _check_unknown_fields(cls, config: Dict[str, Any], schema: Dict[str, Any], prefix: str, warnings: List[ValidationError]):
        """Check for unknown fields in configuration."""
        for field_name, field_value in config.items():
            full_field_name = f"{prefix}.{field_name}" if prefix else field_name
            
            if field_name not in schema:
                warnings.append(ValidationError(
                    field=full_field_name,
                    value=field_value,
                    message=f"Unknown field '{full_field_name}' will be ignored",
                    severity="warning"
                ))
                continue
            
            field_schema = schema[field_name]
            if field_schema.get("type") == dict and "fields" in field_schema and isinstance(field_value, dict):
                cls._check_unknown_fields(field_value, field_schema["fields"], full_field_name, warnings)
    
    @classmethod
    def generate_default_config(cls) -> Dict[str, Any]:
        """Generate default configuration from schema.
        
        Returns:
            Default configuration dictionary
        """
        config = {}
        
        for field_name, field_schema in cls.SCHEMA.items():
            if "default" in field_schema:
                config[field_name] = field_schema["default"]
            elif field_schema.get("type") == dict and "fields" in field_schema:
                config[field_name] = {}
                for sub_field_name, sub_field_schema in field_schema["fields"].items():
                    if "default" in sub_field_schema:
                        config[field_name][sub_field_name] = sub_field_schema["default"]
        
        return config


class ConfigMigrator:
    """Configuration migration system for version updates."""
    
    @staticmethod
    def get_config_version(config: Dict[str, Any]) -> str:
        """Get configuration version.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Version string
        """
        return config.get("version", "1.0")
    
    @staticmethod
    def migrate_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Migrate configuration to current version.
        
        Args:
            config: Configuration dictionary to migrate
            
        Returns:
            Tuple of (migrated_config, migration_messages)
        """
        current_version = ConfigMigrator.get_config_version(config)
        migration_messages = []
        migrated_config = config.copy()
        
        if current_version == "1.0":
            migrated_config, messages = ConfigMigrator._migrate_v1_0_to_v1_1(migrated_config)
            migration_messages.extend(messages)
            current_version = "1.1"
        
        # Set current version
        migrated_config["version"] = ConfigVersion.CURRENT.value
        
        if migration_messages:
            migration_messages.append(f"Configuration migrated to version {ConfigVersion.CURRENT.value}")
        
        return migrated_config, migration_messages
    
    @staticmethod
    def _migrate_v1_0_to_v1_1(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Migrate from version 1.0 to 1.1.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (migrated_config, migration_messages)
        """
        migrated_config = config.copy()
        messages = []
        
        # Add version field if missing
        if "version" not in migrated_config:
            migrated_config["version"] = "1.1"
            messages.append("Added version field")
        
        # Add any new fields with defaults
        default_config = ConfigSchema.generate_default_config()
        
        # Check for missing sections and add them
        for section_name, section_config in default_config.items():
            if section_name not in migrated_config:
                migrated_config[section_name] = section_config
                messages.append(f"Added missing section: {section_name}")
            elif isinstance(section_config, dict):
                # Check for missing fields in existing sections
                for field_name, field_value in section_config.items():
                    if field_name not in migrated_config[section_name]:
                        migrated_config[section_name][field_name] = field_value
                        messages.append(f"Added missing field: {section_name}.{field_name}")
        
        return migrated_config, messages


class ConfigValidator:
    """Main configuration validator and manager."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize configuration validator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.schema = ConfigSchema()
        self.migrator = ConfigMigrator()
    
    def validate_config_file(self, config_path: str) -> ValidationResult:
        """Validate configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            ValidationResult
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except FileNotFoundError:
            return ValidationResult(
                is_valid=False,
                errors=[ValidationError(
                    field="file",
                    value=config_path,
                    message=f"Configuration file not found: {config_path}"
                )],
                warnings=[]
            )
        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                errors=[ValidationError(
                    field="json",
                    value=str(e),
                    message=f"Invalid JSON in configuration file: {e}"
                )],
                warnings=[]
            )
        
        return self.validate_config(config)
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            ValidationResult
        """
        return self.schema.validate_config(config)
    
    def migrate_config_file(self, config_path: str, backup: bool = True) -> Tuple[bool, List[str]]:
        """Migrate configuration file to current version.
        
        Args:
            config_path: Path to configuration file
            backup: Whether to create backup before migration
            
        Returns:
            Tuple of (success, migration_messages)
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to load configuration file: {e}")
            return False, [f"Failed to load configuration file: {e}"]
        
        # Check if migration is needed
        current_version = self.migrator.get_config_version(config)
        if current_version == ConfigVersion.CURRENT.value:
            return True, ["Configuration is already at current version"]
        
        # Create backup if requested
        if backup:
            backup_path = f"{config_path}.backup.{current_version}"
            try:
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                self.logger.info(f"Created backup at {backup_path}")
            except Exception as e:
                self.logger.error(f"Failed to create backup: {e}")
                return False, [f"Failed to create backup: {e}"]
        
        # Migrate configuration
        migrated_config, migration_messages = self.migrator.migrate_config(config)
        
        # Validate migrated configuration
        validation_result = self.validate_config(migrated_config)
        if not validation_result.is_valid:
            error_messages = [f"{error.field}: {error.message}" for error in validation_result.errors]
            self.logger.error(f"Migrated configuration is invalid: {error_messages}")
            return False, [f"Migrated configuration is invalid: {error_messages}"]
        
        # Save migrated configuration
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(migrated_config, f, indent=2)
            self.logger.info(f"Successfully migrated configuration to version {ConfigVersion.CURRENT.value}")
        except Exception as e:
            self.logger.error(f"Failed to save migrated configuration: {e}")
            return False, [f"Failed to save migrated configuration: {e}"]
        
        return True, migration_messages
    
    def generate_default_config_file(self, config_path: str, overwrite: bool = False) -> bool:
        """Generate default configuration file.
        
        Args:
            config_path: Path where to save configuration file
            overwrite: Whether to overwrite existing file
            
        Returns:
            True if successful, False otherwise
        """
        if os.path.exists(config_path) and not overwrite:
            self.logger.warning(f"Configuration file already exists: {config_path}")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Generate default configuration
        default_config = self.schema.generate_default_config()
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
            self.logger.info(f"Generated default configuration at {config_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate default configuration: {e}")
            return False
    
    def validate_runtime_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate configuration at runtime with additional checks.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            ValidationResult
        """
        # First run standard validation
        result = self.validate_config(config)
        
        # Add runtime-specific validations
        runtime_errors = []
        runtime_warnings = []
        
        # Check for logical inconsistencies
        if "decision" in config:
            decision_config = config["decision"]
            
            # Check threshold relationships
            if ("approval_threshold" in decision_config and 
                "rejection_threshold" in decision_config):
                if decision_config["approval_threshold"] <= decision_config["rejection_threshold"]:
                    runtime_errors.append(ValidationError(
                        field="decision.thresholds",
                        value=(decision_config["approval_threshold"], decision_config["rejection_threshold"]),
                        message="Approval threshold must be greater than rejection threshold"
                    ))
        
        # Check processing configuration against system resources
        if "processing" in config:
            processing_config = config["processing"]
            
            # Warn about high batch sizes
            if processing_config.get("batch_size", 0) > 1000:
                runtime_warnings.append(ValidationError(
                    field="processing.batch_size",
                    value=processing_config["batch_size"],
                    message="Large batch size may cause memory issues",
                    severity="warning"
                ))
            
            # Warn about high worker counts
            import os
            cpu_count = os.cpu_count() or 1
            if processing_config.get("max_workers", 0) > cpu_count * 2:
                runtime_warnings.append(ValidationError(
                    field="processing.max_workers",
                    value=processing_config["max_workers"],
                    message=f"Worker count ({processing_config['max_workers']}) is high for system with {cpu_count} CPUs",
                    severity="warning"
                ))
        
        # Combine results
        result.errors.extend(runtime_errors)
        result.warnings.extend(runtime_warnings)
        result.is_valid = result.is_valid and len(runtime_errors) == 0
        
        return result