"""Unit tests for configuration validation system."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
import tempfile
import json
import os
from unittest.mock import patch, MagicMock
import logging

from backend.config.config_validator import (
    ConfigSchema, ConfigValidator, ConfigMigrator, ValidationError, 
    ValidationResult, ConfigVersion
)


class TestValidationError(unittest.TestCase):
    """Test cases for ValidationError dataclass."""
    
    def test_validation_error_creation(self):
        """Test ValidationError creation."""
        error = ValidationError(
            field="test_field",
            value="test_value",
            message="test message"
        )
        
        self.assertEqual(error.field, "test_field")
        self.assertEqual(error.value, "test_value")
        self.assertEqual(error.message, "test message")
        self.assertEqual(error.severity, "error")
    
    def test_validation_error_with_warning(self):
        """Test ValidationError with warning severity."""
        warning = ValidationError(
            field="test_field",
            value="test_value",
            message="test warning",
            severity="warning"
        )
        
        self.assertEqual(warning.severity, "warning")


class TestValidationResult(unittest.TestCase):
    """Test cases for ValidationResult dataclass."""
    
    def test_validation_result_valid(self):
        """Test ValidationResult for valid configuration."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[]
        )
        
        self.assertTrue(result.is_valid)
        self.assertFalse(result.has_errors())
        self.assertFalse(result.has_warnings())
    
    def test_validation_result_with_errors(self):
        """Test ValidationResult with errors."""
        error = ValidationError("field", "value", "message")
        result = ValidationResult(
            is_valid=False,
            errors=[error],
            warnings=[]
        )
        
        self.assertFalse(result.is_valid)
        self.assertTrue(result.has_errors())
        self.assertFalse(result.has_warnings())
    
    def test_validation_result_with_warnings(self):
        """Test ValidationResult with warnings."""
        warning = ValidationError("field", "value", "message", "warning")
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[warning]
        )
        
        self.assertTrue(result.is_valid)
        self.assertFalse(result.has_errors())
        self.assertTrue(result.has_warnings())


class TestConfigSchema(unittest.TestCase):
    """Test cases for ConfigSchema validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.schema = ConfigSchema()
    
    def test_validate_field_integer_valid(self):
        """Test integer field validation with valid value."""
        errors = self.schema.validate_field(
            "test_field", 
            100, 
            {"type": int, "min": 1, "max": 1000}
        )
        self.assertEqual(len(errors), 0)
    
    def test_validate_field_integer_invalid_type(self):
        """Test integer field validation with invalid type."""
        errors = self.schema.validate_field(
            "test_field", 
            "not_an_int", 
            {"type": int}
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("Expected type int", errors[0].message)
    
    def test_validate_field_integer_below_minimum(self):
        """Test integer field validation below minimum."""
        errors = self.schema.validate_field(
            "test_field", 
            0, 
            {"type": int, "min": 1}
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("below minimum", errors[0].message)
    
    def test_validate_field_integer_above_maximum(self):
        """Test integer field validation above maximum."""
        errors = self.schema.validate_field(
            "test_field", 
            1001, 
            {"type": int, "max": 1000}
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("above maximum", errors[0].message)
    
    def test_validate_field_float_valid(self):
        """Test float field validation with valid value."""
        errors = self.schema.validate_field(
            "test_field", 
            0.5, 
            {"type": (int, float), "min": 0, "max": 1}
        )
        self.assertEqual(len(errors), 0)
    
    def test_validate_field_string_pattern_valid(self):
        """Test string field validation with valid pattern."""
        errors = self.schema.validate_field(
            "version", 
            "1.0", 
            {"type": str, "pattern": r"^\d+\.\d+$"}
        )
        self.assertEqual(len(errors), 0)
    
    def test_validate_field_string_pattern_invalid(self):
        """Test string field validation with invalid pattern."""
        errors = self.schema.validate_field(
            "version", 
            "invalid", 
            {"type": str, "pattern": r"^\d+\.\d+$"}
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("does not match pattern", errors[0].message)
    
    def test_validate_field_string_choices_valid(self):
        """Test string field validation with valid choice."""
        errors = self.schema.validate_field(
            "level", 
            "INFO", 
            {"type": str, "choices": ["DEBUG", "INFO", "WARNING", "ERROR"]}
        )
        self.assertEqual(len(errors), 0)
    
    def test_validate_field_string_choices_invalid(self):
        """Test string field validation with invalid choice."""
        errors = self.schema.validate_field(
            "level", 
            "INVALID", 
            {"type": str, "choices": ["DEBUG", "INFO", "WARNING", "ERROR"]}
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("not in allowed choices", errors[0].message)
    
    def test_validate_field_list_valid(self):
        """Test list field validation with valid value."""
        errors = self.schema.validate_field(
            "resolution", 
            [1920, 1080], 
            {"type": list, "length": 2, "item_type": int, "item_min": 1}
        )
        self.assertEqual(len(errors), 0)
    
    def test_validate_field_list_wrong_length(self):
        """Test list field validation with wrong length."""
        errors = self.schema.validate_field(
            "resolution", 
            [1920], 
            {"type": list, "length": 2}
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("does not match required length", errors[0].message)
    
    def test_validate_field_list_wrong_item_type(self):
        """Test list field validation with wrong item type."""
        errors = self.schema.validate_field(
            "resolution", 
            ["1920", "1080"], 
            {"type": list, "item_type": int}
        )
        self.assertEqual(len(errors), 2)  # One error per item
        self.assertIn("does not match required type", errors[0].message)
    
    def test_validate_field_list_item_below_minimum(self):
        """Test list field validation with item below minimum."""
        errors = self.schema.validate_field(
            "resolution", 
            [0, 1080], 
            {"type": list, "item_type": int, "item_min": 1}
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("below minimum", errors[0].message)
    
    def test_validate_config_valid(self):
        """Test complete configuration validation with valid config."""
        config = {
            "processing": {
                "batch_size": 200,
                "max_workers": 4,
                "checkpoint_interval": 50
            },
            "quality": {
                "min_sharpness": 100.0,
                "max_noise_level": 0.1,
                "min_resolution": [1920, 1080]
            },
            "similarity": {
                "hash_threshold": 5,
                "feature_threshold": 0.85,
                "clustering_eps": 0.3
            },
            "compliance": {
                "logo_detection_confidence": 0.7,
                "face_detection_enabled": True,
                "metadata_validation": True
            },
            "decision": {
                "quality_weight": 0.35,
                "defect_weight": 0.25,
                "similarity_weight": 0.20,
                "compliance_weight": 0.15,
                "technical_weight": 0.05,
                "approval_threshold": 0.75,
                "rejection_threshold": 0.40,
                "quality_min_threshold": 0.60,
                "defect_max_threshold": 0.30,
                "similarity_max_threshold": 0.85,
                "compliance_min_threshold": 0.80,
                "quality_critical_threshold": 0.30,
                "defect_critical_threshold": 0.70,
                "compliance_critical_threshold": 0.50
            },
            "output": {
                "images_per_folder": 200,
                "preserve_metadata": True,
                "generate_thumbnails": True
            },
            "logging": {
                "level": "INFO",
                "file": "test.log",
                "max_file_size": "10MB",
                "backup_count": 5
            }
        }
        
        result = self.schema.validate_config(config)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
    
    def test_validate_config_missing_required_field(self):
        """Test configuration validation with missing required field."""
        config = {
            "quality": {
                "min_sharpness": 100.0,
                "max_noise_level": 0.1,
                "min_resolution": [1920, 1080]
            }
            # Missing required 'processing' section
        }
        
        result = self.schema.validate_config(config)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("processing" in error.field for error in result.errors))
    
    def test_validate_config_invalid_decision_weights(self):
        """Test configuration validation with invalid decision weights."""
        config = {
            "processing": {"batch_size": 200, "max_workers": 4, "checkpoint_interval": 50},
            "quality": {"min_sharpness": 100.0, "max_noise_level": 0.1, "min_resolution": [1920, 1080]},
            "similarity": {"hash_threshold": 5, "feature_threshold": 0.85, "clustering_eps": 0.3},
            "compliance": {"logo_detection_confidence": 0.7, "face_detection_enabled": True, "metadata_validation": True},
            "decision": {
                "quality_weight": 0.5,  # These don't sum to 1.0
                "defect_weight": 0.3,
                "similarity_weight": 0.1,
                "compliance_weight": 0.05,
                "technical_weight": 0.01,
                "approval_threshold": 0.75,
                "rejection_threshold": 0.40,
                "quality_min_threshold": 0.60,
                "defect_max_threshold": 0.30,
                "similarity_max_threshold": 0.85,
                "compliance_min_threshold": 0.80,
                "quality_critical_threshold": 0.30,
                "defect_critical_threshold": 0.70,
                "compliance_critical_threshold": 0.50
            },
            "output": {"images_per_folder": 200, "preserve_metadata": True, "generate_thumbnails": True},
            "logging": {"level": "INFO", "file": "test.log", "max_file_size": "10MB", "backup_count": 5}
        }
        
        result = self.schema.validate_config(config)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("weights must sum to 1.0" in error.message for error in result.errors))
    
    def test_validate_config_unknown_fields(self):
        """Test configuration validation with unknown fields."""
        config = {
            "processing": {"batch_size": 200, "max_workers": 4, "checkpoint_interval": 50},
            "quality": {"min_sharpness": 100.0, "max_noise_level": 0.1, "min_resolution": [1920, 1080]},
            "similarity": {"hash_threshold": 5, "feature_threshold": 0.85, "clustering_eps": 0.3},
            "compliance": {"logo_detection_confidence": 0.7, "face_detection_enabled": True, "metadata_validation": True},
            "decision": {
                "quality_weight": 0.35, "defect_weight": 0.25, "similarity_weight": 0.20,
                "compliance_weight": 0.15, "technical_weight": 0.05,
                "approval_threshold": 0.75, "rejection_threshold": 0.40,
                "quality_min_threshold": 0.60, "defect_max_threshold": 0.30,
                "similarity_max_threshold": 0.85, "compliance_min_threshold": 0.80,
                "quality_critical_threshold": 0.30, "defect_critical_threshold": 0.70,
                "compliance_critical_threshold": 0.50
            },
            "output": {"images_per_folder": 200, "preserve_metadata": True, "generate_thumbnails": True},
            "logging": {"level": "INFO", "file": "test.log", "max_file_size": "10MB", "backup_count": 5},
            "unknown_section": {"unknown_field": "unknown_value"}  # Unknown field
        }
        
        result = self.schema.validate_config(config)
        self.assertTrue(result.is_valid)  # Should still be valid
        self.assertTrue(result.has_warnings())
        self.assertTrue(any("Unknown field" in warning.message for warning in result.warnings))
    
    def test_generate_default_config(self):
        """Test default configuration generation."""
        default_config = self.schema.generate_default_config()
        
        # Check that all required sections are present
        self.assertIn("processing", default_config)
        self.assertIn("quality", default_config)
        self.assertIn("similarity", default_config)
        self.assertIn("compliance", default_config)
        self.assertIn("decision", default_config)
        self.assertIn("output", default_config)
        self.assertIn("logging", default_config)
        
        # Check some specific default values
        self.assertEqual(default_config["processing"]["batch_size"], 200)
        self.assertEqual(default_config["quality"]["min_sharpness"], 100.0)
        self.assertEqual(default_config["logging"]["level"], "INFO")
        
        # Validate the generated config
        result = self.schema.validate_config(default_config)
        self.assertTrue(result.is_valid)


class TestConfigMigrator(unittest.TestCase):
    """Test cases for ConfigMigrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.migrator = ConfigMigrator()
    
    def test_get_config_version_with_version(self):
        """Test getting configuration version when version field exists."""
        config = {"version": "1.0", "other": "data"}
        version = self.migrator.get_config_version(config)
        self.assertEqual(version, "1.0")
    
    def test_get_config_version_without_version(self):
        """Test getting configuration version when version field is missing."""
        config = {"other": "data"}
        version = self.migrator.get_config_version(config)
        self.assertEqual(version, "1.0")  # Default version
    
    def test_migrate_config_current_version(self):
        """Test migration when config is already current version."""
        config = {"version": ConfigVersion.CURRENT.value, "other": "data"}
        migrated_config, messages = self.migrator.migrate_config(config)
        
        self.assertEqual(migrated_config["version"], ConfigVersion.CURRENT.value)
        self.assertEqual(len(messages), 0)
    
    def test_migrate_config_from_v1_0(self):
        """Test migration from version 1.0."""
        config = {
            "processing": {"batch_size": 200, "max_workers": 4, "checkpoint_interval": 50},
            "quality": {"min_sharpness": 100.0, "max_noise_level": 0.1, "min_resolution": [1920, 1080]}
            # Missing some sections that should be added
        }
        
        migrated_config, messages = self.migrator.migrate_config(config)
        
        self.assertEqual(migrated_config["version"], ConfigVersion.CURRENT.value)
        self.assertTrue(len(messages) > 0)
        self.assertTrue(any("migrated to version" in msg for msg in messages))
        
        # Check that missing sections were added
        self.assertIn("similarity", migrated_config)
        self.assertIn("compliance", migrated_config)
        self.assertIn("decision", migrated_config)
        self.assertIn("output", migrated_config)
        self.assertIn("logging", migrated_config)


class TestConfigValidator(unittest.TestCase):
    """Test cases for ConfigValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        self.logger = MagicMock()
        self.validator = ConfigValidator(self.logger)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_validate_config_file_not_found(self):
        """Test validation of non-existent configuration file."""
        result = self.validator.validate_config_file("nonexistent.json")
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any("not found" in error.message for error in result.errors))
    
    def test_validate_config_file_invalid_json(self):
        """Test validation of configuration file with invalid JSON."""
        with open(self.config_path, 'w') as f:
            f.write("{ invalid json }")
        
        result = self.validator.validate_config_file(self.config_path)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any("Invalid JSON" in error.message for error in result.errors))
    
    def test_validate_config_file_valid(self):
        """Test validation of valid configuration file."""
        valid_config = {
            "processing": {"batch_size": 200, "max_workers": 4, "checkpoint_interval": 50},
            "quality": {"min_sharpness": 100.0, "max_noise_level": 0.1, "min_resolution": [1920, 1080]},
            "similarity": {"hash_threshold": 5, "feature_threshold": 0.85, "clustering_eps": 0.3},
            "compliance": {"logo_detection_confidence": 0.7, "face_detection_enabled": True, "metadata_validation": True},
            "decision": {
                "quality_weight": 0.35, "defect_weight": 0.25, "similarity_weight": 0.20,
                "compliance_weight": 0.15, "technical_weight": 0.05,
                "approval_threshold": 0.75, "rejection_threshold": 0.40,
                "quality_min_threshold": 0.60, "defect_max_threshold": 0.30,
                "similarity_max_threshold": 0.85, "compliance_min_threshold": 0.80,
                "quality_critical_threshold": 0.30, "defect_critical_threshold": 0.70,
                "compliance_critical_threshold": 0.50
            },
            "output": {"images_per_folder": 200, "preserve_metadata": True, "generate_thumbnails": True},
            "logging": {"level": "INFO", "file": "test.log", "max_file_size": "10MB", "backup_count": 5}
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(valid_config, f)
        
        result = self.validator.validate_config_file(self.config_path)
        self.assertTrue(result.is_valid)
    
    def test_migrate_config_file_success(self):
        """Test successful configuration file migration."""
        old_config = {
            "processing": {"batch_size": 200, "max_workers": 4, "checkpoint_interval": 50},
            "quality": {"min_sharpness": 100.0, "max_noise_level": 0.1, "min_resolution": [1920, 1080]}
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(old_config, f)
        
        success, messages = self.validator.migrate_config_file(self.config_path, backup=False)
        
        self.assertTrue(success)
        self.assertTrue(len(messages) > 0)
        
        # Check that file was updated
        with open(self.config_path, 'r') as f:
            migrated_config = json.load(f)
        
        self.assertEqual(migrated_config["version"], ConfigVersion.CURRENT.value)
        self.assertIn("similarity", migrated_config)
    
    def test_migrate_config_file_with_backup(self):
        """Test configuration file migration with backup."""
        old_config = {
            "processing": {"batch_size": 200, "max_workers": 4, "checkpoint_interval": 50},
            "quality": {"min_sharpness": 100.0, "max_noise_level": 0.1, "min_resolution": [1920, 1080]}
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(old_config, f)
        
        success, messages = self.validator.migrate_config_file(self.config_path, backup=True)
        
        self.assertTrue(success)
        
        # Check that backup was created
        backup_path = f"{self.config_path}.backup.1.0"
        self.assertTrue(os.path.exists(backup_path))
    
    def test_generate_default_config_file_success(self):
        """Test successful default configuration file generation."""
        success = self.validator.generate_default_config_file(self.config_path)
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(self.config_path))
        
        # Validate generated config
        result = self.validator.validate_config_file(self.config_path)
        self.assertTrue(result.is_valid)
    
    def test_generate_default_config_file_exists_no_overwrite(self):
        """Test default config generation when file exists and overwrite is False."""
        # Create existing file
        with open(self.config_path, 'w') as f:
            f.write("{}")
        
        success = self.validator.generate_default_config_file(self.config_path, overwrite=False)
        
        self.assertFalse(success)
    
    def test_generate_default_config_file_exists_overwrite(self):
        """Test default config generation when file exists and overwrite is True."""
        # Create existing file
        with open(self.config_path, 'w') as f:
            f.write("{}")
        
        success = self.validator.generate_default_config_file(self.config_path, overwrite=True)
        
        self.assertTrue(success)
        
        # Validate generated config
        result = self.validator.validate_config_file(self.config_path)
        self.assertTrue(result.is_valid)
    
    def test_validate_runtime_config_threshold_inconsistency(self):
        """Test runtime validation with threshold inconsistencies."""
        config = {
            "processing": {"batch_size": 200, "max_workers": 4, "checkpoint_interval": 50},
            "quality": {"min_sharpness": 100.0, "max_noise_level": 0.1, "min_resolution": [1920, 1080]},
            "similarity": {"hash_threshold": 5, "feature_threshold": 0.85, "clustering_eps": 0.3},
            "compliance": {"logo_detection_confidence": 0.7, "face_detection_enabled": True, "metadata_validation": True},
            "decision": {
                "quality_weight": 0.35, "defect_weight": 0.25, "similarity_weight": 0.20,
                "compliance_weight": 0.15, "technical_weight": 0.05,
                "approval_threshold": 0.30,  # Lower than rejection threshold
                "rejection_threshold": 0.75,  # Higher than approval threshold
                "quality_min_threshold": 0.60, "defect_max_threshold": 0.30,
                "similarity_max_threshold": 0.85, "compliance_min_threshold": 0.80,
                "quality_critical_threshold": 0.30, "defect_critical_threshold": 0.70,
                "compliance_critical_threshold": 0.50
            },
            "output": {"images_per_folder": 200, "preserve_metadata": True, "generate_thumbnails": True},
            "logging": {"level": "INFO", "file": "test.log", "max_file_size": "10MB", "backup_count": 5}
        }
        
        result = self.validator.validate_runtime_config(config)
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any("Approval threshold must be greater than rejection threshold" in error.message 
                          for error in result.errors))
    
    @patch('os.cpu_count')
    def test_validate_runtime_config_high_worker_count(self, mock_cpu_count):
        """Test runtime validation with high worker count warning."""
        mock_cpu_count.return_value = 4
        
        config = {
            "processing": {"batch_size": 200, "max_workers": 16, "checkpoint_interval": 50},  # High worker count
            "quality": {"min_sharpness": 100.0, "max_noise_level": 0.1, "min_resolution": [1920, 1080]},
            "similarity": {"hash_threshold": 5, "feature_threshold": 0.85, "clustering_eps": 0.3},
            "compliance": {"logo_detection_confidence": 0.7, "face_detection_enabled": True, "metadata_validation": True},
            "decision": {
                "quality_weight": 0.35, "defect_weight": 0.25, "similarity_weight": 0.20,
                "compliance_weight": 0.15, "technical_weight": 0.05,
                "approval_threshold": 0.75, "rejection_threshold": 0.40,
                "quality_min_threshold": 0.60, "defect_max_threshold": 0.30,
                "similarity_max_threshold": 0.85, "compliance_min_threshold": 0.80,
                "quality_critical_threshold": 0.30, "defect_critical_threshold": 0.70,
                "compliance_critical_threshold": 0.50
            },
            "output": {"images_per_folder": 200, "preserve_metadata": True, "generate_thumbnails": True},
            "logging": {"level": "INFO", "file": "test.log", "max_file_size": "10MB", "backup_count": 5}
        }
        
        result = self.validator.validate_runtime_config(config)
        
        self.assertTrue(result.is_valid)  # Should still be valid
        self.assertTrue(result.has_warnings())
        self.assertTrue(any("Worker count" in warning.message and "is high for system" in warning.message 
                          for warning in result.warnings))
    
    def test_validate_runtime_config_large_batch_size(self):
        """Test runtime validation with large batch size warning."""
        config = {
            "processing": {"batch_size": 2000, "max_workers": 4, "checkpoint_interval": 50},  # Large batch size
            "quality": {"min_sharpness": 100.0, "max_noise_level": 0.1, "min_resolution": [1920, 1080]},
            "similarity": {"hash_threshold": 5, "feature_threshold": 0.85, "clustering_eps": 0.3},
            "compliance": {"logo_detection_confidence": 0.7, "face_detection_enabled": True, "metadata_validation": True},
            "decision": {
                "quality_weight": 0.35, "defect_weight": 0.25, "similarity_weight": 0.20,
                "compliance_weight": 0.15, "technical_weight": 0.05,
                "approval_threshold": 0.75, "rejection_threshold": 0.40,
                "quality_min_threshold": 0.60, "defect_max_threshold": 0.30,
                "similarity_max_threshold": 0.85, "compliance_min_threshold": 0.80,
                "quality_critical_threshold": 0.30, "defect_critical_threshold": 0.70,
                "compliance_critical_threshold": 0.50
            },
            "output": {"images_per_folder": 200, "preserve_metadata": True, "generate_thumbnails": True},
            "logging": {"level": "INFO", "file": "test.log", "max_file_size": "10MB", "backup_count": 5}
        }
        
        result = self.validator.validate_runtime_config(config)
        
        self.assertTrue(result.is_valid)  # Should still be valid
        self.assertTrue(result.has_warnings())
        self.assertTrue(any("Large batch size may cause memory issues" in warning.message 
                          for warning in result.warnings))


if __name__ == '__main__':
    unittest.main()