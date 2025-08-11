#!/usr/bin/env python3
"""
Demo script for configuration validation and management system.

This script demonstrates the enhanced configuration validation, migration,
and management capabilities of the Adobe Stock Image Processor.
"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
import os
from backend.config.config_validator import ConfigValidator, ConfigSchema
from backend.config.config_loader import ConfigLoader


def demo_schema_validation():
    """Demonstrate configuration schema validation."""
    print("=" * 60)
    print("CONFIGURATION SCHEMA VALIDATION DEMO")
    print("=" * 60)
    
    validator = ConfigValidator()
    
    # Test 1: Valid configuration
    print("\n1. Testing valid configuration:")
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
    
    result = validator.validate_config(valid_config)
    print(f"   Valid: {result.is_valid}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Warnings: {len(result.warnings)}")
    
    # Test 2: Invalid configuration - missing required field
    print("\n2. Testing configuration with missing required field:")
    invalid_config = {
        "quality": {"min_sharpness": 100.0, "max_noise_level": 0.1, "min_resolution": [1920, 1080]}
        # Missing required 'processing' section
    }
    
    result = validator.validate_config(invalid_config)
    print(f"   Valid: {result.is_valid}")
    print(f"   Errors: {len(result.errors)}")
    if result.errors:
        print("   Error details:")
        for error in result.errors[:3]:  # Show first 3 errors
            print(f"     - {error.field}: {error.message}")
    
    # Test 3: Invalid configuration - wrong data types
    print("\n3. Testing configuration with wrong data types:")
    wrong_type_config = {
        "processing": {"batch_size": "not_a_number", "max_workers": 4, "checkpoint_interval": 50},
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
    
    result = validator.validate_config(wrong_type_config)
    print(f"   Valid: {result.is_valid}")
    print(f"   Errors: {len(result.errors)}")
    if result.errors:
        print("   Error details:")
        for error in result.errors:
            print(f"     - {error.field}: {error.message}")
    
    # Test 4: Configuration with warnings (unknown fields)
    print("\n4. Testing configuration with unknown fields (warnings):")
    warning_config = valid_config.copy()
    warning_config["unknown_section"] = {"unknown_field": "value"}
    warning_config["processing"]["unknown_field"] = "value"
    
    result = validator.validate_config(warning_config)
    print(f"   Valid: {result.is_valid}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Warnings: {len(result.warnings)}")
    if result.warnings:
        print("   Warning details:")
        for warning in result.warnings:
            print(f"     - {warning.field}: {warning.message}")


def demo_runtime_validation():
    """Demonstrate runtime configuration validation."""
    print("\n" + "=" * 60)
    print("RUNTIME CONFIGURATION VALIDATION DEMO")
    print("=" * 60)
    
    validator = ConfigValidator()
    
    # Test 1: Threshold inconsistency
    print("\n1. Testing threshold inconsistency:")
    inconsistent_config = {
        "processing": {"batch_size": 200, "max_workers": 4, "checkpoint_interval": 50},
        "quality": {"min_sharpness": 100.0, "max_noise_level": 0.1, "min_resolution": [1920, 1080]},
        "similarity": {"hash_threshold": 5, "feature_threshold": 0.85, "clustering_eps": 0.3},
        "compliance": {"logo_detection_confidence": 0.7, "face_detection_enabled": True, "metadata_validation": True},
        "decision": {
            "quality_weight": 0.35, "defect_weight": 0.25, "similarity_weight": 0.20,
            "compliance_weight": 0.15, "technical_weight": 0.05,
            "approval_threshold": 0.30,  # Lower than rejection threshold!
            "rejection_threshold": 0.75,
            "quality_min_threshold": 0.60, "defect_max_threshold": 0.30,
            "similarity_max_threshold": 0.85, "compliance_min_threshold": 0.80,
            "quality_critical_threshold": 0.30, "defect_critical_threshold": 0.70,
            "compliance_critical_threshold": 0.50
        },
        "output": {"images_per_folder": 200, "preserve_metadata": True, "generate_thumbnails": True},
        "logging": {"level": "INFO", "file": "test.log", "max_file_size": "10MB", "backup_count": 5}
    }
    
    result = validator.validate_runtime_config(inconsistent_config)
    print(f"   Valid: {result.is_valid}")
    print(f"   Errors: {len(result.errors)}")
    if result.errors:
        print("   Error details:")
        for error in result.errors:
            print(f"     - {error.field}: {error.message}")
    
    # Test 2: Performance warnings
    print("\n2. Testing performance warnings:")
    performance_config = {
        "processing": {"batch_size": 2000, "max_workers": 16, "checkpoint_interval": 50},  # High values
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
    
    result = validator.validate_runtime_config(performance_config)
    print(f"   Valid: {result.is_valid}")
    print(f"   Warnings: {len(result.warnings)}")
    if result.warnings:
        print("   Warning details:")
        for warning in result.warnings:
            print(f"     - {warning.field}: {warning.message}")


def demo_default_config_generation():
    """Demonstrate default configuration generation."""
    print("\n" + "=" * 60)
    print("DEFAULT CONFIGURATION GENERATION DEMO")
    print("=" * 60)
    
    # Generate default configuration
    schema = ConfigSchema()
    default_config = schema.generate_default_config()
    
    print("\n1. Generated default configuration structure:")
    for section_name in default_config.keys():
        print(f"   - {section_name}")
        if isinstance(default_config[section_name], dict):
            for field_name in default_config[section_name].keys():
                print(f"     - {field_name}: {default_config[section_name][field_name]}")
    
    # Validate the default configuration
    validator = ConfigValidator()
    result = validator.validate_config(default_config)
    print(f"\n2. Default configuration validation:")
    print(f"   Valid: {result.is_valid}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Warnings: {len(result.warnings)}")
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(default_config, f, indent=2)
        temp_path = f.name
    
    print(f"\n3. Default configuration saved to: {temp_path}")
    
    # Clean up
    os.unlink(temp_path)


def demo_config_migration():
    """Demonstrate configuration migration."""
    print("\n" + "=" * 60)
    print("CONFIGURATION MIGRATION DEMO")
    print("=" * 60)
    
    # Create old version configuration (v1.0)
    old_config = {
        "processing": {"batch_size": 200, "max_workers": 4, "checkpoint_interval": 50},
        "quality": {"min_sharpness": 100.0, "max_noise_level": 0.1, "min_resolution": [1920, 1080]}
        # Missing newer sections
    }
    
    print("\n1. Original configuration (v1.0):")
    print(f"   Sections: {list(old_config.keys())}")
    print(f"   Version: {old_config.get('version', 'not specified (defaults to 1.0)')}")
    
    # Migrate configuration
    from backend.config.config_validator import ConfigMigrator
    migrator = ConfigMigrator()
    migrated_config, messages = migrator.migrate_config(old_config)
    
    print(f"\n2. Migration results:")
    print(f"   Success: {len(messages) > 0}")
    print(f"   Messages: {len(messages)}")
    for message in messages:
        print(f"     - {message}")
    
    print(f"\n3. Migrated configuration:")
    print(f"   Sections: {list(migrated_config.keys())}")
    print(f"   Version: {migrated_config.get('version')}")
    
    # Validate migrated configuration
    validator = ConfigValidator()
    result = validator.validate_config(migrated_config)
    print(f"\n4. Migrated configuration validation:")
    print(f"   Valid: {result.is_valid}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Warnings: {len(result.warnings)}")


def demo_config_file_operations():
    """Demonstrate configuration file operations."""
    print("\n" + "=" * 60)
    print("CONFIGURATION FILE OPERATIONS DEMO")
    print("=" * 60)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    config_path = os.path.join(temp_dir, "demo_config.json")
    
    try:
        validator = ConfigValidator()
        
        # Test 1: Generate default configuration file
        print("\n1. Generating default configuration file:")
        success = validator.generate_default_config_file(config_path)
        print(f"   Success: {success}")
        print(f"   File exists: {os.path.exists(config_path)}")
        
        # Test 2: Validate configuration file
        print("\n2. Validating configuration file:")
        result = validator.validate_config_file(config_path)
        print(f"   Valid: {result.is_valid}")
        print(f"   Errors: {len(result.errors)}")
        print(f"   Warnings: {len(result.warnings)}")
        
        # Test 3: Load configuration with ConfigLoader
        print("\n3. Loading configuration with ConfigLoader:")
        loader = ConfigLoader(config_path)
        try:
            config = loader.load_config()
            print("   Configuration loaded successfully")
            print(f"   Batch size: {config.processing.batch_size}")
            print(f"   Quality min sharpness: {config.quality.min_sharpness}")
            print(f"   Logging level: {config.logging.level}")
        except Exception as e:
            print(f"   Error loading configuration: {e}")
        
        # Test 4: Test validation methods
        print("\n4. Testing ConfigLoader validation methods:")
        errors = loader.get_validation_errors()
        warnings = loader.get_validation_warnings()
        print(f"   Validation errors: {len(errors)}")
        print(f"   Validation warnings: {len(warnings)}")
        
        # Test 5: Create invalid configuration and test error reporting
        print("\n5. Testing error reporting with invalid configuration:")
        invalid_config = {"processing": {"batch_size": "invalid"}}
        with open(config_path, 'w') as f:
            json.dump(invalid_config, f)
        
        result = validator.validate_config_file(config_path)
        print(f"   Valid: {result.is_valid}")
        print(f"   Errors: {len(result.errors)}")
        if result.errors:
            print("   Error details:")
            for error in result.errors[:3]:
                print(f"     - {error.field}: {error.message}")
    
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)


def main():
    """Run all configuration validation demos."""
    print("Adobe Stock Image Processor - Configuration Validation Demo")
    print("This demo showcases the enhanced configuration validation and management system.")
    
    try:
        demo_schema_validation()
        demo_runtime_validation()
        demo_default_config_generation()
        demo_config_migration()
        demo_config_file_operations()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey features demonstrated:")
        print("✓ Comprehensive schema validation with detailed error reporting")
        print("✓ Runtime validation with performance warnings")
        print("✓ Automatic default configuration generation")
        print("✓ Configuration migration system for version updates")
        print("✓ File-based configuration management")
        print("✓ Integration with existing ConfigLoader system")
        print("\nThe configuration system is now ready for production use!")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()