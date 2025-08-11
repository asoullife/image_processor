# Configuration Documentation

This document describes the configuration system for the Adobe Stock Image Processor.

## Configuration Files

### Main Configuration File
- **Location**: `config/settings.json`
- **Format**: JSON
- **Purpose**: Main application configuration with all processing parameters

### Example Configuration File
- **Location**: `config/settings.example.json`
- **Purpose**: Example configuration with comments and explanations

### Default Configuration Template
- **Location**: `config/settings.default.json`
- **Purpose**: Default configuration template for new installations

## Configuration Structure

The configuration file is organized into the following sections:

### 1. Version Information
```json
{
  "version": "1.1"
}
```
- **version**: Configuration schema version (automatically managed)

### 2. Processing Configuration
```json
{
  "processing": {
    "batch_size": 200,
    "max_workers": 4,
    "checkpoint_interval": 50
  }
}
```

#### Parameters:
- **batch_size** (integer, 1-10000): Number of images to process in each batch
  - Default: 200
  - Recommendation: 100-500 for most systems
  - Higher values use more memory but may be faster

- **max_workers** (integer, 1-32): Maximum number of worker threads
  - Default: 4
  - Recommendation: 1-2x CPU core count
  - More workers can improve I/O performance but use more resources

- **checkpoint_interval** (integer, 1-1000): Save progress every N images
  - Default: 50
  - Lower values provide better resume capability but more disk I/O
  - Higher values are more efficient but lose more progress on crashes

### 3. Quality Analysis Configuration
```json
{
  "quality": {
    "min_sharpness": 100.0,
    "max_noise_level": 0.1,
    "min_resolution": [1920, 1080]
  }
}
```

#### Parameters:
- **min_sharpness** (float, ≥0): Minimum sharpness score for image acceptance
  - Default: 100.0
  - Higher values are more strict
  - Typical range: 50-200

- **max_noise_level** (float, 0-1): Maximum acceptable noise level
  - Default: 0.1
  - Lower values are more strict
  - 0.0 = no noise, 1.0 = maximum noise

- **min_resolution** (array of 2 integers): Minimum [width, height] in pixels
  - Default: [1920, 1080]
  - Adobe Stock typically requires high resolution images
  - Both dimensions must be ≥1

### 4. Similarity Detection Configuration
```json
{
  "similarity": {
    "hash_threshold": 5,
    "feature_threshold": 0.85,
    "clustering_eps": 0.3
  }
}
```

#### Parameters:
- **hash_threshold** (integer, 0-64): Hamming distance threshold for perceptual hashing
  - Default: 5
  - Lower values detect more similar images
  - 0 = identical only, 64 = very different allowed

- **feature_threshold** (float, 0-1): Similarity threshold for deep learning features
  - Default: 0.85
  - Higher values detect more similar images
  - 0.0 = very different, 1.0 = identical

- **clustering_eps** (float, 0-2): DBSCAN clustering epsilon parameter
  - Default: 0.3
  - Controls how close images must be to form clusters
  - Lower values create tighter clusters

### 5. Compliance Checking Configuration
```json
{
  "compliance": {
    "logo_detection_confidence": 0.7,
    "face_detection_enabled": true,
    "metadata_validation": true
  }
}
```

#### Parameters:
- **logo_detection_confidence** (float, 0-1): Minimum confidence for logo detection
  - Default: 0.7
  - Higher values reduce false positives but may miss some logos
  - 0.0 = detect everything, 1.0 = only very confident detections

- **face_detection_enabled** (boolean): Enable face detection for privacy checking
  - Default: true
  - Disable if you don't need privacy compliance checking

- **metadata_validation** (boolean): Enable metadata validation
  - Default: true
  - Checks EXIF data and keyword relevance

### 6. Decision Engine Configuration
```json
{
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
  }
}
```

#### Weight Parameters (must sum to 1.0):
- **quality_weight** (float, 0-1): Weight for quality analysis results
- **defect_weight** (float, 0-1): Weight for defect detection results
- **similarity_weight** (float, 0-1): Weight for similarity analysis results
- **compliance_weight** (float, 0-1): Weight for compliance checking results
- **technical_weight** (float, 0-1): Weight for technical specifications

#### Threshold Parameters (all 0-1):
- **approval_threshold**: Overall score needed for automatic approval
- **rejection_threshold**: Overall score below which images are rejected
- **quality_min_threshold**: Minimum quality score required
- **defect_max_threshold**: Maximum defect score allowed
- **similarity_max_threshold**: Maximum similarity score allowed
- **compliance_min_threshold**: Minimum compliance score required
- **quality_critical_threshold**: Quality score below which image is always rejected
- **defect_critical_threshold**: Defect score above which image is always rejected
- **compliance_critical_threshold**: Compliance score below which image is always rejected

### 7. Output Configuration
```json
{
  "output": {
    "images_per_folder": 200,
    "preserve_metadata": true,
    "generate_thumbnails": true
  }
}
```

#### Parameters:
- **images_per_folder** (integer, 1-10000): Number of images per output subfolder
  - Default: 200
  - Adobe Stock recommendation for batch uploads

- **preserve_metadata** (boolean): Preserve original image metadata
  - Default: true
  - Keeps EXIF data, keywords, etc.

- **generate_thumbnails** (boolean): Generate thumbnail images for reports
  - Default: true
  - Creates small preview images for HTML reports

### 8. Logging Configuration
```json
{
  "logging": {
    "level": "INFO",
    "file": "adobe_stock_processor.log",
    "max_file_size": "10MB",
    "backup_count": 5
  }
}
```

#### Parameters:
- **level** (string): Logging level
  - Choices: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
  - Default: "INFO"
  - DEBUG provides most detail, CRITICAL provides least

- **file** (string): Log file name
  - Default: "adobe_stock_processor.log"
  - Relative to application directory

- **max_file_size** (string): Maximum log file size before rotation
  - Default: "10MB"
  - Format: number + unit (B, KB, MB, GB, TB)

- **backup_count** (integer, 0-100): Number of backup log files to keep
  - Default: 5
  - 0 = no backups, higher values keep more history

## Configuration Management

### Validation
The configuration system automatically validates all parameters:
- Type checking (string, integer, float, boolean, array)
- Range validation (minimum/maximum values)
- Format validation (patterns, choices)
- Logical consistency checks
- Runtime environment checks

### Migration
Configuration files are automatically migrated when the schema version changes:
- Backup files are created before migration
- New fields are added with default values
- Deprecated fields are removed or renamed
- Migration messages are logged

### Default Generation
You can generate a default configuration file:
```python
from config.config_loader import ConfigLoader
loader = ConfigLoader("path/to/config.json")
loader.generate_default_config()
```

### Validation Checking
You can validate a configuration file without loading it:
```python
from config.config_loader import ConfigLoader
loader = ConfigLoader("path/to/config.json")
result = loader.validate_config_file()
if not result.is_valid:
    for error in result.errors:
        print(f"Error: {error.field}: {error.message}")
```

## Best Practices

### Performance Tuning
1. **Batch Size**: Start with 200, increase if you have more RAM
2. **Workers**: Use 1-2x your CPU core count
3. **Checkpoints**: Use 50 for most cases, lower if crashes are frequent

### Quality Settings
1. **Sharpness**: 100 is good for most images, lower for artistic/soft images
2. **Noise**: 0.1 is strict, increase to 0.2 for more tolerance
3. **Resolution**: Match Adobe Stock requirements for your market

### Decision Weights
1. **Quality-focused**: Increase quality_weight to 0.4-0.5
2. **Compliance-focused**: Increase compliance_weight to 0.3-0.4
3. **Balanced**: Use default weights (quality=0.35, defect=0.25, etc.)

### Troubleshooting
1. **Memory Issues**: Reduce batch_size and max_workers
2. **Slow Processing**: Increase max_workers (up to CPU count)
3. **Too Many Rejections**: Lower thresholds or adjust weights
4. **Too Few Rejections**: Raise thresholds or increase strict weights

## Error Messages

### Common Validation Errors
- **"Required field 'X' is missing"**: Add the missing configuration section
- **"Expected type int, got str"**: Check data types in JSON
- **"Value X is below minimum Y"**: Increase the parameter value
- **"Decision weights must sum to 1.0"**: Adjust weight values to total 1.0
- **"Unknown field 'X' will be ignored"**: Remove deprecated configuration fields

### Runtime Warnings
- **"Large batch size may cause memory issues"**: Consider reducing batch_size
- **"Worker count is high for system"**: Reduce max_workers for your CPU
- **"Approval threshold must be greater than rejection threshold"**: Fix threshold logic

## Migration Guide

### From Version 1.0 to 1.1
- Version field is automatically added
- All existing configurations remain compatible
- New fields are added with default values
- Backup files are created automatically

### Manual Migration Steps
1. Backup your current configuration
2. Run the application - migration happens automatically
3. Review migration messages in the log
4. Adjust new parameters as needed
5. Test with a small batch of images

## Support

For configuration issues:
1. Check the log file for detailed error messages
2. Validate your configuration using the built-in validator
3. Compare with the example configuration file
4. Reset to defaults if needed and reconfigure gradually