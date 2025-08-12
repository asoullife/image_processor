# Adobe Stock Image Processor - Setup Guide

This guide provides detailed instructions for setting up and configuring the Adobe Stock Image Processor for different use cases and environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Methods](#installation-methods)
3. [Configuration](#configuration)
4. [First Run](#first-run)
5. [Performance Tuning](#performance-tuning)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Setup](#advanced-setup)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- Python 3.8 or higher
- 8GB RAM (16GB recommended)
- 50GB free disk space
- Multi-core CPU (4+ cores recommended)

**Recommended Requirements:**
- Python 3.9 or 3.10
- 16GB+ RAM
- 100GB+ free SSD storage
- 8+ core CPU
- GPU with CUDA support (optional, for deep learning features)

### Operating System Support

- **Windows**: Windows 10/11 (64-bit)
- **macOS**: macOS 10.15 (Catalina) or later
- **Linux**: Ubuntu 18.04+, CentOS 7+, or equivalent

## Installation Methods

### Method 1: Automated Installation (Recommended)

1. **Download the project**
   ```bash
   git clone <repository-url>
   cd adobe-stock-image-processor
   ```

2. **Run the installer**
   ```bash
   python install.py
   ```

   The installer will:
   - Check system requirements
   - Install Python dependencies
   - Create configuration files
   - Set up test data
   - Run verification tests

3. **Installation options**
   ```bash
   # Upgrade existing installation
   python install.py --upgrade
   
   # Skip tests during installation
   python install.py --skip-tests
   ```

### Method 2: Manual Installation

1. **Install Python dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

2. **Create configuration**
   ```bash
   cp config/settings.example.json config/settings.json
   ```

3. **Initialize test data**
   ```bash
   python create_test_images.py
   ```

4. **Verify installation**
   ```bash
   python test_installation.py
   ```

### Method 3: Virtual Environment Installation

1. **Create virtual environment**
   ```bash
   python -m venv adobe_stock_env
   
   # Windows
   adobe_stock_env\Scripts\activate
   
   # macOS/Linux
   source adobe_stock_env/bin/activate
   ```

2. **Install in virtual environment**
   ```bash
   pip install --upgrade pip
   python install.py
   ```

### Method 4: Docker Installation (Advanced)

1. **Build Docker image**
   ```bash
   docker build -t adobe-stock-processor .
   ```

2. **Run container**
   ```bash
   docker run -v /path/to/images:/input -v /path/to/output:/output adobe-stock-processor
   ```

## Configuration

### Basic Configuration

The main configuration file is `config/settings.json`. Start with the example:

```bash
cp config/settings.example.json config/settings.json
```

### Configuration Profiles

Choose a configuration profile based on your use case:

#### Development/Testing
```bash
cp config/settings.development.json config/settings.json
```
- Small batch sizes
- Relaxed quality thresholds
- Minimal resource usage
- Detailed logging

#### Production/Large Batches
```bash
cp config/settings.production.json config/settings.json
```
- Large batch sizes
- Strict quality thresholds
- Optimized for performance
- Comprehensive analysis

### Key Configuration Settings

#### Processing Settings
```json
{
  "processing": {
    "batch_size": 200,        // Images per batch (100-500)
    "max_workers": 4,         // Thread count (CPU cores)
    "checkpoint_interval": 50, // Save progress every N images
    "memory_limit_mb": 8192   // Memory limit in MB
  }
}
```

#### Quality Thresholds
```json
{
  "quality": {
    "min_sharpness": 100.0,      // Minimum sharpness score
    "max_noise_level": 0.1,      // Maximum noise (0.0-1.0)
    "min_resolution": [1920, 1080], // Minimum [width, height]
    "min_file_size_kb": 500      // Minimum file size
  }
}
```

#### Similarity Detection
```json
{
  "similarity": {
    "hash_threshold": 5,         // Perceptual hash difference
    "feature_threshold": 0.85,   // Deep learning similarity
    "clustering_eps": 0.3,       // DBSCAN clustering parameter
    "max_similar_images": 5      // Max images per similarity group
  }
}
```

### Configuration Validation

Validate your configuration:
```bash
python demo_config_validation.py
```

## First Run

### Test with Sample Data

1. **Create test images**
   ```bash
   python create_test_images.py
   ```

2. **Run basic test**
   ```bash
   python backend/main.py process test_input test_output
   ```

3. **Check results**
   - Approved images in `test_output/`
   - Reports in `reports/`
   - Logs in `logs/`

### Process Your Images

1. **Prepare your image folder**
   - Organize images in a single folder
   - Subfolders are supported (recursive scanning)
   - Supported formats: JPG, JPEG, PNG

2. **Run processing**
   ```bash
   python backend/main.py process /path/to/your/images /path/to/output
   ```

3. **Monitor progress**
   - Real-time progress display
   - Checkpoint saves every 50 images
   - Resume capability if interrupted

## Performance Tuning

### Memory Optimization

1. **Adjust batch size based on RAM**
   - 8GB RAM: batch_size = 100-200
   - 16GB RAM: batch_size = 200-400
   - 32GB RAM: batch_size = 400-800

2. **Monitor memory usage**
   ```bash
   python tests/memory_profiler.py
   ```

### CPU Optimization

1. **Set worker threads**
   - max_workers = CPU cores (for I/O bound)
   - max_workers = CPU cores / 2 (for CPU bound)

2. **Run performance tests**
   ```bash
   python tests/test_performance.py
   ```

### Storage Optimization

1. **Use SSD storage** for input/output folders
2. **Ensure sufficient free space** (3x input folder size)
3. **Consider separate drives** for input and output

### GPU Acceleration (Optional)

1. **Install CUDA** (for TensorFlow GPU support)
2. **Enable GPU in configuration**
   ```json
   {
     "processing": {
       "enable_gpu": true
     }
   }
   ```

## Troubleshooting

### Common Issues

#### Installation Problems

**Issue**: Package installation fails
```bash
# Solution: Upgrade pip and try again
pip install --upgrade pip
pip install -r backend/requirements.txt --upgrade
```

**Issue**: Python version too old
```bash
# Solution: Install Python 3.8+
# Check version
python --version

# Install newer Python version
# (method varies by OS)
```

#### Runtime Problems

**Issue**: Out of memory errors
```bash
# Solution: Reduce batch size
# Edit config/settings.json
{
  "processing": {
    "batch_size": 50  // Reduce from default
  }
}
```

**Issue**: Slow processing
```bash
# Solution: Increase workers (but not beyond CPU cores)
{
  "processing": {
    "max_workers": 8  // Adjust based on CPU
  }
}
```

**Issue**: Import errors
```bash
# Solution: Reinstall dependencies
pip install -r backend/requirements.txt --force-reinstall
```

#### Configuration Problems

**Issue**: Invalid configuration
```bash
# Solution: Validate and reset
python demo_config_validation.py
cp config/settings.example.json config/settings.json
```

### Diagnostic Tools

1. **System check**
   ```bash
   python install.py  # Re-run installer for diagnostics
   ```

2. **Installation test**
   ```bash
   python test_installation.py
   ```

3. **Configuration validation**
   ```bash
   python demo_config_validation.py
   ```

4. **Performance test**
   ```bash
   python tests/test_performance.py
   ```

### Log Analysis

Check logs for detailed error information:
- **Application logs**: `logs/adobe_stock_processor.log`
- **Error details**: Look for ERROR and CRITICAL messages
- **Performance metrics**: Look for timing information

## Advanced Setup

### Custom Analysis Models

1. **Download custom models** (if available)
2. **Update model paths** in configuration
3. **Verify model compatibility**

### Integration with Other Tools

1. **Batch scripts** for automated processing
2. **Cron jobs** for scheduled processing
3. **API integration** for web applications

### Cluster/Distributed Processing

1. **Multiple machine setup**
2. **Shared storage configuration**
3. **Load balancing strategies**

### Custom Analyzers

1. **Extend analyzer classes**
2. **Implement custom algorithms**
3. **Register new analyzers**

## Environment-Specific Setup

### Windows Setup

1. **Install Visual C++ Build Tools** (for some packages)
2. **Use PowerShell** for command execution
3. **Consider Windows Subsystem for Linux** (WSL)

### macOS Setup

1. **Install Xcode Command Line Tools**
   ```bash
   xcode-select --install
   ```

2. **Use Homebrew** for Python installation
   ```bash
   brew install python@3.9
   ```

### Linux Setup

1. **Install system dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install python3-dev python3-pip build-essential
   
   # CentOS/RHEL
   sudo yum install python3-devel python3-pip gcc gcc-c++
   ```

2. **Consider using pyenv** for Python version management

## Maintenance

### Regular Updates

1. **Update dependencies**
   ```bash
   pip install -r backend/requirements.txt --upgrade
   ```

2. **Update configuration** as needed
3. **Run tests** after updates
   ```bash
   python test_installation.py
   ```

### Backup and Recovery

1. **Backup configuration files**
2. **Backup processing databases**
3. **Document custom modifications**

### Monitoring

1. **Monitor disk space** usage
2. **Check log file sizes**
3. **Monitor system performance**

## Support

For additional help:

1. **Check the main README.md**
2. **Run diagnostic tools**
3. **Review log files**
4. **Test with sample data**
5. **Verify system requirements**

Remember to always test with a small batch of images before processing large datasets!