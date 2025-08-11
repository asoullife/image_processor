"""Setup script for backend dependencies."""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing backend dependencies...")
    
    # Core FastAPI dependencies
    dependencies = [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.0",
        "sqlalchemy>=2.0.0",
        "asyncpg>=0.29.0",
        "alembic>=1.12.0",
        "pytest-asyncio>=0.21.0"
    ]
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    print("✅ All dependencies installed successfully")
    return True

def create_minimal_config():
    """Create minimal configuration if it doesn't exist."""
    config_dir = Path(__file__).parent / "config"
    config_file = config_dir / "settings.json"
    
    if config_file.exists():
        print("✅ Configuration file already exists")
        return True
    
    print("📝 Creating minimal configuration...")
    
    minimal_config = {
        "processing": {
            "batch_size": 20,
            "max_workers": 4,
            "checkpoint_interval": 50
        },
        "quality": {
            "min_sharpness": 100.0,
            "max_noise_level": 0.1,
            "min_resolution": [800, 600]
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
            "quality_weight": 0.3,
            "defect_weight": 0.25,
            "similarity_weight": 0.2,
            "compliance_weight": 0.15,
            "technical_weight": 0.1,
            "approval_threshold": 0.7,
            "rejection_threshold": 0.3,
            "quality_min_threshold": 0.5,
            "defect_max_threshold": 0.3,
            "similarity_max_threshold": 0.8,
            "compliance_min_threshold": 0.8,
            "quality_critical_threshold": 0.3,
            "defect_critical_threshold": 0.7,
            "compliance_critical_threshold": 0.5
        },
        "output": {
            "images_per_folder": 200,
            "preserve_metadata": True,
            "generate_thumbnails": True
        },
        "logging": {
            "level": "INFO",
            "file": "logs/adobe_stock_processor.log",
            "max_file_size": "10MB",
            "backup_count": 5
        }
    }
    
    try:
        import json
        with open(config_file, 'w') as f:
            json.dump(minimal_config, f, indent=2)
        print("✅ Minimal configuration created")
        return True
    except Exception as e:
        print(f"❌ Failed to create configuration: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 Setting up Adobe Stock Image Processor Backend...")
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        return False
    
    # Create minimal config
    if not create_minimal_config():
        print("❌ Setup failed during configuration creation")
        return False
    
    print("\n✅ Backend setup completed successfully!")
    print("\nNext steps:")
    print("1. Set up PostgreSQL database")
    print("2. Update DATABASE_URL environment variable")
    print("3. Run: python backend/main.py server")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)