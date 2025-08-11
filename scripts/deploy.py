#!/usr/bin/env python3
"""
Adobe Stock Image Processor - Deployment Script

This script handles deployment to different environments (development, staging, production)
with appropriate configuration and optimization settings.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
import shutil
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

class DeploymentManager:
    """Manage deployment to different environments"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_dir = self.project_root / "config"
        
    def deploy_development(self, target_dir: Optional[Path] = None) -> bool:
        """Deploy for development environment"""
        print("Deploying for development environment...")
        
        if target_dir is None:
            target_dir = self.project_root
            
        try:
            # Use development configuration
            dev_config = self.config_dir / "settings.development.json"
            target_config = target_dir / "config" / "settings.json"
            
            if dev_config.exists():
                target_config.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dev_config, target_config)
                print("✓ Development configuration applied")
            
            # Create development-specific directories
            dev_dirs = [
                target_dir / "logs",
                target_dir / "backend/data/output",
                target_dir / "reports"
            ]
            
            for dir_path in dev_dirs:
                dir_path.mkdir(parents=True, exist_ok=True)
                
            print("✓ Development directories created")
            return True
            
        except Exception as e:
            print(f"✗ Development deployment failed: {e}")
            return False
            
    def deploy_production(self, target_dir: Path) -> bool:
        """Deploy for production environment"""
        print("Deploying for production environment...")
        
        try:
            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy essential files
            essential_files = [
                "main.py",
                "requirements.txt",
                "README.md",
                "SETUP_GUIDE.md"
            ]
            
            for file_name in essential_files:
                src_file = self.project_root / file_name
                if src_file.exists():
                    shutil.copy2(src_file, target_dir / file_name)
                    
            # Copy directories
            essential_dirs = [
                "core",
                "analyzers", 
                "utils",
                "config",
                "tests"
            ]
            
            for dir_name in essential_dirs:
                src_dir = self.project_root / dir_name
                target_subdir = target_dir / dir_name
                
                if src_dir.exists():
                    if target_subdir.exists():
                        shutil.rmtree(target_subdir)
                    shutil.copytree(src_dir, target_subdir)
                    
            # Use production configuration
            prod_config = self.config_dir / "settings.production.json"
            target_config = target_dir / "config" / "settings.json"
            
            if prod_config.exists():
                shutil.copy2(prod_config, target_config)
                print("✓ Production configuration applied")
                
            # Create production directories
            prod_dirs = [
                target_dir / "logs",
                target_dir / "data",
                target_dir / "reports",
                target_dir / "backups"
            ]
            
            for dir_path in prod_dirs:
                dir_path.mkdir(parents=True, exist_ok=True)
                
            # Create production-specific files
            self._create_production_scripts(target_dir)
            
            print("✓ Production deployment completed")
            return True
            
        except Exception as e:
            print(f"✗ Production deployment failed: {e}")
            return False
            
    def deploy_docker(self, target_dir: Path) -> bool:
        """Create Docker deployment"""
        print("Creating Docker deployment...")
        
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all necessary files
            self.deploy_production(target_dir)
            
            # Create Dockerfile
            dockerfile_content = self._generate_dockerfile()
            dockerfile_path = target_dir / "Dockerfile"
            dockerfile_path.write_text(dockerfile_content)
            
            # Create docker-compose.yml
            compose_content = self._generate_docker_compose()
            compose_path = target_dir / "docker-compose.yml"
            compose_path.write_text(compose_content)
            
            # Create .dockerignore
            dockerignore_content = self._generate_dockerignore()
            dockerignore_path = target_dir / ".dockerignore"
            dockerignore_path.write_text(dockerignore_content)
            
            print("✓ Docker deployment files created")
            return True
            
        except Exception as e:
            print(f"✗ Docker deployment failed: {e}")
            return False
            
    def _create_production_scripts(self, target_dir: Path):
        """Create production-specific scripts"""
        
        # Create startup script
        startup_script = target_dir / "start.sh"
        startup_content = """#!/bin/bash
# Adobe Stock Image Processor - Production Startup Script

echo "Starting Adobe Stock Image Processor..."

# Check Python version
python3 --version

# Install/update dependencies
pip3 install -r requirements.txt

# Run the application
python3 main.py "$@"
"""
        startup_script.write_text(startup_content)
        startup_script.chmod(0o755)
        
        # Create monitoring script
        monitor_script = target_dir / "monitor.py"
        monitor_content = '''#!/usr/bin/env python3
"""Production monitoring script"""

import psutil
import time
import json
from pathlib import Path

def monitor_system():
    """Monitor system resources"""
    while True:
        stats = {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent
        }
        
        print(json.dumps(stats))
        time.sleep(60)  # Monitor every minute

if __name__ == "__main__":
    monitor_system()
'''
        monitor_script.write_text(monitor_content)
        monitor_script.chmod(0o755)
        
    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile content"""
        return """# Adobe Stock Image Processor - Docker Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs reports data

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port (if needed for web interface)
EXPOSE 8080

# Default command
CMD ["python", "main.py", "--help"]
"""

    def _generate_docker_compose(self) -> str:
        """Generate docker-compose.yml content"""
        return """version: '3.8'

services:
  adobe-stock-processor:
    build: .
    container_name: adobe-stock-processor
    volumes:
      - ./input:/app/input:ro
      - ./output:/app/output
      - ./logs:/app/logs
      - ./reports:/app/reports
      - ./config:/app/config
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    command: python main.py input output
    
  # Optional: monitoring service
  monitor:
    build: .
    container_name: adobe-stock-monitor
    volumes:
      - ./logs:/app/logs:ro
    command: python monitor.py
    restart: unless-stopped
    depends_on:
      - adobe-stock-processor

volumes:
  input:
  output:
  logs:
  reports:
"""

    def _generate_dockerignore(self) -> str:
        """Generate .dockerignore content"""
        return """# Adobe Stock Image Processor - Docker Ignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
test_input/
test_output/
test_output_resume/
logs/*.log
reports/
*.db
*.sqlite

# Development files
.git/
.gitignore
*.md
demo_*.py
test_*.py
run_*_tests.py
"""

    def create_package(self, output_path: Path, include_tests: bool = False) -> bool:
        """Create a distributable package"""
        print("Creating distributable package...")
        
        try:
            # Create package directory
            package_dir = output_path / "adobe-stock-image-processor"
            package_dir.mkdir(parents=True, exist_ok=True)
            
            # Essential files
            essential_files = [
                "main.py",
                "install.py", 
                "requirements.txt",
                "README.md",
                "SETUP_GUIDE.md"
            ]
            
            for file_name in essential_files:
                src_file = self.project_root / file_name
                if src_file.exists():
                    shutil.copy2(src_file, package_dir / file_name)
                    
            # Essential directories
            essential_dirs = ["core", "analyzers", "utils", "config"]
            if include_tests:
                essential_dirs.append("tests")
                
            for dir_name in essential_dirs:
                src_dir = self.project_root / dir_name
                if src_dir.exists():
                    shutil.copytree(src_dir, package_dir / dir_name)
                    
            # Create archive
            archive_path = output_path / "adobe-stock-image-processor"
            shutil.make_archive(str(archive_path), 'zip', str(package_dir))
            
            # Clean up temporary directory
            shutil.rmtree(package_dir)
            
            print(f"✓ Package created: {archive_path}.zip")
            return True
            
        except Exception as e:
            print(f"✗ Package creation failed: {e}")
            return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Adobe Stock Image Processor Deployment")
    parser.add_argument("environment", choices=["dev", "prod", "docker", "package"],
                       help="Deployment environment")
    parser.add_argument("--target", type=Path, 
                       help="Target directory for deployment")
    parser.add_argument("--include-tests", action="store_true",
                       help="Include tests in package (for package environment)")
    
    args = parser.parse_args()
    
    deployer = DeploymentManager()
    success = False
    
    if args.environment == "dev":
        success = deployer.deploy_development(args.target)
    elif args.environment == "prod":
        if not args.target:
            print("Error: --target required for production deployment")
            sys.exit(1)
        success = deployer.deploy_production(args.target)
    elif args.environment == "docker":
        if not args.target:
            print("Error: --target required for Docker deployment")
            sys.exit(1)
        success = deployer.deploy_docker(args.target)
    elif args.environment == "package":
        if not args.target:
            args.target = Path("./dist")
        success = deployer.create_package(args.target, args.include_tests)
        
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()