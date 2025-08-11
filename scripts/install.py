#!/usr/bin/env python3
"""
Adobe Stock Image Processor - Installation Script

This script handles the installation and setup of the Adobe Stock Image Processor,
including dependency verification, system requirements checking, and initial configuration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sys
import os
import subprocess
import platform
import shutil
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# Minimum system requirements
MIN_PYTHON_VERSION = (3, 8)
MIN_RAM_GB = 8
MIN_DISK_SPACE_GB = 50
RECOMMENDED_RAM_GB = 16

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

class InstallationError(Exception):
    """Custom exception for installation errors"""
    pass

class AdobeStockInstaller:
    """Main installer class for Adobe Stock Image Processor"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_dir = self.project_root / "config"
        self.test_dir = self.project_root / "backend/data/input"
        self.requirements_file = self.project_root / "requirements.txt"
        
    def print_header(self):
        """Print installation header"""
        print(f"\n{Colors.BLUE}{Colors.BOLD}=" * 60)
        print("Adobe Stock Image Processor - Installation")
        print("=" * 60 + f"{Colors.END}\n")
        
    def print_success(self, message: str):
        """Print success message"""
        print(f"{Colors.GREEN}✓ {message}{Colors.END}")
        
    def print_error(self, message: str):
        """Print error message"""
        print(f"{Colors.RED}✗ {message}{Colors.END}")
        
    def print_warning(self, message: str):
        """Print warning message"""
        print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")
        
    def print_info(self, message: str):
        """Print info message"""
        print(f"{Colors.BLUE}ℹ {message}{Colors.END}")
        
    def check_python_version(self) -> bool:
        """Check if Python version meets minimum requirements"""
        current_version = sys.version_info[:2]
        if current_version >= MIN_PYTHON_VERSION:
            self.print_success(f"Python {'.'.join(map(str, current_version))} detected")
            return True
        else:
            self.print_error(f"Python {'.'.join(map(str, MIN_PYTHON_VERSION))}+ required, "
                           f"found {'.'.join(map(str, current_version))}")
            return False
            
    def check_system_resources(self) -> bool:
        """Check system RAM and disk space"""
        try:
            import psutil
            
            # Check RAM
            ram_gb = psutil.virtual_memory().total / (1024**3)
            if ram_gb >= MIN_RAM_GB:
                if ram_gb >= RECOMMENDED_RAM_GB:
                    self.print_success(f"RAM: {ram_gb:.1f}GB (recommended)")
                else:
                    self.print_warning(f"RAM: {ram_gb:.1f}GB (minimum met, {RECOMMENDED_RAM_GB}GB recommended)")
            else:
                self.print_error(f"Insufficient RAM: {ram_gb:.1f}GB (minimum {MIN_RAM_GB}GB required)")
                return False
                
            # Check disk space
            disk_usage = shutil.disk_usage(self.project_root)
            free_gb = disk_usage.free / (1024**3)
            if free_gb >= MIN_DISK_SPACE_GB:
                self.print_success(f"Disk space: {free_gb:.1f}GB available")
            else:
                self.print_error(f"Insufficient disk space: {free_gb:.1f}GB "
                               f"(minimum {MIN_DISK_SPACE_GB}GB required)")
                return False
                
            return True
            
        except ImportError:
            self.print_warning("psutil not available, skipping resource check")
            return True
        except Exception as e:
            self.print_warning(f"Could not check system resources: {e}")
            return True
            
    def check_pip_availability(self) -> bool:
        """Check if pip is available"""
        try:
            subprocess.run([sys.executable, "-m", "pip", "--version"], 
                         check=True, capture_output=True)
            self.print_success("pip is available")
            return True
        except subprocess.CalledProcessError:
            self.print_error("pip is not available or not working")
            return False
            
    def install_dependencies(self, upgrade: bool = False) -> bool:
        """Install Python dependencies from requirements.txt"""
        if not self.requirements_file.exists():
            self.print_error(f"Requirements file not found: {self.requirements_file}")
            return False
            
        try:
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)]
            if upgrade:
                cmd.append("--upgrade")
                
            self.print_info("Installing Python dependencies...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            self.print_success("Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to install dependencies: {e}")
            if e.stderr:
                print(f"Error details: {e.stderr}")
            return False
            
    def verify_dependencies(self) -> Tuple[bool, List[str]]:
        """Verify that all required dependencies are properly installed"""
        required_packages = [
            'cv2',           # opencv-python
            'PIL',           # Pillow
            'numpy',
            'tensorflow',
            'sklearn',       # scikit-learn
            'pandas',
            'openpyxl',
            'matplotlib',
            'seaborn',
            'jinja2',
            'imagehash',
            'tqdm',
            'psutil'
        ]
        
        failed_imports = []
        
        for package in required_packages:
            try:
                __import__(package)
                self.print_success(f"✓ {package}")
            except ImportError:
                self.print_error(f"✗ {package}")
                failed_imports.append(package)
                
        return len(failed_imports) == 0, failed_imports
        
    def create_config_files(self) -> bool:
        """Create configuration files if they don't exist"""
        try:
            # Ensure config directory exists
            self.config_dir.mkdir(exist_ok=True)
            
            # Create settings.json from example if it doesn't exist
            settings_file = self.config_dir / "settings.json"
            example_file = self.config_dir / "settings.example.json"
            
            if not settings_file.exists() and example_file.exists():
                shutil.copy2(example_file, settings_file)
                self.print_success("Created settings.json from example")
            elif settings_file.exists():
                self.print_info("settings.json already exists")
            else:
                self.print_warning("No example configuration found")
                
            return True
            
        except Exception as e:
            self.print_error(f"Failed to create config files: {e}")
            return False
            
    def create_test_dataset(self) -> bool:
        """Create test dataset if it doesn't exist"""
        try:
            if self.test_dir.exists() and any(self.test_dir.iterdir()):
                self.print_info("Test dataset already exists")
                return True
                
            # Run the test image creation script
            create_script = self.project_root / "create_test_images.py"
            if create_script.exists():
                self.print_info("Creating test dataset...")
                result = subprocess.run([sys.executable, str(create_script)], 
                                      check=True, capture_output=True, text=True)
                self.print_success("Test dataset created")
                return True
            else:
                self.print_warning("Test image creation script not found")
                return True
                
        except subprocess.CalledProcessError as e:
            self.print_error(f"Failed to create test dataset: {e}")
            return False
        except Exception as e:
            self.print_error(f"Error creating test dataset: {e}")
            return False
            
    def run_basic_tests(self) -> bool:
        """Run basic tests to verify installation"""
        try:
            test_script = self.project_root / "run_simple_tests.py"
            if test_script.exists():
                self.print_info("Running basic tests...")
                result = subprocess.run([sys.executable, str(test_script)], 
                                      check=True, capture_output=True, text=True)
                self.print_success("Basic tests passed")
                return True
            else:
                self.print_warning("Basic test script not found, skipping tests")
                return True
                
        except subprocess.CalledProcessError as e:
            self.print_error(f"Basic tests failed: {e}")
            if e.stdout:
                print("Test output:", e.stdout)
            if e.stderr:
                print("Test errors:", e.stderr)
            return False
        except Exception as e:
            self.print_error(f"Error running tests: {e}")
            return False
            
    def print_installation_summary(self, success: bool):
        """Print installation summary and next steps"""
        print(f"\n{Colors.BLUE}{Colors.BOLD}=" * 60)
        print("Installation Summary")
        print("=" * 60 + f"{Colors.END}\n")
        
        if success:
            self.print_success("Installation completed successfully!")
            print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
            print("1. Review configuration in config/settings.json")
            print("2. Test with sample images: python main.py test_input test_output")
            print("3. Check the documentation in README.md for usage instructions")
            print("4. Run comprehensive tests: python run_comprehensive_tests.py")
        else:
            self.print_error("Installation completed with errors!")
            print(f"\n{Colors.BOLD}Troubleshooting:{Colors.END}")
            print("1. Check error messages above")
            print("2. Ensure you have Python 3.8+ installed")
            print("3. Try running: pip install --upgrade pip")
            print("4. Install dependencies manually: pip install -r requirements.txt")
            
    def install(self, upgrade_deps: bool = False, skip_tests: bool = False) -> bool:
        """Main installation process"""
        self.print_header()
        
        success = True
        
        # Step 1: Check system requirements
        self.print_info("Step 1: Checking system requirements...")
        if not self.check_python_version():
            success = False
        if not self.check_system_resources():
            success = False
        if not self.check_pip_availability():
            success = False
            
        if not success:
            self.print_error("System requirements not met")
            return False
            
        # Step 2: Install dependencies
        self.print_info("\nStep 2: Installing dependencies...")
        if not self.install_dependencies(upgrade=upgrade_deps):
            success = False
            
        # Step 3: Verify dependencies
        self.print_info("\nStep 3: Verifying dependencies...")
        deps_ok, failed_deps = self.verify_dependencies()
        if not deps_ok:
            self.print_error(f"Failed to import: {', '.join(failed_deps)}")
            success = False
            
        # Step 4: Create configuration
        self.print_info("\nStep 4: Setting up configuration...")
        if not self.create_config_files():
            success = False
            
        # Step 5: Create test dataset
        self.print_info("\nStep 5: Setting up test dataset...")
        if not self.create_test_dataset():
            success = False
            
        # Step 6: Run basic tests
        if not skip_tests:
            self.print_info("\nStep 6: Running basic tests...")
            if not self.run_basic_tests():
                success = False
        else:
            self.print_info("\nStep 6: Skipping tests (--skip-tests)")
            
        # Print summary
        self.print_installation_summary(success)
        
        return success

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Adobe Stock Image Processor Installation")
    parser.add_argument("--upgrade", action="store_true", 
                       help="Upgrade existing dependencies")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip running basic tests")
    
    args = parser.parse_args()
    
    installer = AdobeStockInstaller()
    success = installer.install(upgrade_deps=args.upgrade, skip_tests=args.skip_tests)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()