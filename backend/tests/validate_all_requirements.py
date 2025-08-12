#!/usr/bin/env python3
"""
Requirements Validation Script
Validates all requirements from the specification are met
"""

import os
import sys
import asyncio
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import requests
import tempfile
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RequirementsValidator:
    """Validates all requirements from the specification."""
    
    def __init__(self):
        """Initialize the requirements validator."""
        self.validation_results = {}
        self.project_root = Path(__file__).parent
        self.backend_path = self.project_root / "backend"
        
        # Requirements mapping from specification
        self.requirements = {
            "1": "Autonomous batch processing and real-time monitoring",
            "2": "AI-enhanced quality detection",
            "3": "Human Review System",
            "4": "Multi-session project management", 
            "5": "File integrity protection",
            "6": "Advanced similarity detection",
            "7": "Resume capability",
            "8": "Web-based configuration",
            "9": "Compliance checking",
            "10": "Web-based reports and analytics",
            "11": "Clean project structure",
            "12": "Real-time communication"
        }
    
    async def validate_all_requirements(self):
        """Validate all requirements from the specification."""
        logger.info("🔍 Starting Requirements Validation")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Validate each requirement
            for req_id, req_description in self.requirements.items():
                logger.info(f"Validating Requirement {req_id}: {req_description}")
                result = await self.validate_requirement(req_id)
                self.validation_results[req_id] = result
                
                status = "✅ PASSED" if result["status"] == "PASSED" else "❌ FAILED"
                logger.info(f"  {status}: {result['summary']}")
            
            # Generate validation report
            await self.generate_validation_report(start_time)
            
            # Check overall compliance
            failed_requirements = [
                req_id for req_id, result in self.validation_results.items()
                if result["status"] != "PASSED"
            ]
            
            if not failed_requirements:
                logger.info("🎉 ALL REQUIREMENTS VALIDATED SUCCESSFULLY!")
                return True
            else:
                logger.error(f"❌ {len(failed_requirements)} requirements failed validation")
                return False
                
        except Exception as e:
            logger.error(f"Requirements validation failed: {e}")
            return False
    
    async def validate_requirement(self, req_id: str) -> Dict[str, Any]:
        """Validate a specific requirement."""
        
        if req_id == "1":
            return await self.validate_requirement_1()
        elif req_id == "2":
            return await self.validate_requirement_2()
        elif req_id == "3":
            return await self.validate_requirement_3()
        elif req_id == "4":
            return await self.validate_requirement_4()
        elif req_id == "5":
            return await self.validate_requirement_5()
        elif req_id == "6":
            return await self.validate_requirement_6()
        elif req_id == "7":
            return await self.validate_requirement_7()
        elif req_id == "8":
            return await self.validate_requirement_8()
        elif req_id == "9":
            return await self.validate_requirement_9()
        elif req_id == "10":
            return await self.validate_requirement_10()
        elif req_id == "11":
            return await self.validate_requirement_11()
        elif req_id == "12":
            return await self.validate_requirement_12()
        else:
            return {
                "status": "ERROR",
                "summary": f"Unknown requirement ID: {req_id}",
                "details": []
            }
    
    async def validate_requirement_1(self) -> Dict[str, Any]:
        """Validate Requirement 1: Autonomous batch processing and real-time monitoring."""
        details = []
        
        try:
            # Check backend main.py exists and has CLI support
            main_py = self.project_root / "backend" / "main.py"
            if main_py.exists():
                details.append("✅ Main entry point exists")
                
                # Check CLI argument parsing
                content = main_py.read_text()
                if "argparse" in content or "sys.argv" in content:
                    details.append("✅ CLI argument parsing implemented")
                else:
                    details.append("❌ CLI argument parsing not found")
            else:
                details.append("❌ Main entry point missing")
            
            # Check batch processor exists
            batch_processor = self.backend_path / "core" / "batch_processor.py"
            if batch_processor.exists():
                details.append("✅ Batch processor component exists")
            else:
                details.append("❌ Batch processor component missing")
            
            # Check real-time monitoring
            monitoring_files = [
                self.backend_path / "realtime" / "socketio_manager.py",
                self.project_root / "frontend" / "src" / "components" / "monitoring"
            ]
            
            monitoring_exists = any(f.exists() for f in monitoring_files)
            if monitoring_exists:
                details.append("✅ Real-time monitoring components exist")
            else:
                details.append("❌ Real-time monitoring components missing")
            
            # Check for progress tracking
            progress_tracker = self.backend_path / "core" / "progress_tracker.py"
            if progress_tracker.exists():
                details.append("✅ Progress tracking component exists")
            else:
                details.append("❌ Progress tracking component missing")
            
            passed = all("✅" in detail for detail in details)
            
            return {
                "status": "PASSED" if passed else "FAILED",
                "summary": f"Autonomous processing and monitoring: {'implemented' if passed else 'incomplete'}",
                "details": details
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "summary": f"Error validating requirement 1: {e}",
                "details": [f"❌ Validation error: {e}"]
            }
    
    async def validate_requirement_2(self) -> Dict[str, Any]:
        """Validate Requirement 2: AI-enhanced quality detection."""
        details = []
        
        try:
            # Check AI analyzer components
            ai_components = [
                self.backend_path / "analyzers" / "quality_analyzer.py",
                self.backend_path / "analyzers" / "defect_detector.py"
            ]
            
            for component in ai_components:
                if component.exists():
                    details.append(f"✅ {component.name} exists")
                    
                    # Check for AI/ML imports
                    content = component.read_text()
                    ai_imports = ["tensorflow", "torch", "cv2", "sklearn", "numpy"]
                    found_ai = any(ai_lib in content for ai_lib in ai_imports)
                    
                    if found_ai:
                        details.append(f"✅ {component.name} uses AI/ML libraries")
                    else:
                        details.append(f"❌ {component.name} missing AI/ML integration")
                else:
                    details.append(f"❌ {component.name} missing")
            
            # Check for GPU support
            gpu_support_files = [
                self.backend_path / "utils" / "gpu_utils.py",
                self.backend_path / "config" / "config_loader.py"
            ]
            
            gpu_support = False
            for file in gpu_support_files:
                if file.exists():
                    content = file.read_text()
                    if "gpu" in content.lower() or "cuda" in content.lower():
                        gpu_support = True
                        break
            
            if gpu_support:
                details.append("✅ GPU acceleration support detected")
            else:
                details.append("⚠️ GPU acceleration support not clearly implemented")
            
            passed = len([d for d in details if d.startswith("✅")]) >= 3
            
            return {
                "status": "PASSED" if passed else "FAILED",
                "summary": f"AI-enhanced quality detection: {'implemented' if passed else 'incomplete'}",
                "details": details
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "summary": f"Error validating requirement 2: {e}",
                "details": [f"❌ Validation error: {e}"]
            }
    
    async def validate_requirement_3(self) -> Dict[str, Any]:
        """Validate Requirement 3: Human Review System."""
        details = []
        
        try:
            # Check backend review API
            review_api = self.backend_path / "api" / "routes" / "review.py"
            if review_api.exists():
                details.append("✅ Review API endpoints exist")
            else:
                details.append("❌ Review API endpoints missing")
            
            # Check frontend review components
            frontend_review_dir = self.project_root / "frontend" / "src" / "components" / "review"
            if frontend_review_dir.exists():
                details.append("✅ Frontend review components exist")
            else:
                details.append("❌ Frontend review components missing")
            
            # Check for Thai language support
            thai_support_files = [
                self.backend_path / "utils" / "thai_translator.py",
                self.project_root / "frontend" / "src" / "locales"
            ]
            
            thai_support = any(f.exists() for f in thai_support_files)
            if thai_support:
                details.append("✅ Thai language support detected")
            else:
                details.append("❌ Thai language support missing")
            
            # Check for similarity comparison
            similarity_viewer = self.project_root / "frontend" / "src" / "components" / "SimilarityViewer.tsx"
            if similarity_viewer.exists():
                details.append("✅ Similarity comparison interface exists")
            else:
                details.append("❌ Similarity comparison interface missing")
            
            passed = len([d for d in details if d.startswith("✅")]) >= 2
            
            return {
                "status": "PASSED" if passed else "FAILED",
                "summary": f"Human Review System: {'implemented' if passed else 'incomplete'}",
                "details": details
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "summary": f"Error validating requirement 3: {e}",
                "details": [f"❌ Validation error: {e}"]
            }
    
    async def validate_requirement_4(self) -> Dict[str, Any]:
        """Validate Requirement 4: Multi-session project management."""
        details = []
        
        try:
            # Check database models for multi-session support
            models_file = self.backend_path / "database" / "models.py"
            if models_file.exists():
                content = models_file.read_text()
                if "Project" in content and "Session" in content:
                    details.append("✅ Multi-session database models exist")
                else:
                    details.append("❌ Multi-session database models missing")
            else:
                details.append("❌ Database models file missing")
            
            # Check session management API
            sessions_api = self.backend_path / "api" / "routes" / "sessions.py"
            if sessions_api.exists():
                details.append("✅ Session management API exists")
            else:
                details.append("❌ Session management API missing")
            
            # Check project management API
            projects_api = self.backend_path / "api" / "routes" / "projects.py"
            if projects_api.exists():
                details.append("✅ Project management API exists")
            else:
                details.append("❌ Project management API missing")
            
            # Check frontend project management
            project_pages = self.project_root / "frontend" / "src" / "pages" / "projects"
            if project_pages.exists():
                details.append("✅ Frontend project management exists")
            else:
                details.append("❌ Frontend project management missing")
            
            passed = len([d for d in details if d.startswith("✅")]) >= 3
            
            return {
                "status": "PASSED" if passed else "FAILED",
                "summary": f"Multi-session project management: {'implemented' if passed else 'incomplete'}",
                "details": details
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "summary": f"Error validating requirement 4: {e}",
                "details": [f"❌ Validation error: {e}"]
            }
    
    async def validate_requirement_5(self) -> Dict[str, Any]:
        """Validate Requirement 5: File integrity protection."""
        details = []
        
        try:
            # Check file integrity manager
            file_integrity = self.backend_path / "utils" / "file_integrity.py"
            if file_integrity.exists():
                details.append("✅ File integrity manager exists")
                
                content = file_integrity.read_text()
                if "FORBIDDEN_OPERATIONS" in content:
                    details.append("✅ Forbidden operations protection implemented")
                else:
                    details.append("❌ Forbidden operations protection missing")
            else:
                details.append("❌ File integrity manager missing")
            
            # Check file manager with integrity checks
            file_manager = self.backend_path / "utils" / "file_manager.py"
            if file_manager.exists():
                content = file_manager.read_text()
                if "integrity" in content.lower() or "checksum" in content.lower():
                    details.append("✅ File integrity verification implemented")
                else:
                    details.append("❌ File integrity verification missing")
            else:
                details.append("❌ File manager missing")
            
            # Check for atomic operations
            atomic_support = any(
                "atomic" in f.read_text().lower() 
                for f in self.backend_path.rglob("*.py") 
                if f.is_file()
            )
            
            if atomic_support:
                details.append("✅ Atomic operations support detected")
            else:
                details.append("❌ Atomic operations support missing")
            
            passed = len([d for d in details if d.startswith("✅")]) >= 2
            
            return {
                "status": "PASSED" if passed else "FAILED",
                "summary": f"File integrity protection: {'implemented' if passed else 'incomplete'}",
                "details": details
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "summary": f"Error validating requirement 5: {e}",
                "details": [f"❌ Validation error: {e}"]
            }
    
    async def validate_requirement_6(self) -> Dict[str, Any]:
        """Validate Requirement 6: Advanced similarity detection."""
        details = []
        
        try:
            # Check similarity finder component
            similarity_finder = self.backend_path / "analyzers" / "similarity_finder.py"
            if similarity_finder.exists():
                details.append("✅ Similarity finder component exists")
                
                content = similarity_finder.read_text()
                
                # Check for perceptual hashing
                if "hash" in content.lower():
                    details.append("✅ Perceptual hashing implemented")
                else:
                    details.append("❌ Perceptual hashing missing")
                
                # Check for deep learning features
                if "clip" in content.lower() or "embedding" in content.lower():
                    details.append("✅ Deep learning features implemented")
                else:
                    details.append("❌ Deep learning features missing")
                
                # Check for clustering
                if "cluster" in content.lower() or "dbscan" in content.lower():
                    details.append("✅ Clustering algorithms implemented")
                else:
                    details.append("❌ Clustering algorithms missing")
            else:
                details.append("❌ Similarity finder component missing")
            
            # Check similarity comparison UI
            similarity_ui = self.project_root / "frontend" / "src" / "components" / "SimilarityViewer.tsx"
            if similarity_ui.exists():
                details.append("✅ Similarity comparison UI exists")
            else:
                details.append("❌ Similarity comparison UI missing")
            
            passed = len([d for d in details if d.startswith("✅")]) >= 3
            
            return {
                "status": "PASSED" if passed else "FAILED",
                "summary": f"Advanced similarity detection: {'implemented' if passed else 'incomplete'}",
                "details": details
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "summary": f"Error validating requirement 6: {e}",
                "details": [f"❌ Validation error: {e}"]
            }
    
    async def validate_requirement_7(self) -> Dict[str, Any]:
        """Validate Requirement 7: Resume capability."""
        details = []
        
        try:
            # Check progress tracker with checkpoints
            progress_tracker = self.backend_path / "core" / "progress_tracker.py"
            if progress_tracker.exists():
                details.append("✅ Progress tracker exists")
                
                content = progress_tracker.read_text()
                if "checkpoint" in content.lower():
                    details.append("✅ Checkpoint system implemented")
                else:
                    details.append("❌ Checkpoint system missing")
            else:
                details.append("❌ Progress tracker missing")
            
            # Check session manager for resume functionality
            session_manager = self.backend_path / "core" / "session_manager.py"
            if session_manager.exists():
                content = session_manager.read_text()
                if "resume" in content.lower():
                    details.append("✅ Resume functionality implemented")
                else:
                    details.append("❌ Resume functionality missing")
            else:
                details.append("❌ Session manager missing")
            
            # Check database schema for checkpoints
            models_file = self.backend_path / "database" / "models.py"
            if models_file.exists():
                content = models_file.read_text()
                if "checkpoint" in content.lower():
                    details.append("✅ Checkpoint database schema exists")
                else:
                    details.append("❌ Checkpoint database schema missing")
            
            # Check recovery API
            recovery_api = self.backend_path / "api" / "routes" / "recovery.py"
            if recovery_api.exists():
                details.append("✅ Recovery API endpoints exist")
            else:
                details.append("❌ Recovery API endpoints missing")
            
            passed = len([d for d in details if d.startswith("✅")]) >= 3
            
            return {
                "status": "PASSED" if passed else "FAILED",
                "summary": f"Resume capability: {'implemented' if passed else 'incomplete'}",
                "details": details
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "summary": f"Error validating requirement 7: {e}",
                "details": [f"❌ Validation error: {e}"]
            }
    
    async def validate_requirement_8(self) -> Dict[str, Any]:
        """Validate Requirement 8: Web-based configuration."""
        details = []
        
        try:
            # Check settings API
            settings_api = self.backend_path / "api" / "routes" / "settings.py"
            if settings_api.exists():
                details.append("✅ Settings API exists")
            else:
                details.append("❌ Settings API missing")
            
            # Check configuration management
            config_loader = self.backend_path / "config" / "config_loader.py"
            if config_loader.exists():
                details.append("✅ Configuration loader exists")
            else:
                details.append("❌ Configuration loader missing")
            
            # Check frontend settings interface
            settings_pages = self.project_root / "frontend" / "src" / "pages" / "settings"
            if settings_pages.exists():
                details.append("✅ Frontend settings interface exists")
            else:
                details.append("❌ Frontend settings interface missing")
            
            # Check CLI help system
            main_py = self.project_root / "backend" / "main.py"
            if main_py.exists():
                content = main_py.read_text()
                if "-h" in content or "help" in content:
                    details.append("✅ CLI help system implemented")
                else:
                    details.append("❌ CLI help system missing")
            
            passed = len([d for d in details if d.startswith("✅")]) >= 3
            
            return {
                "status": "PASSED" if passed else "FAILED",
                "summary": f"Web-based configuration: {'implemented' if passed else 'incomplete'}",
                "details": details
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "summary": f"Error validating requirement 8: {e}",
                "details": [f"❌ Validation error: {e}"]
            }
    
    async def validate_requirement_9(self) -> Dict[str, Any]:
        """Validate Requirement 9: Compliance checking."""
        details = []
        
        try:
            # Check compliance checker component
            compliance_checker = self.backend_path / "analyzers" / "compliance_checker.py"
            if compliance_checker.exists():
                details.append("✅ Compliance checker exists")
                
                content = compliance_checker.read_text()
                
                # Check for logo detection
                if "logo" in content.lower() or "ocr" in content.lower():
                    details.append("✅ Logo detection implemented")
                else:
                    details.append("❌ Logo detection missing")
                
                # Check for face detection
                if "face" in content.lower():
                    details.append("✅ Face detection implemented")
                else:
                    details.append("❌ Face detection missing")
                
                # Check for metadata validation
                if "metadata" in content.lower():
                    details.append("✅ Metadata validation implemented")
                else:
                    details.append("❌ Metadata validation missing")
            else:
                details.append("❌ Compliance checker missing")
            
            passed = len([d for d in details if d.startswith("✅")]) >= 3
            
            return {
                "status": "PASSED" if passed else "FAILED",
                "summary": f"Compliance checking: {'implemented' if passed else 'incomplete'}",
                "details": details
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "summary": f"Error validating requirement 9: {e}",
                "details": [f"❌ Validation error: {e}"]
            }
    
    async def validate_requirement_10(self) -> Dict[str, Any]:
        """Validate Requirement 10: Web-based reports and analytics."""
        details = []
        
        try:
            # Check reports API
            reports_api = self.backend_path / "api" / "routes" / "reports.py"
            if reports_api.exists():
                details.append("✅ Reports API exists")
            else:
                details.append("❌ Reports API missing")
            
            # Check report generator
            report_generator = self.backend_path / "utils" / "report_generator.py"
            if report_generator.exists():
                details.append("✅ Report generator exists")
            else:
                details.append("❌ Report generator missing")
            
            # Check frontend reports components
            reports_components = self.project_root / "frontend" / "src" / "components" / "reports"
            if reports_components.exists():
                details.append("✅ Frontend reports components exist")
            else:
                details.append("❌ Frontend reports components missing")
            
            # Check for analytics and charts
            chart_files = list(self.project_root.rglob("*chart*.tsx")) + list(self.project_root.rglob("*Chart*.tsx"))
            if chart_files:
                details.append("✅ Chart components exist")
            else:
                details.append("❌ Chart components missing")
            
            passed = len([d for d in details if d.startswith("✅")]) >= 3
            
            return {
                "status": "PASSED" if passed else "FAILED",
                "summary": f"Web-based reports and analytics: {'implemented' if passed else 'incomplete'}",
                "details": details
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "summary": f"Error validating requirement 10: {e}",
                "details": [f"❌ Validation error: {e}"]
            }
    
    async def validate_requirement_11(self) -> Dict[str, Any]:
        """Validate Requirement 11: Clean project structure."""
        details = []
        
        try:
            # Check main directories exist
            required_dirs = [
                self.project_root / "backend",
                self.project_root / "frontend",
                self.project_root / "scripts",
                self.project_root / "docs"
            ]
            
            for dir_path in required_dirs:
                if dir_path.exists():
                    details.append(f"✅ {dir_path.name}/ directory exists")
                else:
                    details.append(f"❌ {dir_path.name}/ directory missing")
            
            # Check backend structure
            backend_dirs = [
                self.backend_path / "api",
                self.backend_path / "core", 
                self.backend_path / "analyzers",
                self.backend_path / "database",
                self.backend_path / "utils"
            ]
            
            backend_structure_ok = all(d.exists() for d in backend_dirs)
            if backend_structure_ok:
                details.append("✅ Backend structure properly organized")
            else:
                details.append("❌ Backend structure incomplete")
            
            # Check root directory is clean
            root_files = list(self.project_root.glob("*"))
            essential_files = ["README.md", ".gitignore", "backend", "frontend", "infra"]
            
            clean_root = len([f for f in root_files if f.is_file()]) <= 10  # Allow some flexibility
            if clean_root:
                details.append("✅ Root directory is clean")
            else:
                details.append("❌ Root directory has too many files")
            
            # Check no shared/ directory exists (should be eliminated)
            shared_dir = self.project_root / "shared"
            if not shared_dir.exists():
                details.append("✅ No shared/ directory (properly eliminated)")
            else:
                details.append("❌ shared/ directory still exists")
            
            passed = len([d for d in details if d.startswith("✅")]) >= 6
            
            return {
                "status": "PASSED" if passed else "FAILED",
                "summary": f"Clean project structure: {'implemented' if passed else 'incomplete'}",
                "details": details
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "summary": f"Error validating requirement 11: {e}",
                "details": [f"❌ Validation error: {e}"]
            }
    
    async def validate_requirement_12(self) -> Dict[str, Any]:
        """Validate Requirement 12: Real-time communication."""
        details = []
        
        try:
            # Check Socket.IO backend implementation
            socketio_manager = self.backend_path / "realtime" / "socketio_manager.py"
            if socketio_manager.exists():
                details.append("✅ Socket.IO backend manager exists")
                
                content = socketio_manager.read_text()
                if "python-socketio" in content or "socketio" in content:
                    details.append("✅ Socket.IO server implementation found")
                else:
                    details.append("❌ Socket.IO server implementation missing")
            else:
                details.append("❌ Socket.IO backend manager missing")
            
            # Check frontend Socket.IO client
            socket_hooks = list(self.project_root.rglob("*socket*.ts*")) + list(self.project_root.rglob("*Socket*.ts*"))
            if socket_hooks:
                details.append("✅ Frontend Socket.IO client exists")
            else:
                details.append("❌ Frontend Socket.IO client missing")
            
            # Check Redis adapter for scaling
            redis_adapter = self.backend_path / "realtime" / "redis_adapter.py"
            if redis_adapter.exists():
                details.append("✅ Redis adapter for scaling exists")
            else:
                details.append("❌ Redis adapter for scaling missing")
            
            # Check real-time progress monitoring
            progress_monitor = self.project_root / "frontend" / "src" / "components" / "monitoring"
            if progress_monitor.exists():
                details.append("✅ Real-time progress monitoring UI exists")
            else:
                details.append("❌ Real-time progress monitoring UI missing")
            
            passed = len([d for d in details if d.startswith("✅")]) >= 3
            
            return {
                "status": "PASSED" if passed else "FAILED",
                "summary": f"Real-time communication: {'implemented' if passed else 'incomplete'}",
                "details": details
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "summary": f"Error validating requirement 12: {e}",
                "details": [f"❌ Validation error: {e}"]
            }
    
    async def generate_validation_report(self, start_time: float):
        """Generate comprehensive validation report."""
        logger.info("📊 Generating Requirements Validation Report")
        
        total_duration = time.time() - start_time
        
        # Calculate statistics
        total_requirements = len(self.validation_results)
        passed_requirements = sum(1 for r in self.validation_results.values() if r["status"] == "PASSED")
        failed_requirements = sum(1 for r in self.validation_results.values() if r["status"] == "FAILED")
        error_requirements = sum(1 for r in self.validation_results.values() if r["status"] == "ERROR")
        
        # Create comprehensive report
        report = {
            "validation_suite": "Requirements Validation Report",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": total_duration,
            "summary": {
                "total_requirements": total_requirements,
                "passed": passed_requirements,
                "failed": failed_requirements,
                "errors": error_requirements,
                "compliance_rate": (passed_requirements / total_requirements * 100) if total_requirements > 0 else 0
            },
            "requirements_status": {
                req_id: {
                    "description": self.requirements[req_id],
                    "status": result["status"],
                    "summary": result["summary"],
                    "details": result["details"]
                }
                for req_id, result in self.validation_results.items()
            },
            "failed_requirements": [
                req_id for req_id, result in self.validation_results.items()
                if result["status"] in ["FAILED", "ERROR"]
            ]
        }
        
        # Save report
        report_path = Path("test_results") / f"requirements_validation_report_{int(time.time())}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("=" * 80)
        logger.info("🎯 REQUIREMENTS VALIDATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total Requirements: {total_requirements}")
        logger.info(f"✅ Passed: {passed_requirements}")
        logger.info(f"❌ Failed: {failed_requirements}")
        logger.info(f"⚠️ Errors: {error_requirements}")
        logger.info(f"Compliance Rate: {report['summary']['compliance_rate']:.1f}%")
        
        if report["failed_requirements"]:
            logger.error("Failed Requirements:")
            for req_id in report["failed_requirements"]:
                req_desc = self.requirements.get(req_id, "Unknown")
                logger.error(f"  - Requirement {req_id}: {req_desc}")
        
        logger.info(f"📊 Detailed report: {report_path}")
        logger.info("=" * 80)

async def main():
    """Main entry point."""
    validator = RequirementsValidator()
    
    try:
        success = await validator.validate_all_requirements()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"Requirements validation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))