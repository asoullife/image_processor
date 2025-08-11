"""Core business logic services."""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload

import sys
from pathlib import Path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from database.connection import DatabaseManager
from database.models import Project, ProcessingSession, ImageResult, Checkpoint
from config.config_loader import AppConfig
from analyzers.analyzer_factory import AnalyzerFactory

logger = logging.getLogger(__name__)

class ProjectService:
    """Service for managing processing projects."""
    
    def __init__(self, db_manager: DatabaseManager, config: AppConfig):
        """Initialize project service.
        
        Args:
            db_manager: Database manager
            config: Application configuration
        """
        self.db_manager = db_manager
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def create_project(
        self,
        name: str,
        description: Optional[str] = None,
        input_folder: str = "",
        output_folder: str = "",
        performance_mode: str = "balanced",
        settings: Optional[Dict[str, Any]] = None
    ) -> Project:
        """Create a new processing project.
        
        Args:
            name: Project name
            description: Project description
            input_folder: Input folder path
            output_folder: Output folder path
            performance_mode: Performance mode (speed/balanced/smart)
            settings: Additional project settings
            
        Returns:
            Created project
        """
        if settings is None:
            settings = {}
        
        # Auto-generate output folder if not provided (input_name → input_name_processed)
        if not output_folder and input_folder:
            from pathlib import Path
            input_path = Path(input_folder)
            output_folder = str(input_path.parent / f"{input_path.name}_processed")
        
        async with self.db_manager.get_session() as session:
            project = Project(
                name=name,
                description=description,
                input_folder=input_folder,
                output_folder=output_folder,
                performance_mode=performance_mode,
                settings=settings,
                status="created"
            )
            
            session.add(project)
            await session.flush()
            await session.refresh(project)
            
            self.logger.info(f"Created project: {project.id} - {name}")
            return project
    
    async def get_project(self, project_id: UUID) -> Optional[Project]:
        """Get project by ID.
        
        Args:
            project_id: Project UUID
            
        Returns:
            Project or None if not found
        """
        async with self.db_manager.get_session() as session:
            stmt = select(Project).where(Project.id == project_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
    
    async def list_projects(
        self,
        skip: int = 0,
        limit: int = 100,
        status_filter: Optional[str] = None
    ) -> List[Project]:
        """List projects with optional filtering.
        
        Args:
            skip: Number of projects to skip
            limit: Maximum number of projects to return
            status_filter: Filter by project status
            
        Returns:
            List of projects
        """
        async with self.db_manager.get_session() as session:
            stmt = select(Project).offset(skip).limit(limit)
            
            if status_filter:
                stmt = stmt.where(Project.status == status_filter)
            
            stmt = stmt.order_by(Project.created_at.desc())
            
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def update_project(
        self,
        project_id: UUID,
        **updates
    ) -> Optional[Project]:
        """Update project information.
        
        Args:
            project_id: Project UUID
            **updates: Fields to update
            
        Returns:
            Updated project or None if not found
        """
        async with self.db_manager.get_session() as session:
            stmt = (
                update(Project)
                .where(Project.id == project_id)
                .values(**updates)
                .returning(Project)
            )
            
            result = await session.execute(stmt)
            project = result.scalar_one_or_none()
            
            if project:
                self.logger.info(f"Updated project: {project_id}")
            
            return project
    
    async def delete_project(self, project_id: UUID) -> bool:
        """Delete a project and all associated data.
        
        Args:
            project_id: Project UUID
            
        Returns:
            True if deleted, False if not found
        """
        async with self.db_manager.get_session() as session:
            stmt = delete(Project).where(Project.id == project_id)
            result = await session.execute(stmt)
            
            deleted = result.rowcount > 0
            if deleted:
                self.logger.info(f"Deleted project: {project_id}")
            
            return deleted
    
    async def start_processing(self, project_id: UUID) -> ProcessingSession:
        """Start processing for a project.
        
        Args:
            project_id: Project UUID
            
        Returns:
            Created processing session
        """
        async with self.db_manager.get_session() as session:
            # Get project
            project = await session.get(Project, project_id)
            if not project:
                raise ValueError("Project not found")
            
            # Check if project is already running
            if project.status == "running":
                # Find active session
                active_session_stmt = (
                    select(ProcessingSession)
                    .where(ProcessingSession.project_id == project_id)
                    .where(ProcessingSession.status.in_(["running", "created"]))
                    .order_by(ProcessingSession.created_at.desc())
                )
                result = await session.execute(active_session_stmt)
                active_session = result.scalar_one_or_none()
                
                if active_session:
                    return active_session
            
            # Scan input folder to count total images
            total_images = await self._count_images_in_folder(project.input_folder)
            
            # Update project status
            project.status = "running"
            
            # Create new processing session
            processing_session = ProcessingSession(
                project_id=project_id,
                total_images=total_images,
                status="created",
                session_config={
                    "performance_mode": project.performance_mode,
                    "settings": project.settings,
                    "input_folder": project.input_folder,
                    "output_folder": project.output_folder
                }
            )
            
            session.add(processing_session)
            await session.flush()
            await session.refresh(processing_session)
            
            self.logger.info(f"Started processing for project: {project_id} with {total_images} images")
            return processing_session
    
    async def _count_images_in_folder(self, folder_path: str) -> int:
        """Count total images in input folder structure.
        
        Args:
            folder_path: Path to input folder
            
        Returns:
            Total number of images
        """
        try:
            from pathlib import Path
            import os
            
            supported_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
            total_count = 0
            
            folder = Path(folder_path)
            if not folder.exists():
                self.logger.warning(f"Input folder does not exist: {folder_path}")
                return 0
            
            # Scan all subdirectories for images
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if Path(file).suffix in supported_extensions:
                        total_count += 1
            
            return total_count
            
        except Exception as e:
            self.logger.error(f"Failed to count images in {folder_path}: {e}")
            return 0
    
    async def pause_processing(self, project_id: UUID) -> bool:
        """Pause processing for a project.
        
        Args:
            project_id: Project UUID
            
        Returns:
            True if paused, False if not found or not running
        """
        async with self.db_manager.get_session() as session:
            # Update project status
            stmt = (
                update(Project)
                .where(Project.id == project_id)
                .where(Project.status == "running")
                .values(status="paused")
            )
            
            result = await session.execute(stmt)
            paused = result.rowcount > 0
            
            if paused:
                # Also pause active sessions
                session_stmt = (
                    update(ProcessingSession)
                    .where(ProcessingSession.project_id == project_id)
                    .where(ProcessingSession.status == "running")
                    .values(status="paused")
                )
                await session.execute(session_stmt)
                
                self.logger.info(f"Paused processing for project: {project_id}")
            
            return paused
    
    async def get_project_sessions(self, project_id: UUID) -> List[ProcessingSession]:
        """Get all processing sessions for a project.
        
        Args:
            project_id: Project UUID
            
        Returns:
            List of processing sessions
        """
        async with self.db_manager.get_session() as session:
            stmt = (
                select(ProcessingSession)
                .where(ProcessingSession.project_id == project_id)
                .order_by(ProcessingSession.created_at.desc())
            )
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def get_active_projects(self) -> List[Project]:
        """Get all currently active (running/paused) projects.
        
        Returns:
            List of active projects
        """
        async with self.db_manager.get_session() as session:
            stmt = (
                select(Project)
                .where(Project.status.in_(["running", "paused"]))
                .order_by(Project.updated_at.desc())
            )
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def get_project_statistics(self, project_id: UUID) -> Dict[str, Any]:
        """Get comprehensive project statistics.
        
        Args:
            project_id: Project UUID
            
        Returns:
            Project statistics
        """
        async with self.db_manager.get_session() as session:
            project = await session.get(Project, project_id)
            if not project:
                return {}
            
            # Get all sessions for this project
            sessions_stmt = (
                select(ProcessingSession)
                .where(ProcessingSession.project_id == project_id)
            )
            sessions_result = await session.execute(sessions_stmt)
            sessions = sessions_result.scalars().all()
            
            # Calculate totals across all sessions
            total_processed = sum(s.processed_images for s in sessions)
            total_approved = sum(s.approved_images for s in sessions)
            total_rejected = sum(s.rejected_images for s in sessions)
            
            # Get latest session for current status
            latest_session = sessions[0] if sessions else None
            
            return {
                "project_id": str(project_id),
                "project_name": project.name,
                "status": project.status,
                "total_sessions": len(sessions),
                "total_images": latest_session.total_images if latest_session else 0,
                "total_processed": total_processed,
                "total_approved": total_approved,
                "total_rejected": total_rejected,
                "approval_rate": (total_approved / total_processed * 100) if total_processed > 0 else 0,
                "latest_session": {
                    "id": str(latest_session.id),
                    "status": latest_session.status,
                    "start_time": latest_session.start_time,
                    "end_time": latest_session.end_time
                } if latest_session else None,
                "created_at": project.created_at,
                "updated_at": project.updated_at
            }

class SessionService:
    """Service for managing processing sessions."""
    
    def __init__(self, db_manager: DatabaseManager, config: AppConfig):
        """Initialize session service.
        
        Args:
            db_manager: Database manager
            config: Application configuration
        """
        self.db_manager = db_manager
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def get_session(self, session_id: UUID) -> Optional[ProcessingSession]:
        """Get session by ID.
        
        Args:
            session_id: Session UUID
            
        Returns:
            Session or None if not found
        """
        async with self.db_manager.get_session() as session:
            stmt = (
                select(ProcessingSession)
                .options(selectinload(ProcessingSession.project))
                .where(ProcessingSession.id == session_id)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
    
    async def list_sessions(
        self,
        project_id: Optional[UUID] = None,
        status_filter: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[ProcessingSession]:
        """List sessions with optional filtering.
        
        Args:
            project_id: Filter by project ID
            status_filter: Filter by session status
            skip: Number of sessions to skip
            limit: Maximum number of sessions to return
            
        Returns:
            List of sessions
        """
        async with self.db_manager.get_session() as session:
            stmt = (
                select(ProcessingSession)
                .options(selectinload(ProcessingSession.project))
                .offset(skip)
                .limit(limit)
                .order_by(ProcessingSession.created_at.desc())
            )
            
            if project_id:
                stmt = stmt.where(ProcessingSession.project_id == project_id)
            
            if status_filter:
                stmt = stmt.where(ProcessingSession.status == status_filter)
            
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def get_session_status(self, session_id: UUID) -> Optional[Dict[str, Any]]:
        """Get detailed session status and progress.
        
        Args:
            session_id: Session UUID
            
        Returns:
            Session status information
        """
        async with self.db_manager.get_session() as session:
            processing_session = await session.get(ProcessingSession, session_id)
            if not processing_session:
                return None
            
            # Calculate progress percentage
            progress_percentage = 0.0
            if processing_session.total_images > 0:
                progress_percentage = (processing_session.processed_images / processing_session.total_images) * 100
            
            # Calculate approval rate
            approval_rate = 0.0
            if processing_session.processed_images > 0:
                approval_rate = (processing_session.approved_images / processing_session.processed_images) * 100
            
            return {
                "session_id": str(session_id),
                "status": processing_session.status,
                "progress": {
                    "total_images": processing_session.total_images,
                    "processed_images": processing_session.processed_images,
                    "approved_images": processing_session.approved_images,
                    "rejected_images": processing_session.rejected_images,
                    "progress_percentage": round(progress_percentage, 2),
                    "approval_rate": round(approval_rate, 2)
                },
                "timing": {
                    "start_time": processing_session.start_time,
                    "end_time": processing_session.end_time,
                    "created_at": processing_session.created_at,
                    "updated_at": processing_session.updated_at
                },
                "error_message": processing_session.error_message
            }
    
    async def get_session_results(
        self,
        session_id: UUID,
        skip: int = 0,
        limit: int = 100,
        decision_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get session processing results.
        
        Args:
            session_id: Session UUID
            skip: Number of results to skip
            limit: Maximum number of results to return
            decision_filter: Filter by decision (approved/rejected)
            
        Returns:
            Session results with pagination
        """
        async with self.db_manager.get_session() as session:
            stmt = (
                select(ImageResult)
                .where(ImageResult.session_id == session_id)
                .offset(skip)
                .limit(limit)
                .order_by(ImageResult.created_at.desc())
            )
            
            if decision_filter:
                stmt = stmt.where(ImageResult.final_decision == decision_filter)
            
            result = await session.execute(stmt)
            results = result.scalars().all()
            
            # Get total count
            count_stmt = select(ImageResult).where(ImageResult.session_id == session_id)
            if decision_filter:
                count_stmt = count_stmt.where(ImageResult.final_decision == decision_filter)
            
            count_result = await session.execute(count_stmt)
            total_count = len(count_result.scalars().all())
            
            return {
                "results": [
                    {
                        "id": str(r.id),
                        "filename": r.filename,
                        "image_path": r.image_path,
                        "final_decision": r.final_decision,
                        "rejection_reasons": r.rejection_reasons,
                        "quality_scores": r.quality_scores,
                        "processing_time": r.processing_time,
                        "human_override": r.human_override,
                        "created_at": r.created_at
                    }
                    for r in results
                ],
                "pagination": {
                    "skip": skip,
                    "limit": limit,
                    "total": total_count,
                    "has_more": skip + limit < total_count
                }
            }
    
    async def resume_session(self, session_id: UUID) -> bool:
        """Resume a paused or interrupted session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            True if resumed, False if not found or cannot be resumed
        """
        async with self.db_manager.get_session() as session:
            stmt = (
                update(ProcessingSession)
                .where(ProcessingSession.id == session_id)
                .where(ProcessingSession.status.in_(["paused", "failed"]))
                .values(status="running")
            )
            
            result = await session.execute(stmt)
            resumed = result.rowcount > 0
            
            if resumed:
                self.logger.info(f"Resumed session: {session_id}")
            
            return resumed
    
    async def pause_session(self, session_id: UUID) -> bool:
        """Pause a running session.
        
        Args:
            session_id: Session UUID
            
        Returns:
            True if paused, False if not found or not running
        """
        async with self.db_manager.get_session() as session:
            stmt = (
                update(ProcessingSession)
                .where(ProcessingSession.id == session_id)
                .where(ProcessingSession.status == "running")
                .values(status="paused")
            )
            
            result = await session.execute(stmt)
            paused = result.rowcount > 0
            
            if paused:
                self.logger.info(f"Paused session: {session_id}")
            
            return paused
    
    async def delete_session(self, session_id: UUID) -> bool:
        """Delete a session and its results.
        
        Args:
            session_id: Session UUID
            
        Returns:
            True if deleted, False if not found
        """
        async with self.db_manager.get_session() as session:
            stmt = delete(ProcessingSession).where(ProcessingSession.id == session_id)
            result = await session.execute(stmt)
            
            deleted = result.rowcount > 0
            if deleted:
                self.logger.info(f"Deleted session: {session_id}")
            
            return deleted
    
    async def get_concurrent_sessions(self) -> List[ProcessingSession]:
        """Get all currently running sessions across all projects.
        
        Returns:
            List of active processing sessions
        """
        async with self.db_manager.get_session() as session:
            stmt = (
                select(ProcessingSession)
                .options(selectinload(ProcessingSession.project))
                .where(ProcessingSession.status.in_(["running", "created"]))
                .order_by(ProcessingSession.start_time.desc())
            )
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def get_session_history(
        self,
        project_id: Optional[UUID] = None,
        limit: int = 50
    ) -> List[ProcessingSession]:
        """Get session history with optional project filtering.
        
        Args:
            project_id: Optional project ID to filter by
            limit: Maximum number of sessions to return
            
        Returns:
            List of historical sessions
        """
        async with self.db_manager.get_session() as session:
            stmt = (
                select(ProcessingSession)
                .options(selectinload(ProcessingSession.project))
                .order_by(ProcessingSession.created_at.desc())
                .limit(limit)
            )
            
            if project_id:
                stmt = stmt.where(ProcessingSession.project_id == project_id)
            
            result = await session.execute(stmt)
            return result.scalars().all()
    
    async def update_session_progress(
        self,
        session_id: UUID,
        processed_count: int,
        approved_count: int,
        rejected_count: int,
        current_image: Optional[str] = None
    ) -> bool:
        """Update session progress counters.
        
        Args:
            session_id: Session UUID
            processed_count: Number of processed images
            approved_count: Number of approved images
            rejected_count: Number of rejected images
            current_image: Currently processing image
            
        Returns:
            True if updated successfully
        """
        async with self.db_manager.get_session() as session:
            stmt = (
                update(ProcessingSession)
                .where(ProcessingSession.id == session_id)
                .values(
                    processed_images=processed_count,
                    approved_images=approved_count,
                    rejected_images=rejected_count,
                    updated_at=datetime.now()
                )
            )
            
            result = await session.execute(stmt)
            updated = result.rowcount > 0
            
            if updated:
                self.logger.debug(f"Updated session progress: {session_id} - {processed_count}/{approved_count}/{rejected_count}")
            
            return updated

class AnalysisService:
    """Service for image analysis operations."""
    
    def __init__(self, config: AppConfig):
        """Initialize analysis service.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.analyzer_factory = AnalyzerFactory(config)
    
    async def analyze_image(
        self,
        image_path: str,
        analysis_types: List[str],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a single image.
        
        Args:
            image_path: Path to image file
            analysis_types: List of analysis types to perform
            options: Analysis options
            
        Returns:
            Analysis results
        """
        start_time = datetime.now()
        results = {
            "image_path": image_path,
            "filename": image_path.split("/")[-1],
            "success": True,
            "error_message": None
        }
        
        try:
            # Perform requested analyses
            if "quality" in analysis_types:
                quality_analyzer = self.analyzer_factory.get_quality_analyzer()
                quality_result = quality_analyzer.analyze(image_path)
                results["quality_result"] = {
                    "sharpness_score": quality_result.sharpness_score,
                    "noise_level": quality_result.noise_level,
                    "exposure_score": quality_result.exposure_score,
                    "color_balance_score": quality_result.color_balance_score,
                    "resolution": quality_result.resolution,
                    "file_size": quality_result.file_size,
                    "overall_score": quality_result.overall_score,
                    "passed": quality_result.passed
                }
            
            if "defect" in analysis_types:
                defect_analyzer = self.analyzer_factory.get_defect_detector()
                defect_result = defect_analyzer.detect_defects(image_path)
                results["defect_result"] = {
                    "defect_count": defect_result.defect_count,
                    "anomaly_score": defect_result.anomaly_score,
                    "defect_types": defect_result.defect_types,
                    "confidence_scores": defect_result.confidence_scores,
                    "detected_objects": [
                        {
                            "object_type": obj.object_type,
                            "defect_type": obj.defect_type,
                            "confidence": obj.confidence,
                            "bounding_box": obj.bounding_box,
                            "description": obj.description
                        }
                        for obj in defect_result.detected_objects
                    ],
                    "passed": defect_result.passed
                }
            
            if "compliance" in analysis_types:
                compliance_analyzer = self.analyzer_factory.get_compliance_checker()
                compliance_result = compliance_analyzer.check_compliance(image_path, {})
                results["compliance_result"] = {
                    "logo_detections": [
                        {
                            "logo_type": logo.logo_type,
                            "confidence": logo.confidence,
                            "bounding_box": logo.bounding_box,
                            "brand_name": logo.brand_name
                        }
                        for logo in compliance_result.logo_detections
                    ],
                    "privacy_violations": [
                        {
                            "violation_type": violation.violation_type,
                            "confidence": violation.confidence,
                            "bounding_box": violation.bounding_box,
                            "description": violation.description
                        }
                        for violation in compliance_result.privacy_violations
                    ],
                    "metadata_issues": compliance_result.metadata_issues,
                    "keyword_relevance": compliance_result.keyword_relevance,
                    "overall_compliance": compliance_result.overall_compliance
                }
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            results["processing_time"] = processing_time
            results["timestamp"] = datetime.now()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {image_path}: {e}")
            results["success"] = False
            results["error_message"] = str(e)
            results["processing_time"] = (datetime.now() - start_time).total_seconds()
            results["timestamp"] = datetime.now()
            return results
    
    async def analyze_batch(
        self,
        image_paths: List[str],
        analysis_types: List[str],
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze a batch of images.
        
        Args:
            image_paths: List of image paths
            analysis_types: List of analysis types to perform
            options: Analysis options
            
        Returns:
            List of analysis results
        """
        results = []
        
        for image_path in image_paths:
            result = await self.analyze_image(image_path, analysis_types, options)
            results.append(result)
        
        return results
    
    async def get_analysis_config(self) -> Dict[str, Any]:
        """Get current analysis configuration.
        
        Returns:
            Analysis configuration
        """
        return {
            "quality": {
                "min_sharpness": self.config.quality.min_sharpness,
                "max_noise_level": self.config.quality.max_noise_level,
                "min_resolution": self.config.quality.min_resolution
            },
            "similarity": {
                "hash_threshold": self.config.similarity.hash_threshold,
                "feature_threshold": self.config.similarity.feature_threshold,
                "clustering_eps": self.config.similarity.clustering_eps
            },
            "compliance": {
                "logo_detection_confidence": self.config.compliance.logo_detection_confidence,
                "face_detection_enabled": self.config.compliance.face_detection_enabled,
                "metadata_validation": self.config.compliance.metadata_validation
            },
            "processing": {
                "batch_size": self.config.processing.batch_size,
                "max_workers": self.config.processing.max_workers,
                "checkpoint_interval": self.config.processing.checkpoint_interval
            }
        }