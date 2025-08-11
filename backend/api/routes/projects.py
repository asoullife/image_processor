"""Project management API routes."""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from uuid import UUID
import logging

import sys
from pathlib import Path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from api.dependencies import get_project_service
from core.services import ProjectService
from database.models import Project
from api.schemas import ProjectCreate, ProjectResponse, ProjectUpdate

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=ProjectResponse)
async def create_project(
    project_data: ProjectCreate,
    service: ProjectService = Depends(get_project_service)
) -> ProjectResponse:
    """Create a new processing project.
    
    Args:
        project_data: Project creation data
        service: Project service
        
    Returns:
        Created project information
    """
    try:
        project = await service.create_project(
            name=project_data.name,
            description=project_data.description,
            input_folder=project_data.input_folder,
            output_folder=project_data.output_folder,
            performance_mode=project_data.performance_mode,
            settings=project_data.settings
        )
        
        return ProjectResponse.from_orm(project)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        raise HTTPException(status_code=500, detail="Failed to create project")

@router.get("/", response_model=List[ProjectResponse])
async def list_projects(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    service: ProjectService = Depends(get_project_service)
) -> List[ProjectResponse]:
    """List all projects with optional filtering.
    
    Args:
        skip: Number of projects to skip
        limit: Maximum number of projects to return
        status: Filter by project status
        service: Project service
        
    Returns:
        List of projects
    """
    try:
        projects = await service.list_projects(
            skip=skip,
            limit=limit,
            status_filter=status
        )
        
        return [ProjectResponse.from_orm(project) for project in projects]
        
    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        raise HTTPException(status_code=500, detail="Failed to list projects")

@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: UUID,
    service: ProjectService = Depends(get_project_service)
) -> ProjectResponse:
    """Get project by ID.
    
    Args:
        project_id: Project UUID
        service: Project service
        
    Returns:
        Project information
    """
    try:
        project = await service.get_project(project_id)
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return ProjectResponse.from_orm(project)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get project")

@router.put("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: UUID,
    project_data: ProjectUpdate,
    service: ProjectService = Depends(get_project_service)
) -> ProjectResponse:
    """Update project information.
    
    Args:
        project_id: Project UUID
        project_data: Project update data
        service: Project service
        
    Returns:
        Updated project information
    """
    try:
        project = await service.update_project(
            project_id=project_id,
            **project_data.dict(exclude_unset=True)
        )
        
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return ProjectResponse.from_orm(project)
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update project {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update project")

@router.delete("/{project_id}")
async def delete_project(
    project_id: UUID,
    service: ProjectService = Depends(get_project_service)
) -> Dict[str, str]:
    """Delete a project.
    
    Args:
        project_id: Project UUID
        service: Project service
        
    Returns:
        Deletion confirmation
    """
    try:
        success = await service.delete_project(project_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return {"message": "Project deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete project {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete project")

@router.post("/{project_id}/start")
async def start_project_processing(
    project_id: UUID,
    service: ProjectService = Depends(get_project_service)
) -> Dict[str, Any]:
    """Start processing for a project.
    
    Args:
        project_id: Project UUID
        service: Project service
        
    Returns:
        Processing start confirmation
    """
    try:
        session = await service.start_processing(project_id)
        
        return {
            "message": "Processing started",
            "project_id": str(project_id),
            "session_id": str(session.id)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start processing for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to start processing")

@router.post("/{project_id}/pause")
async def pause_project_processing(
    project_id: UUID,
    service: ProjectService = Depends(get_project_service)
) -> Dict[str, str]:
    """Pause processing for a project.
    
    Args:
        project_id: Project UUID
        service: Project service
        
    Returns:
        Pause confirmation
    """
    try:
        success = await service.pause_processing(project_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Project not found or not running")
        
        return {"message": "Processing paused"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause processing for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to pause processing")

@router.get("/{project_id}/sessions")
async def get_project_sessions(
    project_id: UUID,
    service: ProjectService = Depends(get_project_service)
) -> List[Dict[str, Any]]:
    """Get all processing sessions for a project.
    
    Args:
        project_id: Project UUID
        service: Project service
        
    Returns:
        List of project sessions
    """
    try:
        sessions = await service.get_project_sessions(project_id)
        
        return [
            {
                "id": str(session.id),
                "project_id": str(session.project_id),
                "status": session.status,
                "total_images": session.total_images,
                "processed_images": session.processed_images,
                "approved_images": session.approved_images,
                "rejected_images": session.rejected_images,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "created_at": session.created_at,
                "updated_at": session.updated_at
            }
            for session in sessions
        ]
        
    except Exception as e:
        logger.error(f"Failed to get sessions for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get project sessions")

@router.get("/{project_id}/statistics")
async def get_project_statistics(
    project_id: UUID,
    service: ProjectService = Depends(get_project_service)
) -> Dict[str, Any]:
    """Get comprehensive project statistics.
    
    Args:
        project_id: Project UUID
        service: Project service
        
    Returns:
        Project statistics
    """
    try:
        stats = await service.get_project_statistics(project_id)
        
        if not stats:
            raise HTTPException(status_code=404, detail="Project not found")
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get statistics for project {project_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get project statistics")

@router.get("/active")
async def get_active_projects(
    service: ProjectService = Depends(get_project_service)
) -> List[ProjectResponse]:
    """Get all currently active projects.
    
    Args:
        service: Project service
        
    Returns:
        List of active projects
    """
    try:
        projects = await service.get_active_projects()
        
        return [ProjectResponse.from_orm(project) for project in projects]
        
    except Exception as e:
        logger.error(f"Failed to get active projects: {e}")
        raise HTTPException(status_code=500, detail="Failed to get active projects")