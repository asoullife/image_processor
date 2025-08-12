"""Path utilities for handling relative paths in the Adobe Stock Image Processor."""

import os
from pathlib import Path
from typing import Union


def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path: Path to the project root directory.
    """
    # Get the directory containing this file (backend/utils/)
    current_file = Path(__file__).resolve()
    # Go up two levels: backend/utils/ -> backend/ -> project_root/
    project_root = current_file.parent.parent.parent
    return project_root


def get_backend_root() -> Path:
    """Get the backend root directory.
    
    Returns:
        Path: Path to the backend directory.
    """
    # Get the directory containing this file (backend/utils/)
    current_file = Path(__file__).resolve()
    # Go up one level: backend/utils/ -> backend/
    backend_root = current_file.parent.parent
    return backend_root


def get_reports_dir() -> str:
    """Get the reports directory path relative to project root.
    
    Returns:
        str: Absolute path to the reports directory.
    """
    project_root = get_project_root()
    reports_dir = project_root / "reports"
    # Ensure the directory exists
    reports_dir.mkdir(exist_ok=True)
    return str(reports_dir)


def get_logs_dir() -> str:
    """Get the logs directory path relative to project root.
    
    Returns:
        str: Absolute path to the logs directory.
    """
    project_root = get_project_root()
    logs_dir = project_root / "logs"
    # Ensure the directory exists
    logs_dir.mkdir(exist_ok=True)
    return str(logs_dir)


def get_data_dir() -> str:
    """Get the backend data directory path.
    
    Returns:
        str: Absolute path to the backend data directory.
    """
    backend_root = get_backend_root()
    data_dir = backend_root / "data"
    # Ensure the directory exists
    data_dir.mkdir(exist_ok=True)
    return str(data_dir)


def ensure_directory_exists(path: Union[str, Path]) -> str:
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory.
        
    Returns:
        str: Absolute path to the directory.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return str(path_obj.resolve())


def make_relative_to_project(path: Union[str, Path]) -> str:
    """Make a path relative to the project root.
    
    Args:
        path: Path to make relative.
        
    Returns:
        str: Path relative to project root.
    """
    project_root = get_project_root()
    path_obj = Path(path).resolve()
    
    try:
        relative_path = path_obj.relative_to(project_root)
        return str(relative_path)
    except ValueError:
        # Path is not relative to project root, return as-is
        return str(path_obj)


def resolve_path_from_project_root(relative_path: Union[str, Path]) -> str:
    """Resolve a path relative to the project root.
    
    Args:
        relative_path: Path relative to project root.
        
    Returns:
        str: Absolute path.
    """
    project_root = get_project_root()
    return str(project_root / relative_path)