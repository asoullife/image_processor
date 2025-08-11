"""Authentication dependencies for FastAPI."""

from fastapi import Depends, HTTPException, status, Request, Cookie
from typing import Optional
import logging

from auth.service import AuthenticationService
from auth.models import User
from database.connection import DatabaseManager, get_database_manager

logger = logging.getLogger(__name__)

async def get_auth_service() -> AuthenticationService:
    """Get authentication service dependency.
    
    Returns:
        Authentication service instance
    """
    db_manager = await get_database_manager()
    return AuthenticationService(db_manager)

async def get_current_user(
    request: Request,
    session_token: Optional[str] = Cookie(None),
    auth_service: AuthenticationService = Depends(get_auth_service)
) -> Optional[User]:
    """Get current user from session.
    
    Args:
        request: FastAPI request
        session_token: Session token from cookie
        auth_service: Authentication service
        
    Returns:
        Current user or None if not authenticated
    """
    if not session_token:
        return None
    
    try:
        user, session = await auth_service.validate_session(session_token)
        return user
    except Exception as e:
        logger.warning(f"Session validation failed: {e}")
        return None

async def get_current_active_user(
    current_user: Optional[User] = Depends(get_current_user)
) -> User:
    """Get current active user, raise exception if not authenticated.
    
    Args:
        current_user: Current user from session
        
    Returns:
        Current active user
        
    Raises:
        HTTPException: If user is not authenticated or inactive
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return current_user

async def get_current_superuser(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current superuser, raise exception if not superuser.
    
    Args:
        current_user: Current active user
        
    Returns:
        Current superuser
        
    Raises:
        HTTPException: If user is not a superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    
    return current_user

def require_auth(current_user: User = Depends(get_current_active_user)) -> User:
    """Dependency to require authentication.
    
    Args:
        current_user: Current active user
        
    Returns:
        Current user
    """
    return current_user

def require_superuser(current_user: User = Depends(get_current_superuser)) -> User:
    """Dependency to require superuser privileges.
    
    Args:
        current_user: Current superuser
        
    Returns:
        Current superuser
    """
    return current_user