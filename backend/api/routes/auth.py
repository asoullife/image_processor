"""Authentication API routes."""

from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from fastapi.security import OAuth2PasswordRequestForm
from typing import Dict, Any
import logging

import sys
from pathlib import Path
backend_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_path))

from auth.service import AuthenticationService
from auth.dependencies import get_auth_service, get_current_active_user
from auth.models import User
from auth.security import security_manager
from api.schemas import UserCreate, UserResponse, LoginResponse, TokenResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    auth_service: AuthenticationService = Depends(get_auth_service)
) -> UserResponse:
    """Register a new user.
    
    Args:
        user_data: User registration data
        auth_service: Authentication service
        
    Returns:
        Created user information
    """
    try:
        user = await auth_service.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name
        )
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            is_superuser=user.is_superuser,
            created_at=user.created_at,
            last_login=user.last_login
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@router.post("/login", response_model=LoginResponse)
async def login(
    response: Response,
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthenticationService = Depends(get_auth_service)
) -> LoginResponse:
    """Login user with username/password.
    
    Args:
        response: FastAPI response
        request: FastAPI request
        form_data: Login form data
        auth_service: Authentication service
        
    Returns:
        Login response with tokens
    """
    try:
        # Authenticate user
        user = await auth_service.authenticate_user(
            username=form_data.username,
            password=form_data.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create session
        user_agent = request.headers.get("user-agent")
        client_ip = request.client.host if request.client else None
        
        session_token, session = await auth_service.create_session(
            user_id=user.id,
            user_agent=user_agent,
            ip_address=client_ip
        )
        
        # Create JWT tokens
        access_token = security_manager.create_access_token(
            data={"sub": str(user.id), "username": user.username}
        )
        refresh_token = security_manager.create_refresh_token(
            data={"sub": str(user.id), "username": user.username}
        )
        
        # Set session cookie
        response.set_cookie(
            key="session_token",
            value=session_token,
            max_age=7 * 24 * 60 * 60,  # 7 days
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax"
        )
        
        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=security_manager.access_token_expire_minutes * 60,
            user=UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                is_active=user.is_active,
                is_superuser=user.is_superuser,
                created_at=user.created_at,
                last_login=user.last_login
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_token: str,
    auth_service: AuthenticationService = Depends(get_auth_service)
) -> TokenResponse:
    """Refresh access token using refresh token.
    
    Args:
        refresh_token: Refresh token
        auth_service: Authentication service
        
    Returns:
        New access token
    """
    try:
        # Verify refresh token
        payload = security_manager.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Get user
        user_id = payload.get("sub")
        user = await auth_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Create new access token
        access_token = security_manager.create_access_token(
            data={"sub": str(user.id), "username": user.username}
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=security_manager.access_token_expire_minutes * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")

@router.post("/logout")
async def logout(
    response: Response,
    request: Request,
    auth_service: AuthenticationService = Depends(get_auth_service)
) -> Dict[str, str]:
    """Logout user and invalidate session.
    
    Args:
        response: FastAPI response
        request: FastAPI request
        auth_service: Authentication service
        
    Returns:
        Logout confirmation
    """
    try:
        # Get session token from cookie
        session_token = request.cookies.get("session_token")
        
        if session_token:
            # Invalidate session
            await auth_service.invalidate_session(session_token)
        
        # Clear session cookie
        response.delete_cookie(key="session_token")
        
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        # Still clear cookie even if invalidation fails
        response.delete_cookie(key="session_token")
        return {"message": "Logged out"}

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
) -> UserResponse:
    """Get current user information.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User information
    """
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_superuser=current_user.is_superuser,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )

@router.post("/change-password")
async def change_password(
    current_password: str,
    new_password: str,
    current_user: User = Depends(get_current_active_user),
    auth_service: AuthenticationService = Depends(get_auth_service)
) -> Dict[str, str]:
    """Change user password.
    
    Args:
        current_password: Current password
        new_password: New password
        current_user: Current authenticated user
        auth_service: Authentication service
        
    Returns:
        Success message
    """
    try:
        # Verify current password
        if not security_manager.verify_password(current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect current password"
            )
        
        # Update password
        success = await auth_service.update_user_password(current_user.id, new_password)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update password"
            )
        
        return {"message": "Password updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change failed: {e}")
        raise HTTPException(status_code=500, detail="Password change failed")