"""Authentication service."""

import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from auth.models import User, Session
from auth.security import security_manager
from database.connection import DatabaseManager

logger = logging.getLogger(__name__)

class AuthenticationService:
    """Service for user authentication and session management."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize authentication service.
        
        Args:
            db_manager: Database manager
        """
        self.db_manager = db_manager
        self.security = security_manager
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None,
        is_superuser: bool = False
    ) -> User:
        """Create a new user.
        
        Args:
            username: Username
            email: Email address
            password: Plain text password
            full_name: Full name
            is_superuser: Whether user is superuser
            
        Returns:
            Created user
            
        Raises:
            ValueError: If username or email already exists
        """
        async with self.db_manager.get_session() as session:
            # Check if username exists
            stmt = select(User).where(User.username == username)
            existing_user = await session.execute(stmt)
            if existing_user.scalar_one_or_none():
                raise ValueError("Username already exists")
            
            # Check if email exists
            stmt = select(User).where(User.email == email)
            existing_email = await session.execute(stmt)
            if existing_email.scalar_one_or_none():
                raise ValueError("Email already exists")
            
            # Create user
            hashed_password = self.security.hash_password(password)
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                full_name=full_name,
                is_superuser=is_superuser
            )
            
            session.add(user)
            await session.flush()
            await session.refresh(user)
            
            self.logger.info(f"Created user: {username}")
            return user
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password.
        
        Args:
            username: Username or email
            password: Plain text password
            
        Returns:
            User if authentication successful, None otherwise
        """
        async with self.db_manager.get_session() as session:
            # Try to find user by username or email
            stmt = select(User).where(
                (User.username == username) | (User.email == username)
            ).where(User.is_active == True)
            
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()
            
            if not user:
                self.logger.warning(f"User not found: {username}")
                return None
            
            if not self.security.verify_password(password, user.hashed_password):
                self.logger.warning(f"Invalid password for user: {username}")
                return None
            
            # Update last login
            user.last_login = datetime.utcnow()
            await session.commit()
            
            self.logger.info(f"User authenticated: {username}")
            return user
    
    async def create_session(
        self,
        user_id: UUID,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Tuple[str, Session]:
        """Create a new user session.
        
        Args:
            user_id: User ID
            user_agent: User agent string
            ip_address: Client IP address
            
        Returns:
            Tuple of (session_token, session_object)
        """
        async with self.db_manager.get_session() as session:
            # Generate session token
            session_token = self.security.generate_session_token()
            hashed_token = self.security.hash_session_token(session_token)
            
            # Create session
            expires_at = datetime.utcnow() + timedelta(days=7)  # 7 days
            user_session = Session(
                user_id=user_id,
                session_token=hashed_token,
                expires_at=expires_at,
                user_agent=user_agent,
                ip_address=ip_address
            )
            
            session.add(user_session)
            await session.flush()
            await session.refresh(user_session)
            
            self.logger.info(f"Created session for user: {user_id}")
            return session_token, user_session
    
    async def validate_session(self, session_token: str) -> Optional[Tuple[User, Session]]:
        """Validate a session token.
        
        Args:
            session_token: Session token
            
        Returns:
            Tuple of (user, session) if valid, None otherwise
        """
        async with self.db_manager.get_session() as db_session:
            hashed_token = self.security.hash_session_token(session_token)
            
            # Find active session
            stmt = (
                select(Session, User)
                .join(User, Session.user_id == User.id)
                .where(Session.session_token == hashed_token)
                .where(Session.is_active == True)
                .where(Session.expires_at > datetime.utcnow())
                .where(User.is_active == True)
            )
            
            result = await db_session.execute(stmt)
            row = result.first()
            
            if not row:
                return None
            
            session_obj, user = row
            return user, session_obj
    
    async def invalidate_session(self, session_token: str) -> bool:
        """Invalidate a session.
        
        Args:
            session_token: Session token
            
        Returns:
            True if session was invalidated
        """
        async with self.db_manager.get_session() as session:
            hashed_token = self.security.hash_session_token(session_token)
            
            stmt = (
                update(Session)
                .where(Session.session_token == hashed_token)
                .values(is_active=False)
            )
            
            result = await session.execute(stmt)
            invalidated = result.rowcount > 0
            
            if invalidated:
                self.logger.info("Session invalidated")
            
            return invalidated
    
    async def invalidate_user_sessions(self, user_id: UUID) -> int:
        """Invalidate all sessions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of sessions invalidated
        """
        async with self.db_manager.get_session() as session:
            stmt = (
                update(Session)
                .where(Session.user_id == user_id)
                .values(is_active=False)
            )
            
            result = await session.execute(stmt)
            count = result.rowcount
            
            self.logger.info(f"Invalidated {count} sessions for user: {user_id}")
            return count
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        async with self.db_manager.get_session() as session:
            stmt = delete(Session).where(Session.expires_at < datetime.utcnow())
            result = await session.execute(stmt)
            count = result.rowcount
            
            self.logger.info(f"Cleaned up {count} expired sessions")
            return count
    
    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User or None if not found
        """
        async with self.db_manager.get_session() as session:
            stmt = select(User).where(User.id == user_id).where(User.is_active == True)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
    
    async def update_user_password(self, user_id: UUID, new_password: str) -> bool:
        """Update user password.
        
        Args:
            user_id: User ID
            new_password: New plain text password
            
        Returns:
            True if password was updated
        """
        async with self.db_manager.get_session() as session:
            hashed_password = self.security.hash_password(new_password)
            
            stmt = (
                update(User)
                .where(User.id == user_id)
                .values(hashed_password=hashed_password, updated_at=datetime.utcnow())
            )
            
            result = await session.execute(stmt)
            updated = result.rowcount > 0
            
            if updated:
                self.logger.info(f"Password updated for user: {user_id}")
                # Invalidate all sessions for security
                await self.invalidate_user_sessions(user_id)
            
            return updated