"""PostgreSQL database connection management."""

import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Database configuration
load_dotenv()
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/stockdb"
)

# SQLAlchemy setup
Base = declarative_base()
engine = None
SessionLocal = None

class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self, database_url: str = None):
        """Initialize database manager.
        
        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url or DATABASE_URL
        self.engine = None
        self.session_factory = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connection and session factory."""
        if self._initialized:
            return
        
        try:
            # Create async engine with enhanced connection pooling
            connect_args = {}
            
            # PostgreSQL-specific settings
            if "postgresql" in self.database_url:
                connect_args = {
                    "server_settings": {
                        "application_name": "adobe_stock_processor_api",
                        "jit": "off"  # Disable JIT for better connection performance
                    }
                }
            
            self.engine = create_async_engine(
                self.database_url,
                echo=False,  # Set to True for SQL logging
                pool_size=20,  # Increased for API workload
                max_overflow=30,  # Allow more overflow connections
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=3600,  # Recycle connections every hour
                pool_timeout=30,  # Timeout for getting connection from pool
                connect_args=connect_args
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            await self._test_connection()
            
            self._initialized = True
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise
    
    async def _test_connection(self):
        """Test database connection."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(text("SELECT 1"))
                assert result.scalar() == 1
                logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session context manager.
        
        Yields:
            AsyncSession: Database session
        """
        if not self._initialized:
            await self.initialize()
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    async def create_tables(self):
        """Create all database tables."""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    async def drop_tables(self):
        """Drop all database tables."""
        if not self._initialized:
            await self.initialize()
        
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup database connections."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections cleaned up")
        self._initialized = False

# Global database manager instance
_db_manager: Optional[DatabaseManager] = None

async def get_database_manager() -> DatabaseManager:
    """Get global database manager instance.
    
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()
    
    return _db_manager

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session for dependency injection.
    
    Yields:
        AsyncSession: Database session
    """
    db_manager = await get_database_manager()
    async with db_manager.get_session() as session:
        yield session

# Initialize database on module import for development
async def init_db():
    """Initialize database for development."""
    try:
        db_manager = await get_database_manager()
        await db_manager.create_tables()
        logger.info("Database initialized for development")
    except Exception as e:
        logger.warning(f"Could not initialize database: {e}")

# Run initialization in background for development
# Note: Removed automatic initialization to avoid issues with Alembic