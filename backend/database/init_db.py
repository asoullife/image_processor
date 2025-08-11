"""Database initialization utilities."""

import asyncio
import logging
from typing import Optional
from sqlalchemy import text

from database.connection import DatabaseManager
from database.models import Base
from auth.service import AuthenticationService

logger = logging.getLogger(__name__)

class DatabaseInitializer:
    """Database initialization and setup utilities."""
    
    def __init__(self, database_url: Optional[str] = None):
        """Initialize database initializer.
        
        Args:
            database_url: Database connection URL
        """
        self.db_manager = DatabaseManager(database_url)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def initialize_database(self, create_tables: bool = True, create_admin: bool = True):
        """Initialize database with tables and default data.
        
        Args:
            create_tables: Whether to create database tables
            create_admin: Whether to create default admin user
        """
        try:
            # Initialize database connection
            await self.db_manager.initialize()
            self.logger.info("Database connection initialized")
            
            if create_tables:
                await self.create_tables()
            
            if create_admin:
                await self.create_default_admin()
            
            self.logger.info("Database initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    async def create_tables(self):
        """Create all database tables."""
        try:
            async with self.db_manager.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create database tables: {e}")
            raise
    
    async def drop_tables(self):
        """Drop all database tables."""
        try:
            async with self.db_manager.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            self.logger.info("Database tables dropped successfully")
        except Exception as e:
            self.logger.error(f"Failed to drop database tables: {e}")
            raise
    
    async def create_default_admin(self):
        """Create default admin user if it doesn't exist."""
        try:
            auth_service = AuthenticationService(self.db_manager)
            
            # Check if admin user already exists
            async with self.db_manager.get_session() as session:
                from sqlalchemy import select
                from auth.models import User
                
                stmt = select(User).where(User.username == "admin")
                result = await session.execute(stmt)
                existing_admin = result.scalar_one_or_none()
                
                if existing_admin:
                    self.logger.info("Admin user already exists")
                    return
            
            # Create admin user
            admin_user = await auth_service.create_user(
                username="admin",
                email="admin@localhost",
                password="admin123",  # Change this in production!
                full_name="System Administrator",
                is_superuser=True
            )
            
            self.logger.info(f"Created default admin user: {admin_user.username}")
            self.logger.warning("Default admin password is 'admin123' - please change it!")
            
        except Exception as e:
            self.logger.error(f"Failed to create default admin user: {e}")
            # Don't raise here as this is not critical for basic functionality
    
    async def verify_database_schema(self) -> bool:
        """Verify that all required tables exist.
        
        Returns:
            True if schema is valid
        """
        try:
            async with self.db_manager.get_session() as session:
                # Check for required tables
                required_tables = [
                    'users', 'user_sessions', 'projects', 'processing_sessions',
                    'image_results', 'checkpoints', 'similarity_groups',
                    'processing_logs', 'system_metrics', 'user_preferences'
                ]
                
                for table_name in required_tables:
                    result = await session.execute(text(f"""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = '{table_name}'
                        )
                    """))
                    
                    exists = result.scalar()
                    if not exists:
                        self.logger.error(f"Required table '{table_name}' does not exist")
                        return False
                
                self.logger.info("Database schema verification passed")
                return True
                
        except Exception as e:
            self.logger.error(f"Database schema verification failed: {e}")
            return False
    
    async def get_database_info(self) -> dict:
        """Get database information and statistics.
        
        Returns:
            Database information dictionary
        """
        try:
            async with self.db_manager.get_session() as session:
                # Get PostgreSQL version
                result = await session.execute(text("SELECT version()"))
                db_version = result.scalar()
                
                # Get database size
                result = await session.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """))
                db_size = result.scalar()
                
                # Get table counts
                table_counts = {}
                tables = ['users', 'projects', 'processing_sessions', 'image_results']
                
                for table in tables:
                    result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    table_counts[table] = result.scalar()
                
                return {
                    "database_version": db_version,
                    "database_size": db_size,
                    "table_counts": table_counts,
                    "connection_pool_size": self.db_manager.engine.pool.size(),
                    "connection_pool_checked_out": self.db_manager.engine.pool.checkedout()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get database info: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup database connections."""
        await self.db_manager.cleanup()

async def init_database(database_url: Optional[str] = None):
    """Initialize database with default settings.
    
    Args:
        database_url: Database connection URL
    """
    initializer = DatabaseInitializer(database_url)
    try:
        await initializer.initialize_database()
    finally:
        await initializer.cleanup()

async def verify_database(database_url: Optional[str] = None) -> bool:
    """Verify database schema.
    
    Args:
        database_url: Database connection URL
        
    Returns:
        True if schema is valid
    """
    initializer = DatabaseInitializer(database_url)
    try:
        await initializer.db_manager.initialize()
        return await initializer.verify_database_schema()
    finally:
        await initializer.cleanup()

if __name__ == "__main__":
    # Command line database initialization
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "init":
            asyncio.run(init_database())
        elif command == "verify":
            result = asyncio.run(verify_database())
            sys.exit(0 if result else 1)
        else:
            print("Usage: python init_db.py [init|verify]")
            sys.exit(1)
    else:
        asyncio.run(init_database())