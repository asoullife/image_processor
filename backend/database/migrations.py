"""Database migration utilities."""

import logging
import asyncio
from typing import List, Dict, Any
from sqlalchemy import text
from .connection import DatabaseManager, Base
from .models import (
    Project, ProcessingSession, ImageResult, Checkpoint,
    SimilarityGroup, ProcessingLog, SystemMetrics, UserPreferences
)

logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """Database migration manager."""
    
    def __init__(self, db_manager: DatabaseManager = None):
        """Initialize database migrator.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager or DatabaseManager()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def create_all_tables(self):
        """Create all database tables."""
        try:
            await self.db_manager.initialize()
            await self.db_manager.create_tables()
            self.logger.info("All database tables created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
            raise
    
    async def drop_all_tables(self):
        """Drop all database tables."""
        try:
            await self.db_manager.initialize()
            await self.db_manager.drop_tables()
            self.logger.info("All database tables dropped successfully")
        except Exception as e:
            self.logger.error(f"Failed to drop tables: {e}")
            raise
    
    async def check_database_exists(self) -> bool:
        """Check if database exists and is accessible.
        
        Returns:
            True if database exists and is accessible
        """
        try:
            await self.db_manager.initialize()
            async with self.db_manager.get_session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            self.logger.error(f"Database check failed: {e}")
            return False
    
    async def get_table_info(self) -> Dict[str, Any]:
        """Get information about existing tables.
        
        Returns:
            Dictionary with table information
        """
        try:
            await self.db_manager.initialize()
            
            async with self.db_manager.get_session() as session:
                # Get list of tables
                result = await session.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """))
                
                tables = [row[0] for row in result.fetchall()]
                
                table_info = {}
                for table in tables:
                    # Get column information for each table
                    col_result = await session.execute(text(f"""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns
                        WHERE table_name = '{table}'
                        ORDER BY ordinal_position
                    """))
                    
                    columns = [
                        {
                            "name": row[0],
                            "type": row[1],
                            "nullable": row[2] == "YES"
                        }
                        for row in col_result.fetchall()
                    ]
                    
                    table_info[table] = {
                        "columns": columns,
                        "column_count": len(columns)
                    }
                
                return {
                    "tables": tables,
                    "table_count": len(tables),
                    "table_info": table_info
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get table info: {e}")
            return {"error": str(e)}
    
    async def migrate_to_latest(self):
        """Migrate database to latest schema version."""
        try:
            # Check if database exists
            if not await self.check_database_exists():
                self.logger.info("Database not found, creating new database")
                await self.create_all_tables()
                return
            
            # Get current table info
            table_info = await self.get_table_info()
            existing_tables = table_info.get("tables", [])
            
            # Define expected tables
            expected_tables = [
                "projects",
                "processing_sessions", 
                "image_results",
                "checkpoints",
                "similarity_groups",
                "processing_logs",
                "system_metrics",
                "user_preferences"
            ]
            
            # Check for missing tables
            missing_tables = set(expected_tables) - set(existing_tables)
            
            if missing_tables:
                self.logger.info(f"Missing tables detected: {missing_tables}")
                self.logger.info("Creating missing tables...")
                await self.db_manager.create_tables()
            else:
                self.logger.info("All expected tables exist")
            
            # TODO: Add column migration logic here if needed
            # This would check for missing columns and add them
            
            self.logger.info("Database migration completed successfully")
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            raise
    
    async def seed_default_data(self):
        """Seed database with default data."""
        try:
            await self.db_manager.initialize()
            
            async with self.db_manager.get_session() as session:
                # Check if default preferences exist
                result = await session.execute(
                    text("SELECT COUNT(*) FROM user_preferences")
                )
                pref_count = result.scalar()
                
                if pref_count == 0:
                    # Insert default preferences
                    default_preferences = [
                        {
                            "preference_key": "default_performance_mode",
                            "preference_value": {"value": "balanced"},
                            "description": "Default performance mode for new projects"
                        },
                        {
                            "preference_key": "default_batch_size",
                            "preference_value": {"value": 20},
                            "description": "Default batch size for processing"
                        },
                        {
                            "preference_key": "ui_theme",
                            "preference_value": {"value": "light"},
                            "description": "User interface theme preference"
                        }
                    ]
                    
                    for pref in default_preferences:
                        await session.execute(text("""
                            INSERT INTO user_preferences (preference_key, preference_value, description)
                            VALUES (:key, :value, :desc)
                        """), {
                            "key": pref["preference_key"],
                            "value": pref["preference_value"],
                            "desc": pref["description"]
                        })
                    
                    self.logger.info("Default preferences seeded successfully")
                else:
                    self.logger.info("Default preferences already exist")
                
        except Exception as e:
            self.logger.error(f"Failed to seed default data: {e}")
            raise
    
    async def cleanup_old_data(self, days_old: int = 30):
        """Cleanup old processing data.
        
        Args:
            days_old: Remove data older than this many days
        """
        try:
            await self.db_manager.initialize()
            
            async with self.db_manager.get_session() as session:
                # Remove old completed sessions
                result = await session.execute(text("""
                    DELETE FROM processing_sessions 
                    WHERE status = 'completed' 
                    AND end_time < NOW() - INTERVAL '%s days'
                """), (days_old,))
                
                deleted_sessions = result.rowcount
                
                # Remove old processing logs
                result = await session.execute(text("""
                    DELETE FROM processing_logs 
                    WHERE created_at < NOW() - INTERVAL '%s days'
                """), (days_old,))
                
                deleted_logs = result.rowcount
                
                # Remove old system metrics
                result = await session.execute(text("""
                    DELETE FROM system_metrics 
                    WHERE recorded_at < NOW() - INTERVAL '%s days'
                """), (days_old,))
                
                deleted_metrics = result.rowcount
                
                self.logger.info(f"Cleanup completed: {deleted_sessions} sessions, {deleted_logs} logs, {deleted_metrics} metrics removed")
                
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            raise

async def run_migrations():
    """Run database migrations."""
    migrator = DatabaseMigrator()
    
    try:
        logger.info("Starting database migration...")
        await migrator.migrate_to_latest()
        await migrator.seed_default_data()
        logger.info("Database migration completed successfully")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        await migrator.db_manager.cleanup()

if __name__ == "__main__":
    # Run migrations when script is executed directly
    asyncio.run(run_migrations())