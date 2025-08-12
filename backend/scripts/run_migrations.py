"""Database migration runner script."""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add backend and repo root to path
backend_path = Path(__file__).parent.parent
repo_root = backend_path.parent
sys.path.insert(0, str(backend_path))

from alembic.config import Config
from alembic import command
from database.init_db import DatabaseInitializer

logger = logging.getLogger(__name__)


def _get_database_url() -> str:
    """Load and normalize DATABASE_URL from environment."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set.")
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace(
            "postgresql://", "postgresql+asyncpg://", 1
        )
    return database_url

async def run_migrations():
    """Run Alembic database migrations."""
    try:
        # Get the alembic.ini path
        alembic_cfg_path = repo_root / "infra" / "alembic.ini"
        
        if not alembic_cfg_path.exists():
            raise FileNotFoundError(f"Alembic config not found: {alembic_cfg_path}")
        
        # Create Alembic config
        alembic_cfg = Config(str(alembic_cfg_path))
        
        # Set the script location
        alembic_cfg.set_main_option("script_location", str(repo_root / "infra" / "migrations"))
        
        # Set database URL from environment
        database_url = _get_database_url()
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)
        
        logger.info("Running database migrations...")
        
        # Run migrations
        command.upgrade(alembic_cfg, "head")
        
        logger.info("Database migrations completed successfully")
        
        # Initialize database with default data
        logger.info("Initializing database with default data...")
        initializer = DatabaseInitializer(database_url)
        await initializer.initialize_database(create_tables=False, create_admin=True)
        await initializer.cleanup()
        
        logger.info("Database initialization completed")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

async def create_migration(message: str):
    """Create a new Alembic migration.
    
    Args:
        message: Migration message
    """
    try:
        # Get the alembic.ini path
        alembic_cfg_path = repo_root / "infra" / "alembic.ini"
        
        if not alembic_cfg_path.exists():
            raise FileNotFoundError(f"Alembic config not found: {alembic_cfg_path}")
        
        # Create Alembic config
        alembic_cfg = Config(str(alembic_cfg_path))
        
        # Set the script location
        alembic_cfg.set_main_option("script_location", str(repo_root / "infra" / "migrations"))
        
        # Set database URL from environment
        database_url = _get_database_url()
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)
        
        logger.info(f"Creating migration: {message}")
        
        # Create migration
        command.revision(alembic_cfg, message=message, autogenerate=True)
        
        logger.info("Migration created successfully")
        
    except Exception as e:
        logger.error(f"Migration creation failed: {e}")
        raise

async def check_migration_status():
    """Check current migration status."""
    try:
        # Get the alembic.ini path
        alembic_cfg_path = repo_root / "infra" / "alembic.ini"
        
        if not alembic_cfg_path.exists():
            raise FileNotFoundError(f"Alembic config not found: {alembic_cfg_path}")
        
        # Create Alembic config
        alembic_cfg = Config(str(alembic_cfg_path))
        
        # Set the script location
        alembic_cfg.set_main_option("script_location", str(repo_root / "infra" / "migrations"))
        
        # Set database URL from environment
        database_url = _get_database_url()
        alembic_cfg.set_main_option("sqlalchemy.url", database_url)
        
        logger.info("Checking migration status...")
        
        # Show current revision
        command.current(alembic_cfg)
        
        # Show migration history
        command.history(alembic_cfg)
        
    except Exception as e:
        logger.error(f"Migration status check failed: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_migrations.py migrate    # Run migrations")
        print("  python run_migrations.py create 'message'  # Create new migration")
        print("  python run_migrations.py status     # Check migration status")
        sys.exit(1)
    
    command_arg = sys.argv[1]
    
    if command_arg == "migrate":
        asyncio.run(run_migrations())
    elif command_arg == "create":
        if len(sys.argv) < 3:
            print("Please provide a migration message")
            sys.exit(1)
        message = sys.argv[2]
        asyncio.run(create_migration(message))
    elif command_arg == "status":
        asyncio.run(check_migration_status())
    else:
        print(f"Unknown command: {command_arg}")
        sys.exit(1)