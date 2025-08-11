"""FastAPI server startup script."""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add backend to Python path
backend_root = Path(__file__).parent
sys.path.insert(0, str(backend_root))

from api.main import create_app
from config.config_loader import load_config
from database.migrations import run_migrations
from core.dependency_injection import initialize_dependencies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def startup_sequence():
    """Run startup sequence for the backend."""
    try:
        logger.info("Starting Adobe Stock Image Processor Backend...")
        
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Run database migrations
        logger.info("Running database migrations...")
        await run_migrations()
        logger.info("Database migrations completed")
        
        # Initialize dependencies
        logger.info("Initializing dependencies...")
        await initialize_dependencies()
        logger.info("Dependencies initialized")
        
        # Create FastAPI app
        logger.info("Creating FastAPI application...")
        app = create_app(config)
        logger.info("FastAPI application created")
        
        return app
        
    except Exception as e:
        logger.error(f"Startup sequence failed: {e}")
        raise

def main():
    """Main entry point for the backend server."""
    import uvicorn
    
    try:
        # Run startup sequence
        app = asyncio.run(startup_sequence())
        
        # Start the server
        logger.info("Starting FastAPI server...")
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            reload=False,  # Disable reload for production
            log_level="info",
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()