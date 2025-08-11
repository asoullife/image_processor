"""FastAPI main application entry point."""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from typing import Dict, Any
import socketio

from .routes import projects, sessions, analysis, health, auth, review, thumbnails, recovery, settings, reports, mock_reports
from .dependencies import get_database, get_config
from ..database.connection import DatabaseManager
from ..config.config_loader import AppConfig
from ..websocket.socketio_manager import sio, socketio_manager
from ..websocket.redis_adapter import redis_adapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Adobe Stock Image Processor API",
    description="""
    RESTful API for Adobe Stock image processing and analysis.
    
    ## Features
    
    * **Authentication**: User registration, login, and session management
    * **Project Management**: Create and manage image processing projects
    * **Session Management**: Track processing sessions with real-time progress
    * **Image Analysis**: Quality, defect, similarity, and compliance checking
    * **Human Review**: Web-based review system for rejected images
    * **Real-time Updates**: WebSocket support for live progress monitoring
    
    ## Authentication
    
    The API uses session-based authentication with HTTP-only cookies.
    Register a new account or login with existing credentials to access protected endpoints.
    
    ## Getting Started
    
    1. Register a new user account at `/api/auth/register`
    2. Login to get a session token at `/api/auth/login`
    3. Create a new project at `/api/projects/`
    4. Start processing with `/api/projects/{project_id}/start`
    5. Monitor progress in real-time via WebSocket or polling
    
    ## Database Schema
    
    The API uses PostgreSQL with the following main entities:
    - **Users**: Authentication and user management
    - **Projects**: Image processing project configuration
    - **Sessions**: Individual processing runs with progress tracking
    - **Results**: Detailed analysis results for each processed image
    """,
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    contact={
        "name": "Adobe Stock Image Processor",
        "url": "https://github.com/your-repo/adobe-stock-processor",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "http://127.0.0.1:8000", 
            "description": "Local development server"
        }
    ]
)

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Socket.IO ASGI app
socket_app = socketio.ASGIApp(sio, app, socketio_path='/socket.io')

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )

# Include API routes
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(review.router, prefix="/api/review", tags=["human-review"])
app.include_router(thumbnails.router, tags=["thumbnails"])
app.include_router(recovery.router, prefix="/api/recovery", tags=["recovery"])
app.include_router(settings.router, tags=["settings"])
app.include_router(reports.router, tags=["reports"])
app.include_router(mock_reports.router, tags=["mock-reports"])

# Import and include WebSocket routes
from .routes import websocket
app.include_router(websocket.router, prefix="/api/websocket", tags=["websocket"])

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting Adobe Stock Image Processor API")
    
    # Initialize Redis adapter for Socket.IO
    try:
        adapter = await redis_adapter.initialize()
        if adapter:
            sio.manager = adapter
            logger.info("Socket.IO Redis adapter initialized")
        else:
            logger.info("Socket.IO running in single-process mode")
    except Exception as e:
        logger.warning(f"Redis adapter initialization failed: {e}")
    
    # Initialize database connection and run startup sequence
    try:
        # Import startup sequence
        import sys
        from pathlib import Path
        backend_path = Path(__file__).parent.parent
        sys.path.insert(0, str(backend_path))
        
        from scripts.startup import startup_sequence
        await startup_sequence()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        # Don't raise here to allow the API to start even if some initialization fails
        logger.warning("API starting with limited functionality")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Adobe Stock Image Processor API")
    
    # Cleanup Redis adapter
    try:
        await redis_adapter.cleanup()
        logger.info("Redis adapter cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up Redis adapter: {e}")
    
    # Cleanup database connections
    try:
        db_manager = DatabaseManager()
        await db_manager.cleanup()
        logger.info("Database connections cleaned up")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Adobe Stock Image Processor API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }

def create_app(config: AppConfig = None) -> socketio.ASGIApp:
    """Create and configure FastAPI application with Socket.IO.
    
    Args:
        config: Application configuration
        
    Returns:
        Configured Socket.IO ASGI application
    """
    if config:
        # Store config in app state for dependency injection
        app.state.config = config
    
    return socket_app

if __name__ == "__main__":
    # Development server with Socket.IO support
    uvicorn.run(
        "backend.api.main:socket_app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )