#!/usr/bin/env python3
"""
Adobe Stock Image Processor - Main Entry Point
Comprehensive image analysis and processing system with AI enhancement and CLI support
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def main():
    """Main application entry point with CLI support"""
    try:
        # Check if CLI arguments are provided
        if len(sys.argv) > 1:
            # Use CLI interface
            from cli.cli_manager import main as cli_main
            return cli_main()
        else:
            # Start web interface
            return asyncio.run(start_web_application())
            
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
        return 0
    except Exception as e:
        print(f"Application error: {e}")
        return 1

async def start_web_application():
    """Start the web application"""
    from api.main import app
    from database.init_db import init_database
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize database
        logger.info("Initializing database...")
        await init_database()
        
        # Start the application
        logger.info("Starting Adobe Stock Image Processor...")
        logger.info("Backend API available at: http://localhost:8000")
        logger.info("Frontend available at: http://localhost:3000")
        logger.info("API Documentation: http://localhost:8000/docs")
        logger.info("Use Ctrl+C to stop the application")
        logger.info("For CLI help, run: python main.py -h")
        
        # Keep the application running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())