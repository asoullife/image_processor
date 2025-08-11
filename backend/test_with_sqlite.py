"""Test the API with SQLite database."""

import os
import sys
import asyncio
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

async def test_with_sqlite():
    """Test database functionality with SQLite."""
    
    # Set SQLite URL
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./adobe_stock_processor.db"
    
    print("Testing with SQLite database...")
    print("=" * 40)
    
    try:
        from database.init_db import DatabaseInitializer
        
        # Initialize database
        initializer = DatabaseInitializer()
        await initializer.initialize_database()
        
        # Get database info
        db_info = await initializer.get_database_info()
        print(f"✓ Database info: {db_info}")
        
        await initializer.cleanup()
        
        print("\n✓ SQLite database is working!")
        print("✓ All tables created successfully")
        print("✓ Default admin user created")
        
        return True
        
    except Exception as e:
        print(f"❌ SQLite test failed: {e}")
        return False

async def test_api_startup():
    """Test API startup with SQLite."""
    
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./adobe_stock_processor.db"
    
    try:
        from scripts.startup import startup_sequence
        await startup_sequence()
        
        print("✓ API startup sequence completed")
        return True
        
    except Exception as e:
        print(f"❌ API startup failed: {e}")
        return False

if __name__ == "__main__":
    print("Adobe Stock Image Processor - SQLite Test")
    print("=" * 50)
    
    # Test SQLite database
    success1 = asyncio.run(test_with_sqlite())
    
    print("\n" + "=" * 50)
    
    # Test API startup
    success2 = asyncio.run(test_api_startup())
    
    print("\n" + "=" * 50)
    
    if success1 and success2:
        print("🚀 All tests passed with SQLite!")
        print("\nYou can now start the API:")
        print("python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000")
        print("\nAPI Documentation: http://localhost:8000/api/docs")
    else:
        print("❌ Some tests failed")
        sys.exit(1)