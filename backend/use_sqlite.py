"""Switch to SQLite for development testing."""

import os
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

def setup_sqlite():
    """Setup SQLite database for development."""
    
    # Install aiosqlite if not available
    try:
        import aiosqlite
        print("✓ aiosqlite is available")
    except ImportError:
        print("Installing aiosqlite...")
        os.system("pip install --user aiosqlite")
    
    # Set environment variable
    os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./adobe_stock_processor.db"
    print("✓ Database URL set to SQLite")
    
    # Test database connection
    try:
        from database.init_db import DatabaseInitializer
        import asyncio
        
        async def test_sqlite():
            initializer = DatabaseInitializer("sqlite+aiosqlite:///./adobe_stock_processor.db")
            await initializer.initialize_database()
            
            # Verify schema
            schema_valid = await initializer.verify_database_schema()
            await initializer.cleanup()
            
            return schema_valid
        
        result = asyncio.run(test_sqlite())
        
        if result:
            print("✓ SQLite database initialized successfully")
            print("✓ Database schema verified")
            print("\nDatabase file: adobe_stock_processor.db")
            print("You can now run the API with: python -m uvicorn api.main:app --reload")
            return True
        else:
            print("❌ Database schema verification failed")
            return False
            
    except Exception as e:
        print(f"❌ SQLite setup failed: {e}")
        return False

if __name__ == "__main__":
    print("Setting up SQLite database for development...")
    print("=" * 50)
    
    success = setup_sqlite()
    
    if success:
        print("\n🚀 SQLite database is ready!")
        print("\nNext steps:")
        print("1. Run API: python -m uvicorn api.main:app --reload")
        print("2. Open docs: http://localhost:8000/api/docs")
        print("3. Test health: http://localhost:8000/api/health/")
    else:
        print("\n❌ SQLite setup failed")
        sys.exit(1)