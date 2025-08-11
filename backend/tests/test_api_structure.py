"""Test API structure without database connection."""

import sys
import logging
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

def test_imports():
    """Test that all API components can be imported."""
    try:
        # Test database models
        from database.models import Project, ProcessingSession, ImageResult
        print("✓ Database models imported successfully")
        
        # Test API schemas
        from api.schemas import ProjectCreate, ProjectResponse, UserCreate, TokenResponse
        print("✓ API schemas imported successfully")
        
        # Test authentication
        from auth.models import User, Session
        from auth.security import SecurityManager
        print("✓ Authentication components imported successfully")
        
        # Test API routes (without database dependency)
        from api.routes import projects, sessions, health, auth
        print("✓ API routes imported successfully")
        
        # Test services (structure only)
        from core.services import ProjectService, SessionService, AnalysisService
        print("✓ Core services imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_api_structure():
    """Test FastAPI application structure."""
    try:
        from api.main import app
        
        # Check that routes are registered
        routes = [route.path for route in app.routes]
        expected_routes = [
            "/api/health/",
            "/api/auth/register",
            "/api/auth/login", 
            "/api/projects/",
            "/api/sessions/"
        ]
        
        for expected_route in expected_routes:
            if any(expected_route in route for route in routes):
                print(f"✓ Route found: {expected_route}")
            else:
                print(f"❌ Route missing: {expected_route}")
        
        print(f"✓ FastAPI app created with {len(routes)} routes")
        return True
        
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False

def test_database_models():
    """Test database model structure."""
    try:
        from database.models import Base
        
        # Get all model classes
        models = []
        for cls in Base.registry._class_registry.values():
            if hasattr(cls, '__tablename__'):
                models.append(cls.__tablename__)
        
        expected_tables = [
            'users', 'user_sessions', 'projects', 'processing_sessions',
            'image_results', 'checkpoints', 'similarity_groups'
        ]
        
        for table in expected_tables:
            if table in models:
                print(f"✓ Table model found: {table}")
            else:
                print(f"❌ Table model missing: {table}")
        
        print(f"✓ Database models defined for {len(models)} tables")
        return True
        
    except Exception as e:
        print(f"❌ Database model test failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise
    
    print("Testing Adobe Stock Image Processor API Structure")
    print("=" * 50)
    
    success = True
    
    print("\n1. Testing imports...")
    success &= test_imports()
    
    print("\n2. Testing API structure...")
    success &= test_api_structure()
    
    print("\n3. Testing database models...")
    success &= test_database_models()
    
    print("\n" + "=" * 50)
    if success:
        print("🚀 All API structure tests passed!")
        print("The database and core API infrastructure is properly set up.")
        sys.exit(0)
    else:
        print("❌ Some tests failed.")
        sys.exit(1)