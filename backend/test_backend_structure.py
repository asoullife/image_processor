"""Test script to verify backend structure and dependencies."""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Add backend to Python path
backend_root = Path(__file__).parent
sys.path.insert(0, str(backend_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_imports():
    """Test that all modules can be imported."""
    try:
        logger.info("Testing imports...")
        
        # Test configuration
        from config.config_loader import load_config, AppConfig
        logger.info("✅ Configuration modules imported")
        
        # Test database
        from database.connection import DatabaseManager
        from database.models import Project, ProcessingSession
        logger.info("✅ Database modules imported")
        
        # Test API
        from api.main import create_app
        from api.dependencies import get_config
        from api.schemas import ProjectCreate, ProjectResponse
        logger.info("✅ API modules imported")
        
        # Test services
        from core.services import ProjectService, SessionService, AnalysisService
        logger.info("✅ Service modules imported")
        
        # Test analyzers
        from analyzers.analyzer_factory import AnalyzerFactory
        logger.info("✅ Analyzer modules imported")
        
        # Test dependency injection
        from core.dependency_injection import DependencyInjector
        logger.info("✅ Dependency injection modules imported")
        
        logger.info("🎉 All imports successful!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

async def test_configuration():
    """Test configuration loading."""
    try:
        logger.info("Testing configuration...")
        
        from config.config_loader import load_config
        
        # This will use default config if no custom config exists
        config = load_config()
        
        # Verify config structure
        assert hasattr(config, 'processing')
        assert hasattr(config, 'quality')
        assert hasattr(config, 'similarity')
        assert hasattr(config, 'compliance')
        
        logger.info("✅ Configuration loaded and validated")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

async def test_dependency_injection():
    """Test dependency injection system."""
    try:
        logger.info("Testing dependency injection...")
        
        from core.dependency_injection import DependencyInjector
        
        # Create injector
        injector = DependencyInjector()
        
        # Test initialization (without database connection)
        try:
            await injector.initialize()
            logger.info("✅ Dependency injection initialized")
        except Exception as e:
            logger.warning(f"⚠️  Dependency injection failed (expected without database): {e}")
        
        # Test config loading
        config = injector.get_config()
        assert config is not None
        logger.info("✅ Configuration retrieved from injector")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dependency injection test failed: {e}")
        return False

async def test_analyzer_factory():
    """Test analyzer factory."""
    try:
        logger.info("Testing analyzer factory...")
        
        from config.config_loader import load_config
        from analyzers.analyzer_factory import AnalyzerFactory
        
        config = load_config()
        factory = AnalyzerFactory(config)
        
        # Test analyzer creation
        quality_analyzer = factory.get_quality_analyzer()
        defect_detector = factory.get_defect_detector()
        similarity_finder = factory.get_similarity_finder()
        compliance_checker = factory.get_compliance_checker()
        
        logger.info("✅ All analyzers created successfully")
        
        # Test factory status
        status = factory.get_analyzer_status()
        assert status['factory_ready'] is True
        logger.info("✅ Analyzer factory status verified")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Analyzer factory test failed: {e}")
        return False

async def test_api_creation():
    """Test FastAPI app creation."""
    try:
        logger.info("Testing FastAPI app creation...")
        
        from api.main import create_app
        from config.config_loader import load_config
        
        config = load_config()
        app = create_app(config)
        
        assert app is not None
        assert hasattr(app, 'routes')
        
        logger.info("✅ FastAPI app created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ FastAPI app creation failed: {e}")
        return False

async def run_all_tests():
    """Run all tests."""
    logger.info("🧪 Starting backend structure tests...")
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Dependency Injection Test", test_dependency_injection),
        ("Analyzer Factory Test", test_analyzer_factory),
        ("API Creation Test", test_api_creation),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Backend structure is ready.")
    else:
        logger.warning("⚠️  Some tests failed. Check the logs above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)