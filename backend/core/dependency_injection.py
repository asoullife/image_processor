"""Dependency injection container for the application."""

import logging
from typing import Dict, Any, Optional, TypeVar, Type
from dataclasses import dataclass

from ..config.config_loader import AppConfig, load_config
from ..database.connection import DatabaseManager
from ..analyzers.analyzer_factory import AnalyzerFactory
from .services import ProjectService, SessionService, AnalysisService

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class ServiceContainer:
    """Container for managing service dependencies."""
    
    # Core dependencies
    config: Optional[AppConfig] = None
    database_manager: Optional[DatabaseManager] = None
    analyzer_factory: Optional[AnalyzerFactory] = None
    
    # Services
    project_service: Optional[ProjectService] = None
    session_service: Optional[SessionService] = None
    analysis_service: Optional[AnalysisService] = None
    
    # Initialization flags
    _initialized: bool = False
    _services_initialized: bool = False

class DependencyInjector:
    """Dependency injection container for managing application dependencies."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize dependency injector.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.container = ServiceContainer()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._singletons: Dict[Type, Any] = {}
    
    async def initialize(self):
        """Initialize all core dependencies."""
        if self.container._initialized:
            return
        
        try:
            # Load configuration
            self.container.config = load_config(self.config_path)
            self.logger.info("Configuration loaded successfully")
            
            # Initialize database manager
            self.container.database_manager = DatabaseManager()
            await self.container.database_manager.initialize()
            self.logger.info("Database manager initialized")
            
            # Initialize analyzer factory
            self.container.analyzer_factory = AnalyzerFactory(self.container.config)
            self.logger.info("Analyzer factory initialized")
            
            self.container._initialized = True
            self.logger.info("Core dependencies initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize dependencies: {e}")
            raise
    
    async def initialize_services(self):
        """Initialize all services."""
        if not self.container._initialized:
            await self.initialize()
        
        if self.container._services_initialized:
            return
        
        try:
            # Initialize services
            self.container.project_service = ProjectService(
                self.container.database_manager,
                self.container.config
            )
            
            self.container.session_service = SessionService(
                self.container.database_manager,
                self.container.config
            )
            
            self.container.analysis_service = AnalysisService(
                self.container.config
            )
            
            self.container._services_initialized = True
            self.logger.info("Services initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize services: {e}")
            raise
    
    def get_config(self) -> AppConfig:
        """Get application configuration.
        
        Returns:
            Application configuration
        """
        if not self.container.config:
            raise RuntimeError("Configuration not initialized. Call initialize() first.")
        return self.container.config
    
    def get_database_manager(self) -> DatabaseManager:
        """Get database manager.
        
        Returns:
            Database manager instance
        """
        if not self.container.database_manager:
            raise RuntimeError("Database manager not initialized. Call initialize() first.")
        return self.container.database_manager
    
    def get_analyzer_factory(self) -> AnalyzerFactory:
        """Get analyzer factory.
        
        Returns:
            Analyzer factory instance
        """
        if not self.container.analyzer_factory:
            raise RuntimeError("Analyzer factory not initialized. Call initialize() first.")
        return self.container.analyzer_factory
    
    async def get_project_service(self) -> ProjectService:
        """Get project service.
        
        Returns:
            Project service instance
        """
        if not self.container.project_service:
            await self.initialize_services()
        return self.container.project_service
    
    async def get_session_service(self) -> SessionService:
        """Get session service.
        
        Returns:
            Session service instance
        """
        if not self.container.session_service:
            await self.initialize_services()
        return self.container.session_service
    
    async def get_analysis_service(self) -> AnalysisService:
        """Get analysis service.
        
        Returns:
            Analysis service instance
        """
        if not self.container.analysis_service:
            await self.initialize_services()
        return self.container.analysis_service
    
    def register_singleton(self, interface: Type[T], implementation: T):
        """Register a singleton instance.
        
        Args:
            interface: Interface type
            implementation: Implementation instance
        """
        self._singletons[interface] = implementation
        self.logger.info(f"Registered singleton: {interface.__name__}")
    
    def get_singleton(self, interface: Type[T]) -> T:
        """Get singleton instance.
        
        Args:
            interface: Interface type
            
        Returns:
            Singleton instance
        """
        if interface not in self._singletons:
            raise RuntimeError(f"Singleton not registered: {interface.__name__}")
        return self._singletons[interface]
    
    async def cleanup(self):
        """Cleanup all resources."""
        try:
            if self.container.database_manager:
                await self.container.database_manager.cleanup()
                self.logger.info("Database manager cleaned up")
            
            # Reset container
            self.container = ServiceContainer()
            self._singletons.clear()
            
            self.logger.info("Dependency injector cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def get_container_status(self) -> Dict[str, Any]:
        """Get container status information.
        
        Returns:
            Container status
        """
        return {
            "initialized": self.container._initialized,
            "services_initialized": self.container._services_initialized,
            "config_loaded": self.container.config is not None,
            "database_manager_ready": self.container.database_manager is not None,
            "analyzer_factory_ready": self.container.analyzer_factory is not None,
            "project_service_ready": self.container.project_service is not None,
            "session_service_ready": self.container.session_service is not None,
            "analysis_service_ready": self.container.analysis_service is not None,
            "registered_singletons": list(self._singletons.keys())
        }

# Global dependency injector instance
_dependency_injector: Optional[DependencyInjector] = None

def get_dependency_injector(config_path: Optional[str] = None) -> DependencyInjector:
    """Get global dependency injector instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dependency injector instance
    """
    global _dependency_injector
    
    if _dependency_injector is None:
        _dependency_injector = DependencyInjector(config_path)
    
    return _dependency_injector

async def initialize_dependencies(config_path: Optional[str] = None):
    """Initialize global dependencies.
    
    Args:
        config_path: Path to configuration file
    """
    injector = get_dependency_injector(config_path)
    await injector.initialize()
    await injector.initialize_services()

async def cleanup_dependencies():
    """Cleanup global dependencies."""
    global _dependency_injector
    
    if _dependency_injector:
        await _dependency_injector.cleanup()
        _dependency_injector = None