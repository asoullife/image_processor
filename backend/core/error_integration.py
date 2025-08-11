"""
Error handling integration utilities for Adobe Stock Image Processor.

This module provides integration utilities to retrofit existing components
with comprehensive error handling capabilities.
"""

import logging
import functools
from typing import Any, Dict, Optional, Callable, Type
from contextlib import contextmanager

from .error_handler import (
    ErrorHandler, ErrorCategory, ErrorSeverity, get_error_handler,
    handle_errors as error_decorator
)


class ErrorIntegration:
    """Integration utilities for adding error handling to existing components."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """Initialize error integration.
        
        Args:
            error_handler: Error handler instance. Uses global if None.
        """
        self.error_handler = error_handler or get_error_handler()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def wrap_analyzer(self, analyzer_class: Type) -> Type:
        """Wrap an analyzer class with error handling.
        
        Args:
            analyzer_class: Analyzer class to wrap.
            
        Returns:
            Wrapped analyzer class.
        """
        error_handler = self.error_handler
        
        class WrappedAnalyzer(analyzer_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._error_handler = error_handler
            
            @error_decorator(
                category=ErrorCategory.PROCESSING,
                retry=True,
                graceful_degradation=True,
                default_return=None
            )
            def analyze(self, *args, **kwargs):
                return super().analyze(*args, **kwargs)
        
        WrappedAnalyzer.__name__ = f"ErrorHandled{analyzer_class.__name__}"
        return WrappedAnalyzer
    
    def wrap_processor(self, processor_class: Type) -> Type:
        """Wrap a processor class with error handling.
        
        Args:
            processor_class: Processor class to wrap.
            
        Returns:
            Wrapped processor class.
        """
        error_handler = self.error_handler
        
        class WrappedProcessor(processor_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._error_handler = error_handler
            
            @error_decorator(
                category=ErrorCategory.PROCESSING,
                retry=True,
                graceful_degradation=False
            )
            def process(self, *args, **kwargs):
                return super().process(*args, **kwargs)
            
            @error_decorator(
                category=ErrorCategory.MEMORY,
                retry=False,
                graceful_degradation=True,
                default_return=True
            )
            def cleanup_memory(self, *args, **kwargs):
                if hasattr(super(), 'cleanup_memory'):
                    return super().cleanup_memory(*args, **kwargs)
                return True
        
        WrappedProcessor.__name__ = f"ErrorHandled{processor_class.__name__}"
        return WrappedProcessor
    
    def wrap_file_operations(self, file_manager_class: Type) -> Type:
        """Wrap file manager class with error handling.
        
        Args:
            file_manager_class: File manager class to wrap.
            
        Returns:
            Wrapped file manager class.
        """
        error_handler = self.error_handler
        
        class WrappedFileManager(file_manager_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._error_handler = error_handler
            
            @error_decorator(
                category=ErrorCategory.FILE_SYSTEM,
                retry=True,
                graceful_degradation=False
            )
            def scan_images(self, *args, **kwargs):
                return super().scan_images(*args, **kwargs)
            
            @error_decorator(
                category=ErrorCategory.FILE_SYSTEM,
                retry=True,
                graceful_degradation=True,
                default_return=False
            )
            def copy_with_verification(self, *args, **kwargs):
                return super().copy_with_verification(*args, **kwargs)
            
            @error_decorator(
                category=ErrorCategory.FILE_SYSTEM,
                retry=False,
                graceful_degradation=True,
                default_return=True
            )
            def organize_output(self, *args, **kwargs):
                return super().organize_output(*args, **kwargs)
        
        WrappedFileManager.__name__ = f"ErrorHandled{file_manager_class.__name__}"
        return WrappedFileManager


@contextmanager
def error_context(component: str, operation: str, **context_data):
    """Context manager for error handling with automatic context collection.
    
    Args:
        component: Component name.
        operation: Operation being performed.
        **context_data: Additional context data.
    """
    error_handler = get_error_handler()
    context = {
        'component': component,
        'operation': operation,
        **context_data
    }
    
    try:
        yield context
    except Exception as e:
        error_handler.handle_error(e, context)
        raise


def safe_execute(func: Callable, 
                *args, 
                category: ErrorCategory = ErrorCategory.PROCESSING,
                default_return: Any = None,
                log_errors: bool = True,
                **kwargs) -> Any:
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute.
        *args: Function arguments.
        category: Error category.
        default_return: Default return value on error.
        log_errors: Whether to log errors.
        **kwargs: Function keyword arguments.
        
    Returns:
        Function result or default_return on error.
    """
    error_handler = get_error_handler()
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        context = {
            'function': func.__name__,
            'module': getattr(func, '__module__', 'unknown'),
            'args_count': len(args),
            'kwargs_keys': list(kwargs.keys())
        }
        
        if log_errors:
            error_handler.handle_error(e, context, category)
        
        return default_return


def batch_safe_execute(functions_and_args: list,
                      category: ErrorCategory = ErrorCategory.PROCESSING,
                      continue_on_error: bool = True,
                      collect_errors: bool = True) -> tuple:
    """Safely execute a batch of functions with error handling.
    
    Args:
        functions_and_args: List of (function, args, kwargs) tuples.
        category: Error category.
        continue_on_error: Whether to continue on individual errors.
        collect_errors: Whether to collect and return errors.
        
    Returns:
        Tuple of (results, errors) lists.
    """
    error_handler = get_error_handler()
    results = []
    errors = []
    
    for i, (func, args, kwargs) in enumerate(functions_and_args):
        try:
            result = func(*args, **kwargs)
            results.append(result)
        except Exception as e:
            context = {
                'batch_index': i,
                'function': func.__name__,
                'module': getattr(func, '__module__', 'unknown')
            }
            
            error_info = error_handler.handle_error(e, context, category)
            
            if collect_errors:
                errors.append(error_info)
            
            if not continue_on_error:
                break
            
            results.append(None)  # Placeholder for failed operation
    
    return results, errors


class GracefulDegradation:
    """Utilities for graceful degradation on errors."""
    
    def __init__(self, error_handler: Optional[ErrorHandler] = None):
        """Initialize graceful degradation utilities.
        
        Args:
            error_handler: Error handler instance.
        """
        self.error_handler = error_handler or get_error_handler()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def degrade_quality_analysis(self, image_path: str) -> dict:
        """Provide degraded quality analysis when full analysis fails.
        
        Args:
            image_path: Path to image file.
            
        Returns:
            Basic quality result.
        """
        try:
            import os
            from PIL import Image
            
            # Basic file checks
            if not os.path.exists(image_path):
                return self._create_failed_result("File not found")
            
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                return self._create_failed_result("Empty file")
            
            # Try to get basic image info
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    format_name = img.format
                
                return {
                    'sharpness_score': 0.5,  # Neutral score
                    'noise_level': 0.5,
                    'exposure_score': 0.5,
                    'color_balance_score': 0.5,
                    'resolution': (width, height),
                    'file_size': file_size,
                    'overall_score': 0.5,
                    'passed': True,  # Allow through with degraded analysis
                    'degraded': True,
                    'degradation_reason': 'Full analysis failed, using basic checks'
                }
            except Exception:
                return self._create_failed_result("Cannot read image")
                
        except Exception as e:
            self.logger.warning(f"Degraded quality analysis failed: {e}")
            return self._create_failed_result("All analysis methods failed")
    
    def degrade_defect_detection(self, image_path: str) -> dict:
        """Provide degraded defect detection when full detection fails.
        
        Args:
            image_path: Path to image file.
            
        Returns:
            Basic defect result.
        """
        return {
            'detected_objects': [],
            'anomaly_score': 0.0,
            'defect_count': 0,
            'defect_types': [],
            'confidence_scores': [],
            'passed': True,  # Allow through with degraded analysis
            'degraded': True,
            'degradation_reason': 'Full defect detection failed, assuming no defects'
        }
    
    def degrade_similarity_detection(self, image_paths: list) -> dict:
        """Provide degraded similarity detection when full detection fails.
        
        Args:
            image_paths: List of image paths.
            
        Returns:
            Basic similarity result.
        """
        # Assign each image to its own group (no similarity detected)
        groups = {i: [path] for i, path in enumerate(image_paths)}
        
        return {
            'similarity_groups': groups,
            'total_groups': len(image_paths),
            'duplicates_found': 0,
            'degraded': True,
            'degradation_reason': 'Full similarity detection failed, treating all as unique'
        }
    
    def degrade_compliance_check(self, image_path: str) -> dict:
        """Provide degraded compliance check when full check fails.
        
        Args:
            image_path: Path to image file.
            
        Returns:
            Basic compliance result.
        """
        return {
            'logo_detections': [],
            'privacy_violations': [],
            'metadata_issues': [],
            'keyword_relevance': 0.5,
            'overall_compliance': True,  # Allow through with degraded analysis
            'degraded': True,
            'degradation_reason': 'Full compliance check failed, assuming compliant'
        }
    
    def _create_failed_result(self, reason: str) -> dict:
        """Create a failed quality result.
        
        Args:
            reason: Failure reason.
            
        Returns:
            Failed result dictionary.
        """
        return {
            'sharpness_score': 0.0,
            'noise_level': 1.0,
            'exposure_score': 0.0,
            'color_balance_score': 0.0,
            'resolution': (0, 0),
            'file_size': 0,
            'overall_score': 0.0,
            'passed': False,
            'degraded': True,
            'degradation_reason': reason
        }


# Global instances
_error_integration = ErrorIntegration()
_graceful_degradation = GracefulDegradation()


def get_error_integration() -> ErrorIntegration:
    """Get global error integration instance."""
    return _error_integration


def get_graceful_degradation() -> GracefulDegradation:
    """Get global graceful degradation instance."""
    return _graceful_degradation


# Convenience functions
def wrap_with_error_handling(component_class: Type, component_type: str = 'processor') -> Type:
    """Wrap a component class with appropriate error handling.
    
    Args:
        component_class: Class to wrap.
        component_type: Type of component ('analyzer', 'processor', 'file_manager').
        
    Returns:
        Wrapped class.
    """
    integration = get_error_integration()
    
    if component_type == 'analyzer':
        return integration.wrap_analyzer(component_class)
    elif component_type == 'processor':
        return integration.wrap_processor(component_class)
    elif component_type == 'file_manager':
        return integration.wrap_file_operations(component_class)
    else:
        raise ValueError(f"Unknown component type: {component_type}")