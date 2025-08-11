"""
Comprehensive Error Handling System for Adobe Stock Image Processor

This module provides centralized error handling with categorized error types,
retry mechanisms, graceful degradation, and detailed error reporting.
"""

import logging
import traceback
import time
import threading
from typing import Dict, Any, Optional, List, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
import psutil
import os


class ErrorCategory(Enum):
    """Categories of errors that can occur during processing."""
    FILE_SYSTEM = "file_system"
    PROCESSING = "processing"
    MEMORY = "memory"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    VALIDATION = "validation"
    CRITICAL = "critical"
    RECOVERABLE = "recoverable"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorInfo:
    """Detailed information about an error occurrence."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    exception_type: str
    message: str
    context: Dict[str, Any]
    timestamp: datetime
    stack_trace: str
    retry_count: int = 0
    resolved: bool = False
    resolution_method: Optional[str] = None
    memory_usage_mb: Optional[float] = None
    thread_id: Optional[str] = None


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    retry_on_exceptions: List[Type[Exception]] = field(default_factory=list)


@dataclass
class ErrorStats:
    """Statistics about error occurrences."""
    total_errors: int = 0
    errors_by_category: Dict[ErrorCategory, int] = field(default_factory=dict)
    errors_by_severity: Dict[ErrorSeverity, int] = field(default_factory=dict)
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    resolved_errors: int = 0
    unresolved_errors: int = 0
    retry_attempts: int = 0
    successful_retries: int = 0


class ErrorHandler:
    """Centralized error handling system with categorization and recovery."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize error handler with configuration.
        
        Args:
            config: Configuration dictionary for error handling.
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Error tracking
        self._errors: Dict[str, ErrorInfo] = {}
        self._error_stats = ErrorStats()
        self._lock = threading.Lock()
        
        # Retry configurations for different error categories
        self._retry_configs = self._setup_retry_configs()
        
        # Error handlers for different categories
        self._category_handlers = self._setup_category_handlers()
        
        # Recovery strategies
        self._recovery_strategies = self._setup_recovery_strategies()
        
        self.logger.info("ErrorHandler initialized")
    
    def _setup_retry_configs(self) -> Dict[ErrorCategory, RetryConfig]:
        """Setup retry configurations for different error categories."""
        return {
            ErrorCategory.FILE_SYSTEM: RetryConfig(
                max_retries=2,
                base_delay=0.5,
                retry_on_exceptions=[OSError, IOError, PermissionError]
            ),
            ErrorCategory.PROCESSING: RetryConfig(
                max_retries=3,
                base_delay=1.0,
                retry_on_exceptions=[RuntimeError, ValueError]
            ),
            ErrorCategory.MEMORY: RetryConfig(
                max_retries=1,
                base_delay=2.0,
                retry_on_exceptions=[MemoryError]
            ),
            ErrorCategory.NETWORK: RetryConfig(
                max_retries=5,
                base_delay=1.0,
                max_delay=30.0,
                retry_on_exceptions=[ConnectionError, TimeoutError]
            ),
            ErrorCategory.VALIDATION: RetryConfig(
                max_retries=0,  # Don't retry validation errors
                retry_on_exceptions=[]
            ),
            ErrorCategory.CRITICAL: RetryConfig(
                max_retries=0,  # Don't retry critical errors
                retry_on_exceptions=[]
            )
        }
    
    def _setup_category_handlers(self) -> Dict[ErrorCategory, Callable]:
        """Setup handlers for different error categories."""
        return {
            ErrorCategory.FILE_SYSTEM: self._handle_file_system_error,
            ErrorCategory.PROCESSING: self._handle_processing_error,
            ErrorCategory.MEMORY: self._handle_memory_error,
            ErrorCategory.CONFIGURATION: self._handle_configuration_error,
            ErrorCategory.NETWORK: self._handle_network_error,
            ErrorCategory.VALIDATION: self._handle_validation_error,
            ErrorCategory.CRITICAL: self._handle_critical_error,
            ErrorCategory.RECOVERABLE: self._handle_recoverable_error
        }
    
    def _setup_recovery_strategies(self) -> Dict[ErrorCategory, Callable]:
        """Setup recovery strategies for different error categories."""
        return {
            ErrorCategory.MEMORY: self._recover_from_memory_error,
            ErrorCategory.FILE_SYSTEM: self._recover_from_file_error,
            ErrorCategory.PROCESSING: self._recover_from_processing_error,
            ErrorCategory.NETWORK: self._recover_from_network_error
        }
    
    def handle_error(self, 
                    exception: Exception, 
                    context: Dict[str, Any], 
                    category: Optional[ErrorCategory] = None,
                    severity: Optional[ErrorSeverity] = None) -> ErrorInfo:
        """Handle an error with categorization and recovery attempts.
        
        Args:
            exception: The exception that occurred.
            context: Context information about the error.
            category: Error category (auto-detected if None).
            severity: Error severity (auto-detected if None).
            
        Returns:
            ErrorInfo: Detailed error information.
        """
        with self._lock:
            # Generate unique error ID
            error_id = self._generate_error_id(exception, context)
            
            # Auto-detect category and severity if not provided
            if category is None:
                category = self._categorize_error(exception, context)
            if severity is None:
                severity = self._assess_severity(exception, category, context)
            
            # Get memory usage
            memory_usage = self._get_memory_usage()
            
            # Create error info
            error_info = ErrorInfo(
                error_id=error_id,
                category=category,
                severity=severity,
                exception_type=type(exception).__name__,
                message=str(exception),
                context=context.copy(),
                timestamp=datetime.now(),
                stack_trace=traceback.format_exc(),
                memory_usage_mb=memory_usage,
                thread_id=threading.current_thread().name
            )
            
            # Store error info
            self._errors[error_id] = error_info
            
            # Update statistics
            self._update_error_stats(error_info)
            
            # Log the error
            self._log_error(error_info)
            
            # Attempt to handle the error
            handled = self._attempt_error_handling(error_info, exception)
            
            return error_info
    
    def _generate_error_id(self, exception: Exception, context: Dict[str, Any]) -> str:
        """Generate unique error ID."""
        import hashlib
        
        error_data = f"{type(exception).__name__}_{str(exception)}_{context.get('component', 'unknown')}"
        return hashlib.md5(error_data.encode()).hexdigest()[:12]
    
    def _categorize_error(self, exception: Exception, context: Dict[str, Any]) -> ErrorCategory:
        """Automatically categorize an error based on exception type and context."""
        exception_type = type(exception).__name__
        
        # Network errors (check first as they can inherit from OSError)
        if isinstance(exception, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
        
        # Memory errors
        if isinstance(exception, MemoryError):
            return ErrorCategory.MEMORY
        
        # File system errors
        if isinstance(exception, (FileNotFoundError, PermissionError, OSError, IOError)):
            return ErrorCategory.FILE_SYSTEM
        
        # Configuration errors
        if isinstance(exception, (ValueError, TypeError)) and 'config' in context:
            return ErrorCategory.CONFIGURATION
        
        # Validation errors
        if isinstance(exception, (ValueError, TypeError, AssertionError)):
            return ErrorCategory.VALIDATION
        
        # Critical system errors
        if isinstance(exception, (SystemError, SystemExit, KeyboardInterrupt)):
            return ErrorCategory.CRITICAL
        
        # Default to processing error
        return ErrorCategory.PROCESSING
    
    def _assess_severity(self, exception: Exception, category: ErrorCategory, context: Dict[str, Any]) -> ErrorSeverity:
        """Assess error severity based on exception type, category, and context."""
        # Critical errors are always high severity
        if category == ErrorCategory.CRITICAL:
            return ErrorSeverity.CRITICAL
        
        # Memory errors are high severity
        if category == ErrorCategory.MEMORY:
            return ErrorSeverity.HIGH
        
        # System-level file errors are high severity
        if category == ErrorCategory.FILE_SYSTEM and isinstance(exception, PermissionError):
            return ErrorSeverity.HIGH
        
        # Configuration errors are medium severity
        if category == ErrorCategory.CONFIGURATION:
            return ErrorSeverity.MEDIUM
        
        # Network errors depend on context
        if category == ErrorCategory.NETWORK:
            return ErrorSeverity.MEDIUM
        
        # Validation errors are typically low severity
        if category == ErrorCategory.VALIDATION:
            return ErrorSeverity.LOW
        
        # Default to medium severity
        return ErrorSeverity.MEDIUM
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return None
    
    def _update_error_stats(self, error_info: ErrorInfo):
        """Update error statistics."""
        self._error_stats.total_errors += 1
        
        # Update category stats
        if error_info.category not in self._error_stats.errors_by_category:
            self._error_stats.errors_by_category[error_info.category] = 0
        self._error_stats.errors_by_category[error_info.category] += 1
        
        # Update severity stats
        if error_info.severity not in self._error_stats.errors_by_severity:
            self._error_stats.errors_by_severity[error_info.severity] = 0
        self._error_stats.errors_by_severity[error_info.severity] += 1
        
        # Update type stats
        if error_info.exception_type not in self._error_stats.errors_by_type:
            self._error_stats.errors_by_type[error_info.exception_type] = 0
        self._error_stats.errors_by_type[error_info.exception_type] += 1
        
        # Update resolution stats
        if error_info.resolved:
            self._error_stats.resolved_errors += 1
        else:
            self._error_stats.unresolved_errors += 1
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level based on severity."""
        log_message = (
            f"Error {error_info.error_id}: {error_info.exception_type} - {error_info.message}\n"
            f"Category: {error_info.category.value}, Severity: {error_info.severity.value}\n"
            f"Context: {error_info.context}\n"
            f"Memory Usage: {error_info.memory_usage_mb:.1f}MB\n"
            f"Thread: {error_info.thread_id}"
        )
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            self.logger.critical(f"Stack trace:\n{error_info.stack_trace}")
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
            self.logger.debug(f"Stack trace:\n{error_info.stack_trace}")
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _attempt_error_handling(self, error_info: ErrorInfo, exception: Exception) -> bool:
        """Attempt to handle the error using category-specific handlers."""
        try:
            handler = self._category_handlers.get(error_info.category)
            if handler:
                return handler(error_info, exception)
            return False
        except Exception as e:
            self.logger.error(f"Error in error handler: {e}")
            return False 
   
    # Category-specific error handlers
    def _handle_file_system_error(self, error_info: ErrorInfo, exception: Exception) -> bool:
        """Handle file system errors."""
        self.logger.debug(f"Handling file system error: {error_info.error_id}")
        
        # Check if file exists and is accessible
        file_path = error_info.context.get('file_path')
        if file_path and os.path.exists(file_path):
            try:
                # Try to access the file
                with open(file_path, 'rb') as f:
                    f.read(1)  # Try to read one byte
                error_info.resolved = True
                error_info.resolution_method = "file_access_retry"
                return True
            except Exception:
                pass
        
        # Try recovery strategy
        recovery_strategy = self._recovery_strategies.get(ErrorCategory.FILE_SYSTEM)
        if recovery_strategy:
            return recovery_strategy(error_info, exception)
        
        return False
    
    def _handle_processing_error(self, error_info: ErrorInfo, exception: Exception) -> bool:
        """Handle processing errors."""
        self.logger.debug(f"Handling processing error: {error_info.error_id}")
        
        # Check if it's a recoverable processing error
        if isinstance(exception, (ValueError, TypeError)):
            # These are usually data validation issues, not recoverable
            return False
        
        # Try recovery strategy
        recovery_strategy = self._recovery_strategies.get(ErrorCategory.PROCESSING)
        if recovery_strategy:
            return recovery_strategy(error_info, exception)
        
        return False
    
    def _handle_memory_error(self, error_info: ErrorInfo, exception: Exception) -> bool:
        """Handle memory errors."""
        self.logger.warning(f"Handling memory error: {error_info.error_id}")
        
        # Always try memory recovery for memory errors
        recovery_strategy = self._recovery_strategies.get(ErrorCategory.MEMORY)
        if recovery_strategy:
            return recovery_strategy(error_info, exception)
        
        return False
    
    def _handle_configuration_error(self, error_info: ErrorInfo, exception: Exception) -> bool:
        """Handle configuration errors."""
        self.logger.debug(f"Handling configuration error: {error_info.error_id}")
        
        # Configuration errors are usually not recoverable at runtime
        # Log detailed information for debugging
        self.logger.error(f"Configuration error details: {error_info.context}")
        return False
    
    def _handle_network_error(self, error_info: ErrorInfo, exception: Exception) -> bool:
        """Handle network errors."""
        self.logger.debug(f"Handling network error: {error_info.error_id}")
        
        # Try recovery strategy
        recovery_strategy = self._recovery_strategies.get(ErrorCategory.NETWORK)
        if recovery_strategy:
            return recovery_strategy(error_info, exception)
        
        return False
    
    def _handle_validation_error(self, error_info: ErrorInfo, exception: Exception) -> bool:
        """Handle validation errors."""
        self.logger.debug(f"Handling validation error: {error_info.error_id}")
        
        # Validation errors are usually not recoverable
        # Log the validation failure for debugging
        self.logger.info(f"Validation failed: {error_info.message}")
        return False
    
    def _handle_critical_error(self, error_info: ErrorInfo, exception: Exception) -> bool:
        """Handle critical errors."""
        self.logger.critical(f"Critical error occurred: {error_info.error_id}")
        
        # Critical errors should not be recovered from
        # Log all available information
        self.logger.critical(f"System state at error: {error_info.context}")
        self.logger.critical(f"Memory usage: {error_info.memory_usage_mb}MB")
        
        return False
    
    def _handle_recoverable_error(self, error_info: ErrorInfo, exception: Exception) -> bool:
        """Handle recoverable errors."""
        self.logger.debug(f"Handling recoverable error: {error_info.error_id}")
        
        # Try generic recovery
        return self._attempt_generic_recovery(error_info, exception)
    
    # Recovery strategies
    def _recover_from_memory_error(self, error_info: ErrorInfo, exception: Exception) -> bool:
        """Attempt to recover from memory errors."""
        self.logger.info("Attempting memory recovery...")
        
        try:
            import gc
            
            # Force garbage collection
            collected = gc.collect()
            self.logger.info(f"Garbage collection freed {collected} objects")
            
            # Additional cleanup
            gc.collect(0)
            gc.collect(1)
            gc.collect(2)
            
            # Check memory usage after cleanup
            memory_after = self._get_memory_usage()
            if memory_after and error_info.memory_usage_mb:
                freed_mb = error_info.memory_usage_mb - memory_after
                self.logger.info(f"Memory recovery freed {freed_mb:.1f}MB")
                
                if freed_mb > 100:  # Significant memory freed
                    error_info.resolved = True
                    error_info.resolution_method = "memory_cleanup"
                    return True
            
        except Exception as e:
            self.logger.error(f"Memory recovery failed: {e}")
        
        return False
    
    def _recover_from_file_error(self, error_info: ErrorInfo, exception: Exception) -> bool:
        """Attempt to recover from file system errors."""
        self.logger.info("Attempting file system recovery...")
        
        file_path = error_info.context.get('file_path')
        if not file_path:
            return False
        
        try:
            # Wait a moment and retry
            time.sleep(0.1)
            
            if os.path.exists(file_path):
                # Try to access the file again
                with open(file_path, 'rb') as f:
                    f.read(1)
                
                error_info.resolved = True
                error_info.resolution_method = "file_retry"
                return True
                
        except Exception as e:
            self.logger.debug(f"File recovery failed: {e}")
        
        return False
    
    def _recover_from_processing_error(self, error_info: ErrorInfo, exception: Exception) -> bool:
        """Attempt to recover from processing errors."""
        self.logger.info("Attempting processing recovery...")
        
        # For processing errors, we might try with different parameters
        # This is context-dependent and would need to be implemented
        # based on the specific processing component
        
        return False
    
    def _recover_from_network_error(self, error_info: ErrorInfo, exception: Exception) -> bool:
        """Attempt to recover from network errors."""
        self.logger.info("Attempting network recovery...")
        
        # Wait and retry for network errors
        try:
            time.sleep(1.0)
            # Network recovery would be context-specific
            # This is a placeholder for actual network retry logic
            return False
        except Exception as e:
            self.logger.debug(f"Network recovery failed: {e}")
        
        return False
    
    def _attempt_generic_recovery(self, error_info: ErrorInfo, exception: Exception) -> bool:
        """Attempt generic recovery for recoverable errors."""
        self.logger.info("Attempting generic recovery...")
        
        # Generic recovery might include:
        # - Clearing caches
        # - Resetting state
        # - Retrying with default parameters
        
        return False
    
    # Retry mechanism
    def retry_on_error(self, 
                      func: Callable, 
                      *args, 
                      category: Optional[ErrorCategory] = None,
                      custom_retry_config: Optional[RetryConfig] = None,
                      **kwargs) -> Any:
        """Execute function with retry mechanism on errors.
        
        Args:
            func: Function to execute.
            *args: Function arguments.
            category: Error category for retry configuration.
            custom_retry_config: Custom retry configuration.
            **kwargs: Function keyword arguments.
            
        Returns:
            Function result.
            
        Raises:
            Exception: Last exception if all retries failed.
        """
        # Determine retry config
        if custom_retry_config:
            retry_config = custom_retry_config
        elif category and category in self._retry_configs:
            retry_config = self._retry_configs[category]
        else:
            retry_config = RetryConfig()  # Default config
        
        last_exception = None
        
        for attempt in range(retry_config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                
                # If we succeeded after retries, log it
                if attempt > 0:
                    self.logger.info(f"Function succeeded after {attempt} retries")
                    with self._lock:
                        self._error_stats.successful_retries += 1
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should retry this exception type
                if retry_config.retry_on_exceptions and not any(
                    isinstance(e, exc_type) for exc_type in retry_config.retry_on_exceptions
                ):
                    break
                
                # If this is the last attempt, don't wait
                if attempt >= retry_config.max_retries:
                    break
                
                # Calculate delay
                delay = self._calculate_retry_delay(attempt, retry_config)
                
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {type(e).__name__} - {str(e)}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                with self._lock:
                    self._error_stats.retry_attempts += 1
                
                time.sleep(delay)
        
        # All retries failed, handle the error
        context = {
            'function': func.__name__,
            'attempts': retry_config.max_retries + 1,
            'args': str(args)[:100],  # Truncate long args
            'kwargs': str(kwargs)[:100]
        }
        
        self.handle_error(last_exception, context, category)
        raise last_exception
    
    def _calculate_retry_delay(self, attempt: int, retry_config: RetryConfig) -> float:
        """Calculate delay for retry attempt."""
        if retry_config.exponential_backoff:
            delay = retry_config.base_delay * (2 ** attempt)
        else:
            delay = retry_config.base_delay
        
        # Apply maximum delay
        delay = min(delay, retry_config.max_delay)
        
        # Add jitter if enabled
        if retry_config.jitter:
            import random
            jitter = random.uniform(0.1, 0.3) * delay
            delay += jitter
        
        return delay
    
    # Decorator for automatic error handling
    def handle_errors(self, 
                     category: Optional[ErrorCategory] = None,
                     severity: Optional[ErrorSeverity] = None,
                     retry: bool = False,
                     graceful_degradation: bool = False,
                     default_return: Any = None):
        """Decorator for automatic error handling.
        
        Args:
            category: Error category.
            severity: Error severity.
            retry: Whether to retry on errors.
            graceful_degradation: Whether to return default value on error.
            default_return: Default return value for graceful degradation.
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                
                try:
                    if retry:
                        return self.retry_on_error(func, *args, category=category, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
                except Exception as e:
                    error_info = self.handle_error(e, context, category, severity)
                    
                    if graceful_degradation:
                        self.logger.warning(
                            f"Graceful degradation for {func.__name__}: returning {default_return}"
                        )
                        return default_return
                    else:
                        raise
            
            return wrapper
        return decorator
    
    # Error reporting and diagnostics
    def get_error_stats(self) -> ErrorStats:
        """Get error statistics."""
        with self._lock:
            return ErrorStats(
                total_errors=self._error_stats.total_errors,
                errors_by_category=self._error_stats.errors_by_category.copy(),
                errors_by_severity=self._error_stats.errors_by_severity.copy(),
                errors_by_type=self._error_stats.errors_by_type.copy(),
                resolved_errors=self._error_stats.resolved_errors,
                unresolved_errors=self._error_stats.unresolved_errors,
                retry_attempts=self._error_stats.retry_attempts,
                successful_retries=self._error_stats.successful_retries
            )
    
    def get_error_by_id(self, error_id: str) -> Optional[ErrorInfo]:
        """Get error information by ID."""
        with self._lock:
            return self._errors.get(error_id)
    
    def get_recent_errors(self, limit: int = 10) -> List[ErrorInfo]:
        """Get most recent errors."""
        with self._lock:
            errors = sorted(self._errors.values(), key=lambda x: x.timestamp, reverse=True)
            return errors[:limit]
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ErrorInfo]:
        """Get errors by category."""
        with self._lock:
            return [error for error in self._errors.values() if error.category == category]
    
    def get_unresolved_errors(self) -> List[ErrorInfo]:
        """Get unresolved errors."""
        with self._lock:
            return [error for error in self._errors.values() if not error.resolved]
    
    def generate_error_report(self) -> Dict[str, Any]:
        """Generate comprehensive error report."""
        stats = self.get_error_stats()
        recent_errors = self.get_recent_errors(20)
        unresolved_errors = self.get_unresolved_errors()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'total_errors': stats.total_errors,
                'resolved_errors': stats.resolved_errors,
                'unresolved_errors': stats.unresolved_errors,
                'retry_attempts': stats.retry_attempts,
                'successful_retries': stats.successful_retries,
                'errors_by_category': {cat.value: count for cat, count in stats.errors_by_category.items()},
                'errors_by_severity': {sev.value: count for sev, count in stats.errors_by_severity.items()},
                'errors_by_type': stats.errors_by_type
            },
            'recent_errors': [
                {
                    'error_id': error.error_id,
                    'category': error.category.value,
                    'severity': error.severity.value,
                    'exception_type': error.exception_type,
                    'message': error.message,
                    'timestamp': error.timestamp.isoformat(),
                    'resolved': error.resolved,
                    'retry_count': error.retry_count
                }
                for error in recent_errors
            ],
            'unresolved_errors': [
                {
                    'error_id': error.error_id,
                    'category': error.category.value,
                    'severity': error.severity.value,
                    'exception_type': error.exception_type,
                    'message': error.message,
                    'timestamp': error.timestamp.isoformat(),
                    'context': error.context
                }
                for error in unresolved_errors
            ]
        }
    
    def clear_resolved_errors(self):
        """Clear resolved errors from memory."""
        with self._lock:
            resolved_count = len([e for e in self._errors.values() if e.resolved])
            self._errors = {k: v for k, v in self._errors.items() if not v.resolved}
            self.logger.info(f"Cleared {resolved_count} resolved errors from memory")
    
    def reset_error_stats(self):
        """Reset error statistics."""
        with self._lock:
            self._error_stats = ErrorStats()
            self.logger.info("Error statistics reset")


# Global error handler instance
_global_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = ErrorHandler()
    return _global_error_handler


def initialize_error_handler(config: Optional[Dict[str, Any]] = None) -> ErrorHandler:
    """Initialize global error handler with configuration."""
    global _global_error_handler
    _global_error_handler = ErrorHandler(config)
    return _global_error_handler


# Convenience functions
def handle_error(exception: Exception, 
                context: Dict[str, Any], 
                category: Optional[ErrorCategory] = None,
                severity: Optional[ErrorSeverity] = None) -> ErrorInfo:
    """Handle error using global error handler."""
    return get_error_handler().handle_error(exception, context, category, severity)


def retry_on_error(func: Callable, 
                  *args, 
                  category: Optional[ErrorCategory] = None,
                  **kwargs) -> Any:
    """Retry function on error using global error handler."""
    return get_error_handler().retry_on_error(func, *args, category=category, **kwargs)


# Decorator shortcuts
def handle_errors(category: Optional[ErrorCategory] = None,
                 severity: Optional[ErrorSeverity] = None,
                 retry: bool = False,
                 graceful_degradation: bool = False,
                 default_return: Any = None):
    """Decorator for automatic error handling."""
    return get_error_handler().handle_errors(
        category=category,
        severity=severity,
        retry=retry,
        graceful_degradation=graceful_degradation,
        default_return=default_return
    )