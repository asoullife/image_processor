"""Logging system with different log levels and file rotation."""

import logging
import logging.handlers
import os
import sys
from typing import Optional
from backend.config.config_loader import LoggingConfig


class LoggerSetup:
    """Setup and configure logging system."""
    
    def __init__(self, config: LoggingConfig):
        """Initialize logger setup with configuration.
        
        Args:
            config: Logging configuration object.
        """
        self.config = config
        self._logger_initialized = False
    
    def setup_logging(self, log_dir: str = "logs") -> logging.Logger:
        """Setup logging with file rotation and console output.
        
        Args:
            log_dir: Directory to store log files.
            
        Returns:
            logging.Logger: Configured root logger.
        """
        if self._logger_initialized:
            return logging.getLogger()
        
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Setup file handler with rotation
        log_file_path = os.path.join(log_dir, self.config.file)
        max_bytes = self._parse_file_size(self.config.max_file_size)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=self.config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        file_handler.setFormatter(detailed_formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.config.level.upper()))
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers to root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Setup specific loggers for different components
        self._setup_component_loggers()
        
        self._logger_initialized = True
        
        # Log initialization message
        root_logger.info("Logging system initialized")
        root_logger.info(f"Log level: {self.config.level}")
        root_logger.info(f"Log file: {log_file_path}")
        
        return root_logger
    
    def _parse_file_size(self, size_str: str) -> int:
        """Parse file size string to bytes.
        
        Args:
            size_str: Size string like '10MB', '1GB', etc.
            
        Returns:
            int: Size in bytes.
        """
        size_str = size_str.upper().strip()
        
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            # Assume bytes if no unit specified
            return int(size_str)
    
    def _setup_component_loggers(self):
        """Setup loggers for specific components with appropriate levels."""
        
        # Quality analyzer logger
        quality_logger = logging.getLogger('QualityAnalyzer')
        quality_logger.setLevel(logging.INFO)
        
        # Defect detector logger
        defect_logger = logging.getLogger('DefectDetector')
        defect_logger.setLevel(logging.INFO)
        
        # Similarity finder logger
        similarity_logger = logging.getLogger('SimilarityFinder')
        similarity_logger.setLevel(logging.INFO)
        
        # Compliance checker logger
        compliance_logger = logging.getLogger('ComplianceChecker')
        compliance_logger.setLevel(logging.INFO)
        
        # Batch processor logger
        batch_logger = logging.getLogger('BatchProcessor')
        batch_logger.setLevel(logging.DEBUG)
        
        # Progress tracker logger
        progress_logger = logging.getLogger('ProgressTracker')
        progress_logger.setLevel(logging.DEBUG)
        
        # File manager logger
        file_logger = logging.getLogger('FileManager')
        file_logger.setLevel(logging.INFO)
        
        # Report generator logger
        report_logger = logging.getLogger('ReportGenerator')
        report_logger.setLevel(logging.INFO)
        
        # Error handler logger
        error_logger = logging.getLogger('ErrorHandler')
        error_logger.setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger for a specific component.
        
        Args:
            name: Name of the logger/component.
            
        Returns:
            logging.Logger: Configured logger instance.
        """
        return logging.getLogger(name)
    
    def set_log_level(self, level: str):
        """Change the logging level at runtime.
        
        Args:
            level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        """
        level_obj = getattr(logging, level.upper())
        
        # Update root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(level_obj)
        
        # Update console handler level
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.handlers.RotatingFileHandler):
                handler.setLevel(level_obj)
        
        logging.info(f"Log level changed to {level.upper()}")


# Global logger setup instance
_logger_setup: Optional[LoggerSetup] = None


def initialize_logging(config: LoggingConfig, log_dir: str = "logs") -> logging.Logger:
    """Initialize the global logging system.
    
    Args:
        config: Logging configuration.
        log_dir: Directory for log files.
        
    Returns:
        logging.Logger: Root logger instance.
    """
    global _logger_setup
    _logger_setup = LoggerSetup(config)
    return _logger_setup.setup_logging(log_dir)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a component.
    
    Args:
        name: Component name.
        
    Returns:
        logging.Logger: Logger instance.
    """
    if _logger_setup is None:
        # Initialize with default config if not already done
        from backend.config.config_loader import get_config
        config = get_config()
        initialize_logging(config.logging)
    
    return _logger_setup.get_logger(name)


def set_log_level(level: str):
    """Set the global log level.
    
    Args:
        level: New log level.
    """
    if _logger_setup is not None:
        _logger_setup.set_log_level(level)


class DiagnosticLogger:
    """Enhanced logger for diagnostic information and error context."""
    
    def __init__(self, name: str = "Diagnostic"):
        self.logger = get_logger(name)
        self._diagnostic_data = {}
    
    def log_system_info(self):
        """Log system information for diagnostics."""
        try:
            import psutil
            import platform
            
            # System information
            self.logger.info(f"System: {platform.system()} {platform.release()}")
            self.logger.info(f"Python: {platform.python_version()}")
            self.logger.info(f"CPU cores: {psutil.cpu_count()}")
            
            # Memory information
            memory = psutil.virtual_memory()
            self.logger.info(f"Total memory: {memory.total / 1024 / 1024 / 1024:.1f} GB")
            self.logger.info(f"Available memory: {memory.available / 1024 / 1024 / 1024:.1f} GB")
            
            # Disk information
            disk = psutil.disk_usage('/')
            self.logger.info(f"Disk space: {disk.free / 1024 / 1024 / 1024:.1f} GB free of {disk.total / 1024 / 1024 / 1024:.1f} GB")
            
        except Exception as e:
            self.logger.warning(f"Could not gather system info: {e}")
    
    def log_component_status(self, component: str, status: str, details: dict = None):
        """Log component status with details."""
        message = f"Component {component}: {status}"
        if details:
            message += f" - {details}"
        self.logger.info(message)
    
    def log_performance_metrics(self, operation: str, duration: float, memory_delta: float = None):
        """Log performance metrics for operations."""
        message = f"Performance - {operation}: {duration:.3f}s"
        if memory_delta is not None:
            message += f", memory delta: {memory_delta:.1f}MB"
        self.logger.info(message)
    
    def log_error_context(self, error_id: str, context: dict):
        """Log detailed error context for debugging."""
        self.logger.error(f"Error context for {error_id}:")
        for key, value in context.items():
            self.logger.error(f"  {key}: {value}")
    
    def add_diagnostic_data(self, key: str, value: any):
        """Add diagnostic data for later reporting."""
        self._diagnostic_data[key] = value
    
    def get_diagnostic_data(self) -> dict:
        """Get collected diagnostic data."""
        return self._diagnostic_data.copy()
    
    def clear_diagnostic_data(self):
        """Clear diagnostic data."""
        self._diagnostic_data.clear()


class ProgressLogger:
    """Special logger for progress tracking with visual indicators."""
    
    def __init__(self, name: str = "Progress"):
        self.logger = get_logger(name)
        self.last_progress = 0
    
    def log_progress(self, current: int, total: int, message: str = "Processing"):
        """Log progress with percentage and visual indicator.
        
        Args:
            current: Current progress count.
            total: Total items to process.
            message: Progress message.
        """
        if total == 0:
            return
        
        percentage = (current / total) * 100
        
        # Only log every 5% to avoid spam
        if percentage - self.last_progress >= 5 or current == total:
            progress_bar = self._create_progress_bar(percentage)
            self.logger.info(f"{message}: {current}/{total} ({percentage:.1f}%) {progress_bar}")
            self.last_progress = percentage
    
    def _create_progress_bar(self, percentage: float, width: int = 20) -> str:
        """Create a visual progress bar.
        
        Args:
            percentage: Progress percentage (0-100).
            width: Width of the progress bar in characters.
            
        Returns:
            str: Visual progress bar.
        """
        filled = int(width * percentage / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]"
    
    def log_batch_start(self, batch_num: int, batch_size: int, total_batches: int):
        """Log batch processing start.
        
        Args:
            batch_num: Current batch number.
            batch_size: Size of current batch.
            total_batches: Total number of batches.
        """
        self.logger.info(f"Starting batch {batch_num}/{total_batches} (size: {batch_size})")
    
    def log_batch_complete(self, batch_num: int, processing_time: float, errors: int = 0):
        """Log batch processing completion.
        
        Args:
            batch_num: Completed batch number.
            processing_time: Time taken to process batch.
            errors: Number of errors in batch.
        """
        error_msg = f" ({errors} errors)" if errors > 0 else ""
        self.logger.info(f"Batch {batch_num} completed in {processing_time:.2f}s{error_msg}")
    
    def log_checkpoint_saved(self, checkpoint_id: str, images_processed: int):
        """Log checkpoint save.
        
        Args:
            checkpoint_id: Checkpoint identifier.
            images_processed: Number of images processed so far.
        """
        self.logger.info(f"Checkpoint saved: {checkpoint_id} ({images_processed} images processed)")
    
    def log_session_resume(self, session_id: str, resume_point: int, total: int):
        """Log session resume.
        
        Args:
            session_id: Session identifier.
            resume_point: Point where processing resumes.
            total: Total images to process.
        """
        self.logger.info(f"Resuming session {session_id} from image {resume_point}/{total}")
    
    def log_memory_usage(self, memory_mb: float):
        """Log current memory usage.
        
        Args:
            memory_mb: Memory usage in megabytes.
        """
        if memory_mb > 1000:  # Log if over 1GB
            self.logger.warning(f"High memory usage: {memory_mb:.1f} MB")
        else:
            self.logger.debug(f"Memory usage: {memory_mb:.1f} MB")