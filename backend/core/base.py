"""Base classes and interfaces for core components."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging


@dataclass
class ProcessingResult:
    """Result of image processing."""
    image_path: str
    filename: str
    quality_result: Optional['QualityResult'] = None
    defect_result: Optional['DefectResult'] = None
    similarity_group: Optional[int] = None
    compliance_result: Optional['ComplianceResult'] = None
    final_decision: str = 'pending'  # 'approved', 'rejected', 'pending'
    rejection_reasons: List[str] = None
    processing_time: float = 0.0
    timestamp: Optional[datetime] = None
    error_message: Optional[str] = None
    decision_result: Optional['DecisionResult'] = None
    
    def __post_init__(self):
        if self.rejection_reasons is None:
            self.rejection_reasons = []
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class QualityResult:
    """Result of quality analysis."""
    sharpness_score: float
    noise_level: float
    exposure_score: float
    color_balance_score: float
    resolution: tuple
    file_size: int
    overall_score: float
    passed: bool


@dataclass
class DefectResult:
    """Result of defect detection."""
    detected_objects: List['ObjectDefect']
    anomaly_score: float
    defect_count: int
    defect_types: List[str]
    confidence_scores: List[float]
    passed: bool


@dataclass
class ObjectDefect:
    """Detected object defect."""
    object_type: str
    defect_type: str
    confidence: float
    bounding_box: tuple
    description: str


@dataclass
class ComplianceResult:
    """Result of compliance checking."""
    logo_detections: List['LogoDetection']
    privacy_violations: List['PrivacyViolation']
    metadata_issues: List[str]
    keyword_relevance: float
    overall_compliance: bool


@dataclass
class LogoDetection:
    """Detected logo or trademark."""
    logo_type: str
    confidence: float
    bounding_box: tuple
    brand_name: str


@dataclass
class PrivacyViolation:
    """Privacy concern detection."""
    violation_type: str  # 'face', 'license_plate', 'personal_info'
    confidence: float
    bounding_box: tuple
    description: str


@dataclass
class ExposureResult:
    """Exposure analysis result."""
    histogram_mean: float
    histogram_std: float
    overexposed_pixels: float
    underexposed_pixels: float
    dynamic_range: float
    exposure_score: float


class BaseAnalyzer(ABC):
    """Base class for all image analyzers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize analyzer with configuration.
        
        Args:
            config: Configuration dictionary for the analyzer.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def analyze(self, image_path: str) -> Any:
        """Analyze an image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Analysis result object.
        """
        pass
    
    def validate_image_path(self, image_path: str) -> bool:
        """Validate that image path exists and is readable.
        
        Args:
            image_path: Path to validate.
            
        Returns:
            bool: True if valid, False otherwise.
        """
        import os
        return os.path.exists(image_path) and os.path.isfile(image_path)


class BaseProcessor(ABC):
    """Base class for processing components."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize processor with configuration.
        
        Args:
            config: Configuration dictionary for the processor.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Process data.
        
        Returns:
            Processing result.
        """
        pass


class BaseManager(ABC):
    """Base class for management components."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize manager with configuration.
        
        Args:
            config: Configuration dictionary for the manager.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the manager.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """Cleanup resources.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        pass


class ProgressTracker(ABC):
    """Abstract base class for progress tracking."""
    
    @abstractmethod
    def save_checkpoint(self, session_id: str, processed_count: int, 
                       total_count: int, results: List[ProcessingResult]) -> bool:
        """Save processing checkpoint.
        
        Args:
            session_id: Unique session identifier.
            processed_count: Number of images processed.
            total_count: Total number of images to process.
            results: Processing results so far.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load processing checkpoint.
        
        Args:
            session_id: Session identifier to load.
            
        Returns:
            Dict with checkpoint data or None if not found.
        """
        pass
    
    @abstractmethod
    def get_session_status(self, session_id: str) -> Optional[str]:
        """Get session processing status.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            Status string or None if session not found.
        """
        pass


class ErrorHandler:
    """Centralized error handling system."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_counts = {}
    
    def handle_file_error(self, error: Exception, file_path: str) -> bool:
        """Handle file-related errors.
        
        Args:
            error: The exception that occurred.
            file_path: Path to the problematic file.
            
        Returns:
            bool: True if error was handled and processing can continue.
        """
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        self.logger.error(f"File error in {file_path}: {error_type} - {str(error)}")
        
        # Handle specific error types
        if isinstance(error, (FileNotFoundError, PermissionError)):
            return False  # Cannot recover from these
        elif isinstance(error, (OSError, IOError)):
            return self.should_retry(error)
        
        return False
    
    def handle_processing_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle processing-related errors.
        
        Args:
            error: The exception that occurred.
            context: Context information about the error.
            
        Returns:
            bool: True if error was handled and processing can continue.
        """
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        self.logger.error(f"Processing error: {error_type} - {str(error)}")
        self.logger.debug(f"Error context: {context}")
        
        # Handle specific error types
        if isinstance(error, MemoryError):
            self.logger.warning("Memory error detected, attempting cleanup")
            return True  # Try to continue after cleanup
        elif isinstance(error, (ValueError, TypeError)):
            return False  # Data validation errors are not recoverable
        
        return self.should_retry(error)
    
    def should_retry(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry.
        
        Args:
            error: The exception to evaluate.
            
        Returns:
            bool: True if retry is recommended.
        """
        error_type = type(error).__name__
        retry_count = self.error_counts.get(error_type, 0)
        
        # Don't retry if we've seen this error type too many times
        if retry_count > 3:
            return False
        
        # Retry for temporary/recoverable errors
        recoverable_errors = [
            'ConnectionError', 'TimeoutError', 'TemporaryFailure',
            'ResourceWarning', 'RuntimeWarning'
        ]
        
        return error_type in recoverable_errors
    
    def log_error(self, error: Exception, context: Dict[str, Any]):
        """Log error with context information.
        
        Args:
            error: The exception to log.
            context: Additional context information.
        """
        self.logger.error(
            f"Error: {type(error).__name__} - {str(error)}\n"
            f"Context: {context}"
        )
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of error counts.
        
        Returns:
            Dict mapping error types to their occurrence counts.
        """
        return self.error_counts.copy()