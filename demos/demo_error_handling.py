#!/usr/bin/env python3
"""
Demonstration of comprehensive error handling and logging system.

This script demonstrates the various error handling capabilities including:
- Centralized error handling with categorization
- Retry mechanisms for recoverable errors
- Graceful degradation for non-critical failures
- Detailed logging with different verbosity levels
- Error reporting and diagnostic information collection
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.error_handler import (
    ErrorHandler, ErrorCategory, ErrorSeverity, get_error_handler,
    initialize_error_handler, handle_errors, retry_on_error
)
from backend.core.error_integration import (
    ErrorIntegration, error_context, safe_execute, batch_safe_execute,
    GracefulDegradation, wrap_with_error_handling
)
from backend.utils.logger import initialize_logging, get_logger, DiagnosticLogger
from backend.config.config_loader import get_config


class DemoImageAnalyzer:
    """Demo image analyzer for error handling demonstration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger('DemoImageAnalyzer')
    
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """Analyze image with various error scenarios."""
        self.logger.info(f"Analyzing image: {image_path}")
        
        # Simulate different error scenarios based on filename
        if "corrupt" in image_path:
            raise ValueError("Image file is corrupted")
        elif "memory_error" in image_path:
            raise MemoryError("Insufficient memory for analysis")
        elif "permission" in image_path:
            raise PermissionError("Cannot access image file")
        elif "network" in image_path:
            raise ConnectionError("Cannot download required model")
        elif "timeout" in image_path:
            time.sleep(0.1)  # Simulate slow operation
            raise TimeoutError("Analysis timed out")
        
        # Simulate successful analysis
        return {
            'quality_score': 0.85,
            'defects_found': 0,
            'compliance_passed': True,
            'processing_time': 0.05
        }


class DemoFileProcessor:
    """Demo file processor for error handling demonstration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger('DemoFileProcessor')
    
    def process_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Process batch of images with error handling."""
        results = []
        
        for image_path in image_paths:
            try:
                with error_context("DemoFileProcessor", "process_image", image_path=image_path):
                    result = self._process_single_image(image_path)
                    results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {image_path}: {e}")
                results.append(None)
        
        return results
    
    def _process_single_image(self, image_path: str) -> Dict[str, Any]:
        """Process single image."""
        if "file_not_found" in image_path:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        return {
            'image_path': image_path,
            'processed': True,
            'output_path': f"output/{os.path.basename(image_path)}"
        }


def demonstrate_basic_error_handling():
    """Demonstrate basic error handling capabilities."""
    print("\n=== Basic Error Handling Demonstration ===")
    
    # Initialize error handler
    error_handler = initialize_error_handler()
    
    # Create demo analyzer
    analyzer = DemoImageAnalyzer({})
    
    # Test different error scenarios
    test_images = [
        "normal_image.jpg",
        "corrupt_image.jpg",
        "memory_error_image.jpg",
        "permission_denied.jpg",
        "network_timeout.jpg"
    ]
    
    for image_path in test_images:
        print(f"\nTesting: {image_path}")
        try:
            result = analyzer.analyze(image_path)
            print(f"  Success: {result}")
        except Exception as e:
            # Handle error with context
            context = {
                'component': 'DemoImageAnalyzer',
                'operation': 'analyze',
                'image_path': image_path
            }
            error_info = error_handler.handle_error(e, context)
            print(f"  Error handled: {error_info.error_id} ({error_info.category.value})")
    
    # Show error statistics
    stats = error_handler.get_error_stats()
    print(f"\nError Statistics:")
    print(f"  Total errors: {stats.total_errors}")
    print(f"  Errors by category: {stats.errors_by_category}")
    print(f"  Errors by type: {stats.errors_by_type}")


def demonstrate_retry_mechanism():
    """Demonstrate retry mechanism for recoverable errors."""
    print("\n=== Retry Mechanism Demonstration ===")
    
    error_handler = get_error_handler()
    
    # Function that fails a few times then succeeds
    attempt_count = 0
    
    def unreliable_network_operation():
        nonlocal attempt_count
        attempt_count += 1
        print(f"    Attempt {attempt_count}")
        
        if attempt_count < 3:
            raise ConnectionError("Network temporarily unavailable")
        return "Network operation successful"
    
    print("Testing retry mechanism with network error:")
    try:
        result = error_handler.retry_on_error(
            unreliable_network_operation,
            category=ErrorCategory.NETWORK
        )
        print(f"  Success after retries: {result}")
    except Exception as e:
        print(f"  Failed after all retries: {e}")
    
    # Reset for next test
    attempt_count = 0
    
    # Function that always fails
    def always_failing_operation():
        nonlocal attempt_count
        attempt_count += 1
        print(f"    Attempt {attempt_count}")
        raise ValueError("This operation always fails")
    
    print("\nTesting retry mechanism with non-recoverable error:")
    try:
        result = error_handler.retry_on_error(
            always_failing_operation,
            category=ErrorCategory.VALIDATION
        )
        print(f"  Unexpected success: {result}")
    except Exception as e:
        print(f"  Failed as expected: {e}")


def demonstrate_error_decorators():
    """Demonstrate error handling decorators."""
    print("\n=== Error Handling Decorators Demonstration ===")
    
    error_handler = get_error_handler()
    
    @error_handler.handle_errors(
        category=ErrorCategory.PROCESSING,
        retry=True,
        graceful_degradation=True,
        default_return={"status": "failed", "score": 0.0}
    )
    def analyze_with_retry(image_path: str) -> Dict[str, Any]:
        """Analysis function with automatic error handling."""
        if "retry_success" in image_path:
            # Fail first time, succeed on retry
            if not hasattr(analyze_with_retry, 'called'):
                analyze_with_retry.called = True
                raise RuntimeError("Temporary failure")
            return {"status": "success", "score": 0.9}
        elif "permanent_failure" in image_path:
            raise ValueError("Permanent analysis failure")
        else:
            return {"status": "success", "score": 0.8}
    
    test_cases = [
        "normal_image.jpg",
        "retry_success_image.jpg",
        "permanent_failure_image.jpg"
    ]
    
    for image_path in test_cases:
        print(f"\nTesting decorated function with: {image_path}")
        result = analyze_with_retry(image_path)
        print(f"  Result: {result}")


def demonstrate_graceful_degradation():
    """Demonstrate graceful degradation capabilities."""
    print("\n=== Graceful Degradation Demonstration ===")
    
    degradation = GracefulDegradation()
    
    # Test degraded quality analysis
    print("Testing degraded quality analysis:")
    test_images = [
        "nonexistent_image.jpg",
        "/dev/null",  # Empty file on Unix systems
        "test_image.jpg"  # This will use mocked PIL
    ]
    
    for image_path in test_images:
        print(f"\n  Testing: {image_path}")
        result = degradation.degrade_quality_analysis(image_path)
        print(f"    Passed: {result.get('passed', False)}")
        print(f"    Degraded: {result.get('degraded', False)}")
        print(f"    Reason: {result.get('degradation_reason', 'N/A')}")
    
    # Test degraded similarity detection
    print("\nTesting degraded similarity detection:")
    image_list = ["img1.jpg", "img2.jpg", "img3.jpg"]
    similarity_result = degradation.degrade_similarity_detection(image_list)
    print(f"  Groups created: {similarity_result['total_groups']}")
    print(f"  Degraded: {similarity_result['degraded']}")


def demonstrate_batch_error_handling():
    """Demonstrate batch processing with error handling."""
    print("\n=== Batch Error Handling Demonstration ===")
    
    # Create test functions with mixed success/failure
    def process_image(image_path: str) -> str:
        if "error" in image_path:
            raise RuntimeError(f"Processing failed for {image_path}")
        return f"processed_{image_path}"
    
    # Test batch with mixed results
    image_paths = [
        "good_image1.jpg",
        "error_image1.jpg",
        "good_image2.jpg",
        "error_image2.jpg",
        "good_image3.jpg"
    ]
    
    functions_and_args = [(process_image, (path,), {}) for path in image_paths]
    
    print("Processing batch with continue_on_error=True:")
    results, errors = batch_safe_execute(
        functions_and_args,
        category=ErrorCategory.PROCESSING,
        continue_on_error=True,
        collect_errors=True
    )
    
    print(f"  Results: {len([r for r in results if r is not None])} successful, {len(errors)} errors")
    for i, result in enumerate(results):
        status = "SUCCESS" if result else "ERROR"
        print(f"    {image_paths[i]}: {status}")


def demonstrate_error_reporting():
    """Demonstrate error reporting and diagnostics."""
    print("\n=== Error Reporting Demonstration ===")
    
    error_handler = get_error_handler()
    
    # Generate some test errors for reporting
    test_errors = [
        (ValueError("Invalid configuration"), {"component": "config_loader"}),
        (FileNotFoundError("Missing image"), {"component": "file_manager", "file": "test.jpg"}),
        (MemoryError("Out of memory"), {"component": "batch_processor", "batch_size": 1000}),
        (ConnectionError("Network failed"), {"component": "model_downloader", "url": "http://example.com"})
    ]
    
    for error, context in test_errors:
        error_handler.handle_error(error, context)
    
    # Generate comprehensive error report
    report = error_handler.generate_error_report()
    
    print("Error Report Summary:")
    print(f"  Total errors: {report['statistics']['total_errors']}")
    print(f"  Resolved errors: {report['statistics']['resolved_errors']}")
    print(f"  Unresolved errors: {report['statistics']['unresolved_errors']}")
    
    print("\nErrors by category:")
    for category, count in report['statistics']['errors_by_category'].items():
        print(f"  {category}: {count}")
    
    print("\nErrors by severity:")
    for severity, count in report['statistics']['errors_by_severity'].items():
        print(f"  {severity}: {count}")
    
    print(f"\nRecent errors: {len(report['recent_errors'])}")
    for error in report['recent_errors'][:3]:  # Show first 3
        print(f"  {error['error_id']}: {error['exception_type']} - {error['message'][:50]}...")


def demonstrate_component_wrapping():
    """Demonstrate wrapping existing components with error handling."""
    print("\n=== Component Wrapping Demonstration ===")
    
    # Wrap the demo analyzer with error handling
    WrappedAnalyzer = wrap_with_error_handling(DemoImageAnalyzer, 'analyzer')
    wrapped_analyzer = WrappedAnalyzer({})
    
    print("Testing wrapped analyzer:")
    test_images = [
        "normal_image.jpg",
        "corrupt_image.jpg",
        "memory_error_image.jpg"
    ]
    
    for image_path in test_images:
        print(f"\n  Analyzing: {image_path}")
        result = wrapped_analyzer.analyze(image_path)
        if result:
            print(f"    Success: Quality score = {result.get('quality_score', 'N/A')}")
        else:
            print(f"    Failed with graceful degradation")


def demonstrate_diagnostic_logging():
    """Demonstrate diagnostic logging capabilities."""
    print("\n=== Diagnostic Logging Demonstration ===")
    
    diagnostic_logger = DiagnosticLogger()
    
    # Log system information
    print("Logging system information...")
    diagnostic_logger.log_system_info()
    
    # Log component status
    diagnostic_logger.log_component_status("QualityAnalyzer", "initialized", {"version": "1.0"})
    diagnostic_logger.log_component_status("DefectDetector", "loading_model", {"model": "yolo_v5"})
    diagnostic_logger.log_component_status("BatchProcessor", "ready", {"batch_size": 200})
    
    # Log performance metrics
    diagnostic_logger.log_performance_metrics("image_analysis", 0.125, 15.5)
    diagnostic_logger.log_performance_metrics("batch_processing", 45.2, -8.3)
    
    # Add diagnostic data
    diagnostic_logger.add_diagnostic_data("images_processed", 1500)
    diagnostic_logger.add_diagnostic_data("errors_encountered", 23)
    diagnostic_logger.add_diagnostic_data("memory_peak_mb", 1024.5)
    
    diagnostic_data = diagnostic_logger.get_diagnostic_data()
    print(f"\nDiagnostic data collected: {diagnostic_data}")


def main():
    """Main demonstration function."""
    print("Adobe Stock Image Processor - Error Handling System Demo")
    print("=" * 60)
    
    # Initialize logging system
    try:
        config = get_config()
        initialize_logging(config.logging)
        print("Logging system initialized successfully")
    except Exception as e:
        print(f"Warning: Could not initialize full logging system: {e}")
        logging.basicConfig(level=logging.INFO)
    
    # Run demonstrations
    try:
        demonstrate_basic_error_handling()
        demonstrate_retry_mechanism()
        demonstrate_error_decorators()
        demonstrate_graceful_degradation()
        demonstrate_batch_error_handling()
        demonstrate_error_reporting()
        demonstrate_component_wrapping()
        demonstrate_diagnostic_logging()
        
        print("\n" + "=" * 60)
        print("Error handling demonstration completed successfully!")
        
        # Final error statistics
        error_handler = get_error_handler()
        final_stats = error_handler.get_error_stats()
        print(f"\nFinal error statistics:")
        print(f"  Total errors handled: {final_stats.total_errors}")
        print(f"  Retry attempts: {final_stats.retry_attempts}")
        print(f"  Successful retries: {final_stats.successful_retries}")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())