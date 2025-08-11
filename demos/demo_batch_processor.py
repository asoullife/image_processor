"""Demo script for batch processor functionality."""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import time
import logging
from typing import List

from backend.core.batch_processor import BatchProcessor
from backend.core.progress_tracker import SQLiteProgressTracker
from backend.core.base import ProcessingResult
from backend.config.config_loader import get_config


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def mock_image_processing_function(image_path: str) -> ProcessingResult:
    """Mock image processing function for demonstration.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        ProcessingResult: Mock processing result.
    """
    # Simulate processing time
    time.sleep(0.05)
    
    # Simulate different outcomes based on filename
    filename = os.path.basename(image_path)
    
    if "bad" in filename.lower():
        # Simulate rejected image
        return ProcessingResult(
            image_path=image_path,
            filename=filename,
            final_decision='rejected',
            rejection_reasons=['quality_too_low'],
            processing_time=0.05
        )
    elif "error" in filename.lower():
        # Simulate processing error
        return ProcessingResult(
            image_path=image_path,
            filename=filename,
            final_decision='error',
            error_message='Simulated processing error',
            processing_time=0.05
        )
    else:
        # Simulate approved image
        return ProcessingResult(
            image_path=image_path,
            filename=filename,
            final_decision='approved',
            processing_time=0.05
        )


def create_demo_image_list(count: int = 25) -> List[str]:
    """Create a list of demo image paths.
    
    Args:
        count: Number of demo images to create.
        
    Returns:
        List[str]: List of demo image paths.
    """
    image_paths = []
    
    for i in range(count):
        if i % 10 == 0:
            # Every 10th image is "bad"
            filename = f"bad_image_{i:03d}.jpg"
        elif i % 15 == 0:
            # Every 15th image has an "error"
            filename = f"error_image_{i:03d}.jpg"
        else:
            # Regular good images
            filename = f"good_image_{i:03d}.jpg"
        
        image_paths.append(f"/demo/images/{filename}")
    
    return image_paths


def demo_basic_batch_processing():
    """Demonstrate basic batch processing functionality."""
    logger.info("=== Demo: Basic Batch Processing ===")
    
    # Configuration for demo
    config = {
        'processing': {
            'batch_size': 5,
            'max_workers': 2,
            'memory_threshold_mb': 100,
            'max_retries': 2,
            'retry_delay': 0.5,
            'enable_memory_monitoring': True,
            'gc_frequency': 2
        }
    }
    
    # Create batch processor
    processor = BatchProcessor(config, mock_image_processing_function)
    
    # Create demo image list
    image_paths = create_demo_image_list(15)
    logger.info(f"Processing {len(image_paths)} demo images")
    
    # Progress callback
    def progress_callback(processed: int, total: int):
        percentage = (processed / total) * 100
        logger.info(f"Progress: {processed}/{total} ({percentage:.1f}%)")
    
    # Process images
    start_time = time.time()
    results = processor.process(image_paths, progress_callback=progress_callback)
    end_time = time.time()
    
    # Analyze results
    approved = sum(1 for r in results if r.final_decision == 'approved')
    rejected = sum(1 for r in results if r.final_decision == 'rejected')
    errors = sum(1 for r in results if r.final_decision == 'error')
    
    logger.info(f"Processing completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Results: {approved} approved, {rejected} rejected, {errors} errors")
    
    # Get statistics
    stats = processor.get_statistics()
    logger.info(f"Statistics: {stats['success_rate']:.1f}% success rate, "
               f"{stats['avg_batch_time']:.2f}s avg batch time")
    
    return results


def demo_batch_processing_with_progress_tracking():
    """Demonstrate batch processing with progress tracking."""
    logger.info("=== Demo: Batch Processing with Progress Tracking ===")
    
    # Configuration
    config = {
        'processing': {
            'batch_size': 4,
            'max_workers': 2,
            'memory_threshold_mb': 100,
            'max_retries': 2,
            'retry_delay': 0.5,
            'enable_memory_monitoring': True,
            'gc_frequency': 3
        }
    }
    
    # Create components
    processor = BatchProcessor(config, mock_image_processing_function)
    progress_tracker = SQLiteProgressTracker("demo_progress.db", checkpoint_interval=3)
    
    # Create demo image list
    image_paths = create_demo_image_list(20)
    
    # Create processing session
    session_id = progress_tracker.create_session(
        input_folder="/demo/input",
        output_folder="/demo/output",
        total_images=len(image_paths)
    )
    
    logger.info(f"Created session: {session_id}")
    
    # Progress callback with checkpointing
    processed_results = []
    
    def progress_callback_with_checkpoints(processed: int, total: int):
        percentage = (processed / total) * 100
        logger.info(f"Progress: {processed}/{total} ({percentage:.1f}%)")
        
        # Save checkpoint every few images
        if processed % 3 == 0 and processed > 0:
            recent_results = processed_results[-3:] if len(processed_results) >= 3 else processed_results
            success = progress_tracker.save_checkpoint(
                session_id=session_id,
                processed_count=processed,
                total_count=total,
                results=recent_results
            )
            if success:
                logger.info(f"Checkpoint saved at {processed} images")
    
    # Process images
    start_time = time.time()
    results = processor.process(image_paths, progress_callback=progress_callback_with_checkpoints)
    processed_results.extend(results)
    end_time = time.time()
    
    # Final checkpoint
    progress_tracker.save_checkpoint(
        session_id=session_id,
        processed_count=len(results),
        total_count=len(image_paths),
        results=results[-5:]  # Save last 5 results
    )
    
    # Complete session
    progress_tracker.complete_session(session_id, 'completed')
    
    # Get progress summary
    summary = progress_tracker.get_progress_summary(session_id)
    logger.info(f"Session completed: {summary['progress_percentage']:.1f}% processed, "
               f"{summary['approval_rate']:.1f}% approval rate")
    
    # Clean up demo database
    if os.path.exists("demo_progress.db"):
        os.remove("demo_progress.db")
    
    return results


def demo_memory_management():
    """Demonstrate memory management features."""
    logger.info("=== Demo: Memory Management ===")
    
    # Configuration with aggressive memory monitoring
    config = {
        'processing': {
            'batch_size': 3,
            'max_workers': 2,
            'memory_threshold_mb': 50,  # Low threshold for demo
            'max_retries': 2,
            'retry_delay': 0.1,
            'enable_memory_monitoring': True,
            'gc_frequency': 1  # Cleanup after every batch
        }
    }
    
    # Create processor
    processor = BatchProcessor(config, mock_image_processing_function)
    
    # Get initial memory usage
    initial_memory = processor.get_memory_usage()
    logger.info(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Create larger demo image list
    image_paths = create_demo_image_list(30)
    
    # Process with memory monitoring
    def memory_progress_callback(processed: int, total: int):
        current_memory = processor.get_memory_usage()
        logger.info(f"Progress: {processed}/{total}, Memory: {current_memory:.1f} MB")
    
    results = processor.process(image_paths, progress_callback=memory_progress_callback)
    
    # Final memory usage
    final_memory = processor.get_memory_usage()
    logger.info(f"Final memory usage: {final_memory:.1f} MB")
    
    # Force memory cleanup
    before_cleanup, after_cleanup = processor.cleanup_memory()
    logger.info(f"Memory cleanup: {before_cleanup:.1f} MB -> {after_cleanup:.1f} MB")
    
    # Get detailed statistics
    stats = processor.get_statistics()
    logger.info(f"Average memory usage during processing: {stats['avg_memory_usage']:.1f} MB")
    
    return results


def demo_error_handling():
    """Demonstrate error handling and retry logic."""
    logger.info("=== Demo: Error Handling ===")
    
    # Configuration with retry settings
    config = {
        'processing': {
            'batch_size': 3,
            'max_workers': 2,
            'memory_threshold_mb': 100,
            'max_retries': 3,
            'retry_delay': 0.2,
            'enable_memory_monitoring': True,
            'gc_frequency': 2
        }
    }
    
    # Create error-prone processing function
    call_count = 0
    
    def error_prone_function(image_path: str) -> ProcessingResult:
        nonlocal call_count
        call_count += 1
        
        filename = os.path.basename(image_path)
        
        # Simulate intermittent errors
        if "flaky" in filename and call_count % 3 == 1:
            raise RuntimeError(f"Intermittent error for {filename}")
        
        return mock_image_processing_function(image_path)
    
    # Create processor with error-prone function
    processor = BatchProcessor(config, error_prone_function)
    
    # Create image list with some flaky images
    image_paths = [
        "/demo/images/good_image_001.jpg",
        "/demo/images/flaky_image_002.jpg",
        "/demo/images/good_image_003.jpg",
        "/demo/images/flaky_image_004.jpg",
        "/demo/images/bad_image_005.jpg",
        "/demo/images/error_image_006.jpg"
    ]
    
    logger.info(f"Processing {len(image_paths)} images with error handling")
    
    # Process with error handling
    results = processor.process(image_paths)
    
    # Analyze results
    approved = sum(1 for r in results if r.final_decision == 'approved')
    rejected = sum(1 for r in results if r.final_decision == 'rejected')
    errors = sum(1 for r in results if r.final_decision == 'error')
    
    logger.info(f"Results: {approved} approved, {rejected} rejected, {errors} errors")
    
    # Show error details
    for result in results:
        if result.final_decision == 'error':
            logger.info(f"Error in {result.filename}: {result.error_message}")
    
    # Get statistics
    stats = processor.get_statistics()
    logger.info(f"Error handling stats: {stats['total_errors']} total errors, "
               f"{stats['success_rate']:.1f}% success rate")
    
    return results


def main():
    """Run all batch processor demos."""
    logger.info("Starting Batch Processor Demos")
    logger.info("=" * 50)
    
    try:
        # Run demos
        demo_basic_batch_processing()
        print()
        
        demo_batch_processing_with_progress_tracking()
        print()
        
        demo_memory_management()
        print()
        
        demo_error_handling()
        print()
        
        logger.info("All demos completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()