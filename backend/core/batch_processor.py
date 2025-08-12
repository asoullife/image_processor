"""Batch processing engine with memory management and multi-threading support."""

import gc
import time
import threading
import psutil
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass
from queue import Queue, Empty
import traceback
from datetime import datetime, timedelta

from .base import BaseProcessor, ProcessingResult, ErrorHandler

# Import monitoring utilities
try:
    from backend.utils.performance_monitor import performance_monitor
    from backend.utils.console_notifier import console_notifier
    from backend.realtime.socketio_manager import socketio_manager, ProgressData
except ImportError:
    # Standalone mode
    from utils.performance_monitor import performance_monitor
    from utils.console_notifier import console_notifier
    from realtime.socketio_manager import socketio_manager, ProgressData


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 200
    max_workers: int = 4
    memory_threshold_mb: int = 1024  # Memory threshold in MB
    max_retries: int = 3
    retry_delay: float = 1.0  # Seconds between retries
    enable_memory_monitoring: bool = True
    gc_frequency: int = 10  # Run GC every N batches


@dataclass
class BatchResult:
    """Result of batch processing."""
    batch_id: int
    processed_count: int
    success_count: int
    error_count: int
    processing_time: float
    memory_usage_mb: float
    results: List[ProcessingResult]
    errors: List[Dict[str, Any]]


class MemoryMonitor:
    """Memory usage monitoring and cleanup."""
    
    def __init__(self, threshold_mb: int = 1024):
        """Initialize memory monitor.
        
        Args:
            threshold_mb: Memory threshold in MB for triggering cleanup.
        """
        self.threshold_mb = threshold_mb
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lock = threading.Lock()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB.
        
        Returns:
            float: Memory usage in MB.
        """
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def check_memory_threshold(self) -> bool:
        """Check if memory usage exceeds threshold.
        
        Returns:
            bool: True if memory usage is above threshold.
        """
        current_usage = self.get_memory_usage()
        return current_usage > self.threshold_mb
    
    def cleanup_memory(self, force: bool = False) -> Tuple[float, float]:
        """Perform memory cleanup.
        
        Args:
            force: Force cleanup regardless of threshold.
            
        Returns:
            Tuple[float, float]: Memory usage before and after cleanup.
        """
        with self._lock:
            memory_before = self.get_memory_usage()
            
            if force or self.check_memory_threshold():
                # Force garbage collection
                collected = gc.collect()
                
                # Additional cleanup for specific object types
                gc.collect(0)  # Collect generation 0
                gc.collect(1)  # Collect generation 1
                gc.collect(2)  # Collect generation 2
                
                memory_after = self.get_memory_usage()
                freed_mb = memory_before - memory_after
                
                self.logger.info(
                    f"Memory cleanup: {memory_before:.1f}MB -> {memory_after:.1f}MB "
                    f"(freed {freed_mb:.1f}MB, collected {collected} objects)"
                )
                
                return memory_before, memory_after
            
            return memory_before, memory_before


class BatchProcessor(BaseProcessor):
    """Batch processing engine with memory management and multi-threading."""
    
    def __init__(self, config: Dict[str, Any], 
                 processing_function: Callable[[str], ProcessingResult],
                 session_id: Optional[str] = None):
        """Initialize batch processor.
        
        Args:
            config: Configuration dictionary.
            processing_function: Function to process individual images.
            session_id: Optional session ID for monitoring.
        """
        super().__init__(config)
        
        # Extract batch configuration
        batch_config_data = config.get('processing', {})
        self.batch_config = BatchConfig(
            batch_size=batch_config_data.get('batch_size', 200),
            max_workers=batch_config_data.get('max_workers', 4),
            memory_threshold_mb=batch_config_data.get('memory_threshold_mb', 1024),
            max_retries=batch_config_data.get('max_retries', 3),
            retry_delay=batch_config_data.get('retry_delay', 1.0),
            enable_memory_monitoring=batch_config_data.get('enable_memory_monitoring', True),
            gc_frequency=batch_config_data.get('gc_frequency', 10)
        )
        
        self.processing_function = processing_function
        self.error_handler = ErrorHandler()
        self.memory_monitor = MemoryMonitor(self.batch_config.memory_threshold_mb)
        
        # Session tracking for monitoring
        self.session_id = session_id
        self.session_start_time: Optional[datetime] = None
        
        # Thread safety
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # Statistics
        self.total_processed = 0
        self.total_approved = 0
        self.total_rejected = 0
        self.total_batches = 0
        self.total_errors = 0
        self.batch_history: List[BatchResult] = []
        
        self.logger.info(f"BatchProcessor initialized with batch_size={self.batch_config.batch_size}, "
                        f"max_workers={self.batch_config.max_workers}")
    
    def process(self, image_paths: List[str], 
                progress_callback: Optional[Callable[[int, int], None]] = None) -> List[ProcessingResult]:
        """Process list of images in batches.
        
        Args:
            image_paths: List of image file paths to process.
            progress_callback: Optional callback for progress updates (processed, total).
            
        Returns:
            List[ProcessingResult]: Results for all processed images.
        """
        if not image_paths:
            self.logger.warning("No images to process")
            return []
        
        self.logger.info(f"Starting batch processing of {len(image_paths)} images")
        start_time = time.time()
        self.session_start_time = datetime.now()
        
        # Reset statistics
        self.total_processed = 0
        self.total_approved = 0
        self.total_rejected = 0
        self.total_batches = 0
        self.total_errors = 0
        self.batch_history.clear()
        self._stop_event.clear()
        
        # Start monitoring
        if self.session_id:
            performance_monitor.start_session(self.session_id)
            console_notifier.start_session(self.session_id, len(image_paths))
        
        all_results = []
        
        try:
            # Split images into batches
            batches = self._create_batches(image_paths)
            self.logger.info(f"Created {len(batches)} batches")
            
            for batch_id, batch_paths in enumerate(batches):
                if self._stop_event.is_set():
                    self.logger.info("Processing stopped by user request")
                    break
                
                self.logger.info(f"Processing batch {batch_id + 1}/{len(batches)} "
                                f"({len(batch_paths)} images)")
                
                # Process batch with retry logic
                batch_result = self._process_batch_with_retry(batch_id, batch_paths)
                
                # Add results to overall list
                all_results.extend(batch_result.results)
                self.batch_history.append(batch_result)
                
                # Update statistics
                with self._lock:
                    self.total_processed += batch_result.processed_count
                    self.total_batches += 1
                    self.total_errors += batch_result.error_count
                    
                    # Count approved/rejected
                    for result in batch_result.results:
                        if result.final_decision == 'approved':
                            self.total_approved += 1
                        else:
                            self.total_rejected += 1
                
                # Update monitoring
                if self.session_id:
                    # Update performance monitor
                    performance_monitor.update_session_progress(
                        self.session_id, 
                        self.total_processed,
                        batch_result.processing_time,
                        batch_result.processing_time
                    )
                    
                    # Get current performance metrics
                    perf_metrics = performance_monitor.get_current_performance(self.session_id)
                    
                    # Update console notifier
                    current_filename = batch_paths[-1].split('/')[-1] if batch_paths else ""
                    console_notifier.update_progress(
                        self.total_processed,
                        self.total_approved,
                        self.total_rejected,
                        perf_metrics.get('images_per_second', 0.0),
                        current_filename,
                        "processing",
                        perf_metrics
                    )
                    
                    # Broadcast real-time progress via Socket.IO
                    elapsed_time = (datetime.now() - self.session_start_time).total_seconds()
                    processing_speed = self.total_processed / elapsed_time if elapsed_time > 0 else 0.0
                    
                    # Calculate ETA
                    eta_seconds = None
                    if processing_speed > 0:
                        remaining = len(image_paths) - self.total_processed
                        eta_seconds = remaining / processing_speed
                    
                    progress_data = ProgressData(
                        session_id=self.session_id,
                        current_image=self.total_processed,
                        total_images=len(image_paths),
                        percentage=(self.total_processed / len(image_paths)) * 100,
                        current_filename=current_filename,
                        approved_count=self.total_approved,
                        rejected_count=self.total_rejected,
                        processing_speed=processing_speed,
                        estimated_completion=datetime.now() + timedelta(seconds=eta_seconds) if eta_seconds else None,
                        current_stage="processing",
                        memory_usage_mb=perf_metrics.get('current_memory_mb', 0.0),
                        gpu_usage_percent=perf_metrics.get('current_gpu_percent', 0.0),
                        cpu_usage_percent=perf_metrics.get('current_cpu_percent', 0.0),
                        batch_processing_time=batch_result.processing_time,
                        avg_image_processing_time=perf_metrics.get('avg_processing_time', 0.0),
                        elapsed_time=elapsed_time,
                        current_batch=batch_id + 1,
                        total_batches=len(batches),
                        images_per_second=processing_speed,
                        eta_seconds=eta_seconds
                    )
                    
                    # Broadcast via Socket.IO (async call)
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        loop.create_task(socketio_manager.broadcast_progress(self.session_id, progress_data))
                    except RuntimeError:
                        # No event loop running, skip Socket.IO broadcast
                        pass
                
                # Progress callback
                if progress_callback:
                    progress_callback(self.total_processed, len(image_paths))
                
                # Memory cleanup between batches
                if self.batch_config.enable_memory_monitoring:
                    if (batch_id + 1) % self.batch_config.gc_frequency == 0:
                        self.memory_monitor.cleanup_memory()
                
                self.logger.info(f"Batch {batch_id + 1} completed: "
                                f"{batch_result.success_count} success, "
                                f"{batch_result.error_count} errors, "
                                f"{batch_result.processing_time:.2f}s")
        
        except Exception as e:
            self.logger.error(f"Critical error in batch processing: {e}")
            self.logger.debug(traceback.format_exc())
            raise
        
        finally:
            # Final cleanup
            if self.batch_config.enable_memory_monitoring:
                self.memory_monitor.cleanup_memory(force=True)
            
            # End monitoring
            if self.session_id:
                final_metrics = performance_monitor.end_session(self.session_id)
                console_notifier.notify_completion(
                    self.total_processed,
                    self.total_approved,
                    self.total_rejected,
                    time.time() - start_time,
                    final_metrics
                )
                console_notifier.stop_session()
        
        total_time = time.time() - start_time
        self.logger.info(f"Batch processing completed: {self.total_processed} images processed "
                        f"in {total_time:.2f}s ({self.total_processed/total_time:.2f} images/sec)")
        
        return all_results
    
    def _create_batches(self, image_paths: List[str]) -> List[List[str]]:
        """Split image paths into batches.
        
        Args:
            image_paths: List of image paths.
            
        Returns:
            List[List[str]]: List of batches.
        """
        batches = []
        batch_size = self.batch_config.batch_size
        
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def _process_batch_with_retry(self, batch_id: int, batch_paths: List[str]) -> BatchResult:
        """Process a batch with retry logic.
        
        Args:
            batch_id: Batch identifier.
            batch_paths: List of image paths in the batch.
            
        Returns:
            BatchResult: Result of batch processing.
        """
        last_exception = None
        
        for attempt in range(self.batch_config.max_retries):
            try:
                if attempt > 0:
                    self.logger.info(f"Retrying batch {batch_id}, attempt {attempt + 1}")
                    time.sleep(self.batch_config.retry_delay * attempt)
                    
                    # Memory cleanup before retry
                    if self.batch_config.enable_memory_monitoring:
                        self.memory_monitor.cleanup_memory(force=True)
                
                return self._process_batch(batch_id, batch_paths)
                
            except MemoryError as e:
                last_exception = e
                self.logger.warning(f"Memory error in batch {batch_id}, attempt {attempt + 1}: {e}")
                
                # Force memory cleanup and reduce batch size for retry
                self.memory_monitor.cleanup_memory(force=True)
                
                if attempt < self.batch_config.max_retries - 1:
                    # Split batch in half for retry
                    if len(batch_paths) > 1:
                        mid = len(batch_paths) // 2
                        first_half = batch_paths[:mid]
                        second_half = batch_paths[mid:]
                        
                        self.logger.info(f"Splitting batch {batch_id} into smaller batches")
                        
                        # Process smaller batches
                        result1 = self._process_batch_with_retry(f"{batch_id}a", first_half)
                        result2 = self._process_batch_with_retry(f"{batch_id}b", second_half)
                        
                        # Combine results
                        combined_result = BatchResult(
                            batch_id=batch_id,
                            processed_count=result1.processed_count + result2.processed_count,
                            success_count=result1.success_count + result2.success_count,
                            error_count=result1.error_count + result2.error_count,
                            processing_time=result1.processing_time + result2.processing_time,
                            memory_usage_mb=max(result1.memory_usage_mb, result2.memory_usage_mb),
                            results=result1.results + result2.results,
                            errors=result1.errors + result2.errors
                        )
                        
                        return combined_result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Error in batch {batch_id}, attempt {attempt + 1}: {e}")
                
                if not self.error_handler.should_retry(e):
                    break
        
        # All retries failed, create error result
        self.logger.error(f"Batch {batch_id} failed after {self.batch_config.max_retries} attempts")
        
        error_results = []
        for path in batch_paths:
            error_result = ProcessingResult(
                image_path=path,
                filename=path.split('/')[-1],
                final_decision='error',
                error_message=str(last_exception)
            )
            error_results.append(error_result)
        
        return BatchResult(
            batch_id=batch_id,
            processed_count=len(batch_paths),
            success_count=0,
            error_count=len(batch_paths),
            processing_time=0.0,
            memory_usage_mb=self.memory_monitor.get_memory_usage(),
            results=error_results,
            errors=[{
                'batch_id': batch_id,
                'error': str(last_exception),
                'paths': batch_paths
            }]
        )
    
    def _process_batch(self, batch_id: int, batch_paths: List[str]) -> BatchResult:
        """Process a single batch using multi-threading.
        
        Args:
            batch_id: Batch identifier.
            batch_paths: List of image paths in the batch.
            
        Returns:
            BatchResult: Result of batch processing.
        """
        start_time = time.time()
        memory_before = self.memory_monitor.get_memory_usage()
        
        results = []
        errors = []
        success_count = 0
        error_count = 0
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=self.batch_config.max_workers,
                               thread_name_prefix=f"batch_{batch_id}") as executor:
            
            # Submit all tasks
            future_to_path = {}
            for path in batch_paths:
                if self._stop_event.is_set():
                    break
                future = executor.submit(self._process_single_image, path)
                future_to_path[future] = path
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                if self._stop_event.is_set():
                    # Cancel remaining futures
                    for f in future_to_path:
                        if not f.done():
                            f.cancel()
                    break
                
                path = future_to_path[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.final_decision == 'error':
                        error_count += 1
                        errors.append({
                            'path': path,
                            'error': result.error_message,
                            'batch_id': batch_id
                        })
                    else:
                        success_count += 1
                        
                except Exception as e:
                    error_count += 1
                    error_message = str(e)
                    
                    # Create error result
                    error_result = ProcessingResult(
                        image_path=path,
                        filename=path.split('/')[-1],
                        final_decision='error',
                        error_message=error_message
                    )
                    results.append(error_result)
                    
                    errors.append({
                        'path': path,
                        'error': error_message,
                        'batch_id': batch_id,
                        'traceback': traceback.format_exc()
                    })
                    
                    self.logger.warning(f"Error processing {path}: {error_message}")
        
        processing_time = time.time() - start_time
        memory_after = self.memory_monitor.get_memory_usage()
        
        batch_result = BatchResult(
            batch_id=batch_id,
            processed_count=len(batch_paths),
            success_count=success_count,
            error_count=error_count,
            processing_time=processing_time,
            memory_usage_mb=memory_after,
            results=results,
            errors=errors
        )
        
        return batch_result
    
    def _process_single_image(self, image_path: str) -> ProcessingResult:
        """Process a single image with error handling.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            ProcessingResult: Result of processing.
        """
        try:
            start_time = time.time()
            
            # Call the provided processing function
            result = self.processing_function(image_path)
            
            # Update processing time
            result.processing_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            error_message = str(e)
            self.error_handler.log_error(e, {'image_path': image_path})
            
            return ProcessingResult(
                image_path=image_path,
                filename=image_path.split('/')[-1],
                final_decision='error',
                error_message=error_message,
                processing_time=time.time() - start_time if 'start_time' in locals() else 0.0
            )
    
    def stop_processing(self):
        """Stop batch processing gracefully."""
        self.logger.info("Stopping batch processing...")
        self._stop_event.set()
    
    def cleanup_memory(self) -> Tuple[float, float]:
        """Force memory cleanup.
        
        Returns:
            Tuple[float, float]: Memory usage before and after cleanup.
        """
        return self.memory_monitor.cleanup_memory(force=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics.
        
        Returns:
            Dict[str, Any]: Processing statistics.
        """
        if not self.batch_history:
            return {
                'total_processed': 0,
                'total_batches': 0,
                'total_errors': 0,
                'success_rate': 0.0,
                'avg_batch_time': 0.0,
                'avg_memory_usage': 0.0
            }
        
        total_time = sum(batch.processing_time for batch in self.batch_history)
        total_memory = sum(batch.memory_usage_mb for batch in self.batch_history)
        
        return {
            'total_processed': self.total_processed,
            'total_batches': self.total_batches,
            'total_errors': self.total_errors,
            'success_rate': ((self.total_processed - self.total_errors) / self.total_processed * 100) 
                           if self.total_processed > 0 else 0.0,
            'avg_batch_time': total_time / len(self.batch_history),
            'avg_memory_usage': total_memory / len(self.batch_history),
            'current_memory_usage': self.memory_monitor.get_memory_usage(),
            'batch_history': [
                {
                    'batch_id': batch.batch_id,
                    'processed_count': batch.processed_count,
                    'success_count': batch.success_count,
                    'error_count': batch.error_count,
                    'processing_time': batch.processing_time,
                    'memory_usage_mb': batch.memory_usage_mb
                }
                for batch in self.batch_history
            ]
        }
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB.
        
        Returns:
            float: Current memory usage in MB.
        """
        return self.memory_monitor.get_memory_usage()
    
    def configure_batch_size(self, new_batch_size: int):
        """Dynamically configure batch size.
        
        Args:
            new_batch_size: New batch size to use.
        """
        if new_batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        old_size = self.batch_config.batch_size
        self.batch_config.batch_size = new_batch_size
        
        self.logger.info(f"Batch size changed from {old_size} to {new_batch_size}")
    
    def configure_workers(self, new_max_workers: int):
        """Dynamically configure number of worker threads.
        
        Args:
            new_max_workers: New maximum number of worker threads.
        """
        if new_max_workers <= 0:
            raise ValueError("Max workers must be positive")
        
        old_workers = self.batch_config.max_workers
        self.batch_config.max_workers = new_max_workers
        
        self.logger.info(f"Max workers changed from {old_workers} to {new_max_workers}")