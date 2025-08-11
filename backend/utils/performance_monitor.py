"""Performance monitoring utilities for real-time metrics collection."""

import psutil
import time
import threading
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import GPUtil

logger = logging.getLogger(__name__)

@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_percent: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_memory_percent: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0

@dataclass
class ProcessingMetrics:
    """Processing performance metrics."""
    session_id: str
    start_time: datetime
    images_processed: int = 0
    images_per_second: float = 0.0
    avg_processing_time: float = 0.0
    batch_processing_times: List[float] = field(default_factory=list)
    peak_memory_mb: float = 0.0
    peak_gpu_usage: float = 0.0
    total_processing_time: float = 0.0
    snapshots: List[PerformanceSnapshot] = field(default_factory=list)

class PerformanceMonitor:
    """Real-time performance monitoring for image processing."""
    
    def __init__(self, update_interval: float = 1.0):
        """Initialize performance monitor.
        
        Args:
            update_interval: Seconds between performance measurements.
        """
        self.update_interval = update_interval
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.session_metrics: Dict[str, ProcessingMetrics] = {}
        self._lock = threading.Lock()
        
        # Initialize baseline measurements
        self.process = psutil.Process()
        self.baseline_disk_io = self.process.io_counters()
        self.baseline_network = psutil.net_io_counters()
        
        # GPU detection
        self.has_gpu = self._detect_gpu()
        
        logger.info(f"Performance monitor initialized (GPU: {'available' if self.has_gpu else 'not available'})")
    
    def _detect_gpu(self) -> bool:
        """Detect if GPU is available for monitoring."""
        try:
            gpus = GPUtil.getGPUs()
            return len(gpus) > 0
        except Exception as e:
            logger.debug(f"GPU detection failed: {e}")
            return False
    
    def start_monitoring(self):
        """Start performance monitoring in background thread."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceMonitor",
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread."""
        while self.is_monitoring:
            try:
                snapshot = self._take_snapshot()
                
                # Update all active session metrics
                with self._lock:
                    for session_id, metrics in self.session_metrics.items():
                        metrics.snapshots.append(snapshot)
                        
                        # Keep only last 100 snapshots per session
                        if len(metrics.snapshots) > 100:
                            metrics.snapshots = metrics.snapshots[-100:]
                        
                        # Update peak values
                        metrics.peak_memory_mb = max(metrics.peak_memory_mb, snapshot.memory_mb)
                        metrics.peak_gpu_usage = max(metrics.peak_gpu_usage, snapshot.gpu_percent)
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _take_snapshot(self) -> PerformanceSnapshot:
        """Take a single performance measurement snapshot."""
        try:
            # CPU and Memory
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = self.process.memory_percent()
            
            # GPU metrics
            gpu_percent = 0.0
            gpu_memory_mb = 0.0
            gpu_memory_percent = 0.0
            
            if self.has_gpu:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        gpu_percent = gpu.load * 100
                        gpu_memory_mb = gpu.memoryUsed
                        gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
                except Exception as e:
                    logger.debug(f"GPU metrics collection failed: {e}")
            
            # Disk I/O
            current_disk_io = self.process.io_counters()
            disk_read_mb = (current_disk_io.read_bytes - self.baseline_disk_io.read_bytes) / 1024 / 1024
            disk_write_mb = (current_disk_io.write_bytes - self.baseline_disk_io.write_bytes) / 1024 / 1024
            
            # Network I/O
            current_network = psutil.net_io_counters()
            network_sent_mb = (current_network.bytes_sent - self.baseline_network.bytes_sent) / 1024 / 1024
            network_recv_mb = (current_network.bytes_recv - self.baseline_network.bytes_recv) / 1024 / 1024
            
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                gpu_percent=gpu_percent,
                gpu_memory_mb=gpu_memory_mb,
                gpu_memory_percent=gpu_memory_percent,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb
            )
            
        except Exception as e:
            logger.error(f"Error taking performance snapshot: {e}")
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_percent=0.0
            )
    
    def start_session(self, session_id: str) -> ProcessingMetrics:
        """Start monitoring a processing session.
        
        Args:
            session_id: Unique session identifier.
            
        Returns:
            ProcessingMetrics: Metrics object for the session.
        """
        with self._lock:
            metrics = ProcessingMetrics(
                session_id=session_id,
                start_time=datetime.now()
            )
            self.session_metrics[session_id] = metrics
            
            # Start monitoring if not already running
            if not self.is_monitoring:
                self.start_monitoring()
            
            logger.info(f"Started performance monitoring for session {session_id}")
            return metrics
    
    def update_session_progress(self, session_id: str, images_processed: int, 
                              processing_time: float = 0.0, batch_time: float = 0.0):
        """Update session processing progress.
        
        Args:
            session_id: Session identifier.
            images_processed: Total images processed so far.
            processing_time: Time taken for last image/batch.
            batch_time: Time taken for last batch.
        """
        with self._lock:
            if session_id not in self.session_metrics:
                logger.warning(f"Session {session_id} not found in metrics")
                return
            
            metrics = self.session_metrics[session_id]
            metrics.images_processed = images_processed
            metrics.total_processing_time += processing_time
            
            if batch_time > 0:
                metrics.batch_processing_times.append(batch_time)
                # Keep only last 50 batch times
                if len(metrics.batch_processing_times) > 50:
                    metrics.batch_processing_times = metrics.batch_processing_times[-50:]
            
            # Calculate processing speed
            elapsed_time = (datetime.now() - metrics.start_time).total_seconds()
            if elapsed_time > 0:
                metrics.images_per_second = images_processed / elapsed_time
            
            # Calculate average processing time
            if images_processed > 0:
                metrics.avg_processing_time = metrics.total_processing_time / images_processed
    
    def get_session_metrics(self, session_id: str) -> Optional[ProcessingMetrics]:
        """Get current metrics for a session.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            ProcessingMetrics: Current metrics or None if session not found.
        """
        with self._lock:
            return self.session_metrics.get(session_id)
    
    def get_current_performance(self, session_id: str) -> Dict[str, Any]:
        """Get current performance metrics for a session.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            Dict containing current performance data.
        """
        with self._lock:
            metrics = self.session_metrics.get(session_id)
            if not metrics:
                return {}
            
            # Get latest snapshot
            latest_snapshot = metrics.snapshots[-1] if metrics.snapshots else None
            if not latest_snapshot:
                latest_snapshot = self._take_snapshot()
            
            elapsed_time = (datetime.now() - metrics.start_time).total_seconds()
            
            return {
                "session_id": session_id,
                "elapsed_time": elapsed_time,
                "images_processed": metrics.images_processed,
                "images_per_second": metrics.images_per_second,
                "avg_processing_time": metrics.avg_processing_time,
                "current_cpu_percent": latest_snapshot.cpu_percent,
                "current_memory_mb": latest_snapshot.memory_mb,
                "current_memory_percent": latest_snapshot.memory_percent,
                "current_gpu_percent": latest_snapshot.gpu_percent,
                "current_gpu_memory_mb": latest_snapshot.gpu_memory_mb,
                "current_gpu_memory_percent": latest_snapshot.gpu_memory_percent,
                "peak_memory_mb": metrics.peak_memory_mb,
                "peak_gpu_usage": metrics.peak_gpu_usage,
                "avg_batch_time": sum(metrics.batch_processing_times) / len(metrics.batch_processing_times) if metrics.batch_processing_times else 0.0,
                "disk_io_read_mb": latest_snapshot.disk_io_read_mb,
                "disk_io_write_mb": latest_snapshot.disk_io_write_mb,
                "network_sent_mb": latest_snapshot.network_sent_mb,
                "network_recv_mb": latest_snapshot.network_recv_mb
            }
    
    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """End monitoring for a session and return final metrics.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            Dict containing final session metrics.
        """
        with self._lock:
            metrics = self.session_metrics.pop(session_id, None)
            if not metrics:
                return None
            
            elapsed_time = (datetime.now() - metrics.start_time).total_seconds()
            
            final_metrics = {
                "session_id": session_id,
                "total_elapsed_time": elapsed_time,
                "total_images_processed": metrics.images_processed,
                "avg_images_per_second": metrics.images_per_second,
                "avg_processing_time": metrics.avg_processing_time,
                "peak_memory_mb": metrics.peak_memory_mb,
                "peak_gpu_usage": metrics.peak_gpu_usage,
                "total_processing_time": metrics.total_processing_time,
                "avg_batch_time": sum(metrics.batch_processing_times) / len(metrics.batch_processing_times) if metrics.batch_processing_times else 0.0,
                "total_snapshots": len(metrics.snapshots)
            }
            
            logger.info(f"Ended performance monitoring for session {session_id}")
            
            # Stop monitoring if no active sessions
            if not self.session_metrics:
                self.stop_monitoring()
            
            return final_metrics
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for performance context.
        
        Returns:
            Dict containing system information.
        """
        try:
            # CPU info
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory info
            memory = psutil.virtual_memory()
            
            # GPU info
            gpu_info = []
            if self.has_gpu:
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_info.append({
                            "name": gpu.name,
                            "memory_total": gpu.memoryTotal,
                            "driver": gpu.driver
                        })
                except Exception as e:
                    logger.debug(f"GPU info collection failed: {e}")
            
            return {
                "cpu_count": cpu_count,
                "cpu_freq_mhz": cpu_freq.current if cpu_freq else None,
                "memory_total_gb": memory.total / 1024 / 1024 / 1024,
                "gpu_available": self.has_gpu,
                "gpu_info": gpu_info,
                "platform": psutil.WINDOWS if hasattr(psutil, 'WINDOWS') else "unknown"
            }
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}

# Global performance monitor instance
performance_monitor = PerformanceMonitor()