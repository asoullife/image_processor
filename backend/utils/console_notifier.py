"""Console notification system for real-time progress updates."""

import sys
import time
import threading
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConsoleNotificationConfig:
    """Configuration for console notifications."""
    show_progress_bar: bool = True
    show_statistics: bool = True
    show_performance_metrics: bool = True
    show_milestones: bool = True
    update_interval: float = 2.0  # seconds
    progress_bar_width: int = 50
    enable_colors: bool = True

class ConsoleColors:
    """ANSI color codes for console output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    
    # Background colors
    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_BLUE = '\033[104m'

class ConsoleNotifier:
    """Real-time console notifications for image processing."""
    
    def __init__(self, config: Optional[ConsoleNotificationConfig] = None):
        """Initialize console notifier.
        
        Args:
            config: Notification configuration.
        """
        self.config = config or ConsoleNotificationConfig()
        self.is_active = False
        self.current_session_id: Optional[str] = None
        self.session_start_time: Optional[datetime] = None
        self.last_update_time: Optional[datetime] = None
        self.notification_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Progress tracking
        self.current_progress = 0
        self.total_progress = 0
        self.approved_count = 0
        self.rejected_count = 0
        self.processing_speed = 0.0
        self.current_filename = ""
        self.current_stage = ""
        self.performance_metrics: Dict[str, Any] = {}
        
        # Milestone tracking
        self.milestones_reached: set = set()
        
        # Console state
        self.last_line_length = 0
        
        logger.info("Console notifier initialized")
    
    def start_session(self, session_id: str, total_images: int):
        """Start console notifications for a processing session.
        
        Args:
            session_id: Session identifier.
            total_images: Total number of images to process.
        """
        with self._lock:
            self.current_session_id = session_id
            self.session_start_time = datetime.now()
            self.last_update_time = datetime.now()
            self.total_progress = total_images
            self.current_progress = 0
            self.approved_count = 0
            self.rejected_count = 0
            self.processing_speed = 0.0
            self.current_filename = ""
            self.current_stage = "initializing"
            self.performance_metrics = {}
            self.milestones_reached = set()
            self.is_active = True
        
        # Print session header
        self._print_session_header(session_id, total_images)
        
        # Start notification thread
        if not self.notification_thread or not self.notification_thread.is_alive():
            self.notification_thread = threading.Thread(
                target=self._notification_loop,
                name="ConsoleNotifier",
                daemon=True
            )
            self.notification_thread.start()
        
        logger.info(f"Started console notifications for session {session_id}")
    
    def update_progress(self, current: int, approved: int, rejected: int, 
                       speed: float, filename: str = "", stage: str = "",
                       performance_metrics: Optional[Dict[str, Any]] = None):
        """Update progress information.
        
        Args:
            current: Current number of processed images.
            approved: Number of approved images.
            rejected: Number of rejected images.
            speed: Processing speed (images per second).
            filename: Currently processing filename.
            stage: Current processing stage.
            performance_metrics: Performance metrics dictionary.
        """
        with self._lock:
            self.current_progress = current
            self.approved_count = approved
            self.rejected_count = rejected
            self.processing_speed = speed
            self.current_filename = filename
            self.current_stage = stage
            self.performance_metrics = performance_metrics or {}
            self.last_update_time = datetime.now()
    
    def notify_milestone(self, milestone_type: str, milestone_value: str, message: str):
        """Notify about a milestone reached.
        
        Args:
            milestone_type: Type of milestone (percentage, count, time).
            milestone_value: Milestone value (25%, 1000_images, etc.).
            message: Milestone message.
        """
        milestone_key = f"{milestone_type}_{milestone_value}"
        if milestone_key in self.milestones_reached:
            return
        
        self.milestones_reached.add(milestone_key)
        
        if self.config.show_milestones:
            self._print_milestone(milestone_type, milestone_value, message)
    
    def notify_completion(self, total_processed: int, total_approved: int, 
                         total_rejected: int, processing_time: float,
                         final_metrics: Optional[Dict[str, Any]] = None):
        """Notify about processing completion.
        
        Args:
            total_processed: Total images processed.
            total_approved: Total images approved.
            total_rejected: Total images rejected.
            processing_time: Total processing time in seconds.
            final_metrics: Final performance metrics.
        """
        self.is_active = False
        self._print_completion_summary(
            total_processed, total_approved, total_rejected, 
            processing_time, final_metrics
        )
    
    def notify_error(self, error_message: str, recoverable: bool = True):
        """Notify about an error.
        
        Args:
            error_message: Error message.
            recoverable: Whether the error is recoverable.
        """
        self._print_error(error_message, recoverable)
    
    def stop_session(self):
        """Stop console notifications for current session."""
        with self._lock:
            self.is_active = False
            self.current_session_id = None
        
        # Wait for notification thread to finish
        if self.notification_thread and self.notification_thread.is_alive():
            self.notification_thread.join(timeout=1.0)
        
        logger.info("Stopped console notifications")
    
    def _notification_loop(self):
        """Main notification loop running in background thread."""
        while self.is_active:
            try:
                if self.config.show_progress_bar or self.config.show_statistics:
                    self._update_console_display()
                
                time.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"Error in console notification loop: {e}")
                time.sleep(1.0)
    
    def _update_console_display(self):
        """Update the console display with current progress."""
        with self._lock:
            if not self.is_active or not self.current_session_id:
                return
            
            # Clear previous line
            if self.last_line_length > 0:
                sys.stdout.write('\r' + ' ' * self.last_line_length + '\r')
            
            # Build progress line
            progress_line = self._build_progress_line()
            
            # Print progress line
            sys.stdout.write(progress_line)
            sys.stdout.flush()
            
            self.last_line_length = len(progress_line)
    
    def _build_progress_line(self) -> str:
        """Build the progress line string."""
        parts = []
        
        # Progress indicator
        if self.config.show_progress_bar and self.total_progress > 0:
            percentage = (self.current_progress / self.total_progress) * 100
            filled_width = int((percentage / 100) * self.config.progress_bar_width)
            bar = '█' * filled_width + '░' * (self.config.progress_bar_width - filled_width)
            
            if self.config.enable_colors:
                bar_color = ConsoleColors.GREEN if percentage >= 100 else ConsoleColors.BLUE
                parts.append(f"{bar_color}{bar}{ConsoleColors.RESET}")
            else:
                parts.append(bar)
            
            parts.append(f" {percentage:.1f}%")
        
        # Current progress
        if self.config.enable_colors:
            parts.append(f" {ConsoleColors.CYAN}{self.current_progress:,}/{self.total_progress:,}{ConsoleColors.RESET}")
        else:
            parts.append(f" {self.current_progress:,}/{self.total_progress:,}")
        
        # Statistics
        if self.config.show_statistics:
            if self.config.enable_colors:
                parts.append(f" | {ConsoleColors.GREEN}✓{self.approved_count:,}{ConsoleColors.RESET}")
                parts.append(f" {ConsoleColors.RED}✗{self.rejected_count:,}{ConsoleColors.RESET}")
            else:
                parts.append(f" | ✓{self.approved_count:,} ✗{self.rejected_count:,}")
        
        # Processing speed
        if self.processing_speed > 0:
            if self.config.enable_colors:
                parts.append(f" | {ConsoleColors.YELLOW}{self.processing_speed:.1f} img/s{ConsoleColors.RESET}")
            else:
                parts.append(f" | {self.processing_speed:.1f} img/s")
        
        # ETA
        if self.processing_speed > 0 and self.total_progress > self.current_progress:
            remaining = self.total_progress - self.current_progress
            eta_seconds = remaining / self.processing_speed
            eta_str = self._format_duration(eta_seconds)
            
            if self.config.enable_colors:
                parts.append(f" | ETA: {ConsoleColors.MAGENTA}{eta_str}{ConsoleColors.RESET}")
            else:
                parts.append(f" | ETA: {eta_str}")
        
        # Performance metrics
        if self.config.show_performance_metrics and self.performance_metrics:
            memory_mb = self.performance_metrics.get('current_memory_mb', 0)
            gpu_percent = self.performance_metrics.get('current_gpu_percent', 0)
            
            if memory_mb > 0:
                if self.config.enable_colors:
                    parts.append(f" | RAM: {ConsoleColors.BLUE}{memory_mb:.0f}MB{ConsoleColors.RESET}")
                else:
                    parts.append(f" | RAM: {memory_mb:.0f}MB")
            
            if gpu_percent > 0:
                if self.config.enable_colors:
                    parts.append(f" GPU: {ConsoleColors.GREEN}{gpu_percent:.0f}%{ConsoleColors.RESET}")
                else:
                    parts.append(f" GPU: {gpu_percent:.0f}%")
        
        # Current stage
        if self.current_stage:
            if self.config.enable_colors:
                parts.append(f" | {ConsoleColors.DIM}{self.current_stage}{ConsoleColors.RESET}")
            else:
                parts.append(f" | {self.current_stage}")
        
        return ''.join(parts)
    
    def _print_session_header(self, session_id: str, total_images: int):
        """Print session start header."""
        print("\n" + "="*80)
        if self.config.enable_colors:
            print(f"{ConsoleColors.BOLD}{ConsoleColors.CYAN}🚀 Adobe Stock Image Processor{ConsoleColors.RESET}")
        else:
            print("🚀 Adobe Stock Image Processor")
        
        print(f"Session ID: {session_id}")
        print(f"Total Images: {total_images:,}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
    
    def _print_milestone(self, milestone_type: str, milestone_value: str, message: str):
        """Print milestone notification."""
        # Clear current progress line
        if self.last_line_length > 0:
            sys.stdout.write('\r' + ' ' * self.last_line_length + '\r')
            self.last_line_length = 0
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if self.config.enable_colors:
            print(f"\n{ConsoleColors.BOLD}{ConsoleColors.YELLOW}🎯 MILESTONE REACHED{ConsoleColors.RESET} [{timestamp}]")
            print(f"{ConsoleColors.GREEN}{message}{ConsoleColors.RESET}")
        else:
            print(f"\n🎯 MILESTONE REACHED [{timestamp}]")
            print(message)
        
        # Show performance snapshot if available
        if self.performance_metrics:
            memory_mb = self.performance_metrics.get('current_memory_mb', 0)
            gpu_percent = self.performance_metrics.get('current_gpu_percent', 0)
            
            perf_parts = []
            if memory_mb > 0:
                perf_parts.append(f"RAM: {memory_mb:.0f}MB")
            if gpu_percent > 0:
                perf_parts.append(f"GPU: {gpu_percent:.0f}%")
            if self.processing_speed > 0:
                perf_parts.append(f"Speed: {self.processing_speed:.1f} img/s")
            
            if perf_parts:
                perf_str = " | ".join(perf_parts)
                if self.config.enable_colors:
                    print(f"{ConsoleColors.DIM}Performance: {perf_str}{ConsoleColors.RESET}")
                else:
                    print(f"Performance: {perf_str}")
        
        print()  # Add blank line
    
    def _print_completion_summary(self, total_processed: int, total_approved: int, 
                                total_rejected: int, processing_time: float,
                                final_metrics: Optional[Dict[str, Any]] = None):
        """Print completion summary."""
        # Clear current progress line
        if self.last_line_length > 0:
            sys.stdout.write('\r' + ' ' * self.last_line_length + '\r')
            self.last_line_length = 0
        
        print("\n" + "="*80)
        if self.config.enable_colors:
            print(f"{ConsoleColors.BOLD}{ConsoleColors.GREEN}✅ PROCESSING COMPLETED{ConsoleColors.RESET}")
        else:
            print("✅ PROCESSING COMPLETED")
        
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Processing Time: {self._format_duration(processing_time)}")
        print()
        
        # Statistics
        approval_rate = (total_approved / total_processed * 100) if total_processed > 0 else 0
        
        if self.config.enable_colors:
            print(f"📊 {ConsoleColors.BOLD}FINAL STATISTICS{ConsoleColors.RESET}")
            print(f"   Total Processed: {ConsoleColors.CYAN}{total_processed:,}{ConsoleColors.RESET}")
            print(f"   Approved: {ConsoleColors.GREEN}{total_approved:,} ({approval_rate:.1f}%){ConsoleColors.RESET}")
            print(f"   Rejected: {ConsoleColors.RED}{total_rejected:,} ({100-approval_rate:.1f}%){ConsoleColors.RESET}")
        else:
            print("📊 FINAL STATISTICS")
            print(f"   Total Processed: {total_processed:,}")
            print(f"   Approved: {total_approved:,} ({approval_rate:.1f}%)")
            print(f"   Rejected: {total_rejected:,} ({100-approval_rate:.1f}%)")
        
        # Performance metrics
        if final_metrics:
            avg_speed = final_metrics.get('avg_images_per_second', 0)
            peak_memory = final_metrics.get('peak_memory_mb', 0)
            peak_gpu = final_metrics.get('peak_gpu_usage', 0)
            
            print()
            if self.config.enable_colors:
                print(f"⚡ {ConsoleColors.BOLD}PERFORMANCE SUMMARY{ConsoleColors.RESET}")
                if avg_speed > 0:
                    print(f"   Average Speed: {ConsoleColors.YELLOW}{avg_speed:.1f} images/second{ConsoleColors.RESET}")
                if peak_memory > 0:
                    print(f"   Peak Memory Usage: {ConsoleColors.BLUE}{peak_memory:.0f} MB{ConsoleColors.RESET}")
                if peak_gpu > 0:
                    print(f"   Peak GPU Usage: {ConsoleColors.GREEN}{peak_gpu:.0f}%{ConsoleColors.RESET}")
            else:
                print("⚡ PERFORMANCE SUMMARY")
                if avg_speed > 0:
                    print(f"   Average Speed: {avg_speed:.1f} images/second")
                if peak_memory > 0:
                    print(f"   Peak Memory Usage: {peak_memory:.0f} MB")
                if peak_gpu > 0:
                    print(f"   Peak GPU Usage: {peak_gpu:.0f}%")
        
        print("="*80)
    
    def _print_error(self, error_message: str, recoverable: bool):
        """Print error notification."""
        # Clear current progress line
        if self.last_line_length > 0:
            sys.stdout.write('\r' + ' ' * self.last_line_length + '\r')
            self.last_line_length = 0
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if self.config.enable_colors:
            error_color = ConsoleColors.YELLOW if recoverable else ConsoleColors.RED
            print(f"\n{error_color}{'⚠️  WARNING' if recoverable else '❌ ERROR'}{ConsoleColors.RESET} [{timestamp}]")
            print(f"{error_color}{error_message}{ConsoleColors.RESET}")
            if recoverable:
                print(f"{ConsoleColors.DIM}Processing will continue...{ConsoleColors.RESET}")
        else:
            print(f"\n{'⚠️  WARNING' if recoverable else '❌ ERROR'} [{timestamp}]")
            print(error_message)
            if recoverable:
                print("Processing will continue...")
        
        print()  # Add blank line
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {minutes}m {secs}s"

# Global console notifier instance
console_notifier = ConsoleNotifier()