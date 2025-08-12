"""Main entry point for Adobe Stock Image Processor."""

import sys
import os
import argparse
import time
import signal
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from threading import Event
from tqdm import tqdm
from pathlib import Path

# Add project root to Python path for integrated execution
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Add backend root to Python path for standalone execution
backend_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_root)

# Try importing with backend prefix first (integrated mode), then without (standalone mode)
try:
    from backend.config.config_loader import load_config, AppConfig
    from backend.utils.logger import initialize_logging, get_logger, ProgressLogger
    from backend.core.base import ErrorHandler
    from backend.core.progress_tracker import PostgresProgressTracker
    from backend.core.batch_processor import BatchProcessor
    from backend.utils.file_manager import FileManager
    from backend.analyzers.quality_analyzer import QualityAnalyzer
    from backend.analyzers.defect_detector import DefectDetector
    from backend.analyzers.similarity_finder import SimilarityFinder
    from backend.analyzers.compliance_checker import ComplianceChecker
    from backend.core.decision_engine import DecisionEngine
    from backend.utils.report_generator import ReportGenerator
except ImportError:
    # Standalone mode - use relative imports
    from config.config_loader import load_config, AppConfig
    from utils.logger import initialize_logging, get_logger, ProgressLogger
    from core.base import ErrorHandler
    from core.progress_tracker import PostgresProgressTracker
    from core.batch_processor import BatchProcessor
    from utils.file_manager import FileManager
    from analyzers.quality_analyzer import QualityAnalyzer
    from analyzers.defect_detector import DefectDetector
    from analyzers.similarity_finder import SimilarityFinder
    from analyzers.compliance_checker import ComplianceChecker
    from core.decision_engine import DecisionEngine
    from utils.report_generator import ReportGenerator


class ImageProcessor:
    """Main image processor application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the image processor.
        
        Args:
            config_path: Path to configuration file.
        """
        try:
            # Load configuration
            self.config = load_config(config_path)
            
            # Initialize logging
            self.logger = initialize_logging(self.config.logging)
            self.progress_logger = ProgressLogger("ImageProcessor")
            
            # Initialize error handler
            self.error_handler = ErrorHandler()
            
            # Initialize shutdown event for graceful termination
            self.shutdown_event = Event()
            self._setup_signal_handlers()
            
            # Initialize core components
            self.progress_tracker = PostgresProgressTracker(
                checkpoint_interval=self.config.processing.checkpoint_interval
            )
            
            # Convert config to dict for components
            self.config_dict = {
                'processing': {
                    'batch_size': self.config.processing.batch_size,
                    'max_workers': self.config.processing.max_workers,
                    'checkpoint_interval': self.config.processing.checkpoint_interval
                },
                'output': {
                    'images_per_folder': self.config.output.images_per_folder,
                    'preserve_metadata': self.config.output.preserve_metadata,
                    'generate_thumbnails': self.config.output.generate_thumbnails
                },
                'quality': {
                    'min_sharpness': self.config.quality.min_sharpness,
                    'max_noise_level': self.config.quality.max_noise_level,
                    'min_resolution': self.config.quality.min_resolution
                },
                'similarity': {
                    'hash_threshold': self.config.similarity.hash_threshold,
                    'feature_threshold': self.config.similarity.feature_threshold,
                    'clustering_eps': self.config.similarity.clustering_eps
                },
                'compliance': {
                    'logo_detection_confidence': self.config.compliance.logo_detection_confidence,
                    'face_detection_enabled': self.config.compliance.face_detection_enabled,
                    'metadata_validation': self.config.compliance.metadata_validation
                }
            }
            
            # Initialize processing components
            self.file_manager = FileManager(self.config_dict['output']['images_per_folder'])
            
            # BatchProcessor will be initialized when needed with processing function
            self.batch_processor = None
            
            # Initialize analyzers
            self.quality_analyzer = QualityAnalyzer(self.config_dict)
            self.defect_detector = DefectDetector(self.config_dict)
            self.similarity_finder = SimilarityFinder(self.config_dict)
            self.compliance_checker = ComplianceChecker(self.config_dict)
            
            # Initialize decision engine and report generator
            self.decision_engine = DecisionEngine(self.config_dict)
            self.report_generator = ReportGenerator(self.config_dict)
            
            # Progress tracking variables
            self.start_time = None
            self.processed_count = 0
            self.total_count = 0
            self.approved_count = 0
            self.rejected_count = 0
            
            self.logger.info("Adobe Stock Image Processor initialized")
            self.logger.info(f"Configuration loaded from: {config_path or 'default'}")
            
        except Exception as e:
            print(f"Failed to initialize Image Processor: {e}")
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            sys.exit(1)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def run(self, input_folder: str, output_folder: str, resume: bool = False):
        """Run the image processing pipeline.
        
        Args:
            input_folder: Path to input folder containing images.
            output_folder: Path to output folder for processed images.
            resume: Whether to resume from previous session.
        """
        self.logger.info("Starting image processing pipeline")
        self.logger.info(f"Input folder: {input_folder}")
        self.logger.info(f"Output folder: {output_folder}")
        self.logger.info(f"Resume mode: {resume}")
        
        # Validate input parameters
        if not self._validate_folders(input_folder, output_folder):
            return False
        
        # Log configuration summary
        self._log_config_summary()
        
        session_id = None
        try:
            # Handle resume functionality
            start_index = 0
            
            if resume:
                session_id, start_index = self._handle_resume_request(input_folder, output_folder)
                if session_id is None:
                    self.logger.info("No resumable session found, starting fresh")
                    resume = False
            
            # Get list of images to process
            print("🔍 Scanning for images...")
            all_image_paths = self.file_manager.scan_images(input_folder)
            
            if not all_image_paths:
                print("⚠️  No images found to process")
                self.logger.warning("No images found to process")
                return True
            
            print(f"✅ Found {len(all_image_paths)} images to process")
            self.logger.info(f"Found {len(all_image_paths)} images to process")
            
            # Initialize progress tracking
            self.total_count = len(all_image_paths)
            self.processed_count = start_index
            self.start_time = datetime.now()
            
            # Create new session if not resuming
            if not resume:
                session_id = self.progress_tracker.create_session(
                    input_folder=input_folder,
                    output_folder=output_folder,
                    total_images=len(all_image_paths),
                    config=self.config_dict
                )
                self.logger.info(f"Created new session: {session_id}")
            else:
                print(f"📄 Resuming session: {session_id}")
                print(f"🔄 Starting from image {start_index + 1}/{len(all_image_paths)}")
                self.logger.info(f"Resuming session {session_id} from image {start_index + 1}")
            
            # Process images starting from the appropriate index
            images_to_process = all_image_paths[start_index:] if resume else all_image_paths
            
            # Run the complete processing pipeline
            success = self._run_processing_pipeline(
                session_id, images_to_process, output_folder, start_index
            )
            
            if success and not self.shutdown_event.is_set():
                # Generate final reports
                print("\n📊 Generating reports...")
                self._generate_final_reports(session_id, output_folder)
                
                # Complete session
                self.progress_tracker.complete_session(session_id, 'completed')
                
                # Display final summary
                self._display_final_summary()
                
                print("✅ Processing completed successfully!")
                self.logger.info("Processing completed successfully")
                return True
            else:
                if self.shutdown_event.is_set():
                    print("\n⏹️  Processing stopped by user")
                    self.progress_tracker.complete_session(session_id, 'cancelled', 'Stopped by user')
                    self.logger.info("Processing cancelled by user")
                else:
                    print("\n❌ Processing failed")
                    self.progress_tracker.complete_session(session_id, 'failed', 'Processing pipeline failed')
                    self.logger.error("Processing pipeline failed")
                return False
            
        except KeyboardInterrupt:
            print("\n⏹️  Processing interrupted by user")
            self.logger.info("Processing interrupted by user")
            if session_id:
                self.progress_tracker.complete_session(session_id, 'cancelled', 'Interrupted by user')
            return False
            
        except Exception as e:
            print(f"\n❌ Processing failed: {e}")
            self.logger.error(f"Processing failed: {e}", exc_info=True)
            if session_id:
                self.progress_tracker.complete_session(session_id, 'failed', str(e))
            return False
    
    def _run_processing_pipeline(self, session_id: str, image_paths: List[str], 
                               output_folder: str, start_index: int = 0) -> bool:
        """Run the complete image processing pipeline with real-time progress.
        
        Args:
            session_id: Session identifier.
            image_paths: List of image paths to process.
            output_folder: Output folder path.
            start_index: Starting index for progress display.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not image_paths:
            return True
        
        print(f"\n🚀 Starting processing pipeline...")
        print(f"📁 Processing {len(image_paths)} images")
        print(f"⚙️  Batch size: {self.config.processing.batch_size}")
        print(f"🧵 Workers: {self.config.processing.max_workers}")
        print(f"💾 Checkpoint interval: {self.config.processing.checkpoint_interval}")
        
        # Initialize progress bar
        progress_bar = tqdm(
            total=len(image_paths),
            desc="Processing images",
            unit="img",
            initial=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        
        try:
            # Initialize batch processor with processing function
            if self.batch_processor is None:
                def processing_function(image_path: str):
                    return self._process_single_image(image_path, {})
                
                self.batch_processor = BatchProcessor(
                    config=self.config_dict,
                    processing_function=processing_function,
                    session_id=session_id
                )
            
            # Phase 1: Similarity analysis (needs all images)
            print("\n🔍 Phase 1: Analyzing image similarity...")
            similarity_groups = self._analyze_similarity(image_paths)
            
            # Phase 2: Process images in batches
            print(f"\n⚡ Phase 2: Processing images in batches...")
            
            batch_size = self.config.processing.batch_size
            approved_images = []
            processing_results = []
            
            for batch_start in range(0, len(image_paths), batch_size):
                if self.shutdown_event.is_set():
                    break
                
                batch_end = min(batch_start + batch_size, len(image_paths))
                batch_paths = image_paths[batch_start:batch_end]
                
                # Process batch
                batch_results = self._process_image_batch(
                    batch_paths, similarity_groups, start_index + batch_start
                )
                
                if batch_results is None:  # Error occurred
                    return False
                
                processing_results.extend(batch_results)
                
                # Update progress and statistics
                for result in batch_results:
                    self.processed_count += 1
                    if result.final_decision == 'approved':
                        self.approved_count += 1
                        approved_images.append(result.image_path)
                    else:
                        self.rejected_count += 1
                    
                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'Approved': self.approved_count,
                        'Rejected': self.rejected_count,
                        'Rate': f"{self.approved_count/(self.processed_count)*100:.1f}%"
                    })
                
                # Save checkpoint
                if self.processed_count % self.config.processing.checkpoint_interval == 0:
                    self._save_checkpoint(session_id, processing_results)
                    print(f"\n💾 Checkpoint saved at {self.processed_count} images")
                
                # Display ETA
                self._update_eta_display(progress_bar)
            
            progress_bar.close()
            
            if self.shutdown_event.is_set():
                print("\n⏹️  Processing stopped by user request")
                return False
            
            # Phase 3: Copy approved images
            if approved_images:
                print(f"\n📁 Phase 3: Organizing {len(approved_images)} approved images...")
                self._organize_approved_images(approved_images, output_folder)
            else:
                print("\n⚠️  No images were approved for copying")
            
            # Save final results
            self._save_final_results(session_id, processing_results)
            
            return True
            
        except Exception as e:
            progress_bar.close()
            self.logger.error(f"Processing pipeline failed: {e}", exc_info=True)
            return False
    
    def _analyze_similarity(self, image_paths: List[str]) -> Dict[str, int]:
        """Analyze similarity across all images.
        
        Args:
            image_paths: List of image paths.
            
        Returns:
            Dict[str, int]: Mapping of image path to similarity group ID.
        """
        try:
            print("🔍 Computing image hashes and features...")
            similarity_groups = self.similarity_finder.find_similar_groups(image_paths)
            
            # Convert to path -> group_id mapping
            path_to_group = {}
            for group_id, paths in similarity_groups.items():
                for path in paths:
                    path_to_group[path] = group_id
            
            # Assign group 0 to images not in any similarity group
            for path in image_paths:
                if path not in path_to_group:
                    path_to_group[path] = 0
            
            similar_count = sum(1 for group_id in path_to_group.values() if group_id > 0)
            print(f"✅ Found {len(similarity_groups)} similarity groups affecting {similar_count} images")
            
            return path_to_group
            
        except Exception as e:
            self.logger.error(f"Similarity analysis failed: {e}", exc_info=True)
            print(f"⚠️  Similarity analysis failed: {e}")
            # Return empty mapping to continue processing
            return {path: 0 for path in image_paths}
    
    def _process_image_batch(self, batch_paths: List[str], similarity_groups: Dict[str, int], 
                           batch_start_index: int) -> Optional[List[Any]]:
        """Process a batch of images through all analyzers.
        
        Args:
            batch_paths: List of image paths in this batch.
            similarity_groups: Similarity group mapping.
            batch_start_index: Starting index for this batch.
            
        Returns:
            Optional[List[Any]]: Processing results or None if failed.
        """
        try:
            batch_results = []
            
            for i, image_path in enumerate(batch_paths):
                if self.shutdown_event.is_set():
                    break
                
                try:
                    # Process single image through all analyzers
                    result = self._process_single_image(image_path, similarity_groups)
                    batch_results.append(result)
                    
                    # Save individual result to database
                    self.progress_tracker.save_image_result(result)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process image {image_path}: {e}")
                    # Create error result
                    from backend.core.base import ProcessingResult
                    error_result = ProcessingResult(
                        image_path=image_path,
                        filename=os.path.basename(image_path),
                        quality_result=None,
                        defect_result=None,
                        similarity_group=similarity_groups.get(image_path, 0),
                        compliance_result=None,
                        final_decision='rejected',
                        rejection_reasons=[f'Processing error: {str(e)}'],
                        processing_time=0.0,
                        timestamp=datetime.now()
                    )
                    batch_results.append(error_result)
            
            return batch_results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}", exc_info=True)
            return None
    
    def _process_single_image(self, image_path: str, similarity_groups: Dict[str, int]) -> Any:
        """Process a single image through all analyzers.
        
        Args:
            image_path: Path to the image.
            similarity_groups: Similarity group mapping.
            
        Returns:
            ProcessingResult: Complete analysis result.
        """
        start_time = time.time()
        
        try:
            # Run all analyzers
            quality_result = self.quality_analyzer.analyze(image_path)
            defect_result = self.defect_detector.detect_defects(image_path)
            compliance_result = self.compliance_checker.check_compliance(image_path, {})
            
            # Get similarity group
            similarity_group = similarity_groups.get(image_path, 0)
            
            # Make final decision
            decision_result = self.decision_engine.make_decision(
                quality_result=quality_result,
                defect_result=defect_result,
                compliance_result=compliance_result,
                similarity_group=similarity_group
            )
            
            processing_time = time.time() - start_time
            
            # Create result object
            from backend.core.base import ProcessingResult
            result = ProcessingResult(
                image_path=image_path,
                filename=os.path.basename(image_path),
                quality_result=quality_result,
                defect_result=defect_result,
                similarity_group=similarity_group,
                compliance_result=compliance_result,
                final_decision=decision_result.decision,
                rejection_reasons=decision_result.rejection_reasons,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Single image processing failed for {image_path}: {e}")
            
            # Create error result
            from backend.core.base import ProcessingResult
            result = ProcessingResult(
                image_path=image_path,
                filename=os.path.basename(image_path),
                quality_result=None,
                defect_result=None,
                similarity_group=similarity_groups.get(image_path, 0),
                compliance_result=None,
                final_decision='rejected',
                rejection_reasons=[f'Analysis error: {str(e)}'],
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            return result
    
    def _organize_approved_images(self, approved_images: List[str], output_folder: str):
        """Organize approved images into output folder structure.
        
        Args:
            approved_images: List of approved image paths.
            output_folder: Output folder path.
        """
        try:
            print("📁 Organizing approved images...")
            organize_progress = tqdm(
                total=len(approved_images),
                desc="Copying images",
                unit="img"
            )
            
            self.file_manager.organize_output(approved_images, output_folder)
            organize_progress.close()
            
            print(f"✅ Successfully organized {len(approved_images)} images")
            
        except Exception as e:
            self.logger.error(f"Failed to organize images: {e}", exc_info=True)
            print(f"❌ Failed to organize images: {e}")
    
    def _save_checkpoint(self, session_id: str, results: List[Any]):
        """Save processing checkpoint.
        
        Args:
            session_id: Session identifier.
            results: Processing results so far.
        """
        try:
            self.progress_tracker.save_checkpoint(
                session_id=session_id,
                processed_count=self.processed_count,
                total_count=self.total_count,
                results=results
            )
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _save_final_results(self, session_id: str, results: List[Any]):
        """Save final processing results.
        
        Args:
            session_id: Session identifier.
            results: All processing results.
        """
        try:
            # Update session with final statistics
            self.progress_tracker.update_session_progress(
                session_id=session_id,
                processed_count=self.processed_count,
                approved_count=self.approved_count,
                rejected_count=self.rejected_count
            )
            
        except Exception as e:
            self.logger.error(f"Failed to save final results: {e}")
    
    def _generate_final_reports(self, session_id: str, output_folder: str):
        """Generate final processing reports.
        
        Args:
            session_id: Session identifier.
            output_folder: Output folder path.
        """
        try:
            # Get all results for this session
            results = self.progress_tracker.get_session_results(session_id)
            
            if results:
                # Generate Excel report
                excel_path = os.path.join(output_folder, f"processing_report_{session_id}.xlsx")
                
                # Create mock decision results and aggregated results for compatibility
                from backend.core.decision_engine import DecisionResult, AggregatedResults, DecisionScores, RejectionReason
                decision_results = []
                for result in results:
                    scores = DecisionScores(
                        quality_score=result.quality_result.overall_score if result.quality_result else 0.0,
                        defect_score=1.0 - (result.defect_result.anomaly_score if result.defect_result else 0.0),
                        similarity_score=0.8,
                        compliance_score=1.0 if (result.compliance_result and result.compliance_result.overall_compliance) else 0.0,
                        technical_score=0.9,
                        overall_score=0.8
                    )
                    
                    rejection_reasons = []
                    for reason in result.rejection_reasons:
                        rejection_reasons.append(RejectionReason(
                            category='quality',
                            reason=reason,
                            severity='medium',
                            details=reason
                        ))
                    
                    decision_result = DecisionResult(
                        image_path=result.image_path,
                        filename=result.filename,
                        decision=result.final_decision,
                        confidence=0.8,
                        scores=scores,
                        rejection_reasons=rejection_reasons,
                        processing_time=result.processing_time,
                        timestamp=result.timestamp
                    )
                    decision_results.append(decision_result)
                
                # Create aggregated results
                approved_count = sum(1 for r in results if r.final_decision == 'approved')
                rejected_count = sum(1 for r in results if r.final_decision == 'rejected')
                
                # Count rejection reasons
                rejection_breakdown = {}
                for result in results:
                    for reason in result.rejection_reasons:
                        rejection_breakdown[reason] = rejection_breakdown.get(reason, 0) + 1
                
                aggregated_results = AggregatedResults(
                    total_images=len(results),
                    approved_count=approved_count,
                    rejected_count=rejected_count,
                    review_required_count=0,
                    approval_rate=approved_count / len(results) if results else 0.0,
                    avg_quality_score=sum(r.quality_result.overall_score if r.quality_result else 0.0 for r in results) / len(results) if results else 0.0,
                    avg_overall_score=0.8,
                    rejection_breakdown=rejection_breakdown,
                    top_rejection_reasons=list(rejection_breakdown.items())[:5],
                    processing_statistics={
                        'avg_processing_time': sum(r.processing_time for r in results) / len(results) if results else 0.0,
                        'total_processing_time': sum(r.processing_time for r in results)
                    }
                )
                
                report_path = self.report_generator.generate_excel_report(
                    session_id=session_id,
                    processing_results=results,
                    decision_results=decision_results,
                    aggregated_results=aggregated_results,
                    output_dir=output_folder
                )
                
                if report_path:
                    print(f"📊 Excel report saved: {report_path}")
                
                # Generate HTML dashboard
                html_path = self.report_generator.generate_html_dashboard(
                    session_id=session_id,
                    processing_results=results,
                    decision_results=decision_results,
                    aggregated_results=aggregated_results,
                    output_dir=output_folder
                )
                
                if html_path:
                    print(f"🌐 HTML dashboard saved: {html_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to generate reports: {e}")
            print(f"⚠️  Report generation failed: {e}")
    
    def _update_eta_display(self, progress_bar):
        """Update ETA display with current progress.
        
        Args:
            progress_bar: Progress bar instance.
        """
        if self.start_time and self.processed_count > 0:
            elapsed = datetime.now() - self.start_time
            rate = self.processed_count / elapsed.total_seconds()
            remaining = self.total_count - self.processed_count
            
            if rate > 0:
                eta_seconds = remaining / rate
                eta = timedelta(seconds=int(eta_seconds))
                progress_bar.set_description(f"Processing images (ETA: {eta})")
    
    def _display_final_summary(self):
        """Display final processing summary."""
        if self.start_time:
            total_time = datetime.now() - self.start_time
            rate = self.processed_count / total_time.total_seconds() if total_time.total_seconds() > 0 else 0
            
            print(f"\n📈 Processing Summary:")
            print(f"   Total processed: {self.processed_count:,} images")
            print(f"   ✅ Approved: {self.approved_count:,} ({self.approved_count/self.processed_count*100:.1f}%)")
            print(f"   ❌ Rejected: {self.rejected_count:,} ({self.rejected_count/self.processed_count*100:.1f}%)")
            print(f"   ⏱️  Total time: {total_time}")
            print(f"   🚀 Average rate: {rate:.2f} images/second")
    
    def _validate_folders(self, input_folder: str, output_folder: str) -> bool:
        """Validate input and output folders.
        
        Args:
            input_folder: Input folder path.
            output_folder: Output folder path.
            
        Returns:
            bool: True if valid, False otherwise.
        """
        if not os.path.exists(input_folder):
            self.logger.error(f"Input folder does not exist: {input_folder}")
            return False
        
        if not os.path.isdir(input_folder):
            self.logger.error(f"Input path is not a directory: {input_folder}")
            return False
        
        # Create output folder if it doesn't exist
        try:
            os.makedirs(output_folder, exist_ok=True)
            self.logger.info(f"Output folder ready: {output_folder}")
        except Exception as e:
            self.logger.error(f"Failed to create output folder: {e}")
            return False
        
        return True
    
    def _log_config_summary(self):
        """Log configuration summary."""
        self.logger.info("Configuration Summary:")
        self.logger.info(f"  Batch size: {self.config.processing.batch_size}")
        self.logger.info(f"  Max workers: {self.config.processing.max_workers}")
        self.logger.info(f"  Checkpoint interval: {self.config.processing.checkpoint_interval}")
        self.logger.info(f"  Images per folder: {self.config.output.images_per_folder}")
        self.logger.info(f"  Min resolution: {self.config.quality.min_resolution}")
        self.logger.info(f"  Similarity threshold: {self.config.similarity.feature_threshold}")
    
    def _handle_resume_request(self, input_folder: str, output_folder: str) -> tuple[Optional[str], int]:
        """Handle resume request and user interaction.
        
        Args:
            input_folder: Input folder path.
            output_folder: Output folder path.
            
        Returns:
            Tuple[Optional[str], int]: Session ID and start index, or (None, 0) if no resume.
        """
        # Find resumable sessions for this input/output combination
        resumable_sessions = self._find_resumable_sessions(input_folder, output_folder)
        
        if not resumable_sessions:
            return None, 0
        
        # If only one resumable session, use it
        if len(resumable_sessions) == 1:
            session = resumable_sessions[0]
            return self._confirm_resume_session(session)
        
        # Multiple sessions found, let user choose
        return self._select_resume_session(resumable_sessions)
    
    def _find_resumable_sessions(self, input_folder: str, output_folder: str) -> List[Dict[str, Any]]:
        """Find sessions that can be resumed for the given folders.
        
        Args:
            input_folder: Input folder path.
            output_folder: Output folder path.
            
        Returns:
            List[Dict[str, Any]]: List of resumable sessions.
        """
        all_resumable = self.progress_tracker.get_resumable_sessions()
        
        # Filter by matching input/output folders
        matching_sessions = []
        for session in all_resumable:
            if (os.path.abspath(session['input_folder']) == os.path.abspath(input_folder) and
                os.path.abspath(session['output_folder']) == os.path.abspath(output_folder)):
                matching_sessions.append(session)
        
        return matching_sessions
    
    def _confirm_resume_session(self, session: Dict[str, Any]) -> tuple[Optional[str], int]:
        """Confirm resuming a specific session with enhanced user interface.
        
        Args:
            session: Session information.
            
        Returns:
            Tuple[Optional[str], int]: Session ID and start index, or (None, 0) if declined.
        """
        session_id = session['session_id']
        
        # Load checkpoint data
        checkpoint_data = self.progress_tracker.load_checkpoint(session_id)
        if not checkpoint_data:
            print(f"⚠️  Could not load checkpoint data for session {session_id}")
            self.logger.warning(f"Could not load checkpoint data for session {session_id}")
            return None, 0
        
        # Calculate progress statistics
        processed = checkpoint_data['processed_count']
        total = checkpoint_data['total_count']
        approved = checkpoint_data['approved_count']
        rejected = checkpoint_data['rejected_count']
        progress_pct = (processed / total * 100) if total > 0 else 0
        approval_rate = (approved / processed * 100) if processed > 0 else 0
        
        # Display enhanced session information
        print(f"\n🔄 Resumable Session Found")
        print(f"{'='*50}")
        print(f"📋 Session ID: {session_id}")
        print(f"📅 Started: {session['start_time']}")
        print(f"📊 Progress: {processed:,}/{total:,} images ({progress_pct:.1f}%)")
        print(f"✅ Approved: {approved:,} images ({approval_rate:.1f}%)")
        print(f"❌ Rejected: {rejected:,} images")
        
        if checkpoint_data['has_checkpoint']:
            print(f"💾 Last checkpoint: {checkpoint_data['last_checkpoint_count']} images")
            
            # Estimate remaining time based on previous rate
            if 'last_checkpoint_data' in checkpoint_data:
                last_data = checkpoint_data['last_checkpoint_data']
                if 'processing_rate' in last_data:
                    remaining = total - processed
                    eta_seconds = remaining / last_data['processing_rate']
                    eta = timedelta(seconds=int(eta_seconds))
                    print(f"⏱️  Estimated remaining time: {eta}")
        
        print(f"{'='*50}")
        
        # Enhanced user prompt with options
        while True:
            print(f"\nOptions:")
            print(f"  [r] Resume from checkpoint ({processed:,} images)")
            print(f"  [s] Start fresh (discard progress)")
            print(f"  [i] Show detailed session info")
            print(f"  [q] Quit")
            
            response = input("\nChoose an option [r/s/i/q]: ").strip().lower()
            
            if response in ['r', 'resume']:
                start_index = checkpoint_data.get('resume_from_index', processed)
                print(f"🔄 Resuming from image {start_index + 1:,}/{total:,}")
                self.logger.info(f"Resuming session {session_id} from index {start_index}")
                return session_id, start_index
                
            elif response in ['s', 'start', 'fresh']:
                print("🆕 Starting fresh session...")
                return None, 0
                
            elif response in ['i', 'info']:
                self._display_detailed_session_info(session_id, checkpoint_data)
                
            elif response in ['q', 'quit']:
                print("👋 Goodbye!")
                sys.exit(0)
                
            else:
                print("❌ Invalid option. Please choose r, s, i, or q.")
    
    def _display_detailed_session_info(self, session_id: str, checkpoint_data: Dict[str, Any]):
        """Display detailed session information.
        
        Args:
            session_id: Session identifier.
            checkpoint_data: Checkpoint data.
        """
        print(f"\n📋 Detailed Session Information")
        print(f"{'='*60}")
        
        # Get session results summary
        try:
            results_summary = self.progress_tracker.get_session_results_summary(session_id)
            
            if results_summary:
                print(f"⚡ Performance Metrics:")
                print(f"   Average processing time: {results_summary['avg_processing_time']:.3f}s per image")
                
                if results_summary.get('avg_quality_score'):
                    print(f"   Average quality score: {results_summary['avg_quality_score']:.2f}")
                
                if results_summary.get('rejection_breakdown'):
                    print(f"\n❌ Rejection Reasons:")
                    for reason, count in results_summary['rejection_breakdown'].items():
                        percentage = (count / checkpoint_data['rejected_count'] * 100) if checkpoint_data['rejected_count'] > 0 else 0
                        print(f"   {reason}: {count} ({percentage:.1f}%)")
            
            # Show recent processing activity
            recent_results = self.progress_tracker.get_recent_results(session_id, limit=10)
            if recent_results:
                print(f"\n📈 Recent Activity (last 10 images):")
                for result in recent_results:
                    status = "✅" if result['final_decision'] == 'approved' else "❌"
                    print(f"   {status} {os.path.basename(result['image_path'])} - {result['final_decision']}")
            
        except Exception as e:
            print(f"⚠️  Could not load detailed session info: {e}")
        
        print(f"{'='*60}")
        input("\nPress Enter to continue...")
    
    def _select_resume_session(self, sessions: List[Dict[str, Any]]) -> tuple[Optional[str], int]:
        """Let user select which session to resume from multiple options.
        
        Args:
            sessions: List of resumable sessions.
            
        Returns:
            Tuple[Optional[str], int]: Session ID and start index, or (None, 0) if declined.
        """
        print(f"\nFound {len(sessions)} resumable sessions:")
        
        # Display all sessions
        for i, session in enumerate(sessions, 1):
            checkpoint_data = self.progress_tracker.load_checkpoint(session['session_id'])
            if checkpoint_data:
                progress_pct = (checkpoint_data['processed_count'] / checkpoint_data['total_count'] * 100) if checkpoint_data['total_count'] > 0 else 0
                print(f"  {i}. Session {session['session_id']}")
                print(f"     Started: {session['start_time']}")
                print(f"     Progress: {checkpoint_data['processed_count']}/{checkpoint_data['total_count']} ({progress_pct:.1f}%)")
                print(f"     Approved: {checkpoint_data['approved_count']}, Rejected: {checkpoint_data['rejected_count']}")
        
        # Let user choose
        while True:
            try:
                response = input(f"\nSelect session to resume (1-{len(sessions)}) or 'n' for new session: ").strip()
                
                if response.lower() in ['n', 'no', 'new']:
                    print("Starting fresh session...")
                    return None, 0
                
                choice = int(response)
                if 1 <= choice <= len(sessions):
                    selected_session = sessions[choice - 1]
                    return self._confirm_resume_session(selected_session)
                else:
                    print(f"Please enter a number between 1 and {len(sessions)}, or 'n' for new session.")
                    
            except ValueError:
                print(f"Please enter a valid number between 1 and {len(sessions)}, or 'n' for new session.")
    
    def list_sessions(self, status_filter: Optional[str] = None):
        """List processing sessions with their status.
        
        Args:
            status_filter: Optional status filter ('running', 'completed', 'failed', 'cancelled').
        """
        sessions = self.progress_tracker.list_sessions(status_filter)
        
        if not sessions:
            print("No sessions found.")
            return
        
        print(f"\nFound {len(sessions)} session(s):")
        print("-" * 80)
        
        for session in sessions:
            print(f"Session ID: {session['session_id']}")
            print(f"Status: {session['status']}")
            print(f"Input: {session['input_folder']}")
            print(f"Output: {session['output_folder']}")
            print(f"Started: {session['start_time']}")
            
            if session.get('end_time'):
                print(f"Ended: {session['end_time']}")
            
            if 'progress_percentage' in session:
                print(f"Progress: {session['processed_images']}/{session['total_images']} ({session['progress_percentage']:.1f}%)")
                print(f"Approved: {session['approved_images']}, Rejected: {session['rejected_images']}")
                
                if session['processed_images'] > 0:
                    print(f"Approval Rate: {session['approval_rate']:.1f}%")
            
            if session.get('error_message'):
                print(f"Error: {session['error_message']}")
            
            print("-" * 80)
    
    def cleanup_old_sessions(self, days_old: int = 30):
        """Clean up old completed sessions.
        
        Args:
            days_old: Remove sessions older than this many days.
        """
        cleaned_count = self.progress_tracker.db_manager.cleanup_old_sessions(days_old)
        print(f"Cleaned up {cleaned_count} old sessions (older than {days_old} days)")
        self.logger.info(f"Cleaned up {cleaned_count} old sessions")
    
    def get_session_info(self, session_id: str):
        """Get detailed information about a specific session.
        
        Args:
            session_id: Session identifier.
        """
        session_info = self.progress_tracker.db_manager.get_session_info(session_id)
        if not session_info:
            print(f"Session '{session_id}' not found.")
            return
        
        print(f"\nSession Information:")
        print(f"  ID: {session_info['session_id']}")
        print(f"  Status: {session_info['status']}")
        print(f"  Input Folder: {session_info['input_folder']}")
        print(f"  Output Folder: {session_info['output_folder']}")
        print(f"  Total Images: {session_info['total_images']}")
        print(f"  Processed: {session_info['processed_images']}")
        print(f"  Approved: {session_info['approved_images']}")
        print(f"  Rejected: {session_info['rejected_images']}")
        print(f"  Started: {session_info['start_time']}")
        
        if session_info.get('end_time'):
            print(f"  Ended: {session_info['end_time']}")
        
        if session_info.get('error_message'):
            print(f"  Error: {session_info['error_message']}")
        
        # Get checkpoint information
        checkpoint_data = self.progress_tracker.load_checkpoint(session_id)
        if checkpoint_data and checkpoint_data['has_checkpoint']:
            print(f"\nCheckpoint Information:")
            print(f"  Last Checkpoint: {checkpoint_data['last_checkpoint_count']} images")
            print(f"  Can Resume: {checkpoint_data['can_resume']}")
            
            if 'last_checkpoint_data' in checkpoint_data:
                last_data = checkpoint_data['last_checkpoint_data']
                print(f"  Checkpoint Time: {last_data.get('timestamp', 'Unknown')}")
                print(f"  Progress: {last_data.get('progress_percentage', 0):.1f}%")
        
        # Get results summary
        results_summary = self.progress_tracker.get_session_results_summary(session_id)
        if results_summary:
            print(f"\nResults Summary:")
            print(f"  Average Processing Time: {results_summary['avg_processing_time']:.3f}s per image")
            
            if results_summary.get('avg_quality_score'):
                print(f"  Average Quality Score: {results_summary['avg_quality_score']:.2f}")
            
            if results_summary.get('rejection_breakdown'):
                print(f"  Rejection Reasons:")
                for reason, count in results_summary['rejection_breakdown'].items():
                    print(f"    {reason}: {count}")
    
    def recover_session(self, session_id: str):
        """Attempt to recover a failed or interrupted session.
        
        Args:
            session_id: Session identifier to recover.
        """
        session_info = self.progress_tracker.db_manager.get_session_info(session_id)
        if not session_info:
            print(f"Session '{session_id}' not found.")
            return False
        
        if session_info['status'] == 'running':
            print(f"Session '{session_id}' is already running.")
            return False
        
        # Check if session can be recovered
        checkpoint_data = self.progress_tracker.load_checkpoint(session_id)
        if not checkpoint_data:
            print(f"No checkpoint data found for session '{session_id}'.")
            return False
        
        print(f"Attempting to recover session '{session_id}'...")
        print(f"  Last processed: {checkpoint_data['processed_count']}/{checkpoint_data['total_count']} images")
        
        # Reset session status to running
        try:
            with self.progress_tracker.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE processing_sessions 
                    SET status = 'running', error_message = NULL, end_time = NULL,
                        last_update = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                ''', (datetime.now(), session_id))
                conn.commit()
            
            print(f"Session '{session_id}' has been reset to running status.")
            print("You can now resume this session using --resume flag.")
            return True
            
        except Exception as e:
            print(f"Failed to recover session: {e}")
            self.logger.error(f"Failed to recover session {session_id}: {e}")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Adobe Stock Image Processor - Analyze and filter images for stock submission"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command (default)
    process_parser = subparsers.add_parser('process', help='Process images')
    process_parser.add_argument(
        "input_folder",
        help="Path to folder containing images to process"
    )
    process_parser.add_argument(
        "output_folder", 
        help="Path to folder where approved images will be copied"
    )
    process_parser.add_argument(
        "--config",
        help="Path to configuration file (optional)"
    )
    process_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous session if available"
    )
    process_parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Override logging level from configuration"
    )
    
    # Sessions command
    sessions_parser = subparsers.add_parser('sessions', help='Manage processing sessions')
    sessions_parser.add_argument(
        '--list',
        action='store_true',
        help='List all sessions'
    )
    sessions_parser.add_argument(
        '--status',
        choices=['running', 'completed', 'failed', 'cancelled'],
        help='Filter sessions by status'
    )
    sessions_parser.add_argument(
        '--info',
        metavar='SESSION_ID',
        help='Get detailed information about a specific session'
    )
    sessions_parser.add_argument(
        '--recover',
        metavar='SESSION_ID',
        help='Attempt to recover a failed session'
    )
    sessions_parser.add_argument(
        '--cleanup',
        type=int,
        metavar='DAYS',
        default=30,
        help='Clean up sessions older than specified days (default: 30)'
    )
    sessions_parser.add_argument(
        '--config',
        help='Path to configuration file (optional)'
    )

    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    server_parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    server_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    server_parser.add_argument('--config', help='Path to configuration file (optional)')
    
    args = parser.parse_args()
    
    # Handle backward compatibility - if no command specified, assume process
    if args.command is None:
        # Check if we have the old-style arguments
        if len(sys.argv) >= 3 and not sys.argv[1].startswith('-'):
            # Old style: script.py input_folder output_folder [options]
            # Convert to new format
            old_args = sys.argv[1:]
            sys.argv = [sys.argv[0], 'process'] + old_args
            args = parser.parse_args()
        else:
            parser.print_help()
            return
    
    try:
        # Initialize processor
        processor = ImageProcessor(getattr(args, 'config', None))
        
        # Override log level if specified
        if hasattr(args, 'log_level') and args.log_level:
            from backend.utils.logger import set_log_level
            set_log_level(args.log_level)
        
        # Handle different commands
        if args.command == 'process':
            # Run processing
            success = processor.run(args.input_folder, args.output_folder, args.resume)
            
            if success:
                print("Processing completed successfully")
                sys.exit(0)
            else:
                print("Processing failed")
                sys.exit(1)
                
        elif args.command == 'sessions':
            # Handle session management commands
            if args.list:
                processor.list_sessions(args.status)
            elif args.info:
                processor.get_session_info(args.info)
            elif args.recover:
                success = processor.recover_session(args.recover)
                sys.exit(0 if success else 1)
            elif hasattr(args, 'cleanup') and not (args.list or args.info or args.recover):
                processor.cleanup_old_sessions(args.cleanup)
            else:
                sessions_parser.print_help()

        elif args.command == 'server':
            start_api_server(host=args.host, port=args.port, reload=args.reload, config_path=args.config)
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def start_api_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False, config_path: Optional[str] = None):
    """Start the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload
        config_path: Path to configuration file
    """
    try:
        import uvicorn
        from api.main import create_app
        from config.config_loader import load_config
        from database.migrations import run_migrations
        from core.dependency_injection import initialize_dependencies
        
        print(f"🚀 Starting Adobe Stock Image Processor API Server...")
        print(f"📍 Server will be available at: http://{host}:{port}")
        print(f"📚 API documentation: http://{host}:{port}/api/docs")
        
        async def startup():
            """Async startup sequence."""
            try:
                # Load configuration
                config = load_config(config_path)
                print("✅ Configuration loaded")
                
                # Run database migrations
                await run_migrations()
                print("✅ Database migrations completed")
                
                # Initialize dependencies
                await initialize_dependencies(config_path)
                print("✅ Dependencies initialized")
                
                return create_app(config)
                
            except Exception as e:
                print(f"❌ Startup failed: {e}")
                raise
        
        # Run startup sequence
        app = asyncio.run(startup())
        
        # Start server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        print(f"❌ Missing dependencies for API server: {e}")
        print("💡 Install with: pip install fastapi uvicorn")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()