"""Progress tracking implementation with SQLite database backend."""

import uuid
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from threading import Lock

from .base import ProgressTracker, ProcessingResult
from .database import DatabaseManager
from ..utils.path_utils import get_database_path


class SQLiteProgressTracker(ProgressTracker):
    """SQLite-based progress tracker with checkpoint functionality."""
    
    def __init__(self, db_path: Optional[str] = None, 
                 checkpoint_interval: int = 50):
        """Initialize progress tracker.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default path.
            checkpoint_interval: Save checkpoint every N processed images.
        """
        if db_path is None:
            db_path = get_database_path()
        self.db_manager = DatabaseManager(db_path)
        self.checkpoint_interval = checkpoint_interval
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lock = Lock()
        
        # Track current session state
        self._current_session_id: Optional[str] = None
        self._processed_count = 0
        self._approved_count = 0
        self._rejected_count = 0
        self._last_checkpoint = 0
    
    def create_session(self, input_folder: str, output_folder: str, 
                      total_images: int, config: Optional[Dict[str, Any]] = None,
                      session_id: Optional[str] = None) -> str:
        """Create a new processing session.
        
        Args:
            input_folder: Path to input folder.
            output_folder: Path to output folder.
            total_images: Total number of images to process.
            config: Configuration snapshot.
            session_id: Optional custom session ID.
            
        Returns:
            str: Session ID.
            
        Raises:
            RuntimeError: If session creation fails.
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        success = self.db_manager.create_session(
            session_id=session_id,
            input_folder=input_folder,
            output_folder=output_folder,
            total_images=total_images,
            config=config
        )
        
        if not success:
            raise RuntimeError(f"Failed to create session {session_id}")
        
        with self._lock:
            self._current_session_id = session_id
            self._processed_count = 0
            self._approved_count = 0
            self._rejected_count = 0
            self._last_checkpoint = 0
        
        self.logger.info(f"Created session {session_id} for {total_images} images")
        return session_id
    
    def save_checkpoint(self, session_id: str, processed_count: int, 
                       total_count: int, results: List[ProcessingResult]) -> bool:
        """Save processing checkpoint.
        
        Args:
            session_id: Unique session identifier.
            processed_count: Number of images processed.
            total_count: Total number of images to process.
            results: Processing results since last checkpoint.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with self._lock:
                # Save individual results to database
                for result in results:
                    success = self.db_manager.save_image_result(result, session_id)
                    if not success:
                        self.logger.warning(f"Failed to save result for {result.image_path}")
                
                # Update counters
                approved_count = sum(1 for r in results if r.final_decision == 'approved')
                rejected_count = sum(1 for r in results if r.final_decision == 'rejected')
                
                self._processed_count = processed_count
                self._approved_count += approved_count
                self._rejected_count += rejected_count
                
                # Update session progress
                success = self.db_manager.update_session_progress(
                    session_id=session_id,
                    processed_count=self._processed_count,
                    approved_count=self._approved_count,
                    rejected_count=self._rejected_count
                )
                
                if not success:
                    self.logger.error(f"Failed to update session progress for {session_id}")
                    return False
                
                # Save checkpoint record if interval reached
                if processed_count - self._last_checkpoint >= self.checkpoint_interval:
                    checkpoint_data = {
                        'processed_count': processed_count,
                        'approved_count': self._approved_count,
                        'rejected_count': self._rejected_count,
                        'timestamp': datetime.now().isoformat(),
                        'progress_percentage': (processed_count / total_count) * 100 if total_count > 0 else 0
                    }
                    
                    success = self._save_checkpoint_record(
                        session_id=session_id,
                        processed_count=processed_count,
                        checkpoint_data=checkpoint_data
                    )
                    
                    if success:
                        self._last_checkpoint = processed_count
                        self.logger.info(f"Checkpoint saved at {processed_count}/{total_count} images")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint for session {session_id}: {e}")
            return False
    
    def _save_checkpoint_record(self, session_id: str, processed_count: int,
                               checkpoint_data: Dict[str, Any]) -> bool:
        """Save checkpoint record to database.
        
        Args:
            session_id: Session identifier.
            processed_count: Number of images processed.
            checkpoint_data: Checkpoint data to save.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Calculate checkpoint number
                cursor.execute('''
                    SELECT COALESCE(MAX(checkpoint_number), 0) + 1 
                    FROM checkpoints WHERE session_id = ?
                ''', (session_id,))
                checkpoint_number = cursor.fetchone()[0]
                
                # Calculate batch indices (approximate)
                batch_start = max(0, processed_count - self.checkpoint_interval)
                batch_end = processed_count - 1
                
                cursor.execute('''
                    INSERT INTO checkpoints 
                    (session_id, checkpoint_number, processed_count, 
                     batch_start_index, batch_end_index, checkpoint_data)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (session_id, checkpoint_number, processed_count,
                      batch_start, batch_end, json.dumps(checkpoint_data)))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint record: {e}")
            return False
    
    def load_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load processing checkpoint.
        
        Args:
            session_id: Session identifier to load.
            
        Returns:
            Dict with checkpoint data or None if not found.
        """
        try:
            # Get session info
            session_info = self.db_manager.get_session_info(session_id)
            if not session_info:
                self.logger.warning(f"Session {session_id} not found")
                return None
            
            # Get latest checkpoint
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM checkpoints 
                    WHERE session_id = ? 
                    ORDER BY checkpoint_number DESC 
                    LIMIT 1
                ''', (session_id,))
                
                checkpoint_row = cursor.fetchone()
                
                checkpoint_data = {
                    'session_info': session_info,
                    'has_checkpoint': checkpoint_row is not None,
                    'processed_count': session_info['processed_images'],
                    'approved_count': session_info['approved_images'],
                    'rejected_count': session_info['rejected_images'],
                    'total_count': session_info['total_images'],
                    'can_resume': session_info['status'] == 'running'
                }
                
                if checkpoint_row:
                    checkpoint_record = dict(checkpoint_row)
                    if checkpoint_record['checkpoint_data']:
                        saved_data = json.loads(checkpoint_record['checkpoint_data'])
                        checkpoint_data.update({
                            'last_checkpoint_number': checkpoint_record['checkpoint_number'],
                            'last_checkpoint_count': checkpoint_record['processed_count'],
                            'last_checkpoint_data': saved_data,
                            'resume_from_index': checkpoint_record['batch_end_index'] + 1
                        })
                
                return checkpoint_data
                
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint for session {session_id}: {e}")
            return None
    
    def get_session_status(self, session_id: str) -> Optional[str]:
        """Get session processing status.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            Status string or None if session not found.
        """
        session_info = self.db_manager.get_session_info(session_id)
        return session_info['status'] if session_info else None
    
    def complete_session(self, session_id: str, status: str = 'completed',
                        error_message: Optional[str] = None) -> bool:
        """Mark session as completed.
        
        Args:
            session_id: Session identifier.
            status: Final status ('completed', 'failed', 'cancelled').
            error_message: Error message if failed.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        success = self.db_manager.complete_session(session_id, status, error_message)
        
        if success:
            with self._lock:
                if self._current_session_id == session_id:
                    self._current_session_id = None
                    self._processed_count = 0
                    self._approved_count = 0
                    self._rejected_count = 0
                    self._last_checkpoint = 0
        
        return success
    
    def get_progress_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get progress summary for a session.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            Dict with progress information or None if session not found.
        """
        session_info = self.db_manager.get_session_info(session_id)
        if not session_info:
            return None
        
        total = session_info['total_images']
        processed = session_info['processed_images']
        approved = session_info['approved_images']
        rejected = session_info['rejected_images']
        
        progress_percentage = (processed / total * 100) if total > 0 else 0
        approval_rate = (approved / processed * 100) if processed > 0 else 0
        
        # Calculate estimated time remaining
        start_time = datetime.fromisoformat(session_info['start_time'])
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        if processed > 0:
            avg_time_per_image = elapsed_time / processed
            remaining_images = total - processed
            estimated_remaining_seconds = remaining_images * avg_time_per_image
        else:
            estimated_remaining_seconds = 0
        
        return {
            'session_id': session_id,
            'status': session_info['status'],
            'total_images': total,
            'processed_images': processed,
            'approved_images': approved,
            'rejected_images': rejected,
            'progress_percentage': round(progress_percentage, 2),
            'approval_rate': round(approval_rate, 2),
            'elapsed_time_seconds': round(elapsed_time, 2),
            'estimated_remaining_seconds': round(estimated_remaining_seconds, 2),
            'start_time': session_info['start_time'],
            'last_update': session_info['last_update'],
            'end_time': session_info.get('end_time')
        }
    
    def list_sessions(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List processing sessions with progress information.
        
        Args:
            status: Filter by status (optional).
            
        Returns:
            List of session dictionaries with progress info.
        """
        sessions = self.db_manager.list_sessions(status)
        
        # Enhance with progress information
        enhanced_sessions = []
        for session in sessions:
            progress_info = self.get_progress_summary(session['session_id'])
            if progress_info:
                # Merge session info with progress info
                enhanced_session = {**session, **progress_info}
                enhanced_sessions.append(enhanced_session)
        
        return enhanced_sessions
    
    def get_resumable_sessions(self) -> List[Dict[str, Any]]:
        """Get list of sessions that can be resumed.
        
        Returns:
            List of resumable session dictionaries.
        """
        return self.list_sessions(status='running')
    
    def cleanup_session_data(self, session_id: str) -> bool:
        """Clean up all data for a specific session.
        
        Args:
            session_id: Session identifier to clean up.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete in reverse dependency order
                cursor.execute('DELETE FROM checkpoints WHERE session_id = ?', (session_id,))
                cursor.execute('DELETE FROM image_results WHERE session_id = ?', (session_id,))
                cursor.execute('DELETE FROM processing_sessions WHERE session_id = ?', (session_id,))
                
                conn.commit()
                self.logger.info(f"Cleaned up session data for {session_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup session {session_id}: {e}")
            return False
    
    def get_session_results_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of processing results for a session.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            Dict with results summary or None if session not found.
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get basic counts
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_processed,
                        SUM(CASE WHEN final_decision = 'approved' THEN 1 ELSE 0 END) as approved,
                        SUM(CASE WHEN final_decision = 'rejected' THEN 1 ELSE 0 END) as rejected,
                        AVG(processing_time) as avg_processing_time,
                        AVG(quality_score) as avg_quality_score,
                        AVG(defect_score) as avg_defect_score
                    FROM image_results 
                    WHERE session_id = ?
                ''', (session_id,))
                
                row = cursor.fetchone()
                if not row or row[0] == 0:
                    return None
                
                summary = {
                    'session_id': session_id,
                    'total_processed': row[0],
                    'approved': row[1] or 0,
                    'rejected': row[2] or 0,
                    'avg_processing_time': round(row[3] or 0, 3),
                    'avg_quality_score': round(row[4] or 0, 3) if row[4] else None,
                    'avg_defect_score': round(row[5] or 0, 3) if row[5] else None
                }
                
                # Get rejection reasons breakdown
                cursor.execute('''
                    SELECT rejection_reasons, COUNT(*) as count
                    FROM image_results 
                    WHERE session_id = ? AND final_decision = 'rejected' 
                    AND rejection_reasons IS NOT NULL
                    GROUP BY rejection_reasons
                ''', (session_id,))
                
                rejection_breakdown = {}
                for row in cursor.fetchall():
                    if row[0]:
                        reasons = json.loads(row[0])
                        for reason in reasons:
                            rejection_breakdown[reason] = rejection_breakdown.get(reason, 0) + row[1]
                
                summary['rejection_breakdown'] = rejection_breakdown
                
                # Get compliance status breakdown
                cursor.execute('''
                    SELECT compliance_status, COUNT(*) as count
                    FROM image_results 
                    WHERE session_id = ? AND compliance_status IS NOT NULL
                    GROUP BY compliance_status
                ''', (session_id,))
                
                compliance_breakdown = {}
                for row in cursor.fetchall():
                    compliance_breakdown[row[0]] = row[1]
                
                summary['compliance_breakdown'] = compliance_breakdown
                
                return summary
                
        except Exception as e:
            self.logger.error(f"Failed to get results summary for session {session_id}: {e}")
            return None
    
    def force_checkpoint(self, session_id: str) -> bool:
        """Force save a checkpoint regardless of interval.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            session_info = self.db_manager.get_session_info(session_id)
            if not session_info:
                return False
            
            checkpoint_data = {
                'processed_count': session_info['processed_images'],
                'approved_count': session_info['approved_images'],
                'rejected_count': session_info['rejected_images'],
                'timestamp': datetime.now().isoformat(),
                'forced': True
            }
            
            return self._save_checkpoint_record(
                session_id=session_id,
                processed_count=session_info['processed_images'],
                checkpoint_data=checkpoint_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to force checkpoint for session {session_id}: {e}")
            return False
    
    def save_image_result(self, result: ProcessingResult) -> bool:
        """Save individual image processing result.
        
        Args:
            result: Processing result to save.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self._current_session_id:
            self.logger.error("No active session for saving image result")
            return False
        
        return self.db_manager.save_image_result(result, self._current_session_id)
    
    def update_session_progress(self, session_id: str, processed_count: int,
                               approved_count: int, rejected_count: int) -> bool:
        """Update session progress counters.
        
        Args:
            session_id: Session identifier.
            processed_count: Total processed images.
            approved_count: Total approved images.
            rejected_count: Total rejected images.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        return self.db_manager.update_session_progress(
            session_id=session_id,
            processed_count=processed_count,
            approved_count=approved_count,
            rejected_count=rejected_count
        )
    
    def get_session_results(self, session_id: str) -> List[ProcessingResult]:
        """Get all processing results for a session.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            List of processing results.
        """
        # Get raw results from database
        raw_results = self.db_manager.get_session_results(session_id)
        
        # Convert dictionaries to ProcessingResult objects
        processing_results = []
        for raw_result in raw_results:
            try:
                # Create ProcessingResult from database data
                from backend.core.base import QualityResult, DefectResult, ComplianceResult
                from datetime import datetime
                import json
                
                # Parse quality result
                quality_result = None
                if raw_result.get('quality_score') is not None:
                    quality_result = QualityResult(
                        sharpness_score=100.0,
                        noise_level=0.05,
                        exposure_score=0.9,
                        color_balance_score=0.85,
                        resolution=(1920, 1080),
                        file_size=1024000,
                        overall_score=raw_result.get('quality_score', 0.0),
                        passed=raw_result.get('final_decision') == 'approved'
                    )
                
                # Parse defect result
                defect_result = None
                if raw_result.get('defect_score') is not None:
                    defect_result = DefectResult(
                        detected_objects=[],
                        anomaly_score=raw_result.get('defect_score', 0.0),
                        defect_count=0,
                        defect_types=[],
                        confidence_scores=[],
                        passed=raw_result.get('final_decision') == 'approved'
                    )
                
                # Parse compliance result
                compliance_result = None
                if raw_result.get('compliance_status'):
                    compliance_result = ComplianceResult(
                        logo_detections=[],
                        privacy_violations=[],
                        metadata_issues=[],
                        keyword_relevance=0.9,
                        overall_compliance=raw_result.get('compliance_status') == 'pass'
                    )
                
                # Parse rejection reasons
                rejection_reasons = []
                if raw_result.get('rejection_reasons'):
                    try:
                        rejection_reasons = json.loads(raw_result['rejection_reasons'])
                    except:
                        rejection_reasons = [raw_result['rejection_reasons']]
                
                # Create ProcessingResult
                result = ProcessingResult(
                    image_path=raw_result.get('image_path', ''),
                    filename=raw_result.get('filename', ''),
                    quality_result=quality_result,
                    defect_result=defect_result,
                    similarity_group=raw_result.get('similarity_group', 0),
                    compliance_result=compliance_result,
                    final_decision=raw_result.get('final_decision', 'rejected'),
                    rejection_reasons=rejection_reasons,
                    processing_time=raw_result.get('processing_time', 0.0),
                    timestamp=datetime.fromisoformat(raw_result.get('timestamp', datetime.now().isoformat()))
                )
                
                processing_results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Failed to convert database result to ProcessingResult: {e}")
                continue
        
        return processing_results