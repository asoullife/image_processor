"""SQLite database schema and connection management for Adobe Stock Image Processor."""

import sqlite3
import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager
import logging
from threading import Lock

from .base import ProcessingResult
from ..utils.path_utils import get_database_path


class DatabaseManager:
    """Manages SQLite database connections and operations."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default path.
        """
        if db_path is None:
            db_path = get_database_path()
        self.db_path = db_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lock = Lock()  # Thread safety for database operations
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with required tables."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create processing_sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS processing_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE NOT NULL,
                        input_folder TEXT NOT NULL,
                        output_folder TEXT NOT NULL,
                        total_images INTEGER NOT NULL DEFAULT 0,
                        processed_images INTEGER NOT NULL DEFAULT 0,
                        approved_images INTEGER NOT NULL DEFAULT 0,
                        rejected_images INTEGER NOT NULL DEFAULT 0,
                        start_time TIMESTAMP NOT NULL,
                        last_update TIMESTAMP NOT NULL,
                        end_time TIMESTAMP NULL,
                        status TEXT NOT NULL DEFAULT 'running',
                        config_snapshot TEXT NULL,
                        error_message TEXT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create image_results table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS image_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        image_path TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        file_size INTEGER NULL,
                        quality_score REAL NULL,
                        defect_score REAL NULL,
                        similarity_group INTEGER NULL,
                        compliance_status TEXT NULL,
                        final_decision TEXT NOT NULL DEFAULT 'pending',
                        rejection_reasons TEXT NULL,
                        processing_time REAL NOT NULL DEFAULT 0.0,
                        error_message TEXT NULL,
                        processed_at TIMESTAMP NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES processing_sessions(session_id)
                    )
                ''')
                
                # Create checkpoints table for granular progress tracking
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        checkpoint_number INTEGER NOT NULL,
                        processed_count INTEGER NOT NULL,
                        batch_start_index INTEGER NOT NULL,
                        batch_end_index INTEGER NOT NULL,
                        checkpoint_data TEXT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES processing_sessions(session_id),
                        UNIQUE(session_id, checkpoint_number)
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_status ON processing_sessions(status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_updated ON processing_sessions(last_update)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_session ON image_results(session_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_decision ON image_results(final_decision)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_checkpoints_session ON checkpoints(session_id)')
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup.
        
        Yields:
            sqlite3.Connection: Database connection.
        """
        conn = None
        try:
            with self._lock:
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=30.0,
                    check_same_thread=False
                )
                conn.row_factory = sqlite3.Row  # Enable column access by name
                conn.execute('PRAGMA foreign_keys = ON')  # Enable foreign key constraints
                conn.execute('PRAGMA journal_mode = WAL')  # Better concurrency
                yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def create_session(self, session_id: str, input_folder: str, output_folder: str,
                      total_images: int, config: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new processing session.
        
        Args:
            session_id: Unique session identifier.
            input_folder: Path to input folder.
            output_folder: Path to output folder.
            total_images: Total number of images to process.
            config: Configuration snapshot.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                config_json = json.dumps(config) if config else None
                now = datetime.now()
                
                cursor.execute('''
                    INSERT INTO processing_sessions 
                    (session_id, input_folder, output_folder, total_images, 
                     start_time, last_update, config_snapshot)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, input_folder, output_folder, total_images,
                      now, now, config_json))
                
                conn.commit()
                self.logger.info(f"Created session {session_id} with {total_images} images")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to create session {session_id}: {e}")
            return False
    
    def update_session_progress(self, session_id: str, processed_count: int,
                               approved_count: int, rejected_count: int) -> bool:
        """Update session progress counters.
        
        Args:
            session_id: Session identifier.
            processed_count: Number of images processed.
            approved_count: Number of images approved.
            rejected_count: Number of images rejected.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE processing_sessions 
                    SET processed_images = ?, approved_images = ?, rejected_images = ?,
                        last_update = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                ''', (processed_count, approved_count, rejected_count,
                      datetime.now(), session_id))
                
                if cursor.rowcount == 0:
                    self.logger.warning(f"Session {session_id} not found for progress update")
                    return False
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to update session progress {session_id}: {e}")
            return False
    
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
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE processing_sessions 
                    SET status = ?, end_time = ?, error_message = ?,
                        last_update = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                ''', (status, datetime.now(), error_message, datetime.now(), session_id))
                
                if cursor.rowcount == 0:
                    self.logger.warning(f"Session {session_id} not found for completion")
                    return False
                
                conn.commit()
                self.logger.info(f"Session {session_id} marked as {status}")
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to complete session {session_id}: {e}")
            return False
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            Dict with session info or None if not found.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM processing_sessions WHERE session_id = ?
                ''', (session_id,))
                
                row = cursor.fetchone()
                if row:
                    session_info = dict(row)
                    # Parse config snapshot if exists
                    if session_info['config_snapshot']:
                        session_info['config_snapshot'] = json.loads(session_info['config_snapshot'])
                    return session_info
                
                return None
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get session info {session_id}: {e}")
            return None
    
    def save_image_result(self, result: ProcessingResult, session_id: str) -> bool:
        """Save image processing result.
        
        Args:
            result: Processing result to save.
            session_id: Session identifier.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get file size if available
                file_size = None
                if os.path.exists(result.image_path):
                    file_size = os.path.getsize(result.image_path)
                
                # Extract scores from results
                quality_score = result.quality_result.overall_score if result.quality_result else None
                defect_score = result.defect_result.anomaly_score if result.defect_result else None
                compliance_status = 'compliant' if (result.compliance_result and 
                                                  result.compliance_result.overall_compliance) else 'non_compliant'
                
                rejection_reasons_json = json.dumps(result.rejection_reasons) if result.rejection_reasons else None
                
                cursor.execute('''
                    INSERT INTO image_results 
                    (session_id, image_path, filename, file_size, quality_score, 
                     defect_score, similarity_group, compliance_status, final_decision,
                     rejection_reasons, processing_time, error_message, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, result.image_path, result.filename, file_size,
                      quality_score, defect_score, result.similarity_group,
                      compliance_status, result.final_decision, rejection_reasons_json,
                      result.processing_time, result.error_message, result.timestamp))
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to save image result for {result.image_path}: {e}")
            return False
    
    def get_session_results(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get processing results for a session.
        
        Args:
            session_id: Session identifier.
            limit: Maximum number of results to return.
            
        Returns:
            List of result dictionaries.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = 'SELECT * FROM image_results WHERE session_id = ? ORDER BY processed_at'
                params = [session_id]
                
                if limit:
                    query += ' LIMIT ?'
                    params.append(limit)
                
                cursor.execute(query, params)
                results = []
                
                for row in cursor.fetchall():
                    result = dict(row)
                    # Parse rejection reasons if exists
                    if result['rejection_reasons']:
                        result['rejection_reasons'] = json.loads(result['rejection_reasons'])
                    results.append(result)
                
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get session results {session_id}: {e}")
            return []
    
    def list_sessions(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List processing sessions.
        
        Args:
            status: Filter by status (optional).
            
        Returns:
            List of session dictionaries.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if status:
                    cursor.execute('''
                        SELECT * FROM processing_sessions 
                        WHERE status = ? 
                        ORDER BY start_time DESC
                    ''', (status,))
                else:
                    cursor.execute('''
                        SELECT * FROM processing_sessions 
                        ORDER BY start_time DESC
                    ''')
                
                sessions = []
                for row in cursor.fetchall():
                    session = dict(row)
                    if session['config_snapshot']:
                        session['config_snapshot'] = json.loads(session['config_snapshot'])
                    sessions.append(session)
                
                return sessions
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to list sessions: {e}")
            return []
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """Clean up old completed sessions.
        
        Args:
            days_old: Remove sessions older than this many days.
            
        Returns:
            Number of sessions removed.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # First get session IDs to delete
                cursor.execute('''
                    SELECT session_id FROM processing_sessions 
                    WHERE status IN ('completed', 'failed', 'cancelled')
                    AND datetime(end_time) < datetime('now', '-{} days')
                '''.format(days_old))
                
                session_ids = [row[0] for row in cursor.fetchall()]
                
                if not session_ids:
                    return 0
                
                # Delete related records
                placeholders = ','.join(['?' for _ in session_ids])
                
                cursor.execute(f'''
                    DELETE FROM checkpoints WHERE session_id IN ({placeholders})
                ''', session_ids)
                
                cursor.execute(f'''
                    DELETE FROM image_results WHERE session_id IN ({placeholders})
                ''', session_ids)
                
                cursor.execute(f'''
                    DELETE FROM processing_sessions WHERE session_id IN ({placeholders})
                ''', session_ids)
                
                conn.commit()
                self.logger.info(f"Cleaned up {len(session_ids)} old sessions")
                return len(session_ids)
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to cleanup old sessions: {e}")
            return 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with database statistics.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Session counts
                cursor.execute('SELECT COUNT(*) FROM processing_sessions')
                stats['total_sessions'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM processing_sessions WHERE status = "running"')
                stats['active_sessions'] = cursor.fetchone()[0]
                
                # Image counts
                cursor.execute('SELECT COUNT(*) FROM image_results')
                stats['total_images_processed'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM image_results WHERE final_decision = "approved"')
                stats['approved_images'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM image_results WHERE final_decision = "rejected"')
                stats['rejected_images'] = cursor.fetchone()[0]
                
                # Database file size
                if os.path.exists(self.db_path):
                    stats['database_size_mb'] = round(os.path.getsize(self.db_path) / (1024 * 1024), 2)
                else:
                    stats['database_size_mb'] = 0
                
                return stats
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {}