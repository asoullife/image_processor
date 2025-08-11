"""Unit tests for database operations and data integrity."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import unittest
import tempfile
import os
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from backend.core.database import DatabaseManager
from backend.core.base import ProcessingResult, QualityResult, DefectResult, ComplianceResult, ObjectDefect


class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager class."""
    
    def setUp(self):
        """Set up test database."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        self.db_manager = DatabaseManager(self.db_path)
        
        # Test data
        self.test_session_id = "test_session_123"
        self.test_input_folder = "/test/input"
        self.test_output_folder = "/test/output"
        self.test_total_images = 100
        self.test_config = {"batch_size": 50, "quality_threshold": 0.8}
    
    def tearDown(self):
        """Clean up test database."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
    
    def test_database_initialization(self):
        """Test database initialization creates all required tables."""
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check that all tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['processing_sessions', 'image_results', 'checkpoints']
            for table in expected_tables:
                self.assertIn(table, tables, f"Table {table} not found")
            
            # Check that indexes exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]
            
            expected_indexes = [
                'idx_sessions_status', 'idx_sessions_updated',
                'idx_results_session', 'idx_results_decision',
                'idx_checkpoints_session'
            ]
            for index in expected_indexes:
                self.assertIn(index, indexes, f"Index {index} not found")
    
    def test_create_session_success(self):
        """Test successful session creation."""
        success = self.db_manager.create_session(
            session_id=self.test_session_id,
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images,
            config=self.test_config
        )
        
        self.assertTrue(success)
        
        # Verify session was created
        session_info = self.db_manager.get_session_info(self.test_session_id)
        self.assertIsNotNone(session_info)
        self.assertEqual(session_info['session_id'], self.test_session_id)
        self.assertEqual(session_info['input_folder'], self.test_input_folder)
        self.assertEqual(session_info['output_folder'], self.test_output_folder)
        self.assertEqual(session_info['total_images'], self.test_total_images)
        self.assertEqual(session_info['status'], 'running')
        self.assertEqual(session_info['config_snapshot'], self.test_config)
    
    def test_create_duplicate_session_fails(self):
        """Test that creating duplicate session fails."""
        # Create first session
        success1 = self.db_manager.create_session(
            session_id=self.test_session_id,
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        self.assertTrue(success1)
        
        # Try to create duplicate
        success2 = self.db_manager.create_session(
            session_id=self.test_session_id,
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        self.assertFalse(success2)
    
    def test_update_session_progress(self):
        """Test updating session progress."""
        # Create session first
        self.db_manager.create_session(
            session_id=self.test_session_id,
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        
        # Update progress
        success = self.db_manager.update_session_progress(
            session_id=self.test_session_id,
            processed_count=50,
            approved_count=30,
            rejected_count=20
        )
        self.assertTrue(success)
        
        # Verify update
        session_info = self.db_manager.get_session_info(self.test_session_id)
        self.assertEqual(session_info['processed_images'], 50)
        self.assertEqual(session_info['approved_images'], 30)
        self.assertEqual(session_info['rejected_images'], 20)
    
    def test_update_nonexistent_session_fails(self):
        """Test updating non-existent session fails."""
        success = self.db_manager.update_session_progress(
            session_id="nonexistent",
            processed_count=10,
            approved_count=5,
            rejected_count=5
        )
        self.assertFalse(success)
    
    def test_complete_session(self):
        """Test completing a session."""
        # Create session first
        self.db_manager.create_session(
            session_id=self.test_session_id,
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        
        # Complete session
        success = self.db_manager.complete_session(
            session_id=self.test_session_id,
            status='completed'
        )
        self.assertTrue(success)
        
        # Verify completion
        session_info = self.db_manager.get_session_info(self.test_session_id)
        self.assertEqual(session_info['status'], 'completed')
        self.assertIsNotNone(session_info['end_time'])
    
    def test_complete_session_with_error(self):
        """Test completing session with error message."""
        # Create session first
        self.db_manager.create_session(
            session_id=self.test_session_id,
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        
        error_message = "Processing failed due to memory error"
        success = self.db_manager.complete_session(
            session_id=self.test_session_id,
            status='failed',
            error_message=error_message
        )
        self.assertTrue(success)
        
        # Verify error was saved
        session_info = self.db_manager.get_session_info(self.test_session_id)
        self.assertEqual(session_info['status'], 'failed')
        self.assertEqual(session_info['error_message'], error_message)
    
    def test_save_image_result(self):
        """Test saving image processing result."""
        # Create session first
        self.db_manager.create_session(
            session_id=self.test_session_id,
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        
        # Create test result
        quality_result = QualityResult(
            sharpness_score=85.5,
            noise_level=0.05,
            exposure_score=90.0,
            color_balance_score=88.0,
            resolution=(1920, 1080),
            file_size=2048000,
            overall_score=87.5,
            passed=True
        )
        
        defect_result = DefectResult(
            detected_objects=[],
            anomaly_score=0.1,
            defect_count=0,
            defect_types=[],
            confidence_scores=[],
            passed=True
        )
        
        compliance_result = ComplianceResult(
            logo_detections=[],
            privacy_violations=[],
            metadata_issues=[],
            keyword_relevance=0.9,
            overall_compliance=True
        )
        
        processing_result = ProcessingResult(
            image_path="/test/image1.jpg",
            filename="image1.jpg",
            quality_result=quality_result,
            defect_result=defect_result,
            similarity_group=1,
            compliance_result=compliance_result,
            final_decision="approved",
            rejection_reasons=[],
            processing_time=2.5,
            timestamp=datetime.now()
        )
        
        # Save result
        success = self.db_manager.save_image_result(processing_result, self.test_session_id)
        self.assertTrue(success)
        
        # Verify result was saved
        results = self.db_manager.get_session_results(self.test_session_id)
        self.assertEqual(len(results), 1)
        
        saved_result = results[0]
        self.assertEqual(saved_result['image_path'], "/test/image1.jpg")
        self.assertEqual(saved_result['filename'], "image1.jpg")
        self.assertEqual(saved_result['quality_score'], 87.5)
        self.assertEqual(saved_result['defect_score'], 0.1)
        self.assertEqual(saved_result['similarity_group'], 1)
        self.assertEqual(saved_result['compliance_status'], 'compliant')
        self.assertEqual(saved_result['final_decision'], 'approved')
        self.assertEqual(saved_result['processing_time'], 2.5)
    
    def test_save_rejected_image_result(self):
        """Test saving rejected image result with reasons."""
        # Create session first
        self.db_manager.create_session(
            session_id=self.test_session_id,
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        
        # Create rejected result
        processing_result = ProcessingResult(
            image_path="/test/image2.jpg",
            filename="image2.jpg",
            final_decision="rejected",
            rejection_reasons=["low_quality", "compliance_issue"],
            processing_time=1.8,
            timestamp=datetime.now()
        )
        
        # Save result
        success = self.db_manager.save_image_result(processing_result, self.test_session_id)
        self.assertTrue(success)
        
        # Verify rejection reasons were saved
        results = self.db_manager.get_session_results(self.test_session_id)
        self.assertEqual(len(results), 1)
        
        saved_result = results[0]
        self.assertEqual(saved_result['final_decision'], 'rejected')
        self.assertEqual(saved_result['rejection_reasons'], ["low_quality", "compliance_issue"])
    
    def test_get_session_results_with_limit(self):
        """Test getting session results with limit."""
        # Create session and add multiple results
        self.db_manager.create_session(
            session_id=self.test_session_id,
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=self.test_total_images
        )
        
        # Add 5 results
        for i in range(5):
            result = ProcessingResult(
                image_path=f"/test/image{i}.jpg",
                filename=f"image{i}.jpg",
                final_decision="approved",
                processing_time=1.0,
                timestamp=datetime.now()
            )
            self.db_manager.save_image_result(result, self.test_session_id)
        
        # Get limited results
        results = self.db_manager.get_session_results(self.test_session_id, limit=3)
        self.assertEqual(len(results), 3)
        
        # Get all results
        all_results = self.db_manager.get_session_results(self.test_session_id)
        self.assertEqual(len(all_results), 5)
    
    def test_list_sessions(self):
        """Test listing sessions."""
        # Create multiple sessions
        sessions_data = [
            ("session1", "completed"),
            ("session2", "running"),
            ("session3", "failed")
        ]
        
        for session_id, status in sessions_data:
            self.db_manager.create_session(
                session_id=session_id,
                input_folder=self.test_input_folder,
                output_folder=self.test_output_folder,
                total_images=10
            )
            if status != "running":
                self.db_manager.complete_session(session_id, status)
        
        # List all sessions
        all_sessions = self.db_manager.list_sessions()
        self.assertEqual(len(all_sessions), 3)
        
        # List only running sessions
        running_sessions = self.db_manager.list_sessions(status="running")
        self.assertEqual(len(running_sessions), 1)
        self.assertEqual(running_sessions[0]['session_id'], "session2")
        
        # List completed sessions
        completed_sessions = self.db_manager.list_sessions(status="completed")
        self.assertEqual(len(completed_sessions), 1)
        self.assertEqual(completed_sessions[0]['session_id'], "session1")
    
    def test_cleanup_old_sessions(self):
        """Test cleaning up old sessions."""
        # Create old session
        old_session_id = "old_session"
        self.db_manager.create_session(
            session_id=old_session_id,
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=10
        )
        
        # Manually set old end time
        with self.db_manager.get_connection() as conn:
            cursor = conn.cursor()
            old_date = datetime.now() - timedelta(days=35)
            cursor.execute('''
                UPDATE processing_sessions 
                SET status = 'completed', end_time = ? 
                WHERE session_id = ?
            ''', (old_date, old_session_id))
            conn.commit()
        
        # Add some results and checkpoints
        result = ProcessingResult(
            image_path="/test/old_image.jpg",
            filename="old_image.jpg",
            final_decision="approved",
            processing_time=1.0,
            timestamp=datetime.now()
        )
        self.db_manager.save_image_result(result, old_session_id)
        
        # Cleanup old sessions (older than 30 days)
        cleaned_count = self.db_manager.cleanup_old_sessions(days_old=30)
        self.assertEqual(cleaned_count, 1)
        
        # Verify session was deleted
        session_info = self.db_manager.get_session_info(old_session_id)
        self.assertIsNone(session_info)
        
        # Verify related records were deleted
        results = self.db_manager.get_session_results(old_session_id)
        self.assertEqual(len(results), 0)
    
    def test_get_database_stats(self):
        """Test getting database statistics."""
        # Create test data
        self.db_manager.create_session(
            session_id="stats_session",
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=10
        )
        
        # Add some results
        for i in range(3):
            decision = "approved" if i < 2 else "rejected"
            result = ProcessingResult(
                image_path=f"/test/stats_image{i}.jpg",
                filename=f"stats_image{i}.jpg",
                final_decision=decision,
                processing_time=1.0,
                timestamp=datetime.now()
            )
            self.db_manager.save_image_result(result, "stats_session")
        
        # Get stats
        stats = self.db_manager.get_database_stats()
        
        self.assertIn('total_sessions', stats)
        self.assertIn('active_sessions', stats)
        self.assertIn('total_images_processed', stats)
        self.assertIn('approved_images', stats)
        self.assertIn('rejected_images', stats)
        self.assertIn('database_size_mb', stats)
        
        self.assertEqual(stats['total_sessions'], 1)
        self.assertEqual(stats['active_sessions'], 1)
        self.assertEqual(stats['total_images_processed'], 3)
        self.assertEqual(stats['approved_images'], 2)
        self.assertEqual(stats['rejected_images'], 1)
        self.assertGreater(stats['database_size_mb'], 0)
    
    def test_database_connection_error_handling(self):
        """Test database connection error handling."""
        # This should raise an exception during initialization
        with self.assertRaises(Exception):
            # Create database manager with invalid path
            invalid_db = DatabaseManager("/invalid/path/database.db")
    
    def test_foreign_key_constraints(self):
        """Test that foreign key constraints are enforced."""
        # Try to save image result without session
        result = ProcessingResult(
            image_path="/test/orphan.jpg",
            filename="orphan.jpg",
            final_decision="approved",
            processing_time=1.0,
            timestamp=datetime.now()
        )
        
        # This should fail due to foreign key constraint
        success = self.db_manager.save_image_result(result, "nonexistent_session")
        self.assertFalse(success)
    
    def test_concurrent_access(self):
        """Test concurrent database access."""
        import threading
        import time
        
        # Create session
        self.db_manager.create_session(
            session_id=self.test_session_id,
            input_folder=self.test_input_folder,
            output_folder=self.test_output_folder,
            total_images=100
        )
        
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(10):
                    result = ProcessingResult(
                        image_path=f"/test/worker{worker_id}_image{i}.jpg",
                        filename=f"worker{worker_id}_image{i}.jpg",
                        final_decision="approved",
                        processing_time=0.1,
                        timestamp=datetime.now()
                    )
                    success = self.db_manager.save_image_result(result, self.test_session_id)
                    results.append(success)
                    time.sleep(0.001)  # Small delay to increase chance of concurrency
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")
        self.assertEqual(len(results), 30)  # 3 workers * 10 results each
        self.assertTrue(all(results), "Some database operations failed")
        
        # Verify all results were saved
        saved_results = self.db_manager.get_session_results(self.test_session_id)
        self.assertEqual(len(saved_results), 30)


if __name__ == '__main__':
    unittest.main()