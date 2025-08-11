"""
Unit tests for ReportGenerator module

Tests cover:
- Excel report generation functionality
- HTML dashboard creation
- Chart and graph generation
- Statistical analysis functions
- Data accuracy validation
- Error handling and edge cases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import os
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

from backend.utils.report_generator import ReportGenerator
from backend.core.base import ProcessingResult, QualityResult, DefectResult, ComplianceResult, ObjectDefect, LogoDetection, PrivacyViolation
from backend.core.decision_engine import DecisionResult, DecisionScores, RejectionReason, DecisionCategory, AggregatedResults


class TestReportGenerator(unittest.TestCase):
    """Test cases for ReportGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'reports': {
                'excel_enabled': True,
                'html_enabled': True,
                'charts_enabled': True,
                'thumbnails_enabled': True,
                'thumbnail_size': [100, 100],
                'max_thumbnails': 10,
                'chart_style': 'default',
                'chart_dpi': 100,
                'chart_figsize': [10, 6]
            }
        }
        
        self.report_generator = ReportGenerator(self.config)
        self.report_generator.initialize()
        
        # Create sample test data
        self.sample_processing_results = self._create_sample_processing_results()
        self.sample_decision_results = self._create_sample_decision_results()
        self.sample_aggregated_results = self._create_sample_aggregated_results()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.report_generator.cleanup()
    
    def _create_sample_processing_results(self) -> list:
        """Create sample processing results for testing"""
        results = []
        
        for i in range(10):
            # Create quality result
            quality_result = QualityResult(
                sharpness_score=0.7 + (i * 0.02),
                noise_level=0.1 + (i * 0.01),
                exposure_score=0.8 + (i * 0.01),
                color_balance_score=0.75 + (i * 0.015),
                resolution=(1920 + i * 100, 1080 + i * 50),
                file_size=2048000 + i * 100000,
                overall_score=0.75 + (i * 0.02),
                passed=i < 7  # First 7 pass quality check
            )
            
            # Create defect result
            defect_result = DefectResult(
                detected_objects=[],
                anomaly_score=0.1 + (i * 0.05),
                defect_count=i % 3,
                defect_types=['blur', 'noise'][:i % 3],
                confidence_scores=[0.8, 0.7][:i % 3],
                passed=i < 8  # First 8 pass defect check
            )
            
            # Create compliance result
            compliance_result = ComplianceResult(
                logo_detections=[],
                privacy_violations=[],
                metadata_issues=['missing_keywords'] if i > 5 else [],
                keyword_relevance=0.8 + (i * 0.01),
                overall_compliance=i < 6  # First 6 pass compliance
            )
            
            # Create processing result
            result = ProcessingResult(
                image_path=f"/test/image_{i:03d}.jpg",
                filename=f"image_{i:03d}.jpg",
                quality_result=quality_result,
                defect_result=defect_result,
                similarity_group=i // 3 if i > 2 else None,
                compliance_result=compliance_result,
                final_decision='approved' if i < 5 else 'rejected',
                rejection_reasons=['low_quality'] if i >= 5 else [],
                processing_time=0.5 + (i * 0.1),
                timestamp=datetime.now()
            )
            
            results.append(result)
        
        return results
    
    def _create_sample_decision_results(self) -> list:
        """Create sample decision results for testing"""
        results = []
        
        for i, processing_result in enumerate(self.sample_processing_results):
            scores = DecisionScores(
                quality_score=0.75 + (i * 0.02),
                defect_score=0.8 - (i * 0.05),
                similarity_score=0.9 - (i * 0.03),
                compliance_score=0.85 - (i * 0.04),
                technical_score=0.9,
                overall_score=0.8 - (i * 0.03),
                weighted_score=0.78 - (i * 0.03)
            )
            
            rejection_reasons = []
            if i >= 5:
                rejection_reasons.append(RejectionReason(
                    category=DecisionCategory.QUALITY,
                    reason="low_quality_score",
                    severity="medium",
                    score=scores.quality_score,
                    threshold=0.6,
                    description=f"Quality score {scores.quality_score:.3f} below threshold"
                ))
            
            decision_result = DecisionResult(
                image_path=processing_result.image_path,
                filename=processing_result.filename,
                decision='approved' if i < 5 else 'rejected',
                confidence=0.9 - (i * 0.02),
                scores=scores,
                rejection_reasons=rejection_reasons,
                approval_factors=['good_quality', 'no_defects'] if i < 5 else [],
                recommendation="Approved for submission" if i < 5 else "Rejected due to quality issues",
                processing_time=0.1 + (i * 0.01)
            )
            
            results.append(decision_result)
        
        return results
    
    def _create_sample_aggregated_results(self) -> AggregatedResults:
        """Create sample aggregated results for testing"""
        return AggregatedResults(
            total_images=10,
            approved_count=5,
            rejected_count=5,
            review_required_count=0,
            approval_rate=0.5,
            avg_quality_score=0.8,
            avg_overall_score=0.75,
            rejection_breakdown={'quality': 3, 'defects': 2},
            top_rejection_reasons=[('low_quality_score', 5), ('high_defect_level', 2)],
            processing_statistics={
                'total_processing_time': 10.5,
                'avg_processing_time': 1.05,
                'min_processing_time': 0.5,
                'max_processing_time': 1.4
            }
        )
    
    def test_initialization(self):
        """Test ReportGenerator initialization"""
        self.assertTrue(self.report_generator._initialized)
        self.assertTrue(self.report_generator.excel_enabled)
        self.assertTrue(self.report_generator.html_enabled)
        self.assertTrue(self.report_generator.charts_enabled)
        self.assertEqual(self.report_generator.thumbnail_size, (100, 100))
    
    def test_initialization_with_minimal_config(self):
        """Test initialization with minimal configuration"""
        minimal_config = {}
        generator = ReportGenerator(minimal_config)
        self.assertTrue(generator.initialize())
        self.assertTrue(generator.excel_enabled)  # Default values
        self.assertTrue(generator.html_enabled)
        generator.cleanup()
    
    @patch('pandas.ExcelWriter')
    def test_excel_report_generation(self, mock_excel_writer):
        """Test Excel report generation"""
        mock_writer = MagicMock()
        mock_excel_writer.return_value.__enter__.return_value = mock_writer
        
        excel_path = self.report_generator.generate_excel_report(
            session_id="test_session",
            processing_results=self.sample_processing_results,
            decision_results=self.sample_decision_results,
            aggregated_results=self.sample_aggregated_results,
            output_dir=self.test_dir
        )
        
        self.assertIsNotNone(excel_path)
        self.assertTrue(excel_path.endswith('.xlsx'))
        
        # Verify that multiple sheets were created
        expected_sheets = [
            'Summary', 'Detailed Results', 'Approved Images', 
            'Rejected Images', 'Quality Analysis', 'Compliance Issues', 'Statistics'
        ]
        
        # Check that to_excel was called for each expected sheet
        call_args_list = [call[1]['sheet_name'] for call in mock_writer.to_excel.call_args_list 
                         if 'sheet_name' in call[1]]
        
        for sheet in expected_sheets:
            self.assertIn(sheet, call_args_list)
    
    def test_excel_report_with_empty_data(self):
        """Test Excel report generation with empty data"""
        empty_aggregated = AggregatedResults(
            total_images=0, approved_count=0, rejected_count=0, review_required_count=0,
            approval_rate=0.0, avg_quality_score=0.0, avg_overall_score=0.0,
            rejection_breakdown={}, top_rejection_reasons=[], processing_statistics={}
        )
        
        with patch('pandas.ExcelWriter') as mock_excel_writer:
            mock_writer = MagicMock()
            mock_excel_writer.return_value.__enter__.return_value = mock_writer
            
            excel_path = self.report_generator.generate_excel_report(
                session_id="empty_session",
                processing_results=[],
                decision_results=[],
                aggregated_results=empty_aggregated,
                output_dir=self.test_dir
            )
            
            self.assertIsNotNone(excel_path)
    
    def test_html_dashboard_generation(self):
        """Test HTML dashboard generation"""
        with patch.object(self.report_generator, '_generate_thumbnails', return_value=[]):
            html_path = self.report_generator.generate_html_dashboard(
                session_id="test_session",
                processing_results=self.sample_processing_results,
                decision_results=self.sample_decision_results,
                aggregated_results=self.sample_aggregated_results,
                output_dir=self.test_dir
            )
        
        self.assertIsNotNone(html_path)
        self.assertTrue(html_path.endswith('.html'))
        self.assertTrue(os.path.exists(html_path))
        
        # Verify HTML content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        self.assertIn('Adobe Stock Processing Report', html_content)
        self.assertIn('test_session', html_content)
        self.assertIn('Total Images', html_content)
        self.assertIn('Approved', html_content)
        self.assertIn('Rejected', html_content)
    
    def test_html_dashboard_with_thumbnails_disabled(self):
        """Test HTML dashboard generation with thumbnails disabled"""
        self.report_generator.thumbnails_enabled = False
        
        html_path = self.report_generator.generate_html_dashboard(
            session_id="test_session",
            processing_results=self.sample_processing_results,
            decision_results=self.sample_decision_results,
            aggregated_results=self.sample_aggregated_results,
            output_dir=self.test_dir
        )
        
        self.assertIsNotNone(html_path)
        self.assertTrue(os.path.exists(html_path))
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_chart_generation(self, mock_close, mock_savefig):
        """Test chart generation"""
        charts_dir = self.report_generator.generate_charts(
            decision_results=self.sample_decision_results,
            aggregated_results=self.sample_aggregated_results,
            charts_dir=os.path.join(self.test_dir, 'charts')
        )
        
        self.assertIsNotNone(charts_dir)
        self.assertTrue(os.path.exists(charts_dir))
        
        # Verify that savefig was called multiple times (for different charts)
        self.assertGreater(mock_savefig.call_count, 0)
        self.assertGreater(mock_close.call_count, 0)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_decision_pie_chart_creation(self, mock_close, mock_savefig):
        """Test decision pie chart creation"""
        charts_dir = os.path.join(self.test_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        chart_path = self.report_generator._create_decision_pie_chart(
            self.sample_aggregated_results, charts_dir
        )
        
        self.assertIsNotNone(chart_path)
        self.assertTrue(chart_path.endswith('decision_distribution.png'))
        mock_savefig.assert_called()
        mock_close.assert_called()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_quality_histogram_creation(self, mock_close, mock_savefig):
        """Test quality histogram creation"""
        charts_dir = os.path.join(self.test_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        chart_path = self.report_generator._create_quality_histogram(
            self.sample_decision_results, charts_dir
        )
        
        self.assertIsNotNone(chart_path)
        self.assertTrue(chart_path.endswith('quality_distribution.png'))
        mock_savefig.assert_called()
        mock_close.assert_called()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_score_scatter_plot_creation(self, mock_close, mock_savefig):
        """Test score scatter plot creation"""
        charts_dir = os.path.join(self.test_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        chart_path = self.report_generator._create_score_scatter_plot(
            self.sample_decision_results, charts_dir
        )
        
        self.assertIsNotNone(chart_path)
        self.assertTrue(chart_path.endswith('score_comparison.png'))
        mock_savefig.assert_called()
        mock_close.assert_called()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_rejection_reasons_chart_creation(self, mock_close, mock_savefig):
        """Test rejection reasons chart creation"""
        charts_dir = os.path.join(self.test_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        chart_path = self.report_generator._create_rejection_reasons_chart(
            self.sample_aggregated_results, charts_dir
        )
        
        self.assertIsNotNone(chart_path)
        self.assertTrue(chart_path.endswith('rejection_reasons.png'))
        mock_savefig.assert_called()
        mock_close.assert_called()
    
    def test_chart_creation_with_empty_data(self):
        """Test chart creation with empty data"""
        empty_aggregated = AggregatedResults(
            total_images=0, approved_count=0, rejected_count=0, review_required_count=0,
            approval_rate=0.0, avg_quality_score=0.0, avg_overall_score=0.0,
            rejection_breakdown={}, top_rejection_reasons=[], processing_statistics={}
        )
        
        charts_dir = os.path.join(self.test_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        # These should return None for empty data
        chart_path = self.report_generator._create_decision_pie_chart(empty_aggregated, charts_dir)
        self.assertIsNone(chart_path)
        
        chart_path = self.report_generator._create_quality_histogram([], charts_dir)
        self.assertIsNone(chart_path)
        
        chart_path = self.report_generator._create_rejection_reasons_chart(empty_aggregated, charts_dir)
        self.assertIsNone(chart_path)
    
    def test_summary_statistics_creation(self):
        """Test summary statistics creation"""
        stats = self.report_generator.create_summary_statistics(self.sample_decision_results)
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['total_images'], 10)
        self.assertIn('decision_counts', stats)
        self.assertIn('approval_rate', stats)
        self.assertIn('score_statistics', stats)
        self.assertIn('rejection_analysis', stats)
        
        # Check score statistics structure
        score_stats = stats['score_statistics']
        self.assertIn('quality_score', score_stats)
        self.assertIn('overall_score', score_stats)
        self.assertIn('weighted_score', score_stats)
        
        # Check that each score statistic has required fields
        for score_type in ['quality_score', 'overall_score', 'weighted_score']:
            score_data = score_stats[score_type]
            self.assertIn('mean', score_data)
            self.assertIn('median', score_data)
            self.assertIn('std', score_data)
            self.assertIn('min', score_data)
            self.assertIn('max', score_data)
            self.assertIn('q25', score_data)
            self.assertIn('q75', score_data)
    
    def test_summary_statistics_with_empty_data(self):
        """Test summary statistics creation with empty data"""
        stats = self.report_generator.create_summary_statistics([])
        self.assertEqual(stats, {})
    
    def test_csv_export(self):
        """Test CSV export functionality"""
        csv_path = os.path.join(self.test_dir, 'test_results.csv')
        
        success = self.report_generator.export_results_to_csv(
            processing_results=self.sample_processing_results,
            decision_results=self.sample_decision_results,
            output_path=csv_path
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(csv_path))
        
        # Verify CSV content
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 10)  # Should have 10 rows
        
        # Check required columns
        required_columns = [
            'filename', 'image_path', 'final_decision', 'processing_time',
            'quality_overall_score', 'defect_anomaly_score', 'compliance_overall'
        ]
        
        for column in required_columns:
            self.assertIn(column, df.columns)
    
    def test_csv_export_with_empty_data(self):
        """Test CSV export with empty data"""
        csv_path = os.path.join(self.test_dir, 'empty_results.csv')
        
        success = self.report_generator.export_results_to_csv(
            processing_results=[],
            decision_results=[],
            output_path=csv_path
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(csv_path))
        
        # Verify empty CSV
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 0)
    
    def test_comprehensive_report_generation(self):
        """Test comprehensive report generation with all formats"""
        with patch.object(self.report_generator, '_generate_thumbnails', return_value=[]):
            with patch('matplotlib.pyplot.savefig'):
                with patch('matplotlib.pyplot.close'):
                    report_files = self.report_generator.generate_comprehensive_report(
                        session_id="comprehensive_test",
                        processing_results=self.sample_processing_results,
                        decision_results=self.sample_decision_results,
                        aggregated_results=self.sample_aggregated_results,
                        output_dir=self.test_dir
                    )
        
        self.assertIsInstance(report_files, dict)
        self.assertIn('html', report_files)
        self.assertIn('charts', report_files)
        
        # Verify files exist
        if 'html' in report_files:
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, report_files['html'])))
        if 'charts' in report_files:
            self.assertTrue(os.path.exists(os.path.join(self.test_dir, report_files['charts'])))
    
    def test_comprehensive_report_with_disabled_features(self):
        """Test comprehensive report generation with some features disabled"""
        # Disable some features
        self.report_generator.excel_enabled = False
        self.report_generator.thumbnails_enabled = False
        
        with patch('matplotlib.pyplot.savefig'):
            with patch('matplotlib.pyplot.close'):
                report_files = self.report_generator.generate_comprehensive_report(
                    session_id="partial_test",
                    processing_results=self.sample_processing_results,
                    decision_results=self.sample_decision_results,
                    aggregated_results=self.sample_aggregated_results,
                    output_dir=self.test_dir
                )
        
        self.assertIsInstance(report_files, dict)
        self.assertNotIn('excel', report_files)  # Should be disabled
        self.assertIn('html', report_files)  # Should still be enabled
    
    def test_error_handling_in_excel_generation(self):
        """Test error handling in Excel report generation"""
        # Test with invalid output directory
        invalid_dir = "/invalid/nonexistent/directory"
        
        excel_path = self.report_generator.generate_excel_report(
            session_id="error_test",
            processing_results=self.sample_processing_results,
            decision_results=self.sample_decision_results,
            aggregated_results=self.sample_aggregated_results,
            output_dir=invalid_dir
        )
        
        # Should handle error gracefully and return None
        self.assertIsNone(excel_path)
    
    def test_error_handling_in_html_generation(self):
        """Test error handling in HTML dashboard generation"""
        # Test with invalid template data
        with patch.object(self.report_generator, '_generate_html_content', side_effect=Exception("Template error")):
            html_path = self.report_generator.generate_html_dashboard(
                session_id="error_test",
                processing_results=self.sample_processing_results,
                decision_results=self.sample_decision_results,
                aggregated_results=self.sample_aggregated_results,
                output_dir=self.test_dir
            )
        
        # Should handle error gracefully and return None
        self.assertIsNone(html_path)
    
    def test_error_handling_in_chart_generation(self):
        """Test error handling in chart generation"""
        # Test with matplotlib error
        with patch('matplotlib.pyplot.savefig', side_effect=Exception("Chart error")):
            charts_dir = self.report_generator.generate_charts(
                decision_results=self.sample_decision_results,
                aggregated_results=self.sample_aggregated_results,
                charts_dir=os.path.join(self.test_dir, 'charts')
            )
        
        # Should handle error gracefully and return None
        self.assertIsNone(charts_dir)
    
    def test_data_accuracy_in_excel_sheets(self):
        """Test data accuracy in generated Excel sheets"""
        with patch('pandas.ExcelWriter') as mock_excel_writer:
            mock_writer = MagicMock()
            mock_excel_writer.return_value.__enter__.return_value = mock_writer
            
            self.report_generator.generate_excel_report(
                session_id="accuracy_test",
                processing_results=self.sample_processing_results,
                decision_results=self.sample_decision_results,
                aggregated_results=self.sample_aggregated_results,
                output_dir=self.test_dir
            )
            
            # Check that DataFrames were created with correct data
            call_args_list = mock_writer.to_excel.call_args_list
            
            # Find the detailed results sheet call
            detailed_results_call = None
            for call in call_args_list:
                if call[1].get('sheet_name') == 'Detailed Results':
                    detailed_results_call = call
                    break
            
            self.assertIsNotNone(detailed_results_call)
            
            # Verify the DataFrame has correct number of rows
            df = detailed_results_call[0][0]  # First positional argument is the DataFrame
            self.assertEqual(len(df), len(self.sample_processing_results))
    
    def test_thumbnail_generation_with_mock_images(self):
        """Test thumbnail generation with mock images"""
        # Create mock image files
        mock_images = []
        for i in range(3):
            img_path = os.path.join(self.test_dir, f'test_image_{i}.jpg')
            
            # Create a simple test image
            img = Image.new('RGB', (200, 200), color=(255, 0, 0))
            img.save(img_path, 'JPEG')
            mock_images.append(img_path)
        
        # Update processing results with real image paths
        test_results = self.sample_processing_results[:3]
        for i, result in enumerate(test_results):
            result.image_path = mock_images[i]
        
        thumbnails_dir = os.path.join(self.test_dir, 'thumbnails')
        thumbnails_data = self.report_generator._generate_thumbnails(test_results, thumbnails_dir)
        
        self.assertEqual(len(thumbnails_data), 3)
        self.assertTrue(os.path.exists(thumbnails_dir))
        
        # Check that thumbnail files were created
        for thumb_data in thumbnails_data:
            thumb_path = os.path.join(self.test_dir, thumb_data['thumbnail_path'])
            self.assertTrue(os.path.exists(thumb_path))
    
    def test_cleanup_functionality(self):
        """Test cleanup functionality"""
        # Generate some charts first
        with patch('matplotlib.pyplot.savefig'):
            with patch('matplotlib.pyplot.close') as mock_close:
                self.report_generator.generate_charts(
                    decision_results=self.sample_decision_results,
                    aggregated_results=self.sample_aggregated_results,
                    charts_dir=os.path.join(self.test_dir, 'charts')
                )
        
        # Test cleanup
        result = self.report_generator.cleanup()
        self.assertTrue(result)
    
    def test_configuration_validation(self):
        """Test configuration validation and defaults"""
        # Test with invalid configuration
        invalid_config = {
            'reports': {
                'thumbnail_size': 'invalid',  # Should be a list/tuple
                'max_thumbnails': -1,  # Should be positive
                'chart_dpi': 'invalid'  # Should be numeric
            }
        }
        
        # Should not crash and use defaults
        generator = ReportGenerator(invalid_config)
        self.assertTrue(generator.initialize())
        generator.cleanup()


class TestReportGeneratorIntegration(unittest.TestCase):
    """Integration tests for ReportGenerator"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            'reports': {
                'excel_enabled': True,
                'html_enabled': True,
                'charts_enabled': True,
                'thumbnails_enabled': False,  # Disable for integration tests
                'chart_style': 'default'
            }
        }
        
        self.report_generator = ReportGenerator(self.config)
        self.report_generator.initialize()
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.report_generator.cleanup()
    
    def test_end_to_end_report_generation(self):
        """Test complete end-to-end report generation workflow"""
        # Create realistic test data
        processing_results = self._create_realistic_processing_results()
        decision_results = self._create_realistic_decision_results(processing_results)
        aggregated_results = self._create_realistic_aggregated_results(decision_results)
        
        # Generate comprehensive report
        with patch('matplotlib.pyplot.savefig'):
            with patch('matplotlib.pyplot.close'):
                report_files = self.report_generator.generate_comprehensive_report(
                    session_id="integration_test",
                    processing_results=processing_results,
                    decision_results=decision_results,
                    aggregated_results=aggregated_results,
                    output_dir=self.test_dir
                )
        
        # Verify all expected files were generated
        self.assertIsInstance(report_files, dict)
        self.assertGreater(len(report_files), 0)
        
        # Verify HTML file content
        if 'html' in report_files:
            html_path = os.path.join(self.test_dir, report_files['html'])
            self.assertTrue(os.path.exists(html_path))
            
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertIn('integration_test', content)
                self.assertIn('Total Images', content)
    
    def _create_realistic_processing_results(self):
        """Create realistic processing results for integration testing"""
        results = []
        decisions = ['approved', 'rejected', 'approved', 'rejected', 'approved']
        
        for i in range(5):
            quality_result = QualityResult(
                sharpness_score=0.6 + (i * 0.1),
                noise_level=0.05 + (i * 0.02),
                exposure_score=0.7 + (i * 0.05),
                color_balance_score=0.8 + (i * 0.03),
                resolution=(1920, 1080),
                file_size=2048000,
                overall_score=0.7 + (i * 0.05),
                passed=decisions[i] == 'approved'
            )
            
            result = ProcessingResult(
                image_path=f"/realistic/test/image_{i}.jpg",
                filename=f"realistic_image_{i}.jpg",
                quality_result=quality_result,
                defect_result=DefectResult([], 0.1, 0, [], [], True),
                compliance_result=ComplianceResult([], [], [], 0.9, True),
                final_decision=decisions[i],
                processing_time=1.0 + (i * 0.2),
                timestamp=datetime.now()
            )
            
            results.append(result)
        
        return results
    
    def _create_realistic_decision_results(self, processing_results):
        """Create realistic decision results for integration testing"""
        results = []
        
        for result in processing_results:
            scores = DecisionScores(
                quality_score=result.quality_result.overall_score,
                defect_score=0.9,
                similarity_score=0.95,
                compliance_score=0.9,
                technical_score=0.85,
                overall_score=result.quality_result.overall_score,
                weighted_score=result.quality_result.overall_score * 0.9
            )
            
            decision_result = DecisionResult(
                image_path=result.image_path,
                filename=result.filename,
                decision=result.final_decision,
                confidence=0.9,
                scores=scores,
                processing_time=0.1
            )
            
            results.append(decision_result)
        
        return results
    
    def _create_realistic_aggregated_results(self, decision_results):
        """Create realistic aggregated results for integration testing"""
        approved = sum(1 for dr in decision_results if dr.decision == 'approved')
        rejected = sum(1 for dr in decision_results if dr.decision == 'rejected')
        
        return AggregatedResults(
            total_images=len(decision_results),
            approved_count=approved,
            rejected_count=rejected,
            review_required_count=0,
            approval_rate=approved / len(decision_results),
            avg_quality_score=0.75,
            avg_overall_score=0.73,
            rejection_breakdown={'quality': rejected},
            top_rejection_reasons=[('low_quality', rejected)] if rejected > 0 else [],
            processing_statistics={'total_processing_time': 5.0}
        )


if __name__ == '__main__':
    unittest.main()