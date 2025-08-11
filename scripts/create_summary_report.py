#!/usr/bin/env python3
"""Create comprehensive summary report"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd
from datetime import datetime
import os
from backend.utils.path_utils import get_database_path, get_reports_dir

def create_summary_report():
    """Create comprehensive text summary report"""
    try:
        # Connect to database
        conn = sqlite3.connect(get_database_path())
        
        # Get latest session
        session_query = '''
            SELECT * FROM processing_sessions 
            ORDER BY created_at DESC LIMIT 1
        '''
        df_session = pd.read_sql_query(session_query, conn)
        
        if df_session.empty:
            print("No sessions found")
            return
        
        session_data = df_session.iloc[0]
        session_id = session_data['session_id']
        
        # Get image results
        results_query = '''
            SELECT * FROM image_results 
            WHERE session_id = ?
            ORDER BY processed_at
        '''
        df_results = pd.read_sql_query(results_query, conn, params=[session_id])
        
        conn.close()
        
        # Calculate statistics
        total_images = len(df_results)
        approved_count = len(df_results[df_results['final_decision'] == 'approved'])
        rejected_count = len(df_results[df_results['final_decision'] == 'rejected'])
        approval_rate = (approved_count / total_images * 100) if total_images > 0 else 0
        
        # Processing time statistics
        avg_time = df_results['processing_time'].mean()
        min_time = df_results['processing_time'].min()
        max_time = df_results['processing_time'].max()
        total_time = df_results['processing_time'].sum()
        
        # Calculate session duration
        start_time = pd.to_datetime(session_data['start_time'])
        end_time = pd.to_datetime(session_data['end_time'])
        session_duration = (end_time - start_time).total_seconds()
        
        # Analyze rejection reasons
        rejection_stats = {}
        for _, row in df_results.iterrows():
            if row['rejection_reasons']:
                try:
                    reasons = eval(row['rejection_reasons']) if isinstance(row['rejection_reasons'], str) else [row['rejection_reasons']]
                    for reason in reasons:
                        rejection_stats[reason] = rejection_stats.get(reason, 0) + 1
                except:
                    reason = str(row['rejection_reasons'])
                    rejection_stats[reason] = rejection_stats.get(reason, 0) + 1
        
        # Create summary report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = os.path.join(get_reports_dir(), f'adobe_stock_summary_{timestamp}.txt')
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\\n")
            f.write("ADOBE STOCK IMAGE PROCESSOR - COMPREHENSIVE SUMMARY REPORT\\n")
            f.write("=" * 80 + "\\n\\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Report Type: Processing Summary\\n")
            f.write(f"Data Source: SQLite Database\\n\\n")
            
            # Session Information
            f.write("SESSION INFORMATION\\n")
            f.write("-" * 40 + "\\n")
            f.write(f"Session ID: {session_data['session_id']}\\n")
            f.write(f"Input Folder: {session_data['input_folder']}\\n")
            f.write(f"Output Folder: {session_data['output_folder']}\\n")
            f.write(f"Processing Status: {session_data['status']}\\n")
            f.write(f"Start Time: {session_data['start_time']}\\n")
            f.write(f"End Time: {session_data['end_time']}\\n")
            f.write(f"Session Duration: {session_duration:.2f} seconds\\n\\n")
            
            # Processing Statistics
            f.write("PROCESSING STATISTICS\\n")
            f.write("-" * 40 + "\\n")
            f.write(f"Total Images Found: {session_data['total_images']}\\n")
            f.write(f"Images Processed: {session_data['processed_images']}\\n")
            f.write(f"Images Approved: {approved_count} ({approval_rate:.1f}%)\\n")
            f.write(f"Images Rejected: {rejected_count} ({100-approval_rate:.1f}%)\\n\\n")
            
            # Performance Metrics
            f.write("PERFORMANCE METRICS\\n")
            f.write("-" * 40 + "\\n")
            f.write(f"Average Processing Time: {avg_time:.4f} seconds/image\\n")
            f.write(f"Minimum Processing Time: {min_time:.4f} seconds\\n")
            f.write(f"Maximum Processing Time: {max_time:.4f} seconds\\n")
            f.write(f"Total Processing Time: {total_time:.4f} seconds\\n")
            f.write(f"Processing Rate: {total_images/session_duration:.2f} images/second\\n")
            f.write(f"Throughput: {total_images/session_duration*3600:.0f} images/hour\\n\\n")
            
            # Quality Analysis
            f.write("QUALITY ANALYSIS\\n")
            f.write("-" * 40 + "\\n")
            quality_scores = df_results['quality_score'].dropna()
            if not quality_scores.empty:
                f.write(f"Average Quality Score: {quality_scores.mean():.3f}\\n")
                f.write(f"Quality Score Range: {quality_scores.min():.3f} - {quality_scores.max():.3f}\\n")
                f.write(f"Quality Score Std Dev: {quality_scores.std():.3f}\\n")
            else:
                f.write("Quality scores not available (processing errors)\\n")
            
            defect_scores = df_results['defect_score'].dropna()
            if not defect_scores.empty:
                f.write(f"Average Defect Score: {defect_scores.mean():.3f}\\n")
                f.write(f"Defect Score Range: {defect_scores.min():.3f} - {defect_scores.max():.3f}\\n")
            else:
                f.write("Defect scores not available (processing errors)\\n")
            f.write("\\n")
            
            # Rejection Analysis
            f.write("REJECTION ANALYSIS\\n")
            f.write("-" * 40 + "\\n")
            if rejection_stats:
                f.write("Top Rejection Reasons:\\n")
                sorted_reasons = sorted(rejection_stats.items(), key=lambda x: x[1], reverse=True)
                for i, (reason, count) in enumerate(sorted_reasons[:10], 1):
                    percentage = (count / total_images * 100)
                    f.write(f"  {i}. {reason}: {count} images ({percentage:.1f}%)\\n")
            else:
                f.write("No rejection reasons found\\n")
            f.write("\\n")
            
            # Detailed Results
            f.write("DETAILED PROCESSING RESULTS\\n")
            f.write("-" * 40 + "\\n")
            f.write(f"{'Filename':<25} {'Decision':<10} {'Time(s)':<8} {'Quality':<8} {'Status'}\\n")
            f.write("-" * 80 + "\\n")
            
            for _, row in df_results.iterrows():
                filename = row['filename'][:24]  # Truncate long filenames
                decision = row['final_decision'][:9]
                proc_time = f"{row['processing_time']:.3f}"
                quality = f"{row['quality_score']:.3f}" if pd.notna(row['quality_score']) else "N/A"
                
                # Get first rejection reason
                status = "OK"
                if row['rejection_reasons']:
                    try:
                        reasons = eval(row['rejection_reasons']) if isinstance(row['rejection_reasons'], str) else [row['rejection_reasons']]
                        status = reasons[0][:30] if reasons else "Unknown"
                    except:
                        status = str(row['rejection_reasons'])[:30]
                
                f.write(f"{filename:<25} {decision:<10} {proc_time:<8} {quality:<8} {status}\\n")
            
            f.write("\\n")
            
            # System Information
            f.write("SYSTEM INFORMATION\\n")
            f.write("-" * 40 + "\\n")
            f.write(f"Database File: adobe_stock_processor.db\\n")
            f.write(f"Configuration: Default settings\\n")
            f.write(f"Processing Mode: Batch processing\\n")
            f.write(f"Error Handling: Graceful degradation\\n\\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\\n")
            f.write("-" * 40 + "\\n")
            if approval_rate == 0:
                f.write("• All images were rejected - check processing configuration\\n")
                f.write("• Review quality thresholds and analyzer settings\\n")
                f.write("• Verify input image quality and format compatibility\\n")
            elif approval_rate < 20:
                f.write("• Low approval rate - consider adjusting quality thresholds\\n")
                f.write("• Review rejection reasons for common issues\\n")
            elif approval_rate > 80:
                f.write("• High approval rate - processing working well\\n")
                f.write("• Consider increasing quality standards if needed\\n")
            
            if avg_time > 1.0:
                f.write("• Processing time is high - consider performance optimization\\n")
            elif avg_time < 0.1:
                f.write("• Very fast processing - may indicate processing errors\\n")
            
            f.write("\\n")
            f.write("=" * 80 + "\\n")
            f.write("END OF REPORT\\n")
            f.write("=" * 80 + "\\n")
        
        print(f"✅ Summary report created: {report_filename}")
        print(f"📁 File location: {os.path.abspath(report_filename)}")
        
        # Show file info
        file_size = os.path.getsize(report_filename)
        print(f"📊 File size: {file_size:,} bytes")
        
        return report_filename
        
    except Exception as e:
        print(f"❌ Error creating summary report: {e}")
        return None

if __name__ == "__main__":
    create_summary_report()