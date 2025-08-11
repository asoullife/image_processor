#!/usr/bin/env python3
"""Create Excel report manually from database"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd
from datetime import datetime
import os
from backend.utils.path_utils import get_database_path, get_reports_dir

def create_excel_report():
    """Create Excel report from database data"""
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
        
        session_id = df_session.iloc[0]['session_id']
        
        # Get image results
        results_query = '''
            SELECT * FROM image_results 
            WHERE session_id = ?
            ORDER BY processed_at
        '''
        df_results = pd.read_sql_query(results_query, conn, params=[session_id])
        
        # Create Excel file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_filename = os.path.join(get_reports_dir(), f'adobe_stock_report_{timestamp}.xlsx')
        
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = {
                'Metric': [
                    'Session ID',
                    'Input Folder', 
                    'Output Folder',
                    'Total Images',
                    'Processed Images',
                    'Approved Images',
                    'Rejected Images',
                    'Approval Rate (%)',
                    'Processing Status',
                    'Start Time',
                    'End Time'
                ],
                'Value': [
                    df_session.iloc[0]['session_id'],
                    df_session.iloc[0]['input_folder'],
                    df_session.iloc[0]['output_folder'],
                    df_session.iloc[0]['total_images'],
                    df_session.iloc[0]['processed_images'],
                    df_session.iloc[0]['approved_images'],
                    df_session.iloc[0]['rejected_images'],
                    f"{(df_session.iloc[0]['approved_images'] / df_session.iloc[0]['total_images'] * 100):.1f}%" if df_session.iloc[0]['total_images'] > 0 else "0.0%",
                    df_session.iloc[0]['status'],
                    df_session.iloc[0]['start_time'],
                    df_session.iloc[0]['end_time']
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Detailed results sheet
            if not df_results.empty:
                # Clean up the data for Excel
                excel_results = df_results.copy()
                excel_results['rejection_reasons'] = excel_results['rejection_reasons'].apply(
                    lambda x: str(x).replace("['", "").replace("']", "").replace("', '", ", ") if x else ""
                )
                
                excel_results.to_excel(writer, sheet_name='Detailed Results', index=False)
                
                # Approved images sheet
                approved_df = excel_results[excel_results['final_decision'] == 'approved']
                if not approved_df.empty:
                    approved_df.to_excel(writer, sheet_name='Approved Images', index=False)
                
                # Rejected images sheet  
                rejected_df = excel_results[excel_results['final_decision'] == 'rejected']
                if not rejected_df.empty:
                    rejected_df.to_excel(writer, sheet_name='Rejected Images', index=False)
                
                # Statistics sheet
                stats_data = {
                    'Statistic': [
                        'Total Images',
                        'Average Processing Time (s)',
                        'Min Processing Time (s)',
                        'Max Processing Time (s)',
                        'Average Quality Score',
                        'Average Defect Score'
                    ],
                    'Value': [
                        len(df_results),
                        f"{df_results['processing_time'].mean():.4f}",
                        f"{df_results['processing_time'].min():.4f}",
                        f"{df_results['processing_time'].max():.4f}",
                        f"{df_results['quality_score'].mean():.3f}" if df_results['quality_score'].notna().any() else "N/A",
                        f"{df_results['defect_score'].mean():.3f}" if df_results['defect_score'].notna().any() else "N/A"
                    ]
                }
                
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        conn.close()
        
        print(f"✅ Excel report created: {excel_filename}")
        print(f"📁 File location: {os.path.abspath(excel_filename)}")
        
        # Show file size
        file_size = os.path.getsize(excel_filename)
        print(f"📊 File size: {file_size:,} bytes")
        
        return excel_filename
        
    except Exception as e:
        print(f"❌ Error creating Excel report: {e}")
        return None

if __name__ == "__main__":
    create_excel_report()