#!/usr/bin/env python3
"""Create CSV report from database"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd
from datetime import datetime
import os
from backend.utils.path_utils import get_database_path, get_reports_dir

def create_csv_report():
    """Create CSV report from database data"""
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
            SELECT 
                filename,
                image_path,
                final_decision,
                rejection_reasons,
                processing_time,
                quality_score,
                defect_score,
                similarity_group,
                compliance_status,
                file_size,
                processed_at
            FROM image_results 
            WHERE session_id = ?
            ORDER BY processed_at
        '''
        df_results = pd.read_sql_query(results_query, conn, params=[session_id])
        
        conn.close()
        
        # Clean up data for CSV
        if not df_results.empty:
            # Clean rejection reasons
            df_results['rejection_reasons'] = df_results['rejection_reasons'].apply(
                lambda x: str(x).replace("['", "").replace("']", "").replace("', '", "; ") if x else ""
            )
            
            # Add session info columns
            df_results['session_id'] = session_id
            df_results['input_folder'] = df_session.iloc[0]['input_folder']
            df_results['output_folder'] = df_session.iloc[0]['output_folder']
        
        # Save CSV file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = os.path.join(get_reports_dir(), f'adobe_stock_results_{timestamp}.csv')
        
        df_results.to_csv(csv_filename, index=False, encoding='utf-8')
        
        print(f"✅ CSV report created: {csv_filename}")
        print(f"📁 File location: {os.path.abspath(csv_filename)}")
        
        # Show file info
        file_size = os.path.getsize(csv_filename)
        print(f"📊 File size: {file_size:,} bytes")
        print(f"📋 Records: {len(df_results)}")
        print(f"📊 Columns: {len(df_results.columns)}")
        
        # Show column names
        print(f"📝 Columns: {', '.join(df_results.columns)}")
        
        return csv_filename
        
    except Exception as e:
        print(f"❌ Error creating CSV report: {e}")
        return None

if __name__ == "__main__":
    create_csv_report()