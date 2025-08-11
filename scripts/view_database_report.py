#!/usr/bin/env python3
"""View database report for Adobe Stock Image Processor"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd
from datetime import datetime
from backend.utils.path_utils import get_database_path

def view_database_report():
    """View processing results from database"""
    try:
        # Connect to database
        conn = sqlite3.connect(get_database_path())
        
        print("=" * 60)
        print("ADOBE STOCK IMAGE PROCESSOR - DATABASE REPORT")
        print("=" * 60)
        
        # Get latest session info
        print("\n📊 LATEST SESSION INFO:")
        print("-" * 40)
        df_sessions = pd.read_sql_query('''
            SELECT session_id, input_folder, output_folder, total_images, 
                   processed_images, approved_images, rejected_images,
                   status, start_time, end_time 
            FROM processing_sessions 
            ORDER BY created_at DESC LIMIT 1
        ''', conn)
        
        if not df_sessions.empty:
            session = df_sessions.iloc[0]
            print(f"Session ID: {session['session_id']}")
            print(f"Input Folder: {session['input_folder']}")
            print(f"Output Folder: {session['output_folder']}")
            print(f"Total Images: {session['total_images']}")
            print(f"Processed: {session['processed_images']}")
            print(f"Approved: {session['approved_images']}")
            print(f"Rejected: {session['rejected_images']}")
            print(f"Status: {session['status']}")
            print(f"Start Time: {session['start_time']}")
            print(f"End Time: {session['end_time']}")
            
            session_id = session['session_id']
        else:
            print("No sessions found")
            return
        
        # Get image results
        print(f"\n🖼️ IMAGE PROCESSING RESULTS:")
        print("-" * 40)
        df_results = pd.read_sql_query('''
            SELECT filename, final_decision, rejection_reasons, 
                   processing_time, quality_score, defect_score,
                   similarity_group, compliance_status, processed_at
            FROM image_results 
            WHERE session_id = ?
            ORDER BY processed_at
        ''', conn, params=[session_id])
        
        if not df_results.empty:
            print(f"Total images processed: {len(df_results)}")
            
            # Summary statistics
            approved = len(df_results[df_results['final_decision'] == 'approved'])
            rejected = len(df_results[df_results['final_decision'] == 'rejected'])
            
            print(f"✅ Approved: {approved} ({approved/len(df_results)*100:.1f}%)")
            print(f"❌ Rejected: {rejected} ({rejected/len(df_results)*100:.1f}%)")
            
            avg_time = df_results['processing_time'].mean()
            print(f"⏱️ Average processing time: {avg_time:.3f}s")
            
            print(f"\n📋 DETAILED RESULTS:")
            print("-" * 40)
            for idx, row in df_results.iterrows():
                status_icon = "✅" if row['final_decision'] == 'approved' else "❌"
                print(f"{status_icon} {row['filename']}")
                print(f"   Decision: {row['final_decision']}")
                print(f"   Time: {row['processing_time']:.3f}s")
                if row['rejection_reasons']:
                    reasons = eval(row['rejection_reasons']) if isinstance(row['rejection_reasons'], str) else row['rejection_reasons']
                    if reasons:
                        print(f"   Reasons: {', '.join(reasons)}")
                print()
        else:
            print("No image results found")
        
        # Get rejection reason statistics
        print(f"\n📈 REJECTION REASONS ANALYSIS:")
        print("-" * 40)
        
        rejection_stats = {}
        for idx, row in df_results.iterrows():
            if row['rejection_reasons']:
                try:
                    reasons = eval(row['rejection_reasons']) if isinstance(row['rejection_reasons'], str) else row['rejection_reasons']
                    for reason in reasons:
                        rejection_stats[reason] = rejection_stats.get(reason, 0) + 1
                except:
                    pass
        
        if rejection_stats:
            for reason, count in sorted(rejection_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"• {reason}: {count} images")
        else:
            print("No rejection reasons found")
        
        conn.close()
        
    except Exception as e:
        print(f"Error viewing database report: {e}")

if __name__ == "__main__":
    view_database_report()