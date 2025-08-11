#!/usr/bin/env python3
"""Create HTML dashboard from database"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd
from datetime import datetime
import os
from backend.utils.path_utils import get_database_path, get_reports_dir

def create_html_dashboard():
    """Create HTML dashboard from database data"""
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
        session_data = df_session.iloc[0]
        
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
        avg_processing_time = df_results['processing_time'].mean()
        
        # Create HTML content
        html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Adobe Stock Image Processor - Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 20px;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
        }}
        .header p {{
            color: #7f8c8d;
            margin: 5px 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .stat-card.approved {{
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        }}
        .stat-card.rejected {{
            background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
        }}
        .stat-card.rate {{
            background: linear-gradient(135deg, #2196F3 0%, #0b7dda 100%);
        }}
        .stat-card.time {{
            background: linear-gradient(135deg, #FF9800 0%, #e68900 100%);
        }}
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 0;
        }}
        .stat-label {{
            font-size: 1.1em;
            margin: 5px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .info-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        .info-card h3 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
        }}
        .info-item {{
            margin: 5px 0;
            display: flex;
            justify-content: space-between;
        }}
        .info-label {{
            font-weight: bold;
            color: #34495e;
        }}
        .info-value {{
            color: #7f8c8d;
        }}
        .results-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .results-table th,
        .results-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .results-table th {{
            background-color: #3498db;
            color: white;
        }}
        .results-table tr:hover {{
            background-color: #f5f5f5;
        }}
        .status-approved {{
            color: #27ae60;
            font-weight: bold;
        }}
        .status-rejected {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🖼️ Adobe Stock Image Processor</h1>
            <p>Processing Dashboard</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{total_images}</div>
                <div class="stat-label">Total Images</div>
            </div>
            <div class="stat-card approved">
                <div class="stat-number">{approved_count}</div>
                <div class="stat-label">Approved</div>
            </div>
            <div class="stat-card rejected">
                <div class="stat-number">{rejected_count}</div>
                <div class="stat-label">Rejected</div>
            </div>
            <div class="stat-card rate">
                <div class="stat-number">{approval_rate:.1f}%</div>
                <div class="stat-label">Approval Rate</div>
            </div>
            <div class="stat-card time">
                <div class="stat-number">{avg_processing_time:.3f}s</div>
                <div class="stat-label">Avg Time</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Session Information</h2>
            <div class="info-grid">
                <div class="info-card">
                    <h3>Session Details</h3>
                    <div class="info-item">
                        <span class="info-label">Session ID:</span>
                        <span class="info-value">{session_data['session_id']}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Status:</span>
                        <span class="info-value">{session_data['status']}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Start Time:</span>
                        <span class="info-value">{session_data['start_time']}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">End Time:</span>
                        <span class="info-value">{session_data['end_time']}</span>
                    </div>
                </div>
                <div class="info-card">
                    <h3>Folder Paths</h3>
                    <div class="info-item">
                        <span class="info-label">Input:</span>
                        <span class="info-value">{session_data['input_folder']}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Output:</span>
                        <span class="info-value">{session_data['output_folder']}</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>📋 Processing Results</h2>
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Filename</th>
                        <th>Decision</th>
                        <th>Processing Time</th>
                        <th>Quality Score</th>
                        <th>Rejection Reasons</th>
                    </tr>
                </thead>
                <tbody>
'''
        
        # Add table rows
        for _, row in df_results.iterrows():
            status_class = "status-approved" if row['final_decision'] == 'approved' else "status-rejected"
            status_icon = "✅" if row['final_decision'] == 'approved' else "❌"
            
            # Clean up rejection reasons
            reasons = str(row['rejection_reasons']).replace("['", "").replace("']", "").replace("', '", ", ") if row['rejection_reasons'] else ""
            
            quality_score = f"{row['quality_score']:.3f}" if pd.notna(row['quality_score']) else "N/A"
            
            html_content += f'''
                    <tr>
                        <td>{row['filename']}</td>
                        <td class="{status_class}">{status_icon} {row['final_decision'].title()}</td>
                        <td>{row['processing_time']:.3f}s</td>
                        <td>{quality_score}</td>
                        <td>{reasons}</td>
                    </tr>
'''
        
        html_content += '''
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Adobe Stock Image Processor Dashboard</p>
            <p>Powered by Python & SQLite</p>
        </div>
    </div>
</body>
</html>
'''
        
        # Save HTML file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_filename = os.path.join(get_reports_dir(), f'adobe_stock_dashboard_{timestamp}.html')
        
        with open(html_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML dashboard created: {html_filename}")
        print(f"📁 File location: {os.path.abspath(html_filename)}")
        
        # Show file size
        file_size = os.path.getsize(html_filename)
        print(f"📊 File size: {file_size:,} bytes")
        
        return html_filename
        
    except Exception as e:
        print(f"❌ Error creating HTML dashboard: {e}")
        return None

if __name__ == "__main__":
    create_html_dashboard()