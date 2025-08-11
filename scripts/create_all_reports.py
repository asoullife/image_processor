#!/usr/bin/env python3
"""Create all types of reports at once"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
from datetime import datetime
from backend.utils.path_utils import get_reports_dir

def create_all_reports():
    """Create all report types"""
    print("🚀 Creating all report types...")
    print("=" * 50)
    
    reports_created = []
    
    # Database report (display only)
    print("\n1. 🗃️ Database Report:")
    os.system("python scripts/view_database_report.py")
    
    # Excel report
    print("\n2. 📊 Creating Excel Report...")
    result = os.system("python scripts/create_excel_report.py")
    if result == 0:
        reports_created.append("Excel")
    
    # HTML dashboard
    print("\n3. 🌐 Creating HTML Dashboard...")
    result = os.system("python scripts/create_html_dashboard.py")
    if result == 0:
        reports_created.append("HTML")
    
    # CSV report
    print("\n4. 📈 Creating CSV Report...")
    result = os.system("python scripts/create_csv_report.py")
    if result == 0:
        reports_created.append("CSV")
    
    # Summary report
    print("\n5. 📄 Creating Summary Report...")
    result = os.system("python scripts/create_summary_report.py")
    if result == 0:
        reports_created.append("Summary")
    
    print("\n" + "=" * 50)
    print("🎉 Report Generation Complete!")
    print(f"✅ Created: {', '.join(reports_created)}")
    
    # List all report files
    print("\n📁 Generated Files:")
    import glob
    reports_dir = get_reports_dir()
    report_files = glob.glob(os.path.join(reports_dir, "adobe_stock_*.*"))
    for file in sorted(report_files):
        size = os.path.getsize(file)
        filename = os.path.basename(file)
        print(f"  📄 {filename} ({size:,} bytes)")
    
    print(f"\n💡 To view reports:")
    excel_files = [f for f in report_files if f.endswith('.xlsx')]
    html_files = [f for f in report_files if f.endswith('.html')]
    csv_files = [f for f in report_files if f.endswith('.csv')]
    txt_files = [f for f in report_files if f.endswith('.txt')]
    
    if excel_files:
        print(f"  📊 Excel: start {max(excel_files, default='')}")
    if html_files:
        print(f"  🌐 HTML: start {max(html_files, default='')}")
    if csv_files:
        print(f"  📈 CSV: start {max(csv_files, default='')}")
    if txt_files:
        print(f"  📄 Summary: start {max(txt_files, default='')}")

if __name__ == "__main__":
    create_all_reports()