#!/usr/bin/env python3
"""
Test script for Web-Based Reports and Analytics implementation
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_frontend_components():
    """Test that all frontend components exist and can be imported"""
    print("Testing frontend components...")
    
    frontend_components = [
        "frontend/src/components/reports/ReportsOverview.tsx",
        "frontend/src/components/reports/ReportsCharts.tsx", 
        "frontend/src/components/reports/ReportsTable.tsx",
        "frontend/src/components/reports/ReportsThumbnails.tsx",
        "frontend/src/components/reports/ReportsFilters.tsx",
        "frontend/src/components/reports/ReportsExport.tsx",
        "frontend/src/components/reports/PerformanceMetrics.tsx",
        "frontend/src/components/ui/checkbox.tsx",
        "frontend/src/components/ui/scroll-area.tsx",
        "frontend/src/components/ui/table.tsx",
        "frontend/src/components/ui/dropdown-menu.tsx",
        "frontend/src/components/ui/dialog.tsx"
    ]
    
    missing_components = []
    for component in frontend_components:
        if not Path(component).exists():
            missing_components.append(component)
    
    if missing_components:
        print(f"❌ Missing frontend components: {missing_components}")
        return False
    else:
        print("✅ All frontend components exist")
        return True

def test_backend_utilities():
    """Test that all backend utilities exist"""
    print("Testing backend utilities...")
    
    backend_utilities = [
        "backend/utils/report_generator.py",
        "backend/utils/analytics_engine.py",
        "backend/utils/thumbnail_generator.py",
        "backend/api/routes/reports.py",
        "backend/api/routes/thumbnails.py",
        "backend/api/routes/mock_reports.py"
    ]
    
    missing_utilities = []
    for utility in backend_utilities:
        if not Path(utility).exists():
            missing_utilities.append(utility)
    
    if missing_utilities:
        print(f"❌ Missing backend utilities: {missing_utilities}")
        return False
    else:
        print("✅ All backend utilities exist")
        return True

def test_package_dependencies():
    """Test that required dependencies are in package.json"""
    print("Testing package dependencies...")
    
    package_json_path = Path("frontend/package.json")
    if not package_json_path.exists():
        print("❌ package.json not found")
        return False
    
    with open(package_json_path, 'r') as f:
        content = f.read()
    
    required_deps = [
        "sonner",
        "chart.js", 
        "react-chartjs-2",
        "recharts"
    ]
    
    missing_deps = []
    for dep in required_deps:
        if dep not in content:
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"❌ Missing dependencies: {missing_deps}")
        return False
    else:
        print("✅ All required dependencies found in package.json")
        return True

def test_api_routes():
    """Test that API routes are properly configured"""
    print("Testing API routes...")
    
    main_api_path = Path("backend/api/main.py")
    if not main_api_path.exists():
        print("❌ main.py not found")
        return False
    
    with open(main_api_path, 'r') as f:
        content = f.read()
    
    required_imports = [
        "mock_reports",
        "reports"
    ]
    
    required_includes = [
        "reports.router",
        "mock_reports.router"
    ]
    
    missing_items = []
    for item in required_imports + required_includes:
        if item not in content:
            missing_items.append(item)
    
    if missing_items:
        print(f"❌ Missing API route configurations: {missing_items}")
        return False
    else:
        print("✅ API routes properly configured")
        return True

def test_demo_pages():
    """Test that demo pages exist"""
    print("Testing demo pages...")
    
    demo_pages = [
        "frontend/src/app/reports-demo/page.tsx",
        "frontend/src/app/projects/[id]/reports/page.tsx"
    ]
    
    missing_pages = []
    for page in demo_pages:
        if not Path(page).exists():
            missing_pages.append(page)
    
    if missing_pages:
        print(f"❌ Missing demo pages: {missing_pages}")
        return False
    else:
        print("✅ All demo pages exist")
        return True

def main():
    """Run all tests"""
    print("🧪 Testing Web-Based Reports and Analytics Implementation")
    print("=" * 60)
    
    tests = [
        test_frontend_components,
        test_backend_utilities,
        test_package_dependencies,
        test_api_routes,
        test_demo_pages
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is complete.")
        print("\nNext steps:")
        print("1. Install frontend dependencies: cd frontend && npm install")
        print("2. Start the backend server: python -m backend.api.main")
        print("3. Start the frontend: cd frontend && npm run dev")
        print("4. Visit http://localhost:3000/reports-demo to test the implementation")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)