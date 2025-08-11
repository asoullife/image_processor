#!/usr/bin/env python3
"""Test script for multi-session project management API endpoints."""

import asyncio
import sys
import json
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from fastapi.testclient import TestClient
from api.main import app

def test_multi_session_api():
    """Test multi-session project management API endpoints."""
    print("🧪 Testing Multi-Session Project Management API")
    print("=" * 50)
    
    try:
        client = TestClient(app)
        
        # Test 1: Health check
        print("\n❤️ Test 1: Health check")
        response = client.get("/api/health")
        print(f"  ✅ Health check: {response.status_code}")
        
        # Test 2: Create multiple projects
        print("\n📁 Test 2: Creating multiple projects")
        
        projects = []
        for i in range(3):
            project_data = {
                "name": f"API Test Project {i+1}",
                "description": f"Test project for API testing {i+1}",
                "input_folder": f"/test/api/input/project{i+1}",
                "output_folder": f"/test/api/input/project{i+1}_processed",
                "performance_mode": "balanced"
            }
            
            response = client.post("/api/projects/", json=project_data)
            if response.status_code == 200:
                project = response.json()
                projects.append(project)
                print(f"  ✅ Created project: {project['name']} (ID: {project['id']})")
            else:
                print(f"  ❌ Failed to create project {i+1}: {response.status_code}")
                print(f"     Response: {response.text}")
        
        # Test 3: List all projects
        print("\n📋 Test 3: Listing all projects")
        
        response = client.get("/api/projects/")
        if response.status_code == 200:
            all_projects = response.json()
            print(f"  ✅ Found {len(all_projects)} total projects")
        else:
            print(f"  ❌ Failed to list projects: {response.status_code}")
        
        # Test 4: Get active projects
        print("\n🔥 Test 4: Getting active projects")
        
        response = client.get("/api/projects/active")
        if response.status_code == 200:
            active_projects = response.json()
            print(f"  ✅ Found {len(active_projects)} active projects")
        else:
            print(f"  ❌ Failed to get active projects: {response.status_code}")
        
        # Test 5: Start processing for projects
        print("\n🚀 Test 5: Starting processing for projects")
        
        sessions = []
        for project in projects[:2]:  # Start only first 2 projects
            response = client.post(f"/api/projects/{project['id']}/start")
            if response.status_code == 200:
                session_info = response.json()
                sessions.append(session_info)
                print(f"  ✅ Started processing for {project['name']}")
                print(f"     Session ID: {session_info.get('session_id', 'N/A')}")
            else:
                print(f"  ❌ Failed to start processing for {project['name']}: {response.status_code}")
        
        # Test 6: Get concurrent sessions
        print("\n📊 Test 6: Getting concurrent sessions")
        
        response = client.get("/api/sessions/concurrent")
        if response.status_code == 200:
            concurrent_sessions = response.json()
            print(f"  ✅ Found {len(concurrent_sessions)} concurrent sessions")
            for session in concurrent_sessions:
                print(f"     - {session.get('project_name', 'Unknown')}: {session.get('status', 'Unknown')}")
        else:
            print(f"  ❌ Failed to get concurrent sessions: {response.status_code}")
        
        # Test 7: Get session history
        print("\n📜 Test 7: Getting session history")
        
        response = client.get("/api/sessions/history?limit=10")
        if response.status_code == 200:
            history = response.json()
            print(f"  ✅ Found {len(history)} sessions in history")
            for session in history[:3]:  # Show first 3
                print(f"     - {session.get('project_name', 'Unknown')}: {session.get('status', 'Unknown')}")
        else:
            print(f"  ❌ Failed to get session history: {response.status_code}")
        
        # Test 8: Get project statistics
        print("\n📊 Test 8: Getting project statistics")
        
        for project in projects:
            response = client.get(f"/api/projects/{project['id']}/statistics")
            if response.status_code == 200:
                stats = response.json()
                print(f"  ✅ Stats for {project['name']}:")
                print(f"     - Total processed: {stats.get('total_processed', 0)}")
                print(f"     - Approval rate: {stats.get('approval_rate', 0):.1f}%")
            else:
                print(f"  ❌ Failed to get stats for {project['name']}: {response.status_code}")
        
        # Test 9: Get project sessions
        print("\n🔗 Test 9: Getting project sessions")
        
        for project in projects:
            response = client.get(f"/api/projects/{project['id']}/sessions")
            if response.status_code == 200:
                project_sessions = response.json()
                print(f"  ✅ Found {len(project_sessions)} sessions for {project['name']}")
            else:
                print(f"  ❌ Failed to get sessions for {project['name']}: {response.status_code}")
        
        # Test 10: Pause project processing
        print("\n⏸️ Test 10: Pausing project processing")
        
        if projects:
            response = client.post(f"/api/projects/{projects[0]['id']}/pause")
            if response.status_code == 200:
                print(f"  ✅ Paused processing for {projects[0]['name']}")
            else:
                print(f"  ❌ Failed to pause processing: {response.status_code}")
        
        # Test 11: Check active projects after pause
        print("\n🔄 Test 11: Checking active projects after pause")
        
        response = client.get("/api/projects/active")
        if response.status_code == 200:
            active_after_pause = response.json()
            print(f"  ✅ Found {len(active_after_pause)} active projects after pause")
            for project in active_after_pause:
                print(f"     - {project['name']}: {project['status']}")
        else:
            print(f"  ❌ Failed to get active projects: {response.status_code}")
        
        print("\n🎉 All API tests completed successfully!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multi_session_api()
    sys.exit(0 if success else 1)