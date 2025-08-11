#!/usr/bin/env python3
"""Test script for multi-session project management functionality."""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from database.connection import DatabaseManager
from database.models import Project, ProcessingSession
from sqlalchemy import select

async def test_multi_session_management():
    """Test multi-session project management functionality."""
    print("🧪 Testing Multi-Session Project Management")
    print("=" * 50)
    
    try:
        # Initialize database with SQLite
        db_manager = DatabaseManager("sqlite+aiosqlite:///./test_multi_session.db")
        await db_manager.initialize()
        await db_manager.create_tables()  # Create tables for testing
        
        print("✅ Database initialized successfully")
        
        # Test 1: Create multiple projects directly
        print("\n📁 Test 1: Creating multiple projects")
        
        projects = []
        async with db_manager.get_session() as session:
            for i in range(3):
                input_folder = f"/test/input/project{i+1}"
                output_folder = f"{input_folder}_processed"  # Auto-generate
                
                project = Project(
                    name=f"Test Project {i+1}",
                    description=f"Test project for multi-session management {i+1}",
                    input_folder=input_folder,
                    output_folder=output_folder,
                    performance_mode="balanced",
                    status="created"
                )
                session.add(project)
                projects.append(project)
            
            await session.flush()
            for project in projects:
                await session.refresh(project)
                print(f"  ✅ Created project: {project.name} (ID: {project.id})")
                print(f"     Output folder: {project.output_folder}")
        
        # Test 2: Create processing sessions
        print("\n🚀 Test 2: Creating processing sessions")
        
        sessions = []
        async with db_manager.get_session() as db_session:
            for project in projects:
                processing_session = ProcessingSession(
                    project_id=project.id,
                    total_images=100,
                    status="running",
                    session_config={
                        "performance_mode": project.performance_mode,
                        "input_folder": project.input_folder,
                        "output_folder": project.output_folder
                    }
                )
                db_session.add(processing_session)
                sessions.append(processing_session)
            
            await db_session.flush()
            for session in sessions:
                await db_session.refresh(session)
                print(f"  ✅ Created session for project {session.project_id} (Session ID: {session.id})")
        
        # Test 3: Query concurrent sessions
        print("\n📊 Test 3: Querying concurrent sessions")
        
        async with db_manager.get_session() as db_session:
            stmt = (
                select(ProcessingSession)
                .where(ProcessingSession.status.in_(["running", "created"]))
                .order_by(ProcessingSession.created_at.desc())
            )
            result = await db_session.execute(stmt)
            concurrent_sessions = result.scalars().all()
            
            print(f"  ✅ Found {len(concurrent_sessions)} concurrent sessions")
            for session in concurrent_sessions:
                print(f"     - Session {session.id}: {session.status}")
        
        # Test 4: Update session progress
        print("\n📈 Test 4: Updating session progress")
        
        async with db_manager.get_session() as db_session:
            for i, session in enumerate(sessions):
                session.processed_images = 10 * (i + 1)
                session.approved_images = 8 * (i + 1)
                session.rejected_images = 2 * (i + 1)
                print(f"  ✅ Updated progress for session {session.id}")
        
        # Test 5: Query project statistics
        print("\n📊 Test 5: Querying project statistics")
        
        async with db_manager.get_session() as db_session:
            for project in projects:
                # Get sessions for this project
                stmt = select(ProcessingSession).where(ProcessingSession.project_id == project.id)
                result = await db_session.execute(stmt)
                project_sessions = result.scalars().all()
                
                total_processed = sum(s.processed_images for s in project_sessions)
                total_approved = sum(s.approved_images for s in project_sessions)
                approval_rate = (total_approved / total_processed * 100) if total_processed > 0 else 0
                
                print(f"  ✅ Stats for {project.name}:")
                print(f"     - Total processed: {total_processed}")
                print(f"     - Approval rate: {approval_rate:.1f}%")
        
        # Test 6: Query all sessions (history)
        print("\n📜 Test 6: Querying session history")
        
        async with db_manager.get_session() as db_session:
            stmt = (
                select(ProcessingSession)
                .order_by(ProcessingSession.created_at.desc())
                .limit(10)
            )
            result = await db_session.execute(stmt)
            history = result.scalars().all()
            
            print(f"  ✅ Found {len(history)} sessions in history")
            for session in history[:3]:  # Show first 3
                print(f"     - Session {session.id}: {session.status} ({session.processed_images} processed)")
        
        # Test 7: Query active projects
        print("\n🔥 Test 7: Querying active projects")
        
        async with db_manager.get_session() as db_session:
            stmt = (
                select(Project)
                .where(Project.status.in_(["running", "paused"]))
                .order_by(Project.updated_at.desc())
            )
            result = await db_session.execute(stmt)
            active_projects = result.scalars().all()
            
            print(f"  ✅ Found {len(active_projects)} active projects")
            for project in active_projects:
                print(f"     - {project.name}: {project.status}")
        
        # Test 8: Update project status (pause)
        print("\n⏸️ Test 8: Updating project status")
        
        async with db_manager.get_session() as db_session:
            # Pause first project
            projects[0].status = "paused"
            print(f"  ✅ Paused project {projects[0].name}")
            
            # Count running vs paused
            running_projects = [p for p in projects if p.status == "running"]
            paused_projects = [p for p in projects if p.status == "paused"]
            print(f"  ✅ Status: {len(running_projects)} running, {len(paused_projects)} paused")
        
        print("\n🎉 All tests completed successfully!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'db_manager' in locals():
            await db_manager.cleanup()

if __name__ == "__main__":
    success = asyncio.run(test_multi_session_management())
    sys.exit(0 if success else 1)