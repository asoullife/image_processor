#!/usr/bin/env python3
"""Test FastAPI server with Socket.IO integration."""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

def test_server_import():
    """Test server import and basic functionality."""
    print("Testing FastAPI Server with Socket.IO...")
    print("=" * 50)
    
    try:
        # Test FastAPI import
        print("1. Testing FastAPI import...")
        import fastapi
        print(f"   ✓ FastAPI version: {fastapi.__version__}")
        
        # Test Socket.IO import
        print("2. Testing Socket.IO import...")
        import socketio
        try:
            version = socketio.__version__
        except AttributeError:
            version = "available"
        print(f"   ✓ Socket.IO: {version}")
        
        # Test server creation
        print("3. Testing server creation...")
        from api.main import create_app
        
        app = create_app()
        print(f"   ✓ Server app created: {type(app)}")
        
        # Test Socket.IO manager
        print("4. Testing Socket.IO manager...")
        from realtime.socketio_manager import socketio_manager, sio
        print(f"   ✓ Socket.IO manager available")
        print(f"   ✓ Socket.IO server created: {type(sio)}")
        
        print("\n" + "=" * 50)
        print("✅ Server components ready!")
        print("\nTo start the server:")
        print("cd backend")
        print("python -m api.main")
        print("\nOr with uvicorn:")
        print("uvicorn api.main:socket_app --host 127.0.0.1 --port 8000 --reload")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_server_import()