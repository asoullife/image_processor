#!/usr/bin/env python3
"""Demo script to test resume functionality."""

import os
import sys
import time
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.progress_tracker import SQLiteProgressTracker
from backend.core.base import ProcessingResult


def create_interrupted_session():
    """Create a session that appears to be interrupted."""
    print("Creating interrupted session for resume testing...")
    
    # Initialize progress tracker
    from backend.utils.path_utils import get_database_path
    tracker = SQLiteProgressTracker(get_database_path(), checkpoint_interval=5)
    
    # Create session
    session_id = tracker.create_session(
        input_folder="backend/data/input",
        output_folder="test_output_resume",
        total_images=17,
        config={"demo": True, "batch_size": 5}
    )
    
    print(f"Created session: {session_id}")
    
    # Simulate processing first 8 images
    results = []
    for i in range(8):
        result = ProcessingResult(
            image_path=f"test_input/image_{i:03d}.jpg",
            filename=f"image_{i:03d}.jpg",
            final_decision="approved" if i % 3 != 0 else "rejected",
            rejection_reasons=["demo_reason"] if i % 3 == 0 else [],
            processing_time=0.1,
            timestamp=datetime.now()
        )
        results.append(result)
    
    # Save checkpoint (should trigger at 5 images)
    success = tracker.save_checkpoint(
        session_id=session_id,
        processed_count=8,
        total_count=17,
        results=results
    )
    
    if success:
        print(f"Saved checkpoint at 8/17 images")
        print("Session is now ready for resume testing")
        print(f"Session ID: {session_id}")
        
        # Show session info
        checkpoint_data = tracker.load_checkpoint(session_id)
        if checkpoint_data:
            print(f"Can resume: {checkpoint_data['can_resume']}")
            print(f"Processed: {checkpoint_data['processed_count']}/{checkpoint_data['total_count']}")
            print(f"Approved: {checkpoint_data['approved_count']}")
            print(f"Rejected: {checkpoint_data['rejected_count']}")
            if checkpoint_data['has_checkpoint']:
                print(f"Last checkpoint: {checkpoint_data['last_checkpoint_count']} images")
                print(f"Resume from index: {checkpoint_data.get('resume_from_index', 'N/A')}")
    else:
        print("Failed to save checkpoint")
    
    return session_id


def test_resume_detection():
    """Test resume detection functionality."""
    print("\nTesting resume detection...")
    
    tracker = SQLiteProgressTracker(get_database_path(), checkpoint_interval=5)
    
    # Get resumable sessions
    resumable = tracker.get_resumable_sessions()
    
    if resumable:
        print(f"Found {len(resumable)} resumable session(s):")
        for session in resumable:
            print(f"  - {session['session_id']}: {session['processed_images']}/{session['total_images']} images")
    else:
        print("No resumable sessions found")


def main():
    """Main demo function."""
    print("Resume Functionality Demo")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        session_id = create_interrupted_session()
        print(f"\nTo test resume functionality, run:")
        print(f"python main.py process test_input test_output_resume --resume")
        print(f"\nOr get session info:")
        print(f"python main.py sessions --info {session_id}")
    else:
        test_resume_detection()
        print("\nTo create an interrupted session for testing, run:")
        print("python demo_resume_functionality.py create")


if __name__ == "__main__":
    main()