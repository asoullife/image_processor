"""Test Socket.IO implementation."""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

# Test the Socket.IO manager
from ..websocket.socketio_manager import socketio_manager, ProgressData, ErrorData, CompletionData
from ..utils.socketio_integration import ProcessingProgressBroadcaster, SocketIOProcessingIntegration

class TestSocketIOManager:
    """Test Socket.IO manager functionality."""

    def test_progress_data_creation(self):
        """Test ProgressData model creation."""
        progress = ProgressData(
            session_id="test-session",
            current_image=100,
            total_images=1000,
            percentage=10.0,
            current_filename="test.jpg",
            approved_count=80,
            rejected_count=20,
            processing_speed=5.5,
            current_stage="processing"
        )
        
        assert progress.session_id == "test-session"
        assert progress.current_image == 100
        assert progress.total_images == 1000
        assert progress.percentage == 10.0
        assert progress.current_filename == "test.jpg"
        assert progress.approved_count == 80
        assert progress.rejected_count == 20
        assert progress.processing_speed == 5.5
        assert progress.current_stage == "processing"

    def test_error_data_creation(self):
        """Test ErrorData model creation."""
        error = ErrorData(
            session_id="test-session",
            error_type="processing_error",
            error_message="Test error message",
            timestamp=datetime.now(),
            recoverable=True
        )
        
        assert error.session_id == "test-session"
        assert error.error_type == "processing_error"
        assert error.error_message == "Test error message"
        assert error.recoverable is True

    def test_completion_data_creation(self):
        """Test CompletionData model creation."""
        completion = CompletionData(
            session_id="test-session",
            total_processed=1000,
            total_approved=800,
            total_rejected=200,
            processing_time=300.5,
            completion_time=datetime.now(),
            output_folder="/path/to/output"
        )
        
        assert completion.session_id == "test-session"
        assert completion.total_processed == 1000
        assert completion.total_approved == 800
        assert completion.total_rejected == 200
        assert completion.processing_time == 300.5
        assert completion.output_folder == "/path/to/output"

    @pytest.mark.asyncio
    async def test_socketio_manager_session_management(self):
        """Test session management in Socket.IO manager."""
        session_id = "test-session-123"
        client_id = "client-456"
        
        # Test joining session
        await socketio_manager.join_session_room(client_id, session_id)
        
        # Verify client tracking
        assert session_id in socketio_manager.connected_clients
        assert client_id in socketio_manager.connected_clients[session_id]
        assert socketio_manager.client_sessions[client_id] == session_id
        
        # Test client count
        count = socketio_manager.get_connected_clients_count(session_id)
        assert count == 1
        
        # Test leaving session
        await socketio_manager.leave_session_room(client_id, session_id)
        
        # Verify cleanup
        assert client_id not in socketio_manager.client_sessions
        count = socketio_manager.get_connected_clients_count(session_id)
        assert count == 0

    @pytest.mark.asyncio
    async def test_progress_broadcasting(self):
        """Test progress data broadcasting."""
        session_id = "test-session-progress"
        
        progress_data = ProgressData(
            session_id=session_id,
            current_image=50,
            total_images=100,
            percentage=50.0,
            current_filename="test50.jpg",
            approved_count=40,
            rejected_count=10,
            processing_speed=2.5,
            current_stage="processing"
        )
        
        # Mock the Socket.IO emit function
        original_emit = socketio_manager.sio.emit
        socketio_manager.sio.emit = AsyncMock()
        
        try:
            # Test broadcasting
            await socketio_manager.broadcast_progress(session_id, progress_data)
            
            # Verify progress was cached
            assert session_id in socketio_manager.progress_cache
            cached_progress = socketio_manager.progress_cache[session_id]
            assert cached_progress.current_image == 50
            assert cached_progress.percentage == 50.0
            
            # Verify emit was called
            socketio_manager.sio.emit.assert_called_once()
            
        finally:
            # Restore original emit function
            socketio_manager.sio.emit = original_emit

class TestProcessingProgressBroadcaster:
    """Test processing progress broadcaster."""

    @pytest.mark.asyncio
    async def test_broadcaster_initialization(self):
        """Test broadcaster initialization."""
        session_id = "test-broadcaster-session"
        broadcaster = ProcessingProgressBroadcaster(session_id)
        
        assert broadcaster.session_id == session_id
        assert broadcaster.processed_count == 0
        assert broadcaster.total_count == 0
        assert broadcaster.approved_count == 0
        assert broadcaster.rejected_count == 0
        assert broadcaster.current_stage == "initializing"

    @pytest.mark.asyncio
    async def test_progress_calculation(self):
        """Test progress calculation and speed tracking."""
        session_id = "test-progress-calc"
        broadcaster = ProcessingProgressBroadcaster(session_id)
        
        # Mock the socketio_manager
        original_broadcast = socketio_manager.broadcast_progress
        socketio_manager.broadcast_progress = AsyncMock()
        
        try:
            # Initialize with total images
            await broadcaster.initialize(100, "scanning")
            
            # Update progress multiple times
            await broadcaster.update_progress(1, "image1.jpg", approved=True)
            await broadcaster.update_progress(2, "image2.jpg", approved=False)
            await broadcaster.update_progress(3, "image3.jpg", approved=True)
            
            # Check counts
            assert broadcaster.processed_count == 3
            assert broadcaster.approved_count == 2
            assert broadcaster.rejected_count == 1
            assert broadcaster.current_filename == "image3.jpg"
            
            # Verify broadcast was called
            assert socketio_manager.broadcast_progress.call_count == 3
            
        finally:
            # Restore original function
            socketio_manager.broadcast_progress = original_broadcast

    @pytest.mark.asyncio
    async def test_error_broadcasting(self):
        """Test error broadcasting."""
        session_id = "test-error-broadcast"
        broadcaster = ProcessingProgressBroadcaster(session_id)
        
        # Mock the socketio_manager
        original_broadcast = socketio_manager.broadcast_error
        socketio_manager.broadcast_error = AsyncMock()
        
        try:
            # Broadcast error
            await broadcaster.broadcast_error(
                "test_error",
                "This is a test error",
                recoverable=True
            )
            
            # Verify broadcast was called
            socketio_manager.broadcast_error.assert_called_once()
            
            # Check the error data
            call_args = socketio_manager.broadcast_error.call_args
            assert call_args[0][0] == session_id  # session_id
            error_data = call_args[0][1]  # error_data
            assert error_data.error_type == "test_error"
            assert error_data.error_message == "This is a test error"
            assert error_data.recoverable is True
            
        finally:
            # Restore original function
            socketio_manager.broadcast_error = original_broadcast

    @pytest.mark.asyncio
    async def test_completion_broadcasting(self):
        """Test completion broadcasting."""
        session_id = "test-completion-broadcast"
        broadcaster = ProcessingProgressBroadcaster(session_id)
        
        # Set some progress data
        broadcaster.processed_count = 100
        broadcaster.approved_count = 80
        broadcaster.rejected_count = 20
        
        # Mock the socketio_manager
        original_broadcast = socketio_manager.broadcast_completion
        socketio_manager.broadcast_completion = AsyncMock()
        
        try:
            # Broadcast completion
            await broadcaster.broadcast_completion("/path/to/output")
            
            # Verify broadcast was called
            socketio_manager.broadcast_completion.assert_called_once()
            
            # Check the completion data
            call_args = socketio_manager.broadcast_completion.call_args
            assert call_args[0][0] == session_id  # session_id
            completion_data = call_args[0][1]  # completion_data
            assert completion_data.total_processed == 100
            assert completion_data.total_approved == 80
            assert completion_data.total_rejected == 20
            assert completion_data.output_folder == "/path/to/output"
            
        finally:
            # Restore original function
            socketio_manager.broadcast_completion = original_broadcast

class TestSocketIOIntegration:
    """Test Socket.IO integration utilities."""

    @pytest.mark.asyncio
    async def test_create_broadcaster(self):
        """Test broadcaster creation."""
        session_id = "test-integration-session"
        broadcaster = await SocketIOProcessingIntegration.create_broadcaster(session_id)
        
        assert isinstance(broadcaster, ProcessingProgressBroadcaster)
        assert broadcaster.session_id == session_id

    @pytest.mark.asyncio
    async def test_session_notifications(self):
        """Test session notification methods."""
        session_id = "test-notifications"
        
        # Mock the socketio_manager
        original_broadcast = socketio_manager.broadcast_stage_change
        socketio_manager.broadcast_stage_change = AsyncMock()
        
        try:
            # Test start notification
            await SocketIOProcessingIntegration.notify_session_start(session_id, 1000)
            
            # Test pause notification
            await SocketIOProcessingIntegration.notify_session_pause(session_id)
            
            # Test resume notification
            await SocketIOProcessingIntegration.notify_session_resume(session_id)
            
            # Test batch complete notification
            await SocketIOProcessingIntegration.notify_batch_complete(session_id, 5, 10)
            
            # Test checkpoint notification
            await SocketIOProcessingIntegration.notify_checkpoint_saved(session_id, 3)
            
            # Verify all notifications were sent
            assert socketio_manager.broadcast_stage_change.call_count == 5
            
        finally:
            # Restore original function
            socketio_manager.broadcast_stage_change = original_broadcast

# Integration test
@pytest.mark.asyncio
async def test_full_socketio_integration():
    """Test full Socket.IO integration flow."""
    session_id = "test-full-integration"
    
    # Mock all socketio_manager methods
    original_methods = {
        'broadcast_stage_change': socketio_manager.broadcast_stage_change,
        'broadcast_progress': socketio_manager.broadcast_progress,
        'broadcast_completion': socketio_manager.broadcast_completion,
        'cleanup_session': socketio_manager.cleanup_session
    }
    
    for method_name in original_methods:
        setattr(socketio_manager, method_name, AsyncMock())
    
    try:
        # Simulate a small processing session
        image_paths = [f"image_{i}.jpg" for i in range(1, 6)]  # 5 images
        
        # Run the example processing function
        from ..utils.socketio_integration import example_processing_with_socketio
        await example_processing_with_socketio(session_id, image_paths)
        
        # Verify methods were called
        assert socketio_manager.broadcast_stage_change.call_count >= 1
        assert socketio_manager.broadcast_progress.call_count == 5  # One per image
        assert socketio_manager.broadcast_completion.call_count == 1
        assert socketio_manager.cleanup_session.call_count == 1
        
    finally:
        # Restore original methods
        for method_name, original_method in original_methods.items():
            setattr(socketio_manager, method_name, original_method)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])