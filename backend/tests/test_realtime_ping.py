"""Simple ping/pong round-trip test for Socket.IO."""

import asyncio
import pytest
import socketio
import uvicorn
from fastapi import FastAPI

from ..realtime.socketio_manager import sio


@pytest.mark.asyncio
async def test_ping_pong_roundtrip():
    """Ensure a client receives pong in response to ping."""

    app = FastAPI()
    socket_app = socketio.ASGIApp(sio, app, socketio_path="/socket.io")

    config = uvicorn.Config(socket_app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())

    while not server.started:
        await asyncio.sleep(0.1)

    port = list(server.servers)[0].sockets[0].getsockname()[1]

    client = socketio.AsyncClient()
    pong_future: asyncio.Future = asyncio.get_event_loop().create_future()

    @client.on("pong")
    def on_pong(data):
        pong_future.set_result(data)

    await client.connect(f"http://127.0.0.1:{port}", socketio_path="/socket.io")
    await client.emit("ping", {"test": True})

    data = await asyncio.wait_for(pong_future, timeout=3)
    assert "timestamp" in data

    await client.disconnect()

    server.should_exit = True
    await task
