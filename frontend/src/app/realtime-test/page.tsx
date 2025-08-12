'use client';

import { useEffect, useState } from 'react';
import { useSocket } from '@/hooks/useSocket';

export default function RealtimeTestPage() {
  const { socket, ping, socketState } = useSocket();
  const [pong, setPong] = useState<string | null>(null);

  useEffect(() => {
    if (!socket) return;
    const handler = (data: unknown) => setPong(JSON.stringify(data));
    socket.on('pong', handler);
    return () => {
      socket.off('pong', handler);
    };
  }, [socket]);

  return (
    <div className="p-4 space-y-2">
      <h1 className="text-xl font-bold">Socket.IO Ping Test</h1>
      <p>Connected: {socketState.isConnected ? 'yes' : 'no'}</p>
      <button
        className="px-3 py-1 bg-blue-500 text-white rounded"
        onClick={() => ping()}
        disabled={!socketState.isConnected}
      >
        Send Ping
      </button>
      {pong && <p>Pong: {pong}</p>}
    </div>
  );
}

