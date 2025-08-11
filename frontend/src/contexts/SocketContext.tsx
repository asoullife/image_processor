/**
 * Socket.IO context provider for React application
 */

import React, { createContext, useContext, useEffect, useState } from "react";
import { useSocket, UseSocketReturn, ProgressData, ErrorData, CompletionData } from "../hooks/useSocket";

interface SocketContextType extends UseSocketReturn {
  // Additional context-specific methods
  clearData: () => void;
  isSessionActive: (sessionId: string) => boolean;
}

const SocketContext = createContext<SocketContextType | null>(null);

interface SocketProviderProps {
  children: React.ReactNode;
  sessionId?: string;
  autoConnect?: boolean;
}

export function SocketProvider({ 
  children, 
  sessionId, 
  autoConnect = true 
}: SocketProviderProps) {
  const socketHook = useSocket(autoConnect ? sessionId : undefined);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(sessionId || null);

  // Update active session when sessionId prop changes
  useEffect(() => {
    if (sessionId !== activeSessionId) {
      setActiveSessionId(sessionId || null);
    }
  }, [sessionId, activeSessionId]);

  // Clear all data
  const clearData = () => {
    // This would typically reset all state in the hook
    // For now, we'll just log it
    console.log("Clearing Socket.IO data");
  };

  // Check if a session is currently active
  const isSessionActive = (sessionId: string): boolean => {
    return activeSessionId === sessionId && socketHook.socketState.isConnected;
  };

  const contextValue: SocketContextType = {
    ...socketHook,
    clearData,
    isSessionActive,
  };

  return (
    <SocketContext.Provider value={contextValue}>
      {children}
    </SocketContext.Provider>
  );
}

export function useSocketContext(): SocketContextType {
  const context = useContext(SocketContext);
  if (!context) {
    throw new Error("useSocketContext must be used within a SocketProvider");
  }
  return context;
}

// Convenience hooks for specific data types
export function useProgressData(): ProgressData | null {
  const { progressData } = useSocketContext();
  return progressData;
}

export function useErrorData(): ErrorData | null {
  const { errorData } = useSocketContext();
  return errorData;
}

export function useCompletionData(): CompletionData | null {
  const { completionData } = useSocketContext();
  return completionData;
}

export function useSocketConnection() {
  const { socketState, socket } = useSocketContext();
  return {
    isConnected: socketState.isConnected,
    isConnecting: socketState.isConnecting,
    error: socketState.error,
    socket,
    reconnectAttempts: socketState.reconnectAttempts,
    lastReconnectAttempt: socketState.lastReconnectAttempt,
  };
}