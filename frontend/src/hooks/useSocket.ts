/**
 * Socket.IO client hook for real-time communication with backend
 */

import { useEffect, useState, useRef, useCallback } from "react";
import { io, Socket } from "socket.io-client";
import { ProgressData } from "@/types";

export interface ErrorData {
  session_id: string;
  error_type: string;
  error_message: string;
  timestamp: string;
  recoverable: boolean;
}

export interface CompletionData {
  session_id: string;
  total_processed: number;
  total_approved: number;
  total_rejected: number;
  processing_time: number;
  completion_time: string;
  output_folder: string;
}

export interface StageChangeData {
  session_id: string;
  stage: string;
  message: string;
  timestamp: string;
}

export interface SocketState {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  lastReconnectAttempt: Date | null;
  reconnectAttempts: number;
}

export interface UseSocketReturn {
  // Connection state
  socket: Socket | null;
  socketState: SocketState;
  
  // Data
  progressData: ProgressData | null;
  errorData: ErrorData | null;
  completionData: CompletionData | null;
  stageData: StageChangeData | null;
  
  // Actions
  joinSession: (sessionId: string) => void;
  leaveSession: (sessionId: string) => void;
  pauseProcessing: (sessionId: string) => void;
  resumeProcessing: (sessionId: string) => void;
  getSessionStatus: (sessionId: string) => void;
  ping: () => void;
  
  // Cleanup
  disconnect: () => void;
}

const SOCKET_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const MAX_RECONNECT_ATTEMPTS = 10;
const RECONNECT_DELAY = 1000;

export function useSocket(sessionId?: string): UseSocketReturn {
  // State
  const [progressData, setProgressData] = useState<ProgressData | null>(null);
  const [errorData, setErrorData] = useState<ErrorData | null>(null);
  const [completionData, setCompletionData] = useState<CompletionData | null>(null);
  const [stageData, setStageData] = useState<StageChangeData | null>(null);
  const [socketState, setSocketState] = useState<SocketState>({
    isConnected: false,
    isConnecting: false,
    error: null,
    lastReconnectAttempt: null,
    reconnectAttempts: 0,
  });

  // Refs
  const socketRef = useRef<Socket | null>(null);
  const currentSessionRef = useRef<string | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize socket connection
  useEffect(() => {
    if (socketRef.current) return; // Already connected

    setSocketState(prev => ({ ...prev, isConnecting: true }));

    const socket = io(SOCKET_URL, {
      path: "/socket.io/",
      transports: ["websocket", "polling"], // Fallback support
      reconnection: true,
      reconnectionDelay: RECONNECT_DELAY,
      reconnectionAttempts: MAX_RECONNECT_ATTEMPTS,
      timeout: 20000,
      forceNew: true,
    });

    socketRef.current = socket;

    // Connection events
    socket.on("connect", () => {
      console.log("Socket.IO connected:", socket.id);
      setSocketState(prev => ({
        ...prev,
        isConnected: true,
        isConnecting: false,
        error: null,
        reconnectAttempts: 0,
      }));

      // Auto-join session if provided
      if (sessionId) {
        socket.emit("join_session", { session_id: sessionId });
        currentSessionRef.current = sessionId;
      }
    });

    socket.on("disconnect", (reason) => {
      console.log("Socket.IO disconnected:", reason);
      setSocketState(prev => ({
        ...prev,
        isConnected: false,
        error: `Disconnected: ${reason}`,
      }));
    });

    socket.on("connect_error", (error) => {
      console.error("Socket.IO connection error:", error);
      setSocketState(prev => ({
        ...prev,
        isConnected: false,
        isConnecting: false,
        error: `Connection error: ${error.message}`,
        lastReconnectAttempt: new Date(),
        reconnectAttempts: prev.reconnectAttempts + 1,
      }));
    });

    socket.on("reconnect", (attemptNumber) => {
      console.log("Socket.IO reconnected after", attemptNumber, "attempts");
      setSocketState(prev => ({
        ...prev,
        isConnected: true,
        error: null,
        reconnectAttempts: 0,
      }));

      // Re-join session if we were in one
      if (currentSessionRef.current) {
        socket.emit("join_session", { session_id: currentSessionRef.current });
      }
    });

    socket.on("reconnect_error", (error) => {
      console.error("Socket.IO reconnection error:", error);
      setSocketState(prev => ({
        ...prev,
        error: `Reconnection error: ${error.message}`,
        lastReconnectAttempt: new Date(),
      }));
    });

    socket.on("reconnect_failed", () => {
      console.error("Socket.IO reconnection failed");
      setSocketState(prev => ({
        ...prev,
        error: "Reconnection failed after maximum attempts",
      }));
    });

    // Server events
    socket.on("connected", (data) => {
      console.log("Server connection confirmed:", data);
    });

    socket.on("session_joined", (data) => {
      console.log("Joined session:", data);
      currentSessionRef.current = data.session_id;
    });

    socket.on("progress_update", (data: ProgressData) => {
      console.log("Progress update:", data);
      setProgressData(data);
    });

    socket.on("processing_error", (data: ErrorData) => {
      console.error("Processing error:", data);
      setErrorData(data);
    });

    socket.on("processing_complete", (data: CompletionData) => {
      console.log("Processing complete:", data);
      setCompletionData(data);
      // Clear progress data when complete
      setProgressData(null);
    });

    socket.on("stage_change", (data: StageChangeData) => {
      console.log("Stage change:", data);
      setStageData(data);
    });

    socket.on("processing_paused", (data) => {
      console.log("Processing paused:", data);
    });

    socket.on("processing_resumed", (data) => {
      console.log("Processing resumed:", data);
    });

    socket.on("session_status", (data) => {
      console.log("Session status:", data);
    });

    socket.on("milestone_reached", (data) => {
      console.log("Milestone reached:", data);
      // You can add milestone handling here if needed
      // For now, we'll let components handle this via custom events
    });

    socket.on("pong", (data) => {
      console.log("Pong received:", data);
    });

    socket.on("error", (data) => {
      console.error("Socket.IO error:", data);
      setSocketState(prev => ({
        ...prev,
        error: data.message || "Unknown socket error",
      }));
    });

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      socket.disconnect();
      socketRef.current = null;
      currentSessionRef.current = null;
    };
  }, []); // Empty dependency array - only run once

  // Update session when sessionId prop changes
  useEffect(() => {
    if (!socketRef.current || !socketState.isConnected) return;

    if (sessionId && sessionId !== currentSessionRef.current) {
      // Leave current session if any
      if (currentSessionRef.current) {
        socketRef.current.emit("leave_session", { session_id: currentSessionRef.current });
      }
      
      // Join new session
      socketRef.current.emit("join_session", { session_id: sessionId });
      currentSessionRef.current = sessionId;
    } else if (!sessionId && currentSessionRef.current) {
      // Leave session if sessionId is cleared
      socketRef.current.emit("leave_session", { session_id: currentSessionRef.current });
      currentSessionRef.current = null;
    }
  }, [sessionId, socketState.isConnected]);

  // Action functions
  const joinSession = useCallback((sessionId: string) => {
    if (!socketRef.current || !socketState.isConnected) {
      console.warn("Socket not connected, cannot join session");
      return;
    }

    socketRef.current.emit("join_session", { session_id: sessionId });
    currentSessionRef.current = sessionId;
  }, [socketState.isConnected]);

  const leaveSession = useCallback((sessionId: string) => {
    if (!socketRef.current || !socketState.isConnected) {
      console.warn("Socket not connected, cannot leave session");
      return;
    }

    socketRef.current.emit("leave_session", { session_id: sessionId });
    if (currentSessionRef.current === sessionId) {
      currentSessionRef.current = null;
    }
  }, [socketState.isConnected]);

  const pauseProcessing = useCallback((sessionId: string) => {
    if (!socketRef.current || !socketState.isConnected) {
      console.warn("Socket not connected, cannot pause processing");
      return;
    }

    socketRef.current.emit("pause_processing", { session_id: sessionId });
  }, [socketState.isConnected]);

  const resumeProcessing = useCallback((sessionId: string) => {
    if (!socketRef.current || !socketState.isConnected) {
      console.warn("Socket not connected, cannot resume processing");
      return;
    }

    socketRef.current.emit("resume_processing", { session_id: sessionId });
  }, [socketState.isConnected]);

  const getSessionStatus = useCallback((sessionId: string) => {
    if (!socketRef.current || !socketState.isConnected) {
      console.warn("Socket not connected, cannot get session status");
      return;
    }

    socketRef.current.emit("get_session_status", { session_id: sessionId });
  }, [socketState.isConnected]);

  const ping = useCallback(() => {
    if (!socketRef.current || !socketState.isConnected) {
      console.warn("Socket not connected, cannot ping");
      return;
    }

    socketRef.current.emit("ping", { timestamp: new Date().toISOString() });
  }, [socketState.isConnected]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
      currentSessionRef.current = null;
    }
  }, []);

  return {
    socket: socketRef.current,
    socketState,
    progressData,
    errorData,
    completionData,
    stageData,
    joinSession,
    leaveSession,
    pauseProcessing,
    resumeProcessing,
    getSessionStatus,
    ping,
    disconnect,
  };
}