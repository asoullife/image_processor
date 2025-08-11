/**
 * Real-time processing monitor component using Socket.IO
 */

import React, { useEffect, useState } from "react";
import { useSocketContext, useProgressData, useSocketConnection } from "../contexts/SocketContext";

interface ProcessingMonitorProps {
  sessionId: string;
  onComplete?: (data: any) => void;
  onError?: (error: any) => void;
}

export function ProcessingMonitor({ 
  sessionId, 
  onComplete, 
  onError 
}: ProcessingMonitorProps) {
  const { 
    joinSession, 
    leaveSession, 
    pauseProcessing, 
    resumeProcessing,
    completionData,
    errorData,
    stageData
  } = useSocketContext();
  
  const progressData = useProgressData();
  const { isConnected, isConnecting, error } = useSocketConnection();
  const [isPaused, setIsPaused] = useState(false);

  // Join session on mount
  useEffect(() => {
    if (isConnected && sessionId) {
      joinSession(sessionId);
    }

    return () => {
      if (sessionId) {
        leaveSession(sessionId);
      }
    };
  }, [sessionId, isConnected, joinSession, leaveSession]);

  // Handle completion
  useEffect(() => {
    if (completionData && completionData.session_id === sessionId) {
      onComplete?.(completionData);
    }
  }, [completionData, sessionId, onComplete]);

  // Handle errors
  useEffect(() => {
    if (errorData && errorData.session_id === sessionId) {
      onError?.(errorData);
    }
  }, [errorData, sessionId, onError]);

  const handlePauseResume = () => {
    if (isPaused) {
      resumeProcessing(sessionId);
      setIsPaused(false);
    } else {
      pauseProcessing(sessionId);
      setIsPaused(true);
    }
  };

  const formatTime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  const formatETA = (isoString?: string): string => {
    if (!isoString) return "Calculating...";
    
    const eta = new Date(isoString);
    const now = new Date();
    const diffMs = eta.getTime() - now.getTime();
    
    if (diffMs <= 0) return "Almost done!";
    
    const diffSeconds = Math.floor(diffMs / 1000);
    return formatTime(diffSeconds);
  };

  if (!isConnected && !isConnecting) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-red-500 rounded-full mr-3"></div>
          <div>
            <h3 className="text-red-800 font-medium">Connection Error</h3>
            <p className="text-red-600 text-sm">{error || "Failed to connect to server"}</p>
          </div>
        </div>
      </div>
    );
  }

  if (isConnecting) {
    return (
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-blue-500 rounded-full mr-3 animate-pulse"></div>
          <div>
            <h3 className="text-blue-800 font-medium">Connecting...</h3>
            <p className="text-blue-600 text-sm">Establishing real-time connection</p>
          </div>
        </div>
      </div>
    );
  }

  if (!progressData) {
    return (
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-gray-400 rounded-full mr-3"></div>
          <div>
            <h3 className="text-gray-800 font-medium">Waiting for Processing</h3>
            <p className="text-gray-600 text-sm">No active processing session</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-6 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <div className="w-3 h-3 bg-green-500 rounded-full mr-3 animate-pulse"></div>
          <h3 className="text-lg font-semibold text-gray-900">Processing Images</h3>
        </div>
        <button
          onClick={handlePauseResume}
          className={`px-4 py-2 rounded-md text-sm font-medium ${
            isPaused
              ? "bg-green-600 hover:bg-green-700 text-white"
              : "bg-yellow-600 hover:bg-yellow-700 text-white"
          }`}
        >
          {isPaused ? "Resume" : "Pause"}
        </button>
      </div>

      {/* Progress Bar */}
      <div className="space-y-2">
        <div className="flex justify-between text-sm text-gray-600">
          <span>{progressData.current_image.toLocaleString()} / {progressData.total_images.toLocaleString()}</span>
          <span>{progressData.percentage.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3">
          <div
            className="bg-blue-600 h-3 rounded-full transition-all duration-300 ease-out"
            style={{ width: `${progressData.percentage}%` }}
          ></div>
        </div>
      </div>

      {/* Current File */}
      <div className="bg-gray-50 rounded-md p-3">
        <p className="text-sm text-gray-600">Currently processing:</p>
        <p className="font-mono text-sm text-gray-900 truncate">{progressData.current_filename}</p>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="text-center">
          <div className="text-2xl font-bold text-green-600">{progressData.approved_count.toLocaleString()}</div>
          <div className="text-xs text-gray-500">Approved</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-red-600">{progressData.rejected_count.toLocaleString()}</div>
          <div className="text-xs text-gray-500">Rejected</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-600">{progressData.processing_speed.toFixed(1)}</div>
          <div className="text-xs text-gray-500">Images/sec</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-purple-600">{formatETA(progressData.estimated_completion)}</div>
          <div className="text-xs text-gray-500">ETA</div>
        </div>
      </div>

      {/* Current Stage */}
      {stageData && (
        <div className="bg-blue-50 border border-blue-200 rounded-md p-3">
          <div className="flex items-center">
            <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
            <span className="text-sm font-medium text-blue-800">
              Stage: {stageData.stage}
            </span>
          </div>
          {stageData.message && (
            <p className="text-sm text-blue-600 mt-1">{stageData.message}</p>
          )}
        </div>
      )}

      {/* Error Display */}
      {errorData && errorData.session_id === sessionId && (
        <div className="bg-red-50 border border-red-200 rounded-md p-3">
          <div className="flex items-center">
            <div className="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
            <span className="text-sm font-medium text-red-800">
              Error: {errorData.error_type}
            </span>
          </div>
          <p className="text-sm text-red-600 mt-1">{errorData.error_message}</p>
          {errorData.recoverable && (
            <p className="text-xs text-red-500 mt-1">This error is recoverable. Processing will continue.</p>
          )}
        </div>
      )}
    </div>
  );
}