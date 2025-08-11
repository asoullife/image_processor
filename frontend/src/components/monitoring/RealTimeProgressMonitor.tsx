/**
 * Enhanced real-time progress monitor with performance metrics and milestone notifications
 */

import React, { useEffect, useState, useCallback } from "react";
import { useSocket } from "@/hooks/useSocket";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AnimatedCounter } from "@/components/ui/animated-counter";
import { 
  Play, 
  Pause, 
  Activity, 
  Clock, 
  Cpu, 
  HardDrive, 
  Zap,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Trophy,
  TrendingUp
} from "lucide-react";
import { cn } from "@/lib/utils";

interface RealTimeProgressMonitorProps {
  sessionId: string;
  onComplete?: (data: any) => void;
  onError?: (error: any) => void;
  onMilestone?: (milestone: any) => void;
  className?: string;
}

interface MilestoneData {
  session_id: string;
  milestone_type: string;
  milestone_value: string;
  current_progress: number;
  total_progress: number;
  message: string;
  timestamp: string;
  performance_snapshot: Record<string, any>;
}

export function RealTimeProgressMonitor({ 
  sessionId, 
  onComplete, 
  onError,
  onMilestone,
  className 
}: RealTimeProgressMonitorProps) {
  const {
    socketState,
    progressData,
    errorData,
    completionData,
    stageData,
    pauseProcessing,
    resumeProcessing,
  } = useSocket(sessionId);

  const [isPaused, setIsPaused] = useState(false);
  const [milestones, setMilestones] = useState<MilestoneData[]>([]);
  const [showMilestones, setShowMilestones] = useState(true);

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

  // Handle milestones (would need to be added to Socket.IO events)
  const handleMilestone = useCallback((milestone: MilestoneData) => {
    if (milestone.session_id === sessionId) {
      setMilestones(prev => [milestone, ...prev.slice(0, 4)]); // Keep last 5 milestones
      onMilestone?.(milestone);
    }
  }, [sessionId, onMilestone]);

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
    if (seconds < 60) return `${Math.floor(seconds)}s`;
    if (seconds < 3600) {
      const mins = Math.floor(seconds / 60);
      const secs = Math.floor(seconds % 60);
      return `${mins}m ${secs}s`;
    }
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${mins}m`;
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

  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes.toFixed(0)} MB`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} GB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} TB`;
  };

  // Connection status
  if (!socketState.isConnected && !socketState.isConnecting) {
    return (
      <Card className={cn("border-red-200 bg-red-50", className)}>
        <CardContent className="pt-6">
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              Connection Error: {socketState.error || "Failed to connect to server"}
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  if (socketState.isConnecting) {
    return (
      <Card className={cn("border-blue-200 bg-blue-50", className)}>
        <CardContent className="pt-6">
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
            <div>
              <h3 className="text-blue-800 font-medium">Connecting...</h3>
              <p className="text-blue-600 text-sm">Establishing real-time connection</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!progressData) {
    return (
      <Card className={cn("border-gray-200 bg-gray-50", className)}>
        <CardContent className="pt-6">
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
            <div>
              <h3 className="text-gray-800 font-medium">Waiting for Processing</h3>
              <p className="text-gray-600 text-sm">No active processing session</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Main Progress Card */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <CardTitle className="text-lg">Processing Images</CardTitle>
              <Badge variant="outline" className="text-xs">
                {stageData?.stage || progressData.current_stage}
              </Badge>
            </div>
            <Button
              onClick={handlePauseResume}
              variant={isPaused ? "default" : "secondary"}
              size="sm"
              className="min-w-[80px]"
            >
              {isPaused ? (
                <>
                  <Play className="w-4 h-4 mr-1" />
                  Resume
                </>
              ) : (
                <>
                  <Pause className="w-4 h-4 mr-1" />
                  Pause
                </>
              )}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm text-gray-600">
              <span>
                <AnimatedCounter value={progressData.current_image} /> / {progressData.total_images.toLocaleString()}
              </span>
              <span>{progressData.percentage.toFixed(1)}%</span>
            </div>
            <Progress value={progressData.percentage} className="h-3" />
          </div>

          {/* Current File */}
          <div className="bg-gray-50 rounded-md p-3">
            <p className="text-sm text-gray-600 mb-1">Currently processing:</p>
            <p className="font-mono text-sm text-gray-900 truncate">
              {progressData.current_filename || "Preparing next batch..."}
            </p>
          </div>

          {/* Statistics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-3 bg-green-50 rounded-lg">
              <div className="flex items-center justify-center mb-1">
                <CheckCircle className="w-4 h-4 text-green-600 mr-1" />
                <span className="text-2xl font-bold text-green-600">
                  <AnimatedCounter value={progressData.approved_count} />
                </span>
              </div>
              <div className="text-xs text-gray-500">Approved</div>
            </div>
            
            <div className="text-center p-3 bg-red-50 rounded-lg">
              <div className="flex items-center justify-center mb-1">
                <XCircle className="w-4 h-4 text-red-600 mr-1" />
                <span className="text-2xl font-bold text-red-600">
                  <AnimatedCounter value={progressData.rejected_count} />
                </span>
              </div>
              <div className="text-xs text-gray-500">Rejected</div>
            </div>
            
            <div className="text-center p-3 bg-blue-50 rounded-lg">
              <div className="flex items-center justify-center mb-1">
                <TrendingUp className="w-4 h-4 text-blue-600 mr-1" />
                <span className="text-2xl font-bold text-blue-600">
                  {progressData.processing_speed.toFixed(1)}
                </span>
              </div>
              <div className="text-xs text-gray-500">Images/sec</div>
            </div>
            
            <div className="text-center p-3 bg-purple-50 rounded-lg">
              <div className="flex items-center justify-center mb-1">
                <Clock className="w-4 h-4 text-purple-600 mr-1" />
                <span className="text-2xl font-bold text-purple-600">
                  {formatETA(progressData.estimated_completion)}
                </span>
              </div>
              <div className="text-xs text-gray-500">ETA</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Performance Metrics Card */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center">
            <Activity className="w-5 h-5 mr-2" />
            Performance Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="flex items-center space-x-3">
              <Cpu className="w-5 h-5 text-blue-500" />
              <div>
                <div className="text-sm font-medium">CPU</div>
                <div className="text-xs text-gray-500">
                  {progressData.cpu_usage_percent?.toFixed(1) || 0}%
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <HardDrive className="w-5 h-5 text-green-500" />
              <div>
                <div className="text-sm font-medium">Memory</div>
                <div className="text-xs text-gray-500">
                  {formatBytes(progressData.memory_usage_mb || 0)}
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <Zap className="w-5 h-5 text-yellow-500" />
              <div>
                <div className="text-sm font-medium">GPU</div>
                <div className="text-xs text-gray-500">
                  {progressData.gpu_usage_percent?.toFixed(1) || 0}%
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <Clock className="w-5 h-5 text-purple-500" />
              <div>
                <div className="text-sm font-medium">Avg Time</div>
                <div className="text-xs text-gray-500">
                  {(progressData.avg_image_processing_time || 0).toFixed(2)}s
                </div>
              </div>
            </div>
          </div>

          {/* Batch Progress */}
          {progressData.current_batch && progressData.total_batches && (
            <div className="mt-4 pt-4 border-t">
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <span>Batch Progress</span>
                <span>
                  {progressData.current_batch} / {progressData.total_batches}
                </span>
              </div>
              <Progress 
                value={(progressData.current_batch / progressData.total_batches) * 100} 
                className="h-2"
              />
            </div>
          )}
        </CardContent>
      </Card>

      {/* Milestones Card */}
      {showMilestones && milestones.length > 0 && (
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg flex items-center">
                <Trophy className="w-5 h-5 mr-2 text-yellow-500" />
                Recent Milestones
              </CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowMilestones(false)}
              >
                Hide
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {milestones.map((milestone, index) => (
                <div
                  key={`${milestone.milestone_value}-${milestone.timestamp}`}
                  className={cn(
                    "p-3 rounded-lg border-l-4",
                    index === 0 ? "bg-yellow-50 border-yellow-400" : "bg-gray-50 border-gray-300"
                  )}
                >
                  <div className="flex items-center justify-between mb-1">
                    <Badge variant={index === 0 ? "default" : "secondary"}>
                      {milestone.milestone_value}
                    </Badge>
                    <span className="text-xs text-gray-500">
                      {new Date(milestone.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <p className="text-sm text-gray-700">{milestone.message}</p>
                  {milestone.performance_snapshot && (
                    <div className="flex space-x-4 mt-2 text-xs text-gray-500">
                      {milestone.performance_snapshot.processing_speed && (
                        <span>Speed: {milestone.performance_snapshot.processing_speed.toFixed(1)} img/s</span>
                      )}
                      {milestone.performance_snapshot.memory_usage && (
                        <span>RAM: {milestone.performance_snapshot.memory_usage.toFixed(0)}MB</span>
                      )}
                      {milestone.performance_snapshot.approved_rate && (
                        <span>Approval: {milestone.performance_snapshot.approved_rate.toFixed(1)}%</span>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Current Stage */}
      {stageData && (
        <Alert>
          <Activity className="h-4 w-4" />
          <AlertDescription>
            <strong>Stage: {stageData.stage}</strong>
            {stageData.message && <span className="ml-2">{stageData.message}</span>}
          </AlertDescription>
        </Alert>
      )}

      {/* Error Display */}
      {errorData && errorData.session_id === sessionId && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            <strong>Error: {errorData.error_type}</strong>
            <p className="mt-1">{errorData.error_message}</p>
            {errorData.recoverable && (
              <p className="text-xs mt-1 opacity-75">
                This error is recoverable. Processing will continue.
              </p>
            )}
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}