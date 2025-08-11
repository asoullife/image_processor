/**
 * Real-time monitoring dashboard for processing sessions
 */

import React, { useState, useEffect } from "react";
import { useRouter } from "next/router";
import { useQuery } from "@tanstack/react-query";
import { apiClient, queryKeys } from "@/lib/api";
import { RealTimeProgressMonitor } from "@/components/monitoring/RealTimeProgressMonitor";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  ArrowLeft, 
  Download, 
  RefreshCw, 
  Settings,
  BarChart3,
  Activity,
  Clock,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Info
} from "lucide-react";
import { toast } from "sonner";

export default function MonitoringPage() {
  const router = useRouter();
  const { sessionId } = router.query;
  const [isCompleted, setIsCompleted] = useState(false);
  const [completionData, setCompletionData] = useState<any>(null);

  // Fetch session data
  const { data: session, isLoading, error, refetch } = useQuery({
    queryKey: queryKeys.session(sessionId as string),
    queryFn: () => apiClient.getSession(sessionId as string),
    enabled: !!sessionId,
    refetchInterval: isCompleted ? false : 5000, // Stop refetching when completed
  });

  // Fetch session results for statistics
  const { data: results } = useQuery({
    queryKey: queryKeys.sessionResults(sessionId as string),
    queryFn: () => apiClient.getSessionResults(sessionId as string),
    enabled: !!sessionId,
    refetchInterval: isCompleted ? false : 10000,
  });

  const handleComplete = (data: any) => {
    setIsCompleted(true);
    setCompletionData(data);
    toast.success("Processing completed successfully!", {
      description: `Processed ${data.total_processed} images in ${formatDuration(data.processing_time)}`,
    });
    refetch(); // Refresh session data
  };

  const handleError = (error: any) => {
    toast.error("Processing error occurred", {
      description: error.error_message,
    });
  };

  const handleMilestone = (milestone: any) => {
    toast.info(`Milestone reached: ${milestone.milestone_value}`, {
      description: milestone.message,
    });
  };

  const formatDuration = (seconds: number): string => {
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

  const getStatusColor = (status: string) => {
    switch (status) {
      case "running": return "bg-blue-100 text-blue-800";
      case "completed": return "bg-green-100 text-green-800";
      case "paused": return "bg-yellow-100 text-yellow-800";
      case "failed": return "bg-red-100 text-red-800";
      default: return "bg-gray-100 text-gray-800";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "running": return <Activity className="w-4 h-4" />;
      case "completed": return <CheckCircle className="w-4 h-4" />;
      case "paused": return <Clock className="w-4 h-4" />;
      case "failed": return <XCircle className="w-4 h-4" />;
      default: return <Info className="w-4 h-4" />;
    }
  };

  if (isLoading) {
    return (
      <div className="container mx-auto py-8">
        <div className="flex items-center justify-center h-64">
          <div className="flex items-center space-x-2">
            <RefreshCw className="w-6 h-6 animate-spin" />
            <span>Loading session data...</span>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto py-8">
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Failed to load session data: {error.message}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="container mx-auto py-8">
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            Session not found or no longer available.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => router.back()}
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>
          <div>
            <h1 className="text-2xl font-bold">Processing Monitor</h1>
            <p className="text-gray-600">Session: {sessionId}</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Badge className={getStatusColor(session.status)}>
            {getStatusIcon(session.status)}
            <span className="ml-1 capitalize">{session.status}</span>
          </Badge>
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Session Overview */}
      <Card>
        <CardHeader>
          <CardTitle>Session Overview</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {session.total_images.toLocaleString()}
              </div>
              <div className="text-sm text-gray-500">Total Images</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {session.processed_images.toLocaleString()}
              </div>
              <div className="text-sm text-gray-500">Processed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {session.approved_images.toLocaleString()}
              </div>
              <div className="text-sm text-gray-500">Approved</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">
                {session.rejected_images.toLocaleString()}
              </div>
              <div className="text-sm text-gray-500">Rejected</div>
            </div>
          </div>
          
          <div className="mt-4 pt-4 border-t">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-gray-500">Started:</span>
                <span className="ml-2 font-medium">
                  {new Date(session.start_time).toLocaleString()}
                </span>
              </div>
              {session.end_time && (
                <div>
                  <span className="text-gray-500">Completed:</span>
                  <span className="ml-2 font-medium">
                    {new Date(session.end_time).toLocaleString()}
                  </span>
                </div>
              )}
              {session.estimated_completion && (
                <div>
                  <span className="text-gray-500">ETA:</span>
                  <span className="ml-2 font-medium">
                    {new Date(session.estimated_completion).toLocaleString()}
                  </span>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Tabs for different views */}
      <Tabs defaultValue="monitor" className="space-y-4">
        <TabsList>
          <TabsTrigger value="monitor">Real-time Monitor</TabsTrigger>
          <TabsTrigger value="statistics">Statistics</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="monitor" className="space-y-4">
          {/* Real-time Progress Monitor */}
          <RealTimeProgressMonitor
            sessionId={sessionId as string}
            onComplete={handleComplete}
            onError={handleError}
            onMilestone={handleMilestone}
          />

          {/* Completion Summary */}
          {isCompleted && completionData && (
            <Card className="border-green-200 bg-green-50">
              <CardHeader>
                <CardTitle className="text-green-800 flex items-center">
                  <CheckCircle className="w-5 h-5 mr-2" />
                  Processing Completed
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {completionData.total_processed.toLocaleString()}
                    </div>
                    <div className="text-sm text-gray-600">Total Processed</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {completionData.total_approved.toLocaleString()}
                    </div>
                    <div className="text-sm text-gray-600">Approved</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-red-600">
                      {completionData.total_rejected.toLocaleString()}
                    </div>
                    <div className="text-sm text-gray-600">Rejected</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {formatDuration(completionData.processing_time)}
                    </div>
                    <div className="text-sm text-gray-600">Total Time</div>
                  </div>
                </div>
                
                <div className="mt-4 flex justify-center space-x-2">
                  <Button onClick={() => router.push(`/review/${sessionId}`)}>
                    <BarChart3 className="w-4 h-4 mr-2" />
                    View Results
                  </Button>
                  <Button variant="outline">
                    <Download className="w-4 h-4 mr-2" />
                    Download Report
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="statistics" className="space-y-4">
          {/* Statistics will be implemented here */}
          <Card>
            <CardHeader>
              <CardTitle>Processing Statistics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-gray-500">
                <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Detailed statistics will be available here</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="space-y-4">
          {/* Settings will be implemented here */}
          <Card>
            <CardHeader>
              <CardTitle>Monitor Settings</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8 text-gray-500">
                <Settings className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Monitor configuration options will be available here</p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}