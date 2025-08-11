"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  PlayIcon, 
  PauseIcon, 
  ClockIcon,
  ActivityIcon,
  FolderIcon,
  TrendingUpIcon,
  AlertCircleIcon
} from "lucide-react";
import { useConcurrentSessions, useActiveProjects, useMultiSessionManager } from "@/hooks/useMultiSession";
import { AnimatedCounter } from "@/components/ui/animated-counter";
import { MagicCard } from "@/components/ui/magic-card";
import { LoadingSpinner } from "@/components/ui/loading-spinner";

const getStatusBadge = (status: string) => {
  switch (status) {
    case "running":
      return <Badge className="bg-green-500">Running</Badge>;
    case "paused":
      return <Badge className="bg-yellow-500">Paused</Badge>;
    case "completed":
      return <Badge className="bg-blue-500">Completed</Badge>;
    case "failed":
      return <Badge className="bg-red-500">Failed</Badge>;
    default:
      return <Badge variant="secondary">{status}</Badge>;
  }
};

const getStatusIcon = (status: string) => {
  switch (status) {
    case "running":
      return <PlayIcon className="w-4 h-4 text-green-500" />;
    case "paused":
      return <PauseIcon className="w-4 h-4 text-yellow-500" />;
    case "completed":
      return <ClockIcon className="w-4 h-4 text-blue-500" />;
    case "failed":
      return <AlertCircleIcon className="w-4 h-4 text-red-500" />;
    default:
      return <ActivityIcon className="w-4 h-4 text-gray-500" />;
  }
};

export function MultiSessionDashboard() {
  const { data: concurrentSessions, isLoading: sessionsLoading } = useConcurrentSessions();
  const { data: activeProjects, isLoading: projectsLoading } = useActiveProjects();
  const { pauseAllSessions, resumeAllSessions } = useMultiSessionManager();

  if (sessionsLoading || projectsLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  const runningSessions = concurrentSessions?.filter(s => s.status === "running") || [];
  const pausedSessions = concurrentSessions?.filter(s => s.status === "paused") || [];
  
  // Calculate totals
  const totalProcessed = concurrentSessions?.reduce((sum, s) => sum + s.processed_images, 0) || 0;
  const totalApproved = concurrentSessions?.reduce((sum, s) => sum + s.approved_images, 0) || 0;
  const totalRejected = concurrentSessions?.reduce((sum, s) => sum + s.rejected_images, 0) || 0;
  const overallApprovalRate = totalProcessed > 0 ? (totalApproved / totalProcessed) * 100 : 0;

  return (
    <div className="space-y-6">
      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MagicCard glowEffect={runningSessions.length > 0}>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <ActivityIcon className="w-4 h-4" />
              Active Sessions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              <AnimatedCounter value={concurrentSessions?.length || 0} />
            </div>
            <p className="text-xs text-muted-foreground">
              {runningSessions.length} running, {pausedSessions.length} paused
            </p>
          </CardContent>
        </MagicCard>

        <MagicCard>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <FolderIcon className="w-4 h-4" />
              Active Projects
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              <AnimatedCounter value={activeProjects?.length || 0} />
            </div>
            <p className="text-xs text-muted-foreground">
              Currently processing
            </p>
          </CardContent>
        </MagicCard>

        <MagicCard>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUpIcon className="w-4 h-4" />
              Images Processed
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              <AnimatedCounter value={totalProcessed} />
            </div>
            <p className="text-xs text-muted-foreground">
              Across all sessions
            </p>
          </CardContent>
        </MagicCard>

        <MagicCard>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">
              Approval Rate
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              <AnimatedCounter value={overallApprovalRate} suffix="%" />
            </div>
            <p className="text-xs text-muted-foreground">
              {totalApproved} approved, {totalRejected} rejected
            </p>
          </CardContent>
        </MagicCard>
      </div>

      {/* Bulk Actions */}
      {concurrentSessions && concurrentSessions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              Bulk Session Management
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => pauseAllSessions.mutate(runningSessions.map(s => s.id))}
                  disabled={runningSessions.length === 0 || pauseAllSessions.isPending}
                >
                  <PauseIcon className="w-4 h-4 mr-2" />
                  Pause All ({runningSessions.length})
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => resumeAllSessions.mutate(pausedSessions.map(s => s.id))}
                  disabled={pausedSessions.length === 0 || resumeAllSessions.isPending}
                >
                  <PlayIcon className="w-4 h-4 mr-2" />
                  Resume All ({pausedSessions.length})
                </Button>
              </div>
            </CardTitle>
            <CardDescription>
              Manage multiple processing sessions simultaneously
            </CardDescription>
          </CardHeader>
        </Card>
      )}

      {/* Active Sessions List */}
      <Card>
        <CardHeader>
          <CardTitle>Concurrent Sessions</CardTitle>
          <CardDescription>
            Real-time view of all active processing sessions
          </CardDescription>
        </CardHeader>
        <CardContent>
          {!concurrentSessions || concurrentSessions.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <ActivityIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No active sessions</p>
              <p className="text-sm">Start a project to see sessions here</p>
            </div>
          ) : (
            <div className="space-y-4">
              {concurrentSessions.map((session, index) => (
                <motion.div
                  key={session.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    {getStatusIcon(session.status)}
                    <div>
                      <h4 className="font-medium">{session.project_name}</h4>
                      <p className="text-sm text-muted-foreground">
                        Session {session.id.slice(0, 8)}...
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-6">
                    <div className="text-right">
                      <div className="text-sm font-medium">
                        <AnimatedCounter value={session.processed_images} /> / <AnimatedCounter value={session.total_images} />
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {session.total_images > 0 
                          ? `${Math.round((session.processed_images / session.total_images) * 100)}%`
                          : '0%'
                        }
                      </div>
                    </div>

                    <div className="text-right">
                      <div className="text-sm">
                        <span className="text-green-600">
                          <AnimatedCounter value={session.approved_images} />
                        </span>
                        {' / '}
                        <span className="text-red-600">
                          <AnimatedCounter value={session.rejected_images} />
                        </span>
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Approved / Rejected
                      </div>
                    </div>

                    {getStatusBadge(session.status)}
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}