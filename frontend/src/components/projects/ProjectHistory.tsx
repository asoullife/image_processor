"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  ClockIcon,
  FolderIcon,
  PlayIcon,
  CheckCircleIcon,
  XCircleIcon,
  PauseIcon,
  AlertCircleIcon,
  EyeIcon
} from "lucide-react";
import { useSessionHistory, useProjectSessions } from "@/hooks/useMultiSession";
import { AnimatedCounter } from "@/components/ui/animated-counter";
import { LoadingSpinner } from "@/components/ui/loading-spinner";
import Link from "next/link";

interface ProjectHistoryProps {
  projectId?: string;
  limit?: number;
  showProjectNames?: boolean;
}

const getStatusBadge = (status: string) => {
  switch (status) {
    case "completed":
      return <Badge className="bg-green-500">Completed</Badge>;
    case "running":
      return <Badge className="bg-blue-500">Running</Badge>;
    case "paused":
      return <Badge className="bg-yellow-500">Paused</Badge>;
    case "failed":
      return <Badge className="bg-red-500">Failed</Badge>;
    default:
      return <Badge variant="secondary">{status}</Badge>;
  }
};

const getStatusIcon = (status: string) => {
  switch (status) {
    case "completed":
      return <CheckCircleIcon className="w-4 h-4 text-green-500" />;
    case "running":
      return <PlayIcon className="w-4 h-4 text-blue-500" />;
    case "paused":
      return <PauseIcon className="w-4 h-4 text-yellow-500" />;
    case "failed":
      return <AlertCircleIcon className="w-4 h-4 text-red-500" />;
    default:
      return <ClockIcon className="w-4 h-4 text-gray-500" />;
  }
};

export function ProjectHistory({ 
  projectId, 
  limit = 50, 
  showProjectNames = true 
}: ProjectHistoryProps) {
  const { data: sessions, isLoading } = useSessionHistory(projectId, limit);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (!sessions || sessions.length === 0) {
    return (
      <Card>
        <CardContent className="text-center py-8">
          <FolderIcon className="w-12 h-12 mx-auto mb-4 opacity-50 text-muted-foreground" />
          <h3 className="text-lg font-semibold mb-2">No Processing History</h3>
          <p className="text-muted-foreground">
            {projectId 
              ? "This project hasn't been processed yet"
              : "No processing sessions found"
            }
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <ClockIcon className="w-5 h-5" />
          Processing History
        </CardTitle>
        <CardDescription>
          {projectId 
            ? "All processing sessions for this project"
            : `Recent processing sessions across all projects (${sessions.length})`
          }
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {sessions.map((session, index) => {
            const duration = session.end_time && session.start_time
              ? new Date(session.end_time).getTime() - new Date(session.start_time).getTime()
              : null;
            
            const approvalRate = session.processed_images > 0
              ? (session.approved_images / session.processed_images) * 100
              : 0;

            return (
              <motion.div
                key={session.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
                className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50 transition-colors"
              >
                <div className="flex items-center gap-4">
                  {getStatusIcon(session.status)}
                  <div>
                    {showProjectNames && (
                      <h4 className="font-medium">{session.project_name}</h4>
                    )}
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <span>Session {session.id.slice(0, 8)}...</span>
                      <span>•</span>
                      <span>{new Date(session.created_at).toLocaleDateString()}</span>
                      {duration && (
                        <>
                          <span>•</span>
                          <span>{Math.round(duration / 1000 / 60)} min</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-6">
                  {/* Progress Stats */}
                  <div className="text-right">
                    <div className="text-sm font-medium">
                      <AnimatedCounter value={session.processed_images} /> / <AnimatedCounter value={session.total_images} />
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {session.total_images > 0 
                        ? `${Math.round((session.processed_images / session.total_images) * 100)}% processed`
                        : 'No images'
                      }
                    </div>
                  </div>

                  {/* Approval Stats */}
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
                      {approvalRate.toFixed(1)}% approval rate
                    </div>
                  </div>

                  {/* Status and Actions */}
                  <div className="flex items-center gap-2">
                    {getStatusBadge(session.status)}
                    
                    {session.status === "completed" && (
                      <Button
                        variant="outline"
                        size="sm"
                        asChild
                      >
                        <Link href={`/projects/${session.project_id}/review?session=${session.id}`}>
                          <EyeIcon className="w-3 h-3 mr-1" />
                          Review
                        </Link>
                      </Button>
                    )}
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

// Specialized component for single project history
export function SingleProjectHistory({ projectId }: { projectId: string }) {
  const { data: sessions, isLoading } = useProjectSessions(projectId);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-4">
        <LoadingSpinner />
      </div>
    );
  }

  return (
    <ProjectHistory 
      projectId={projectId} 
      showProjectNames={false}
      limit={20}
    />
  );
}