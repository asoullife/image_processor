'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { 
  AlertTriangle, 
  RefreshCw, 
  CheckCircle, 
  Clock,
  Database,
  Zap,
  HardDrive
} from 'lucide-react';
import { useRecovery, CrashedSession } from '@/hooks/useRecovery';
import { RecoveryDialog } from './RecoveryDialog';
import { formatDistanceToNow } from 'date-fns';

export function RecoveryDashboard() {
  const [selectedSession, setSelectedSession] = useState<CrashedSession | null>(null);
  const [recoveryDialogOpen, setRecoveryDialogOpen] = useState(false);
  
  const { 
    crashedSessions, 
    loading, 
    error, 
    detectCrashedSessions 
  } = useRecovery();

  const handleRecoverSession = (session: CrashedSession) => {
    setSelectedSession(session);
    setRecoveryDialogOpen(true);
  };

  const handleRecoveryComplete = (sessionId: string, result: any) => {
    // Refresh the crashed sessions list
    detectCrashedSessions();
  };

  const getCrashTypeIcon = (crashType: string) => {
    switch (crashType) {
      case 'power_failure':
        return <Zap className="h-4 w-4" />;
      case 'system_crash':
        return <AlertTriangle className="h-4 w-4" />;
      case 'application_crash':
        return <RefreshCw className="h-4 w-4" />;
      case 'out_of_memory':
        return <HardDrive className="h-4 w-4" />;
      default:
        return <AlertTriangle className="h-4 w-4" />;
    }
  };

  const getCrashTypeColor = (crashType: string) => {
    switch (crashType) {
      case 'power_failure':
        return 'destructive';
      case 'system_crash':
        return 'destructive';
      case 'application_crash':
        return 'secondary';
      case 'out_of_memory':
        return 'secondary';
      default:
        return 'secondary';
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-8">
          <RefreshCw className="h-6 w-6 animate-spin mr-2" />
          Detecting crashed sessions...
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent className="py-8">
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
          <Button 
            onClick={detectCrashedSessions} 
            className="mt-4"
            variant="outline"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry Detection
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (crashedSessions.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-green-600" />
            No Crashed Sessions
          </CardTitle>
          <CardDescription>
            All processing sessions completed successfully or are running normally.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button 
            onClick={detectCrashedSessions} 
            variant="outline"
            size="sm"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </CardContent>
      </Card>
    );
  }

  const recoverableSessions = crashedSessions.filter(s => s.can_recover);
  const nonRecoverableSessions = crashedSessions.filter(s => !s.can_recover);

  return (
    <div className="space-y-6">
      {/* Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-orange-600" />
            Session Recovery Dashboard
          </CardTitle>
          <CardDescription>
            {crashedSessions.length} interrupted session{crashedSessions.length !== 1 ? 's' : ''} detected
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <div className="font-medium">Total Crashed</div>
              <div className="text-2xl font-bold text-orange-600">
                {crashedSessions.length}
              </div>
            </div>
            <div>
              <div className="font-medium">Recoverable</div>
              <div className="text-2xl font-bold text-green-600">
                {recoverableSessions.length}
              </div>
            </div>
            <div>
              <div className="font-medium">Non-Recoverable</div>
              <div className="text-2xl font-bold text-red-600">
                {nonRecoverableSessions.length}
              </div>
            </div>
          </div>
          <Button 
            onClick={detectCrashedSessions} 
            className="mt-4"
            variant="outline"
            size="sm"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh Detection
          </Button>
        </CardContent>
      </Card>

      {/* Recoverable Sessions */}
      {recoverableSessions.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-green-600">
            Recoverable Sessions ({recoverableSessions.length})
          </h3>
          <div className="grid gap-4">
            {recoverableSessions.map((session) => (
              <Card key={session.session_id} className="border-green-200">
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Badge variant={getCrashTypeColor(session.crash_type)}>
                        {getCrashTypeIcon(session.crash_type)}
                        {session.crash_type.replace('_', ' ').toUpperCase()}
                      </Badge>
                      <span className="font-mono text-sm">
                        {session.session_id.slice(0, 8)}...
                      </span>
                    </div>
                    <Button 
                      onClick={() => handleRecoverSession(session)}
                      size="sm"
                      className="bg-green-600 hover:bg-green-700"
                    >
                      <Database className="h-4 w-4 mr-2" />
                      Recover
                    </Button>
                  </CardTitle>
                  <CardDescription>
                    Last updated {formatDistanceToNow(new Date(session.last_update))} ago
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Progress</span>
                        <span>
                          {session.processed_count.toLocaleString()} / {session.total_images.toLocaleString()}
                        </span>
                      </div>
                      <Progress 
                        value={(session.processed_count / session.total_images) * 100} 
                        className="h-2"
                      />
                      <div className="text-xs text-muted-foreground mt-1">
                        {((session.processed_count / session.total_images) * 100).toFixed(1)}% complete
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <div className="font-medium">Project ID</div>
                        <div className="font-mono text-xs">{session.project_id.slice(0, 8)}...</div>
                      </div>
                      <div>
                        <div className="font-medium">Checkpoint Status</div>
                        <div className="flex items-center gap-1">
                          <CheckCircle className="h-3 w-3 text-green-600" />
                          <span className="text-green-600">Valid</span>
                        </div>
                      </div>
                    </div>

                    {session.error_message && (
                      <Alert variant="destructive">
                        <AlertTriangle className="h-4 w-4" />
                        <AlertDescription className="text-xs">
                          {session.error_message}
                        </AlertDescription>
                      </Alert>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Non-Recoverable Sessions */}
      {nonRecoverableSessions.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-red-600">
            Non-Recoverable Sessions ({nonRecoverableSessions.length})
          </h3>
          <div className="grid gap-4">
            {nonRecoverableSessions.map((session) => (
              <Card key={session.session_id} className="border-red-200 opacity-75">
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Badge variant="destructive">
                        {getCrashTypeIcon(session.crash_type)}
                        {session.crash_type.replace('_', ' ').toUpperCase()}
                      </Badge>
                      <span className="font-mono text-sm">
                        {session.session_id.slice(0, 8)}...
                      </span>
                    </div>
                    <Badge variant="destructive">
                      Corrupted Data
                    </Badge>
                  </CardTitle>
                  <CardDescription>
                    Last updated {formatDistanceToNow(new Date(session.last_update))} ago
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Progress (Lost)</span>
                        <span>
                          {session.processed_count.toLocaleString()} / {session.total_images.toLocaleString()}
                        </span>
                      </div>
                      <Progress 
                        value={(session.processed_count / session.total_images) * 100} 
                        className="h-2 opacity-50"
                      />
                      <div className="text-xs text-muted-foreground mt-1">
                        {((session.processed_count / session.total_images) * 100).toFixed(1)}% progress lost
                      </div>
                    </div>
                    
                    <Alert variant="destructive">
                      <AlertTriangle className="h-4 w-4" />
                      <AlertDescription className="text-xs">
                        Checkpoint data is corrupted and cannot be recovered. 
                        The session will need to be restarted from the beginning.
                      </AlertDescription>
                    </Alert>

                    {session.error_message && (
                      <div className="text-xs bg-muted p-2 rounded">
                        <div className="font-medium">Error:</div>
                        <div>{session.error_message}</div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* Recovery Dialog */}
      <RecoveryDialog
        open={recoveryDialogOpen}
        onOpenChange={setRecoveryDialogOpen}
        crashedSession={selectedSession}
        onRecoveryComplete={handleRecoveryComplete}
      />
    </div>
  );
}