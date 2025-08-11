'use client';

import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Database, 
  HardDrive, 
  Zap,
  RefreshCw,
  RotateCcw,
  Square
} from 'lucide-react';
import { useRecovery, CrashedSession, RecoveryOptions } from '@/hooks/useRecovery';
import { formatDistanceToNow } from 'date-fns';

interface RecoveryDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  crashedSession: CrashedSession | null;
  onRecoveryComplete?: (sessionId: string, result: any) => void;
}

export function RecoveryDialog({ 
  open, 
  onOpenChange, 
  crashedSession,
  onRecoveryComplete 
}: RecoveryDialogProps) {
  const [recoveryOptions, setRecoveryOptions] = useState<RecoveryOptions | null>(null);
  const [selectedOption, setSelectedOption] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const { getRecoveryOptions, executeRecovery, verifySessionIntegrity } = useRecovery();

  // Load recovery options when dialog opens
  useEffect(() => {
    if (open && crashedSession) {
      loadRecoveryOptions();
    }
  }, [open, crashedSession]);

  const loadRecoveryOptions = async () => {
    if (!crashedSession) return;
    
    try {
      setLoading(true);
      setError(null);
      
      const options = await getRecoveryOptions(crashedSession.session_id);
      setRecoveryOptions(options);
      
      // Auto-select recommended option
      const recommendedOption = options?.available_options.find(opt => opt.recommended);
      if (recommendedOption) {
        setSelectedOption(recommendedOption.option);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRecovery = async () => {
    if (!crashedSession || !selectedOption) return;
    
    try {
      setExecuting(true);
      setError(null);
      
      const result = await executeRecovery(crashedSession.session_id, selectedOption, true);
      
      if (result.success) {
        onRecoveryComplete?.(crashedSession.session_id, result);
        onOpenChange(false);
      } else {
        setError(result.message);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setExecuting(false);
    }
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

  const getOptionIcon = (option: string) => {
    switch (option) {
      case 'continue':
        return <RefreshCw className="h-4 w-4" />;
      case 'restart_batch':
        return <RotateCcw className="h-4 w-4" />;
      case 'fresh_start':
        return <Square className="h-4 w-4" />;
      default:
        return <RefreshCw className="h-4 w-4" />;
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low':
        return 'text-green-600';
      case 'medium':
        return 'text-yellow-600';
      case 'high':
        return 'text-red-600';
      default:
        return 'text-gray-600';
    }
  };

  if (!crashedSession) return null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {getCrashTypeIcon(crashedSession.crash_type)}
            Session Recovery Required
          </DialogTitle>
          <DialogDescription>
            A processing session was interrupted and can be recovered. Choose how to proceed.
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="h-6 w-6 animate-spin mr-2" />
            Loading recovery options...
          </div>
        ) : error ? (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        ) : recoveryOptions ? (
          <Tabs defaultValue="overview" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="options">Recovery Options</TabsTrigger>
              <TabsTrigger value="details">Technical Details</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Badge variant={getCrashTypeColor(crashedSession.crash_type)}>
                      {crashedSession.crash_type.replace('_', ' ').toUpperCase()}
                    </Badge>
                    Session {crashedSession.session_id.slice(0, 8)}
                  </CardTitle>
                  <CardDescription>
                    Last updated {formatDistanceToNow(new Date(crashedSession.last_update))} ago
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm font-medium">Progress</div>
                      <div className="text-2xl font-bold">
                        {recoveryOptions.session_info.processed_count.toLocaleString()} / {recoveryOptions.session_info.total_images.toLocaleString()}
                      </div>
                      <Progress 
                        value={recoveryOptions.session_info.progress_percentage} 
                        className="mt-2"
                      />
                      <div className="text-xs text-muted-foreground mt-1">
                        {recoveryOptions.session_info.progress_percentage.toFixed(1)}% complete
                      </div>
                    </div>
                    <div>
                      <div className="text-sm font-medium">Approval Rate</div>
                      <div className="text-2xl font-bold text-green-600">
                        {recoveryOptions.session_info.approval_rate.toFixed(1)}%
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {recoveryOptions.session_info.approved_count.toLocaleString()} approved, {recoveryOptions.session_info.rejected_count.toLocaleString()} rejected
                      </div>
                    </div>
                  </div>

                  <Separator />

                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <div className="font-medium">Processing Rate</div>
                      <div>{recoveryOptions.session_info.last_processing_rate.toFixed(2)} images/sec</div>
                    </div>
                    <div>
                      <div className="font-medium">Memory Usage</div>
                      <div>{recoveryOptions.session_info.memory_usage_mb.toFixed(0)} MB</div>
                    </div>
                    <div>
                      <div className="font-medium">GPU Memory</div>
                      <div>{recoveryOptions.session_info.gpu_memory_usage_mb.toFixed(0)} MB</div>
                    </div>
                  </div>

                  {recoveryOptions.session_info.estimated_time_remaining_seconds > 0 && (
                    <Alert>
                      <Clock className="h-4 w-4" />
                      <AlertDescription>
                        Estimated time remaining: {Math.round(recoveryOptions.session_info.estimated_time_remaining_seconds / 60)} minutes
                      </AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Database className="h-4 w-4" />
                    Recovery Information
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span>Checkpoint Integrity</span>
                    <Badge variant={recoveryOptions.recovery_info.checkpoint_integrity === 'valid' ? 'default' : 'destructive'}>
                      {recoveryOptions.recovery_info.checkpoint_integrity}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Data Loss Risk</span>
                    <span className={`font-medium ${getRiskColor(recoveryOptions.recovery_info.data_loss_risk)}`}>
                      {recoveryOptions.recovery_info.data_loss_risk.toUpperCase()}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Recovery Confidence</span>
                    <span className="font-medium">
                      {(recoveryOptions.recovery_info.recovery_confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <Alert>
                    <CheckCircle className="h-4 w-4" />
                    <AlertDescription>
                      {recoveryOptions.recovery_info.recommended_action}
                    </AlertDescription>
                  </Alert>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="options" className="space-y-4">
              <div className="space-y-3">
                {recoveryOptions.available_options.map((option) => (
                  <Card 
                    key={option.option}
                    className={`cursor-pointer transition-colors ${
                      selectedOption === option.option 
                        ? 'ring-2 ring-primary' 
                        : 'hover:bg-muted/50'
                    }`}
                    onClick={() => setSelectedOption(option.option)}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-start gap-3">
                        <div className="mt-1">
                          {getOptionIcon(option.option)}
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <h4 className="font-medium capitalize">
                              {option.option.replace('_', ' ')}
                            </h4>
                            {option.recommended && (
                              <Badge variant="default" className="text-xs">
                                Recommended
                              </Badge>
                            )}
                          </div>
                          <p className="text-sm text-muted-foreground mt-1">
                            {option.description}
                          </p>
                          {option.start_from_image && (
                            <p className="text-xs text-muted-foreground mt-2">
                              Will resume from image {option.start_from_image.toLocaleString()}
                            </p>
                          )}
                          {option.warning && (
                            <Alert variant="destructive" className="mt-2">
                              <AlertTriangle className="h-4 w-4" />
                              <AlertDescription className="text-xs">
                                {option.warning}
                              </AlertDescription>
                            </Alert>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="details" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Session Details</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="font-medium">Session ID</div>
                      <div className="font-mono text-xs">{crashedSession.session_id}</div>
                    </div>
                    <div>
                      <div className="font-medium">Project ID</div>
                      <div className="font-mono text-xs">{crashedSession.project_id}</div>
                    </div>
                    <div>
                      <div className="font-medium">Crash Type</div>
                      <div>{crashedSession.crash_type}</div>
                    </div>
                    <div>
                      <div className="font-medium">Last Update</div>
                      <div>{new Date(crashedSession.last_update).toLocaleString()}</div>
                    </div>
                    <div>
                      <div className="font-medium">Checkpoint Created</div>
                      <div>{new Date(recoveryOptions.session_info.checkpoint_created).toLocaleString()}</div>
                    </div>
                    <div>
                      <div className="font-medium">Has Valid Checkpoint</div>
                      <div>{crashedSession.has_valid_checkpoint ? 'Yes' : 'No'}</div>
                    </div>
                  </div>
                  {crashedSession.error_message && (
                    <div>
                      <div className="font-medium">Error Message</div>
                      <div className="text-xs bg-muted p-2 rounded mt-1">
                        {crashedSession.error_message}
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        ) : null}

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button 
            onClick={handleRecovery}
            disabled={!selectedOption || executing}
          >
            {executing ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Recovering...
              </>
            ) : (
              'Execute Recovery'
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}