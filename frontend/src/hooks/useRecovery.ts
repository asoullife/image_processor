import { useState, useEffect } from 'react';
import { api } from '@/lib/api';

export interface CrashedSession {
  session_id: string;
  project_id: string;
  crash_type: string;
  last_update: string;
  processed_count: number;
  total_images: number;
  has_valid_checkpoint: boolean;
  can_recover: boolean;
  error_message?: string;
}

export interface RecoveryOptions {
  session_info: {
    session_id: string;
    total_images: number;
    processed_count: number;
    approved_count: number;
    rejected_count: number;
    progress_percentage: number;
    approval_rate: number;
    last_processing_rate: number;
    estimated_time_remaining_seconds: number;
    checkpoint_created: string;
    memory_usage_mb: number;
    gpu_memory_usage_mb: number;
  };
  available_options: Array<{
    option: string;
    description: string;
    recommended: boolean;
    start_from_image?: number;
    warning?: string;
  }>;
  recovery_info: {
    recovery_available: boolean;
    checkpoint_integrity: string;
    data_loss_risk: string;
    recommended_action: string;
    recovery_confidence: number;
  };
}

export interface RecoveryResult {
  success: boolean;
  message: string;
  start_index: number;
  integrity_check: {
    integrity_check_passed: boolean;
    issues_found: string[];
    recommendations: string[];
    safe_to_continue: boolean;
  };
}

export function useRecovery() {
  const [crashedSessions, setCrashedSessions] = useState<CrashedSession[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Detect crashed sessions on mount
  useEffect(() => {
    detectCrashedSessions();
  }, []);

  const detectCrashedSessions = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await api.get('/recovery/crashed-sessions');
      setCrashedSessions(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to detect crashed sessions');
      console.error('Failed to detect crashed sessions:', err);
    } finally {
      setLoading(false);
    }
  };

  const getRecoveryOptions = async (sessionId: string): Promise<RecoveryOptions | null> => {
    try {
      const response = await api.get(`/recovery/sessions/${sessionId}/recovery-options`);
      return response.data;
    } catch (err: any) {
      console.error('Failed to get recovery options:', err);
      throw new Error(err.response?.data?.detail || 'Failed to get recovery options');
    }
  };

  const executeRecovery = async (
    sessionId: string,
    recoveryOption: string,
    userConfirmed: boolean = true
  ): Promise<RecoveryResult> => {
    try {
      const response = await api.post(`/recovery/sessions/${sessionId}/recover`, {
        resume_option: recoveryOption,
        user_confirmed: userConfirmed
      });
      
      // Remove the recovered session from crashed sessions list
      setCrashedSessions(prev => prev.filter(s => s.session_id !== sessionId));
      
      return response.data;
    } catch (err: any) {
      console.error('Failed to execute recovery:', err);
      throw new Error(err.response?.data?.detail || 'Recovery failed');
    }
  };

  const resumeSession = async (
    sessionId: string,
    resumeOption: string,
    userConfirmed: boolean = true
  ): Promise<{ success: boolean; message: string; start_index: number }> => {
    try {
      const response = await api.post(`/recovery/sessions/${sessionId}/resume`, {
        resume_option: resumeOption,
        user_confirmed: userConfirmed
      });
      
      return response.data;
    } catch (err: any) {
      console.error('Failed to resume session:', err);
      throw new Error(err.response?.data?.detail || 'Failed to resume session');
    }
  };

  const getSessionCheckpoints = async (sessionId: string) => {
    try {
      const response = await api.get(`/recovery/sessions/${sessionId}/checkpoints`);
      return response.data;
    } catch (err: any) {
      console.error('Failed to get session checkpoints:', err);
      throw new Error(err.response?.data?.detail || 'Failed to get checkpoints');
    }
  };

  const getLatestCheckpoint = async (sessionId: string) => {
    try {
      const response = await api.get(`/recovery/sessions/${sessionId}/latest-checkpoint`);
      return response.data;
    } catch (err: any) {
      console.error('Failed to get latest checkpoint:', err);
      throw new Error(err.response?.data?.detail || 'Failed to get latest checkpoint');
    }
  };

  const createManualCheckpoint = async (
    sessionId: string,
    checkpointType: string = 'manual',
    force: boolean = false
  ) => {
    try {
      const response = await api.post(`/recovery/sessions/${sessionId}/checkpoint`, {
        checkpoint_type: checkpointType,
        force
      });
      return response.data;
    } catch (err: any) {
      console.error('Failed to create manual checkpoint:', err);
      throw new Error(err.response?.data?.detail || 'Failed to create checkpoint');
    }
  };

  const verifySessionIntegrity = async (sessionId: string) => {
    try {
      const response = await api.post(`/recovery/sessions/${sessionId}/verify-integrity`);
      return response.data;
    } catch (err: any) {
      console.error('Failed to verify session integrity:', err);
      throw new Error(err.response?.data?.detail || 'Failed to verify integrity');
    }
  };

  const cleanupOldCheckpoints = async (sessionId: string, keepCount: number = 5) => {
    try {
      const response = await api.delete(`/recovery/sessions/${sessionId}/checkpoints?keep_count=${keepCount}`);
      return response.data;
    } catch (err: any) {
      console.error('Failed to cleanup checkpoints:', err);
      throw new Error(err.response?.data?.detail || 'Failed to cleanup checkpoints');
    }
  };

  const createEmergencyCheckpoint = async (sessionId: string, errorMessage: string) => {
    try {
      const response = await api.post(
        `/recovery/sessions/${sessionId}/emergency-checkpoint?error_message=${encodeURIComponent(errorMessage)}`
      );
      return response.data;
    } catch (err: any) {
      console.error('Failed to create emergency checkpoint:', err);
      throw new Error(err.response?.data?.detail || 'Failed to create emergency checkpoint');
    }
  };

  const getRecoveryStatistics = async () => {
    try {
      const response = await api.get('/recovery/recovery-statistics');
      return response.data;
    } catch (err: any) {
      console.error('Failed to get recovery statistics:', err);
      throw new Error(err.response?.data?.detail || 'Failed to get recovery statistics');
    }
  };

  return {
    // State
    crashedSessions,
    loading,
    error,
    
    // Actions
    detectCrashedSessions,
    getRecoveryOptions,
    executeRecovery,
    resumeSession,
    getSessionCheckpoints,
    getLatestCheckpoint,
    createManualCheckpoint,
    verifySessionIntegrity,
    cleanupOldCheckpoints,
    createEmergencyCheckpoint,
    getRecoveryStatistics
  };
}