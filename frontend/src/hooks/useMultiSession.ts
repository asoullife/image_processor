import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient, queryKeys } from "@/lib/api";
import { useUIStore } from "@/stores/useUIStore";

// Get concurrent sessions across all projects
export function useConcurrentSessions() {
  return useQuery({
    queryKey: queryKeys.concurrentSessions,
    queryFn: () => apiClient.getConcurrentSessions(),
    refetchInterval: 5000, // Refresh every 5 seconds for real-time updates
    staleTime: 2000, // Consider data stale after 2 seconds
  });
}

// Get session history with optional project filtering
export function useSessionHistory(projectId?: string, limit: number = 50) {
  return useQuery({
    queryKey: queryKeys.sessionHistory(projectId, limit),
    queryFn: () => apiClient.getSessionHistory(projectId, limit),
    staleTime: 30000, // 30 seconds
  });
}

// Get active projects
export function useActiveProjects() {
  return useQuery({
    queryKey: queryKeys.activeProjects,
    queryFn: () => apiClient.getActiveProjects(),
    refetchInterval: 10000, // Refresh every 10 seconds
    staleTime: 5000, // Consider data stale after 5 seconds
  });
}

// Get project sessions
export function useProjectSessions(projectId: string) {
  return useQuery({
    queryKey: queryKeys.projectSessions(projectId),
    queryFn: () => apiClient.getProjectSessions(projectId),
    enabled: !!projectId,
    staleTime: 10000, // 10 seconds
  });
}

// Get project statistics
export function useProjectStatistics(projectId: string) {
  return useQuery({
    queryKey: queryKeys.projectStatistics(projectId),
    queryFn: () => apiClient.getProjectStatistics(projectId),
    enabled: !!projectId,
    staleTime: 5000, // 5 seconds for real-time stats
  });
}

// Update session progress
export function useUpdateSessionProgress() {
  const queryClient = useQueryClient();
  const { addNotification } = useUIStore();

  return useMutation({
    mutationFn: ({ sessionId, progressData }: { sessionId: string; progressData: any }) =>
      apiClient.updateSessionProgress(sessionId, progressData),
    onSuccess: (_, { sessionId }) => {
      // Invalidate related queries
      queryClient.invalidateQueries({ queryKey: queryKeys.session(sessionId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.concurrentSessions });
    },
    onError: (error: any) => {
      addNotification({
        type: "error",
        title: "Failed to Update Progress",
        message: error.message || "Could not update session progress.",
      });
    },
  });
}

// Multi-session management utilities
export function useMultiSessionManager() {
  const queryClient = useQueryClient();
  const { addNotification } = useUIStore();

  const pauseAllSessions = useMutation({
    mutationFn: async (sessionIds: string[]) => {
      const results = await Promise.allSettled(
        sessionIds.map(id => apiClient.pauseProject(id))
      );
      return results;
    },
    onSuccess: (results) => {
      const successful = results.filter(r => r.status === 'fulfilled').length;
      const failed = results.filter(r => r.status === 'rejected').length;
      
      addNotification({
        type: successful > 0 ? "success" : "error",
        title: "Bulk Pause Operation",
        message: `${successful} sessions paused successfully${failed > 0 ? `, ${failed} failed` : ''}.`,
      });
      
      // Refresh concurrent sessions
      queryClient.invalidateQueries({ queryKey: queryKeys.concurrentSessions });
      queryClient.invalidateQueries({ queryKey: queryKeys.activeProjects });
    },
  });

  const resumeAllSessions = useMutation({
    mutationFn: async (sessionIds: string[]) => {
      const results = await Promise.allSettled(
        sessionIds.map(id => apiClient.resumeProject(id))
      );
      return results;
    },
    onSuccess: (results) => {
      const successful = results.filter(r => r.status === 'fulfilled').length;
      const failed = results.filter(r => r.status === 'rejected').length;
      
      addNotification({
        type: successful > 0 ? "success" : "error",
        title: "Bulk Resume Operation",
        message: `${successful} sessions resumed successfully${failed > 0 ? `, ${failed} failed` : ''}.`,
      });
      
      // Refresh concurrent sessions
      queryClient.invalidateQueries({ queryKey: queryKeys.concurrentSessions });
      queryClient.invalidateQueries({ queryKey: queryKeys.activeProjects });
    },
  });

  return {
    pauseAllSessions,
    resumeAllSessions,
  };
}