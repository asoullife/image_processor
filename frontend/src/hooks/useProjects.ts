import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient, queryKeys, Project } from "@/lib/api";
import { useUIStore } from "@/stores/useUIStore";
import { useProjectStore } from "@/stores/useProjectStore";

// Get all projects
export function useProjects() {
  const { setLoading } = useUIStore();
  
  return useQuery({
    queryKey: queryKeys.projects,
    queryFn: async () => {
      setLoading("projects", true);
      try {
        return await apiClient.getProjects();
      } finally {
        setLoading("projects", false);
      }
    },
    staleTime: 30000, // 30 seconds
    refetchOnWindowFocus: false,
  });
}

// Get single project
export function useProject(projectId: string | null) {
  return useQuery({
    queryKey: queryKeys.project(projectId!),
    queryFn: () => apiClient.getProject(projectId!),
    enabled: !!projectId,
    staleTime: 10000, // 10 seconds
  });
}

// Create project mutation
export function useCreateProject() {
  const queryClient = useQueryClient();
  const { addNotification } = useUIStore();
  const { setCurrentProject, setIsCreatingProject } = useProjectStore();

  return useMutation({
    mutationFn: apiClient.createProject,
    onMutate: () => {
      setIsCreatingProject(true);
    },
    onSuccess: (newProject: Project) => {
      // Update projects cache
      queryClient.setQueryData(queryKeys.projects, (old: Project[] = []) => [
        ...old,
        newProject,
      ]);
      
      // Set as current project
      setCurrentProject(newProject);
      
      // Show success notification
      addNotification({
        type: "success",
        title: "Project Created",
        message: `Project "${newProject.name}" has been created successfully.`,
      });
      
      setIsCreatingProject(false);
    },
    onError: (error: any) => {
      addNotification({
        type: "error",
        title: "Failed to Create Project",
        message: error.message || "An unexpected error occurred.",
      });
      
      setIsCreatingProject(false);
    },
  });
}

// Start project mutation
export function useStartProject() {
  const queryClient = useQueryClient();
  const { addNotification } = useUIStore();
  const { setCurrentSession } = useProjectStore();

  return useMutation({
    mutationFn: apiClient.startProject,
    onSuccess: (session, projectId) => {
      // Update project cache
      queryClient.invalidateQueries({ queryKey: queryKeys.project(projectId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.projects });
      
      // Set current session
      setCurrentSession(session);
      
      // Show success notification
      addNotification({
        type: "success",
        title: "Processing Started",
        message: "Image processing has begun. You can monitor progress in real-time.",
      });
    },
    onError: (error: any) => {
      addNotification({
        type: "error",
        title: "Failed to Start Processing",
        message: error.message || "Could not start the processing session.",
      });
    },
  });
}

// Pause project mutation
export function usePauseProject() {
  const queryClient = useQueryClient();
  const { addNotification } = useUIStore();

  return useMutation({
    mutationFn: apiClient.pauseProject,
    onSuccess: (_, projectId) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.project(projectId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.projects });
      
      addNotification({
        type: "info",
        title: "Processing Paused",
        message: "Image processing has been paused. You can resume it anytime.",
      });
    },
    onError: (error: any) => {
      addNotification({
        type: "error",
        title: "Failed to Pause Processing",
        message: error.message || "Could not pause the processing session.",
      });
    },
  });
}

// Resume project mutation
export function useResumeProject() {
  const queryClient = useQueryClient();
  const { addNotification } = useUIStore();

  return useMutation({
    mutationFn: apiClient.resumeProject,
    onSuccess: (_, projectId) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.project(projectId) });
      queryClient.invalidateQueries({ queryKey: queryKeys.projects });
      
      addNotification({
        type: "success",
        title: "Processing Resumed",
        message: "Image processing has been resumed.",
      });
    },
    onError: (error: any) => {
      addNotification({
        type: "error",
        title: "Failed to Resume Processing",
        message: error.message || "Could not resume the processing session.",
      });
    },
  });
}