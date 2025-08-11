import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";
import { Project, ProcessingSession } from "@/lib/api";

interface ProjectState {
  // Current project data
  currentProject: Project | null;
  currentSession: ProcessingSession | null;
  
  // UI state
  isCreatingProject: boolean;
  selectedProjectId: string | null;
  
  // Actions
  setCurrentProject: (project: Project | null) => void;
  setCurrentSession: (session: ProcessingSession | null) => void;
  setIsCreatingProject: (isCreating: boolean) => void;
  setSelectedProjectId: (id: string | null) => void;
  
  // Computed values
  getProjectProgress: () => number;
  getApprovalRate: () => number;
}

export const useProjectStore = create<ProjectState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        currentProject: null,
        currentSession: null,
        isCreatingProject: false,
        selectedProjectId: null,

        // Actions
        setCurrentProject: (project) => 
          set({ currentProject: project }, false, "setCurrentProject"),
          
        setCurrentSession: (session) => 
          set({ currentSession: session }, false, "setCurrentSession"),
          
        setIsCreatingProject: (isCreating) => 
          set({ isCreatingProject: isCreating }, false, "setIsCreatingProject"),
          
        setSelectedProjectId: (id) => 
          set({ selectedProjectId: id }, false, "setSelectedProjectId"),

        // Computed values
        getProjectProgress: () => {
          const { currentProject } = get();
          if (!currentProject || currentProject.total_images === 0) return 0;
          return (currentProject.processed_images / currentProject.total_images) * 100;
        },

        getApprovalRate: () => {
          const { currentProject } = get();
          if (!currentProject || currentProject.processed_images === 0) return 0;
          return (currentProject.approved_images / currentProject.processed_images) * 100;
        },
      }),
      {
        name: "project-store",
        partialize: (state) => ({
          selectedProjectId: state.selectedProjectId,
        }),
      }
    ),
    {
      name: "project-store",
    }
  )
);