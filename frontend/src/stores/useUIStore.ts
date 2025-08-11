import { create } from "zustand";
import { devtools } from "zustand/middleware";

interface UIState {
  // Theme
  theme: "light" | "dark" | "system";
  
  // Navigation
  sidebarOpen: boolean;
  currentPage: string;
  
  // Modals and dialogs
  modals: {
    createProject: boolean;
    deleteProject: boolean;
    settings: boolean;
    imagePreview: boolean;
  };
  
  // Loading states
  loading: {
    projects: boolean;
    processing: boolean;
    saving: boolean;
  };
  
  // Notifications
  notifications: Array<{
    id: string;
    type: "success" | "error" | "warning" | "info";
    title: string;
    message: string;
    timestamp: number;
    duration?: number;
  }>;
  
  // Actions
  setTheme: (theme: "light" | "dark" | "system") => void;
  setSidebarOpen: (open: boolean) => void;
  setCurrentPage: (page: string) => void;
  
  // Modal actions
  openModal: (modal: keyof UIState["modals"]) => void;
  closeModal: (modal: keyof UIState["modals"]) => void;
  closeAllModals: () => void;
  
  // Loading actions
  setLoading: (key: keyof UIState["loading"], loading: boolean) => void;
  
  // Notification actions
  addNotification: (notification: Omit<UIState["notifications"][0], "id" | "timestamp">) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
}

export const useUIStore = create<UIState>()(
  devtools(
    (set, get) => ({
      // Initial state
      theme: "system",
      sidebarOpen: true,
      currentPage: "/",
      
      modals: {
        createProject: false,
        deleteProject: false,
        settings: false,
        imagePreview: false,
      },
      
      loading: {
        projects: false,
        processing: false,
        saving: false,
      },
      
      notifications: [],

      // Theme actions
      setTheme: (theme) => 
        set({ theme }, false, "setTheme"),

      // Navigation actions
      setSidebarOpen: (open) => 
        set({ sidebarOpen: open }, false, "setSidebarOpen"),
        
      setCurrentPage: (page) => 
        set({ currentPage: page }, false, "setCurrentPage"),

      // Modal actions
      openModal: (modal) => 
        set(
          (state) => ({
            modals: { ...state.modals, [modal]: true }
          }),
          false,
          `openModal:${modal}`
        ),
        
      closeModal: (modal) => 
        set(
          (state) => ({
            modals: { ...state.modals, [modal]: false }
          }),
          false,
          `closeModal:${modal}`
        ),
        
      closeAllModals: () => 
        set(
          (state) => ({
            modals: Object.keys(state.modals).reduce(
              (acc, key) => ({ ...acc, [key]: false }),
              {} as UIState["modals"]
            )
          }),
          false,
          "closeAllModals"
        ),

      // Loading actions
      setLoading: (key, loading) => 
        set(
          (state) => ({
            loading: { ...state.loading, [key]: loading }
          }),
          false,
          `setLoading:${key}:${loading}`
        ),

      // Notification actions
      addNotification: (notification) => {
        const id = Math.random().toString(36).substr(2, 9);
        const timestamp = Date.now();
        
        set(
          (state) => ({
            notifications: [
              ...state.notifications,
              { ...notification, id, timestamp }
            ]
          }),
          false,
          "addNotification"
        );
        
        // Auto-remove notification after duration
        if (notification.duration !== 0) {
          setTimeout(() => {
            get().removeNotification(id);
          }, notification.duration || 5000);
        }
      },
      
      removeNotification: (id) => 
        set(
          (state) => ({
            notifications: state.notifications.filter(n => n.id !== id)
          }),
          false,
          "removeNotification"
        ),
        
      clearNotifications: () => 
        set({ notifications: [] }, false, "clearNotifications"),
    }),
    {
      name: "ui-store",
    }
  )
);