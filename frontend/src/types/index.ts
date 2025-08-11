// Re-export API types for convenience
export type {
  Project,
  ProcessingSession,
  ImageResult,
  ProgressData,
} from "@/lib/api";

// Additional UI-specific types
export interface SystemHealth {
  backend: "online" | "offline";
  database: "connected" | "disconnected";
  ai_models: "loaded" | "loading" | "error";
}

export interface ActivityItem {
  id: string;
  type: "project_created" | "processing_started" | "processing_completed" | "error";
  title: string;
  description: string;
  timestamp: string;
  projectId?: string;
}

export interface DashboardData {
  summary: SummaryStats;
  charts: ChartData;
  thumbnails: ThumbnailData[];
  filters: FilterOptions;
}

export interface SummaryStats {
  total_images: number;
  approved_count: number;
  rejected_count: number;
  approval_rate: number;
  rejection_breakdown: Record<string, number>;
  processing_time: number;
}

export interface ChartData {
  approvalTrend: Array<{ date: string; approved: number; rejected: number }>;
  rejectionReasons: Array<{ reason: string; count: number }>;
  processingSpeed: Array<{ time: string; speed: number }>;
}

export interface ThumbnailData {
  image_id: string;
  thumbnail_url: string;
  filename: string;
  decision: "approved" | "rejected" | "pending";
  rejection_reasons: string[];
  confidence_scores: Record<string, number>;
  human_override: boolean;
}

export interface FilterOptions {
  decisions: Array<{ value: string; label: string; count: number }>;
  rejectionReasons: Array<{ value: string; label: string; count: number }>;
  folders: Array<{ value: string; label: string; count: number }>;
}

export interface InputStructure {
  base_name: string;
  subfolders: Record<string, string[]>;
}

// Form types
export interface ProjectFormData {
  name: string;
  description?: string;
  input_folder: string;
  output_folder: string;
  performance_mode: "speed" | "balanced" | "smart";
}

export interface ReviewFilters {
  decision?: "approved" | "rejected" | "pending";
  rejection_reason?: string;
  human_reviewed?: boolean;
  source_folder?: string;
}

// Socket event types
export interface SocketEvents {
  // Client to server
  join_session: { session_id: string };
  leave_session: { session_id: string };
  pause_processing: { session_id: string };
  resume_processing: { session_id: string };
  
  // Server to client
  connected: { status: string };
  progress_update: ProgressData;
  processing_error: { session_id: string; error: string; timestamp: string };
  processing_complete: { session_id: string; summary: SummaryStats; timestamp: string };
  processing_paused: { session_id: string };
  processing_resumed: { session_id: string };
}