/**
 * Types for Reports and Analytics
 */

export interface ReportSummary {
  session_id: string;
  total_images: number;
  processed_images: number;
  approved_images: number;
  rejected_images: number;
  pending_images: number;
  approval_rate: number;
  processing_time: number;
  average_processing_time_per_image: number;
  rejection_breakdown: Record<string, number>;
  quality_score_average: number;
  human_overrides: number;
  created_at: string;
  updated_at: string;
}

export interface ChartDataPoint {
  label: string;
  value: number;
  color?: string;
  metadata?: Record<string, any>;
}

export interface ChartData {
  chart_type: 'pie' | 'bar' | 'line' | 'histogram';
  title: string;
  data: ChartDataPoint[];
  chart_config: Record<string, any>;
}

export interface ProcessingMetrics {
  session_id: string;
  current_speed: number;
  estimated_completion?: string;
  memory_usage: number;
  gpu_usage?: number;
  cpu_usage: number;
  batch_size: number;
  current_image_index: number;
  errors_count: number;
  warnings_count: number;
}

export interface ThumbnailData {
  image_id: string;
  filename: string;
  thumbnail_url: string;
  decision: 'approved' | 'rejected' | 'pending';
  rejection_reasons: string[];
  confidence_scores: Record<string, number>;
  human_override: boolean;
  processing_time: number;
  source_folder: string;
  created_at: string;
}

export interface FilterOptions {
  decisions: string[];
  rejection_reasons: string[];
  source_folders: string[];
  date_range: {
    min: string;
    max: string;
  };
  quality_score_range: {
    min: number;
    max: number;
  };
}

export interface AnalyticsData {
  session_id: string;
  summary: ReportSummary;
  charts: ChartData[];
  performance_metrics: ProcessingMetrics;
  filter_options: FilterOptions;
  recent_activity: ActivityItem[];
}

export interface ActivityItem {
  type: 'processed' | 'approved' | 'rejected' | 'human_override';
  filename: string;
  decision: string;
  timestamp: string;
  processing_time: number;
}

export interface FilteredResults {
  items: ProcessedImage[];
  total_count: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface ProcessedImage {
  id: string;
  filename: string;
  image_path: string;
  source_folder?: string;
  final_decision: 'approved' | 'rejected' | 'pending';
  rejection_reasons: string[];
  quality_scores?: Record<string, any>;
  human_override: boolean;
  processing_time: number;
  created_at: string;
  thumbnail_url: string;
}

export interface RealtimeStats {
  session_id: string;
  current_status: string;
  processed_count: number;
  approved_count: number;
  rejected_count: number;
  processing_speed: number;
  eta_minutes?: number;
  current_image?: string;
  last_updated: string;
}

export interface SessionComparison {
  sessions: string[];
  comparison_metrics: Record<string, Record<string, number>>;
  performance_comparison: Record<string, any>;
  recommendations: string[];
}

export interface ExportOptions {
  format: 'json' | 'csv' | 'excel';
  session_id: string;
}

export interface ExportResult {
  data: any;
  filename: string;
}

// Filter types for UI components
export interface ReportsFilter {
  decision?: string;
  rejection_reason?: string;
  source_folder?: string;
  human_override?: boolean;
  search?: string;
  date_from?: string;
  date_to?: string;
  quality_score_min?: number;
  quality_score_max?: number;
}

export interface SortOptions {
  sort_by: 'filename' | 'created_at' | 'confidence' | 'processing_time';
  sort_order: 'asc' | 'desc';
}

export interface PaginationOptions {
  page: number;
  page_size: number;
}

// Chart configuration types
export interface ChartConfig {
  responsive: boolean;
  plugins?: {
    legend?: {
      position: 'top' | 'bottom' | 'left' | 'right';
    };
    tooltip?: {
      enabled: boolean;
    };
  };
  scales?: {
    x?: {
      title?: {
        display: boolean;
        text: string;
      };
    };
    y?: {
      title?: {
        display: boolean;
        text: string;
      };
      beginAtZero?: boolean;
      max?: number;
    };
  };
}

// Dashboard layout types
export interface DashboardLayout {
  summary: boolean;
  charts: boolean;
  performance: boolean;
  recent_activity: boolean;
  thumbnails: boolean;
}

export interface DashboardSettings {
  layout: DashboardLayout;
  refresh_interval: number; // seconds
  auto_refresh: boolean;
  chart_types: string[];
  thumbnail_size: 'small' | 'medium' | 'large';
}