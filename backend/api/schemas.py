"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from uuid import UUID
from datetime import datetime
from enum import Enum

class ProjectStatus(str, Enum):
    """Project status enumeration."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SessionStatus(str, Enum):
    """Session status enumeration."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class PerformanceMode(str, Enum):
    """Performance mode enumeration."""
    SPEED = "speed"
    BALANCED = "balanced"
    SMART = "smart"

# Project schemas
class ProjectCreate(BaseModel):
    """Schema for creating a new project."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    input_folder: str = Field(..., min_length=1)
    output_folder: str = Field(..., min_length=1)
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('input_folder', 'output_folder')
    def validate_folder_paths(cls, v):
        if not v or not v.strip():
            raise ValueError('Folder path cannot be empty')
        return v.strip()

class ProjectUpdate(BaseModel):
    """Schema for updating a project."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    performance_mode: Optional[PerformanceMode] = None
    settings: Optional[Dict[str, Any]] = None

class ProjectResponse(BaseModel):
    """Schema for project response."""
    id: UUID
    name: str
    description: Optional[str]
    input_folder: str
    output_folder: str
    performance_mode: PerformanceMode
    status: ProjectStatus
    settings: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Session schemas
class SessionCreate(BaseModel):
    """Schema for creating a new session."""
    project_id: UUID
    total_images: Optional[int] = None
    resume_from: Optional[int] = None

class SessionResponse(BaseModel):
    """Schema for session response."""
    id: UUID
    project_id: UUID
    total_images: int
    processed_images: int
    approved_images: int
    rejected_images: int
    status: SessionStatus
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Analysis schemas
class AnalysisRequest(BaseModel):
    """Schema for image analysis request."""
    image_path: str = Field(..., min_length=1)
    analysis_types: List[str] = Field(default=["quality", "defect", "compliance"])
    options: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('analysis_types')
    def validate_analysis_types(cls, v):
        valid_types = {"quality", "defect", "compliance", "similarity"}
        invalid_types = set(v) - valid_types
        if invalid_types:
            raise ValueError(f'Invalid analysis types: {invalid_types}')
        return v

class QualityResult(BaseModel):
    """Schema for quality analysis result."""
    sharpness_score: float = Field(..., ge=0.0, le=1.0)
    noise_level: float = Field(..., ge=0.0, le=1.0)
    exposure_score: float = Field(..., ge=0.0, le=1.0)
    color_balance_score: float = Field(..., ge=0.0, le=1.0)
    resolution: tuple[int, int]
    file_size: int
    overall_score: float = Field(..., ge=0.0, le=1.0)
    passed: bool

class DefectResult(BaseModel):
    """Schema for defect detection result."""
    defect_count: int = Field(..., ge=0)
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    defect_types: List[str]
    confidence_scores: List[float]
    detected_objects: List[Dict[str, Any]]
    passed: bool

class ComplianceResult(BaseModel):
    """Schema for compliance checking result."""
    logo_detections: List[Dict[str, Any]]
    privacy_violations: List[Dict[str, Any]]
    metadata_issues: List[str]
    keyword_relevance: float = Field(..., ge=0.0, le=1.0)
    overall_compliance: bool

class SimilarityResult(BaseModel):
    """Schema for similarity detection result."""
    similarity_hash: str
    feature_vector: Optional[List[float]] = None
    similar_images: List[Dict[str, Any]]
    similarity_group: int

class AnalysisResponse(BaseModel):
    """Schema for analysis response."""
    image_path: str
    filename: str
    quality_result: Optional[QualityResult] = None
    defect_result: Optional[DefectResult] = None
    compliance_result: Optional[ComplianceResult] = None
    similarity_result: Optional[SimilarityResult] = None
    processing_time: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

# Image result schemas
class ImageResultResponse(BaseModel):
    """Schema for image processing result."""
    id: UUID
    session_id: UUID
    image_path: str
    filename: str
    source_folder: Optional[str]
    quality_scores: Optional[Dict[str, Any]]
    defect_results: Optional[Dict[str, Any]]
    similarity_group: Optional[int]
    similar_images: Optional[List[Dict[str, Any]]]
    compliance_results: Optional[Dict[str, Any]]
    final_decision: str
    rejection_reasons: List[str]
    human_override: bool
    human_review_at: Optional[datetime]
    processing_time: float
    created_at: datetime
    
    class Config:
        from_attributes = True

# Human review schemas
class ReviewDecision(BaseModel):
    """Schema for human review decision."""
    decision: str = Field(..., pattern="^(approve|reject)$")
    reason: Optional[str] = None

class BulkReviewRequest(BaseModel):
    """Schema for bulk review request."""
    image_ids: List[str] = Field(..., min_items=1)  # Accept strings, validate as UUID in endpoint
    decision: str = Field(..., pattern="^(approve|reject)$")
    reason: Optional[str] = None

# Progress and status schemas
class ProgressUpdate(BaseModel):
    """Schema for progress updates."""
    session_id: UUID
    processed_count: int
    total_count: int
    approved_count: int
    rejected_count: int
    current_image: Optional[str] = None
    processing_speed: Optional[float] = None
    estimated_completion: Optional[datetime] = None
    status: SessionStatus

class SystemStatus(BaseModel):
    """Schema for system status."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_sessions: int
    total_projects: int
    database_status: str
    
# Authentication schemas
class UserCreate(BaseModel):
    """Schema for user creation."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None, max_length=100)

class UserLogin(BaseModel):
    """Schema for user login."""
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)

class UserResponse(BaseModel):
    """Schema for user response."""
    id: UUID
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_superuser: bool
    created_at: datetime
    last_login: Optional[datetime]
    
    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    """Schema for token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Optional[UserResponse] = None

class LoginResponse(BaseModel):
    """Schema for login response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

# Configuration schemas
class ProcessingConfig(BaseModel):
    """Schema for processing configuration."""
    batch_size: int = Field(..., gt=0)
    max_workers: int = Field(..., gt=0)
    checkpoint_interval: int = Field(..., gt=0)

class QualityConfig(BaseModel):
    """Schema for quality configuration."""
    min_sharpness: float = Field(..., ge=0.0)
    max_noise_level: float = Field(..., ge=0.0, le=1.0)
    min_resolution: tuple[int, int]

class ConfigResponse(BaseModel):
    """Schema for configuration response."""
    processing: ProcessingConfig
    quality: QualityConfig
    similarity: Dict[str, Any]
    compliance: Dict[str, Any]
    decision: Dict[str, Any]

# Reports and Analytics Schemas
class ReportSummary(BaseModel):
    """Schema for session report summary."""
    session_id: UUID
    total_images: int
    processed_images: int
    approved_images: int
    rejected_images: int
    pending_images: int
    approval_rate: float = Field(..., ge=0.0, le=1.0)
    processing_time: float
    average_processing_time_per_image: float
    rejection_breakdown: Dict[str, int]
    quality_score_average: float
    human_overrides: int
    created_at: datetime
    updated_at: datetime

class ChartDataPoint(BaseModel):
    """Schema for chart data point."""
    label: str
    value: float
    color: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ChartData(BaseModel):
    """Schema for chart data."""
    chart_type: str
    title: str
    data: List[ChartDataPoint]
    chart_config: Dict[str, Any]

class ProcessingMetrics(BaseModel):
    """Schema for real-time processing metrics."""
    session_id: UUID
    current_speed: float  # images per minute
    estimated_completion: Optional[datetime]
    memory_usage: float = Field(..., ge=0.0, le=100.0)
    gpu_usage: Optional[float] = Field(None, ge=0.0, le=100.0)
    cpu_usage: float = Field(..., ge=0.0, le=100.0)
    batch_size: int = Field(..., gt=0)
    current_image_index: int = Field(..., ge=0)
    errors_count: int = Field(..., ge=0)
    warnings_count: int = Field(..., ge=0)

class ThumbnailData(BaseModel):
    """Schema for thumbnail data."""
    image_id: UUID
    filename: str
    thumbnail_url: str
    decision: str
    rejection_reasons: List[str]
    confidence_scores: Dict[str, float]
    human_override: bool
    processing_time: float
    source_folder: str
    created_at: datetime

class FilterOptions(BaseModel):
    """Schema for filter options."""
    decisions: List[str]
    rejection_reasons: List[str]
    source_folders: List[str]
    date_range: Dict[str, datetime]
    quality_score_range: Dict[str, float]

class AnalyticsData(BaseModel):
    """Schema for comprehensive analytics data."""
    session_id: UUID
    summary: ReportSummary
    charts: List[ChartData]
    performance_metrics: ProcessingMetrics
    filter_options: FilterOptions
    recent_activity: List[Dict[str, Any]]

class SessionComparison(BaseModel):
    """Schema for session comparison."""
    sessions: List[UUID]
    comparison_metrics: Dict[str, Dict[str, float]]
    performance_comparison: Dict[str, Any]
    recommendations: List[str]

class RealtimeStats(BaseModel):
    """Schema for real-time statistics."""
    session_id: UUID
    current_status: SessionStatus
    processed_count: int
    approved_count: int
    rejected_count: int
    processing_speed: float
    eta_minutes: Optional[float]
    current_image: Optional[str]
    last_updated: datetime