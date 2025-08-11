"""SQLAlchemy database models for PostgreSQL."""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from datetime import datetime

from .connection import Base

class Project(Base):
    """Project model for managing processing projects."""
    
    __tablename__ = "projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    input_folder = Column(Text, nullable=False)
    output_folder = Column(Text, nullable=False)
    performance_mode = Column(String(20), default='balanced')
    status = Column(String(20), default='created')
    settings = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    sessions = relationship("ProcessingSession", back_populates="project", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Project(id={self.id}, name='{self.name}', status='{self.status}')>"

class ProcessingSession(Base):
    """Processing session model for tracking individual processing runs."""
    
    __tablename__ = "processing_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey('projects.id'), nullable=False)
    total_images = Column(Integer, nullable=False, default=0)
    processed_images = Column(Integer, default=0)
    approved_images = Column(Integer, default=0)
    rejected_images = Column(Integer, default=0)
    status = Column(String(20), default='created')
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    session_config = Column(JSON, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    project = relationship("Project", back_populates="sessions")
    image_results = relationship("ImageResult", back_populates="session", cascade="all, delete-orphan")
    checkpoints = relationship("Checkpoint", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ProcessingSession(id={self.id}, project_id={self.project_id}, status='{self.status}')>"

class ImageResult(Base):
    """Image processing result model with enhanced metadata."""
    
    __tablename__ = "image_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('processing_sessions.id'), nullable=False)
    image_path = Column(Text, nullable=False)
    filename = Column(String(255), nullable=False)
    source_folder = Column(String(10), nullable=True)  # 1, 2, 3, etc.
    
    # Analysis results stored as JSON
    quality_scores = Column(JSON, nullable=True)  # {sharpness: 0.8, noise: 0.2, ...}
    defect_results = Column(JSON, nullable=True)  # {detected_objects: [...], confidence: 0.9}
    similarity_group = Column(Integer, nullable=True)
    similar_images = Column(JSON, nullable=True)  # [{path: "...", similarity: 0.95}, ...]
    compliance_results = Column(JSON, nullable=True)  # {logos: [], faces: [], privacy: []}
    
    # Decision and review
    final_decision = Column(String(20), nullable=False)  # approved/rejected
    rejection_reasons = Column(JSON, default=list)  # Array of reasons in Thai (JSON for SQLite compatibility)
    human_override = Column(Boolean, default=False)
    human_review_at = Column(DateTime(timezone=True), nullable=True)
    
    # Processing metadata
    processing_time = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("ProcessingSession", back_populates="image_results")
    
    def __repr__(self):
        return f"<ImageResult(id={self.id}, filename='{self.filename}', decision='{self.final_decision}')>"

class Checkpoint(Base):
    """Checkpoint model for resume capability."""
    
    __tablename__ = "checkpoints"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('processing_sessions.id'), nullable=False)
    checkpoint_type = Column(String(20), nullable=False)  # batch/image/milestone
    processed_count = Column(Integer, nullable=False)
    current_batch = Column(Integer, nullable=True)
    current_image_index = Column(Integer, nullable=True)
    session_state = Column(JSON, default=dict)  # Serialized session state
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("ProcessingSession", back_populates="checkpoints")
    
    def __repr__(self):
        return f"<Checkpoint(id={self.id}, session_id={self.session_id}, type='{self.checkpoint_type}')>"

class SimilarityGroup(Base):
    """Similarity group model for clustering similar images."""
    
    __tablename__ = "similarity_groups"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey('processing_sessions.id'), nullable=False)
    group_hash = Column(String(64), nullable=False)  # Hash representing the group
    representative_image = Column(Text, nullable=True)  # Path to representative image
    image_count = Column(Integer, default=0)
    similarity_threshold = Column(Float, default=0.85)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<SimilarityGroup(id={self.id}, session_id={self.session_id}, count={self.image_count})>"

class ProcessingLog(Base):
    """Processing log model for detailed logging and debugging."""
    
    __tablename__ = "processing_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('processing_sessions.id'), nullable=True)
    level = Column(String(10), nullable=False)  # DEBUG, INFO, WARNING, ERROR
    message = Column(Text, nullable=False)
    context = Column(JSON, default=dict)  # Additional context data
    image_path = Column(Text, nullable=True)  # Associated image if applicable
    processing_step = Column(String(50), nullable=True)  # quality, defect, similarity, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<ProcessingLog(id={self.id}, level='{self.level}', step='{self.processing_step}')>"

class SystemMetrics(Base):
    """System metrics model for performance monitoring."""
    
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('processing_sessions.id'), nullable=True)
    metric_type = Column(String(50), nullable=False)  # cpu, memory, disk, gpu, processing_speed
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(20), nullable=True)  # %, MB, images/sec, etc.
    context = Column(JSON, default=dict)  # Additional metric context
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<SystemMetrics(id={self.id}, type='{self.metric_type}', value={self.metric_value})>"

class UserPreferences(Base):
    """User preferences model for storing user settings."""
    
    __tablename__ = "user_preferences"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    preference_key = Column(String(100), nullable=False, unique=True)
    preference_value = Column(JSON, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<UserPreferences(key='{self.preference_key}')>"