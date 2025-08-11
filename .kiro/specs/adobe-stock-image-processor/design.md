# Design Document

## Overview

Adobe Stock Image Processor เป็นระบบประมวลผลภาพแบบ Desktop Application ที่ใช้ Python FastAPI Backend + Next.js Frontend พร้อม AI/ML enhancement และ Real-time Web Monitoring ระบบออกแบบให้ทำงานบนเครื่องส่วนตัว (Local Desktop) โดยใช้ PostgreSQL database, GPU acceleration (RTX2060), และ WebSocket สำหรับ real-time updates พร้อม Human Review System ที่ช่วยให้ผู้ใช้สามารถตรวจสอบและแก้ไขผลการวิเคราะห์ได้

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Desktop Application                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐│
│  │   Next.js       │    │   Python        │    │   PostgreSQL    ││
│  │   Frontend      │◄──►│   FastAPI       │◄──►│   Database      ││
│  │   (Port 3000)   │    │   Backend       │    │                 ││
│  │                 │    │   (Port 8000)   │    │                 ││
│  └─────────────────┘    └─────────────────┘    └─────────────────┘│
│           │                       │                               │
│           │                       ▼                               │
│           │              ┌─────────────────┐                      │
│           │              │   AI/ML Models  │                      │
│           │              │   TensorFlow    │                      │
│           │              │   YOLO v8       │                      │
│           │              │   OpenCV        │                      │
│           │              └─────────────────┘                      │
│           │                       │                               │
│           ▼                       ▼                               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              WebSocket Real-time Updates                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Folder  │───▶│  Batch Processor │───▶│  Output Folder  │
│   (Read-only)   │    │   GPU Accel.     │    │  (_processed)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Modern Tech Stack

```
Frontend: Next.js 14 + TypeScript + Tailwind CSS + Framer Motion
Backend: Python FastAPI + SQLAlchemy + Socket.IO + Celery
Database: PostgreSQL + Redis (caching & Socket.IO adapter)
AI/ML: TensorFlow + YOLO v8 + OpenCV (GPU accelerated)
State Management: Zustand + React Query
UI Components: Shadcn/UI + Magic UI + Headless UI + Chart.js + SweetAlert2
Real-time: Socket.IO (python-socketio) ↔ Socket.IO Client (Next.js)
```

### Socket.IO Real-time Communication Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Socket.IO Real-time Communication           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Next.js Frontend                    Python FastAPI Backend    │
│  ┌─────────────────┐                 ┌─────────────────┐       │
│  │  socket.io-     │    Socket.IO    │  python-        │       │
│  │  client         │◄───────────────►│  socketio       │       │
│  │                 │                 │  (ASGI)         │       │
│  └─────────────────┘                 └─────────────────┘       │
│           │                                   │                 │
│           │                                   │                 │
│  ┌─────────────────┐                 ┌─────────────────┐       │
│  │  useSocket      │                 │  SocketIO       │       │
│  │  Hook           │                 │  Manager        │       │
│  └─────────────────┘                 └─────────────────┘       │
│                                               │                 │
│                                       ┌─────────────────┐       │
│                                       │  Redis Adapter  │       │
│                                       │  (Multi-process)│       │
│                                       └─────────────────┘       │
└─────────────────────────────────────────────────────────────────┘

Socket.IO Benefits:
✅ Auto-reconnection with exponential backoff
✅ Fallback: WebSocket → HTTP long-polling → HTTP polling
✅ Room-based sessions (join session rooms)
✅ Built-in error handling and connection management
✅ Redis adapter for multi-process scaling
✅ Event-based communication (progress, error, complete)
```

### Corrected Project Structure

Based on architectural review, the project structure has been refined to eliminate redundancy and improve maintainability:

```
adobe-stock-processor/
├── backend/                    # Python FastAPI backend
│   ├── api/                   # REST API endpoints
│   ├── core/                  # Core processing logic
│   ├── analyzers/             # AI/ML image analysis modules
│   ├── utils/                 # Backend utilities
│   ├── database/              # Database models and migrations
│   ├── config/                # Backend configuration
│   ├── data/                  # Data files (moved from root)
│   │   ├── input/            # Test input images
│   │   ├── output/           # Processing results
│   │   └── temp/             # Temporary files
│   ├── tests/                 # Backend tests
│   └── main.py               # Backend entry point
├── frontend/                   # Next.js frontend
│   ├── src/                   # Frontend source code
│   ├── tests/                 # Frontend tests
│   └── package.json          # Frontend dependencies
├── scripts/                    # Build/deploy scripts and utilities
├── demos/                      # Demo applications
├── docs/                       # Documentation
├── reports/                    # Generated reports
├── logs/                       # Application logs
├── main.py                     # Main entry point (delegates to backend)
├── docker-compose.yml          # Docker orchestration
└── README.md                   # Project documentation
```

**Key Changes:**

- **Removed shared/**: Shared code moved to backend/, frontend accesses via API
- **Moved data/**: Now backend/data/ since data processing belongs to backend
- **Separated tests**: backend/tests/ and frontend/tests/ for clear ownership
- **Simplified tools**: Development utilities moved to scripts/ or external repos

## Components and Interfaces

### Backend Components (Python FastAPI)

#### 1. Main Application (`main.py`)

**Purpose:** CLI entry point และ server orchestration
**Key Functions:**

- Command-line interface with comprehensive help (-h)
- FastAPI server startup/shutdown
- Web interface launcher
- Resume capability detection

**Interface:**

```python
# CLI Commands
python main.py process --input /path/to/images --output /path/to/output
python main.py resume --session-id abc123
python main.py web --start --port 3000
python main.py -h  # Show all available commands

class ApplicationOrchestrator:
    def start_web_server(self, port: int = 8000)
    def start_frontend(self, port: int = 3000)
    def detect_incomplete_sessions(self) -> List[Session]
```

#### 2. API Routes (`api/routes/`)

**Purpose:** RESTful API endpoints สำหรับ Frontend
**Key Endpoints:**

```python
# Project Management
POST /api/projects/create
GET /api/projects/{project_id}
GET /api/projects/{project_id}/status
POST /api/projects/{project_id}/start
POST /api/projects/{project_id}/pause

# Human Review
GET /api/projects/{project_id}/results
POST /api/projects/{project_id}/review/{image_id}/approve
GET /api/projects/{project_id}/similar/{image_id}

# WebSocket
WS /ws/progress/{project_id}  # Real-time updates
```

#### 3. Unified Batch Processor (`core/batch_processor.py`)

**Purpose:** จัดการการประมวลผลแบบ batch พร้อม performance modes
**Key Functions:**

- Unified processing logic สำหรับทุก performance modes
- GPU-accelerated batch processing (RTX2060)
- Adaptive batch sizing ตาม system resources
- Memory management และ cleanup

**Interface:**

```python
class UnifiedBatchProcessor:
    def __init__(self, mode: str = "balanced"):  # speed/balanced/smart
        self.batch_sizes = {
            "speed": 50,      # เร็ว แต่ใช้ memory มาก
            "balanced": 20,   # สมดุล
            "smart": "auto"   # ปรับตาม system resources
        }

    def process_batch_unified(self, batch: List[str]) -> List[ProcessingResult]:
        """Core logic ที่ใช้ร่วมกันทุก mode"""
        # Quality analysis (TensorFlow + GPU)
        # Defect detection (YOLO v8)
        # Similarity check (CLIP embeddings)
        # Compliance check (OCR + Face detection)
        return results

    def optimize_performance(self) -> dict:
        """Auto-adjust settings based on hardware"""
        gpu_memory = self.get_gpu_memory()
        system_ram = self.get_system_ram()
        return optimized_settings
```

#### 4. Session Manager (`core/session_manager.py`)

**Purpose:** PostgreSQL-based multi-session management และ checkpoint system
**Key Functions:**

- Multi-project concurrent processing
- Advanced checkpoint system (every 10 images)
- Resume options (continue/restart batch/fresh start)
- Real-time progress broadcasting

**PostgreSQL Schema:**

```sql
-- Projects table
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    input_folder TEXT NOT NULL,
    output_folder TEXT NOT NULL,
    performance_mode VARCHAR(20) DEFAULT 'balanced',
    status VARCHAR(20) DEFAULT 'created',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Processing sessions
CREATE TABLE processing_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    total_images INTEGER,
    processed_images INTEGER DEFAULT 0,
    approved_images INTEGER DEFAULT 0,
    rejected_images INTEGER DEFAULT 0,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status VARCHAR(20) DEFAULT 'running'
);

-- Image results with enhanced metadata
CREATE TABLE image_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES processing_sessions(id),
    image_path TEXT NOT NULL,
    filename VARCHAR(255) NOT NULL,
    source_folder VARCHAR(10), -- 1, 2, 3, etc.
    quality_scores JSONB,      -- {sharpness: 0.8, noise: 0.2, ...}
    defect_results JSONB,      -- {detected_objects: [...], confidence: 0.9}
    similarity_group INTEGER,
    similar_images JSONB,      -- [{path: "...", similarity: 0.95}, ...]
    compliance_results JSONB,  -- {logos: [], faces: [], privacy: []}
    final_decision VARCHAR(20), -- approved/rejected
    rejection_reasons TEXT[],   -- Array of reasons in Thai
    human_override BOOLEAN DEFAULT FALSE,
    human_review_at TIMESTAMP,
    processing_time REAL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Checkpoints for resume capability
CREATE TABLE checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES processing_sessions(id),
    checkpoint_type VARCHAR(20), -- batch/image/milestone
    processed_count INTEGER,
    current_batch INTEGER,
    current_image_index INTEGER,
    session_state JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Frontend Components (Next.js + TypeScript)

#### 1. Project Management Interface

```typescript
// pages/index.tsx - Dashboard
interface DashboardProps {
  activeProjects: Project[];
  recentActivity: ActivityItem[];
  systemStatus: SystemHealth;
}

// pages/projects/new.tsx - Project Creation
interface ProjectCreationFlow {
  steps: [
    { name: "Project Setup"; fields: ["name", "description"] },
    { name: "Folder Selection"; fields: ["inputFolder", "outputFolder"] },
    { name: "AI Settings"; fields: ["performanceMode", "qualityThresholds"] },
    { name: "Review & Start"; fields: ["confirmation"] }
  ];
}

// pages/projects/[id]/monitor.tsx - Real-time Monitoring
interface MonitoringInterface {
  progressBar: ProgressIndicator; // 1002/20000 (5.01%)
  currentImage: ImagePreview; // Currently processing image
  liveStats: ProcessingStats; // Approved/Rejected counts
  performanceMetrics: PerformanceData; // Speed, ETA, GPU usage
  logStream: LogEntry[]; // Real-time processing logs
}
```

#### 2. Human Review System

```typescript
// pages/projects/[id]/review.tsx - Advanced Review Interface
interface ReviewInterface {
  filterTabs: ["ทั้งหมด", "รอตรวจสอบ", "ตรวจสอบแล้ว", "ไม่ผ่าน", "ผ่าน"];
  rejectionFilters: [
    "คุณภาพต่ำ",
    "พบความผิดปกติ",
    "ภาพซ้ำ",
    "ปัญหาลิขสิทธิ์",
    "ปัญหาเทคนิค"
  ];

  imageGrid: {
    layout: "grid" | "list" | "comparison";
    sortBy: "timestamp" | "confidence" | "filename";
    pagination: PaginationControls;
  };

  reviewPanel: {
    imageViewer: ZoomableImageViewer; // Zoom, pan, full-screen
    rejectionReasons: ReasonChips; // Thai language explanations
    similarityViewer: SimilarityComparison; // Side-by-side comparison
    overrideControls: ApproveRejectButtons;
    bulkActions: BulkSelectionTools;
  };
}

// components/SimilarityViewer.tsx - Advanced Image Comparison
interface SimilarityViewerProps {
  mainImage: ImageData;
  similarImages: SimilarImageData[];
  onApprove: (imageId: string) => void;
  onReject: (imageId: string) => void;
  onSelectBest: (imageIds: string[]) => void;
}
```

### AI/ML Enhancement Components

#### 3. Enhanced Quality Analyzer (`analyzers/ai_quality_analyzer.py`)

```python
class AIQualityAnalyzer:
    def __init__(self):
        # Load models at startup for RTX2060
        self.quality_model = tf.keras.applications.ResNet50(
            weights='imagenet', include_top=False
        )
        self.aesthetic_model = self.load_aesthetic_model()

    def analyze_with_ai(self, image_path: str) -> EnhancedQualityResult:
        """AI-enhanced quality analysis"""
        # Resize for analysis (preserve original)
        image = self.load_and_resize_for_analysis(image_path)

        # Traditional metrics (fallback)
        traditional_scores = self.analyze_traditional(image)

        # AI-enhanced analysis
        ai_scores = self.analyze_with_models(image)

        return EnhancedQualityResult(
            traditional=traditional_scores,
            ai_enhanced=ai_scores,
            confidence=ai_scores.confidence,
            reasoning_thai=self.generate_thai_explanation(ai_scores)
        )
```

#### 4. Advanced Defect Detector (`analyzers/ai_defect_detector.py`)

```python
class AIDefectDetector:
    def __init__(self):
        # YOLO v8 for object detection
        self.defect_model = YOLO('yolov8n.pt')  # 6MB model
        self.confidence_threshold = 0.7

    def detect_defects_ai(self, image_path: str) -> DefectResult:
        """AI-powered defect detection"""
        # Load and resize for analysis
        image = self.load_and_resize_for_analysis(image_path)

        # YOLO detection
        results = self.defect_model(image)

        # Analyze detected objects for defects
        defects = self.analyze_object_defects(results)

        return DefectResult(
            detected_objects=defects,
            confidence_scores=[d.confidence for d in defects],
            reasoning_thai=self.generate_thai_defect_explanation(defects)
        )
```

#### 5. File Integrity Manager (`utils/file_integrity.py`)

```python
class FileIntegrityManager:
    """Critical: Protect original files at all costs"""

    FORBIDDEN_OPERATIONS = [
        "rename_source", "resize_source", "modify_source", "move_source"
    ]

    def safe_copy_to_output(self, source: str, output_folder: str) -> bool:
        """Copy approved image with 100% integrity"""
        original_name = Path(source).name
        dest_path = Path(output_folder) / original_name

        # Atomic copy with metadata preservation
        shutil.copy2(source, dest_path)

        # Verify file integrity
        return self.verify_integrity(source, dest_path)

    def load_for_analysis(self, image_path: str) -> np.ndarray:
        """Load and optionally resize for analysis (never touch original)"""
        image = cv2.imread(image_path)

        # Resize in memory for faster processing
        if max(image.shape[:2]) > 2000:
            scale = 2000 / max(image.shape[:2])
            new_size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

        return image

    def organize_output_structure(self, input_path: str, approved_results: List[ImageResult]):
        """Create output structure: imageTest1 → imageTest1_processed"""
        input_name = Path(input_path).name
        output_base = f"{input_name}_processed"

        # Group by source folder (1, 2, 3, etc.)
        approved_by_folder = {}
        for result in approved_results:
            if result.approved:
                folder_num = result.source_folder
                if folder_num not in approved_by_folder:
                    approved_by_folder[folder_num] = []
                approved_by_folder[folder_num].append(result)

        # Create output folders only for folders with approved images
        for folder_num, images in approved_by_folder.items():
            output_folder = Path(output_base) / str(folder_num)
            output_folder.mkdir(parents=True, exist_ok=True)

            # Copy images with original names
            for image_result in images:
                self.safe_copy_to_output(image_result.image_path, output_folder)
```

#### 6. Advanced Similarity Finder (`analyzers/ai_similarity_finder.py`)

```python
class AISimilarityFinder:
    def __init__(self):
        # CLIP model for semantic similarity
        self.clip_model = torch.hub.load('openai/CLIP-ViT-B/32', 'clip')
        self.similarity_threshold = 0.85

    def find_similar_images_ai(self, image_paths: List[str]) -> Dict[int, List[SimilarImage]]:
        """AI-powered similarity detection with CLIP embeddings"""
        # Extract CLIP features
        embeddings = self.extract_clip_features(image_paths)

        # Combine with perceptual hashing
        hashes = [self.compute_perceptual_hash(path) for path in image_paths]

        # Clustering with DBSCAN
        similarity_groups = self.cluster_similar_images(embeddings, hashes)

        return similarity_groups

    def extract_clip_features(self, image_paths: List[str]) -> np.ndarray:
        """Extract CLIP embeddings for semantic similarity"""
        features = []
        for path in image_paths:
            image = self.load_and_resize_for_analysis(path)
            feature = self.clip_model.encode_image(image)
            features.append(feature.cpu().numpy())
        return np.array(features)
```

#### 7. AI Compliance Checker (`analyzers/ai_compliance_checker.py`)

```python
class AIComplianceChecker:
    def __init__(self):
        # Face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # OCR for text/logo detection
        self.ocr_reader = easyocr.Reader(['en', 'th'])

    def check_compliance_ai(self, image_path: str, metadata: dict) -> ComplianceResult:
        """AI-enhanced compliance checking"""
        image = self.load_and_resize_for_analysis(image_path)

        # Face detection for privacy
        faces = self.detect_faces_ai(image)

        # Logo/text detection with OCR
        text_detections = self.detect_text_and_logos(image)

        # License plate detection
        license_plates = self.detect_license_plates(image)

        # Metadata validation
        metadata_issues = self.validate_metadata(metadata)

        return ComplianceResult(
            face_detections=faces,
            logo_detections=text_detections,
            license_plates=license_plates,
            metadata_issues=metadata_issues,
            reasoning_thai=self.generate_thai_compliance_explanation(faces, text_detections, license_plates)
        )
```

#### 8. Enhanced File Manager (`utils/enhanced_file_manager.py`)

```python
class EnhancedFileManager:
    """Enhanced file management with strict integrity protection"""

    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png'}
        self.max_file_size = 100 * 1024 * 1024  # 100MB

    def scan_input_structure(self, input_folder: str) -> InputStructure:
        """Scan input folder structure (imageTest1/1, imageTest1/2, etc.)"""
        input_path = Path(input_folder)
        structure = InputStructure(
            base_name=input_path.name,
            subfolders={}
        )

        # Scan numbered subfolders (1, 2, 3, etc.)
        for subfolder in input_path.iterdir():
            if subfolder.is_dir() and subfolder.name.isdigit():
                images = self.scan_images_in_folder(subfolder)
                if images:  # Only include folders with images
                    structure.subfolders[subfolder.name] = images

        return structure

    def create_output_structure(self, input_structure: InputStructure, approved_results: List[ImageResult]) -> str:
        """Create output structure: imageTest1 → imageTest1_processed"""
        output_base = f"{input_structure.base_name}_processed"

        # Group approved images by source folder
        approved_by_folder = self.group_by_source_folder(approved_results)

        # Create output folders only for folders with approved images
        for folder_num, approved_images in approved_by_folder.items():
            output_folder = Path(output_base) / folder_num
            output_folder.mkdir(parents=True, exist_ok=True)

            # Copy approved images with integrity verification
            for image_result in approved_images:
                self.safe_copy_with_verification(
                    image_result.image_path,
                    output_folder
                )

        return output_base
```

#### 9. Web Report Generator (`utils/web_report_generator.py`)

```python
class WebReportGenerator:
    """Generate web-based reports and analytics"""

    def generate_web_dashboard_data(self, session_id: str) -> DashboardData:
        """Generate data for web dashboard"""
        results = self.db.get_session_results(session_id)

        return DashboardData(
            summary=self.create_summary_statistics(results),
            charts=self.create_chart_data(results),
            thumbnails=self.generate_thumbnail_data(results),
            filters=self.create_filter_options(results)
        )

    def create_summary_statistics(self, results: List[ImageResult]) -> SummaryStats:
        """Create comprehensive statistics"""
        total = len(results)
        approved = len([r for r in results if r.final_decision == 'approved'])
        rejected = len([r for r in results if r.final_decision == 'rejected'])

        # Rejection reason breakdown
        rejection_reasons = {}
        for result in results:
            if result.final_decision == 'rejected':
                for reason in result.rejection_reasons:
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        return SummaryStats(
            total_images=total,
            approved_count=approved,
            rejected_count=rejected,
            approval_rate=approved/total if total > 0 else 0,
            rejection_breakdown=rejection_reasons,
            processing_time=self.calculate_total_processing_time(results)
        )

    def generate_thumbnail_data(self, results: List[ImageResult]) -> List[ThumbnailData]:
        """Generate thumbnail data for web interface"""
        thumbnails = []
        for result in results:
            thumbnail_path = self.create_thumbnail_if_needed(result.image_path)
            thumbnails.append(ThumbnailData(
                image_id=result.id,
                thumbnail_url=f"/api/thumbnails/{result.id}",
                filename=result.filename,
                decision=result.final_decision,
                rejection_reasons=result.rejection_reasons,
                confidence_scores=result.quality_scores,
                human_override=result.human_override
            ))
        return thumbnails
```

### Real-Time Communication System

#### 10. Socket.IO Manager (`websocket/socketio_manager.py`)

```python
import socketio
from typing import Dict, List
import asyncio
import json
from fastapi import FastAPI

# Create Socket.IO server with Redis adapter for scaling
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True
)

# Redis adapter for multi-process support
sio.attach(app, socketio_path='/socket.io/')

class SocketIOManager:
    """Manage real-time progress updates via Socket.IO"""

    def __init__(self):
        self.progress_cache: Dict[str, ProgressData] = {}

    async def join_session_room(self, sid: str, session_id: str):
        """Add client to session room"""
        await sio.enter_room(sid, f"session_{session_id}")

        # Send cached progress if available
        if session_id in self.progress_cache:
            await sio.emit(
                'progress_update',
                self.progress_cache[session_id].dict(),
                room=sid
            )

    async def leave_session_room(self, sid: str, session_id: str):
        """Remove client from session room"""
        await sio.leave_room(sid, f"session_{session_id}")

    async def broadcast_progress(self, session_id: str, progress_data: ProgressData):
        """Broadcast progress to all clients in session room"""
        self.progress_cache[session_id] = progress_data

        await sio.emit(
            'progress_update',
            progress_data.dict(),
            room=f"session_{session_id}"
        )

    async def broadcast_error(self, session_id: str, error_data: dict):
        """Broadcast error to session room"""
        await sio.emit(
            'processing_error',
            error_data,
            room=f"session_{session_id}"
        )

    async def broadcast_completion(self, session_id: str, completion_data: dict):
        """Broadcast completion to session room"""
        await sio.emit(
            'processing_complete',
            completion_data,
            room=f"session_{session_id}"
        )

# Socket.IO Event Handlers
@sio.event
async def connect(sid, environ):
    """Handle client connection"""
    print(f"Client {sid} connected")
    await sio.emit('connected', {'status': 'Connected to server'}, room=sid)

@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    print(f"Client {sid} disconnected")

@sio.event
async def join_session(sid, data):
    """Handle client joining session room"""
    session_id = data.get('session_id')
    if session_id:
        await socketio_manager.join_session_room(sid, session_id)
        print(f"Client {sid} joined session {session_id}")

@sio.event
async def leave_session(sid, data):
    """Handle client leaving session room"""
    session_id = data.get('session_id')
    if session_id:
        await socketio_manager.leave_session_room(sid, session_id)
        print(f"Client {sid} left session {session_id}")

@sio.event
async def pause_processing(sid, data):
    """Handle pause processing request"""
    session_id = data.get('session_id')
    # Implement pause logic here
    await sio.emit('processing_paused', {'session_id': session_id}, room=sid)

@sio.event
async def resume_processing(sid, data):
    """Handle resume processing request"""
    session_id = data.get('session_id')
    # Implement resume logic here
    await sio.emit('processing_resumed', {'session_id': session_id}, room=sid)

# Global Socket.IO manager instance
socketio_manager = SocketIOManager()
```

#### Frontend Socket.IO Client (`hooks/useSocket.ts`)

```typescript
import { useEffect, useState, useRef } from "react";
import { io, Socket } from "socket.io-client";
import { ProgressData } from "@/types/processing";

export function useSocket(sessionId: string) {
  const [progressData, setProgressData] = useState<ProgressData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const socketRef = useRef<Socket | null>(null);

  useEffect(() => {
    if (!sessionId) return;

    // Connect to Socket.IO server
    const socket = io("http://localhost:8000", {
      path: "/socket.io/",
      transports: ["websocket", "polling"], // Fallback support
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5,
      timeout: 20000,
    });

    socketRef.current = socket;

    // Connection events
    socket.on("connect", () => {
      setIsConnected(true);
      setError(null);
      console.log(`Connected to server`);

      // Join session room
      socket.emit("join_session", { session_id: sessionId });
    });

    socket.on("disconnect", (reason) => {
      setIsConnected(false);
      console.log(`Disconnected: ${reason}`);
    });

    socket.on("connect_error", (error) => {
      setError(`Connection error: ${error.message}`);
      console.error("Socket.IO connection error:", error);
    });

    // Progress events
    socket.on("progress_update", (data: ProgressData) => {
      setProgressData(data);
    });

    socket.on("processing_error", (data) => {
      setError(data.message);
      console.error("Processing error:", data);
    });

    socket.on("processing_complete", (data) => {
      console.log("Processing completed:", data);
      // Handle completion
    });

    // Auto-reconnection handling
    socket.on("reconnect", (attemptNumber) => {
      console.log(`Reconnected after ${attemptNumber} attempts`);
      setIsConnected(true);
      setError(null);
      // Rejoin session room
      socket.emit("join_session", { session_id: sessionId });
    });

    socket.on("reconnect_error", (error) => {
      console.error("Reconnection failed:", error);
      setError("Failed to reconnect");
    });

    // Cleanup on unmount
    return () => {
      socket.emit("leave_session", { session_id: sessionId });
      socket.disconnect();
    };
  }, [sessionId]);

  const pauseProcessing = () => {
    socketRef.current?.emit("pause_processing", { session_id: sessionId });
  };

  const resumeProcessing = () => {
    socketRef.current?.emit("resume_processing", { session_id: sessionId });
  };

  return {
    progressData,
    isConnected,
    error,
    pauseProcessing,
    resumeProcessing,
    disconnect: () => socketRef.current?.disconnect(),
  };
}
```

#### 11. Notification System (`utils/notification_manager.py`)

```python
class NotificationManager:
    """Handle Python console and web notifications"""

    def __init__(self):
        self.console_enabled = True
        self.web_enabled = True

    def notify_progress(self, current: int, total: int, current_file: str):
        """Console progress notification"""
        if self.console_enabled:
            percentage = (current / total) * 100
            print(f"📊 Progress: {current:,}/{total:,} ({percentage:.1f}%) - Current: {Path(current_file).name}")

    def notify_milestone(self, percentage: int, stats: ProcessingStats):
        """Milestone notification (25%, 50%, 75%, 100%)"""
        if self.console_enabled:
            print(f"🎯 Milestone: {percentage}% completed!")
            print(f"   ✅ Approved: {stats.approved:,}")
            print(f"   ❌ Rejected: {stats.rejected:,}")
            print(f"   ⚡ Speed: {stats.images_per_minute:.1f} images/min")

    def notify_completion(self, session_id: str, final_stats: FinalStats):
        """Final completion notification"""
        if self.console_enabled:
            print(f"\n🎉 Processing completed for session {session_id}!")
            print(f"   📈 Total processed: {final_stats.total_processed:,}")
            print(f"   ✅ Approved: {final_stats.approved:,} ({final_stats.approval_rate:.1f}%)")
            print(f"   ❌ Rejected: {final_stats.rejected:,} ({final_stats.rejection_rate:.1f}%)")
            print(f"   ⏱️  Total time: {final_stats.total_time}")
            print(f"   📁 Output folder: {final_stats.output_folder}")

    def notify_error(self, error_type: str, message: str, context: dict = None):
        """Error notification"""
        if self.console_enabled:
            print(f"❌ Error ({error_type}): {message}")
            if context:
                print(f"   Context: {context}")
```

## Data Models

### Enhanced Data Models

#### Core Processing Models

```python
@dataclass
class ProcessingResult:
    image_path: str
    filename: str
    source_folder: str  # "1", "2", "3", etc.
    quality_result: QualityResult
    defect_result: DefectResult
    similarity_group: int
    similar_images: List[SimilarImage]
    compliance_result: ComplianceResult
    final_decision: str  # 'approved', 'rejected'
    rejection_reasons: List[str]  # Thai language reasons
    confidence_scores: Dict[str, float]
    human_override: bool
    human_review_at: Optional[datetime]
    processing_time: float
    timestamp: datetime

@dataclass
class SimilarImage:
    image_path: str
    filename: str
    similarity_score: float
    similarity_type: str  # 'perceptual', 'semantic', 'combined'

@dataclass
class InputStructure:
    base_name: str  # "imageTest1"
    subfolders: Dict[str, List[str]]  # {"1": [list of image paths], "2": [...]}
    total_images: int
```

#### AI-Enhanced Results Models

```python
@dataclass
class EnhancedQualityResult:
    # Traditional metrics (fallback)
    sharpness_score: float
    noise_level: float
    exposure_score: float
    color_balance_score: float
    resolution: Tuple[int, int]
    file_size: int

    # AI-enhanced metrics
    aesthetic_score: float  # AI aesthetic assessment
    composition_score: float  # Rule of thirds, leading lines
    lighting_quality: float  # AI lighting analysis
    ai_confidence: float  # Confidence in AI assessment

    # Combined results
    overall_score: float
    passed: bool
    reasoning_thai: str  # Thai explanation
```

@dataclass
class EnhancedDefectResult: # YOLO detection results
detected_objects: List[ObjectDefect]
defect_locations: List[BoundingBox] # Where defects were found
defect_types: List[str] # Types of defects detected
confidence_scores: List[float] # Confidence for each detection

    # Analysis results
    anomaly_score: float
    defect_count: int
    severity_level: str  # "low", "medium", "high"

    # Decision
    passed: bool
    reasoning_thai: str  # Thai explanation

@dataclass
class ObjectDefect:
object_type: str # "glass", "utensil", "person", etc.
defect_type: str # "crack", "chip", "stain", etc.
bounding_box: BoundingBox
confidence: float
severity: str

@dataclass
class BoundingBox:
x: int
y: int
width: int
height: int

```

@dataclass
class EnhancedComplianceResult:
    # Detection results
    face_detections: List[FaceDetection]
    logo_detections: List[LogoDetection]
    text_detections: List[TextDetection]
    license_plates: List[LicensePlateDetection]

    # Analysis results
    privacy_violations: List[PrivacyViolation]
    copyright_issues: List[CopyrightIssue]
    metadata_issues: List[str]

    # Scores
    privacy_risk_score: float
    copyright_risk_score: float
    overall_compliance_score: float

    # Decision
    overall_compliance: bool
    reasoning_thai: str  # Thai explanation

@dataclass
class FaceDetection:
    bounding_box: BoundingBox
    confidence: float
    estimated_age: Optional[str]  # "child", "adult" if detectable
    privacy_risk: str  # "low", "medium", "high"

@dataclass
class LogoDetection:
    text_content: str
    bounding_box: BoundingBox
    confidence: float
    logo_type: str  # "brand", "trademark", "watermark"

@dataclass
class PrivacyViolation:
    violation_type: str  # "face", "license_plate", "personal_info"
    description: str
    severity: str  # "low", "medium", "high"
    location: BoundingBox
```

#### Real-time and Web Interface Models

```python
@dataclass
class ProgressData:
    session_id: str
    current: int
    total: int
    percentage: float
    current_file: str
    approved_count: int
    rejected_count: int
    processing_speed: float  # images per minute
    eta_minutes: int
    gpu_usage: float  # 0-100%
    memory_usage: float  # 0-100%
    timestamp: datetime

@dataclass
class ProcessingStats:
    approved: int
    rejected: int
    pending: int
    images_per_minute: float
    estimated_time_remaining: int  # minutes
    gpu_utilization: float
    memory_usage: float
    current_batch: int
    total_batches: int

@dataclass
class DashboardData:
    summary: SummaryStats
    charts: ChartData
    thumbnails: List[ThumbnailData]
    filters: FilterOptions

@dataclass
class SummaryStats:
    total_images: int
    approved_count: int
    rejected_count: int
    approval_rate: float
    rejection_breakdown: Dict[str, int]  # reason -> count
    processing_time: str
    average_confidence: float

@dataclass
class ThumbnailData:
    image_id: str
    thumbnail_url: str
    filename: str
    decision: str  # "approved", "rejected", "pending"
    rejection_reasons: List[str]
    confidence_scores: Dict[str, float]
    human_override: bool
    similar_images: List[str]  # IDs of similar images

@dataclass
class FilterOptions:
    rejection_reasons: List[str]
    confidence_ranges: List[str]
    date_ranges: List[str]
    source_folders: List[str]
    review_status: List[str]  # "all", "pending", "reviewed"
```

## Error Handling

### Error Categories

1. **File System Errors:** ไฟล์เสียหาย, permission issues, disk space
2. **Processing Errors:** AI model failures, memory issues, format incompatibility
3. **Configuration Errors:** invalid settings, missing dependencies
4. **Network Errors:** model download failures (if applicable)

### Error Handling Strategy

```python
class ErrorHandler:
    def handle_file_error(self, error: Exception, file_path: str) -> bool
    def handle_processing_error(self, error: Exception, context: dict) -> bool
    def log_error(self, error: Exception, context: dict)
    def should_retry(self, error: Exception) -> bool
```

### Recovery Mechanisms

- **Checkpoint Recovery:** Resume จาก last successful checkpoint
- **Batch Retry:** Retry failed batches with smaller batch size
- **Individual Image Skip:** Skip problematic images และ continue
- **Memory Recovery:** Force garbage collection และ restart batch

## Testing Strategy

### Unit Testing

- **Analyzer Modules:** Test แต่ละ analyzer กับ known test images
- **File Operations:** Test file scanning, copying, organization
- **Database Operations:** Test SQLite operations และ data integrity
- **Configuration:** Test config loading และ validation

### Integration Testing

- **End-to-End Pipeline:** Test complete processing pipeline
- **Resume Functionality:** Test checkpoint และ resume capabilities
- **Large Batch Processing:** Test กับ dataset ขนาดใหญ่
- **Error Scenarios:** Test error handling และ recovery

### Performance Testing

- **Memory Usage:** Monitor memory consumption กับ large batches
- **Processing Speed:** Benchmark processing time per image
- **Scalability:** Test กับ different batch sizes และ thread counts
- **Database Performance:** Test SQLite performance กับ large datasets

### Test Data Requirements

- **Quality Test Images:** High/low quality, various defects
- **Similarity Test Sets:** Groups of similar/identical images
- **Compliance Test Images:** Images with logos, faces, privacy issues
- **Large Test Dataset:** 1000+ images สำหรับ performance testing

## Performance Optimization for RTX2060

### GPU Acceleration Strategy

```python
# Optimized for RTX2060 8GB VRAM
GPU_CONFIG = {
    "tensorflow": {
        "memory_growth": True,
        "memory_limit": 6144,  # 6GB for TensorFlow
        "mixed_precision": True
    },
    "batch_processing": {
        "speed_mode": 50,      # High memory usage
        "balanced_mode": 20,   # Optimal for RTX2060
        "smart_mode": "auto"   # Adaptive based on available memory
    },
    "model_loading": {
        "strategy": "preload_all",  # Load all models at startup
        "models": ["ResNet50", "VGG16", "YOLO_v8", "CLIP"]
    }
}
```

### Web Interface Performance

```typescript
// Frontend Performance Optimizations
const PERFORMANCE_CONFIG = {
  imageLoading: {
    thumbnailSize: 300, // 300px thumbnails
    lazyLoadThreshold: 100, // Load when 100px from viewport
    preloadCount: 5, // Preload 5 images ahead
    maxConcurrent: 3, // Max 3 concurrent image loads
  },

  realTimeUpdates: {
    websocketThrottle: 100, // Update every 100ms
    batchUpdates: true, // Batch multiple updates
    maxUpdatesPerSecond: 10, // Limit to 10 updates/sec
  },

  uiOptimization: {
    virtualScrolling: true, // For large image lists
    imageVirtualization: true, // Virtualize off-screen images
    debounceSearch: 300, // 300ms search debounce
  },
};
```

## Configuration Management

### Web-Based Configuration System

```typescript
// Frontend Configuration Interface
interface ConfigurationInterface {
  projectSettings: {
    performanceMode: "speed" | "balanced" | "smart";
    batchSize: number;
    checkpointInterval: number;
  };

  aiSettings: {
    qualityThresholds: QualityThresholds;
    similarityThresholds: SimilarityThresholds;
    confidenceThresholds: ConfidenceThresholds;
  };

  outputSettings: {
    folderNaming: "input_name_processed" | "custom";
    preserveStructure: boolean;
    generateThumbnails: boolean;
  };
}
```

### Database Configuration (PostgreSQL)

```sql
-- Configuration stored in database per project
CREATE TABLE project_configurations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id),
    performance_mode VARCHAR(20) DEFAULT 'balanced',
    batch_size INTEGER DEFAULT 20,
    checkpoint_interval INTEGER DEFAULT 10,
    quality_thresholds JSONB DEFAULT '{}',
    similarity_thresholds JSONB DEFAULT '{}',
    ai_model_settings JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

## Deployment and Dependencies

## Project Structure and Organization

### Complete Project Structure (Corrected)

```
adobe-stock-processor/
├── 📁 backend/                     # Python FastAPI Backend
│   ├── 📁 api/
│   │   ├── 📁 routes/
│   │   │   ├── projects.py         # Project CRUD operations
│   │   │   ├── processing.py       # Start/stop/pause processing
│   │   │   ├── review.py           # Human review endpoints
│   │   │   └── reports.py          # Report generation APIs
│   │   └── 📁 websocket/
│   │       └── socketio_manager.py # Socket.IO event handlers
│   ├── 📁 core/
│   │   ├── batch_processor.py      # Main processing engine
│   │   ├── session_manager.py      # Multi-session management
│   │   └── unified_processor.py    # AI/ML processing logic
│   ├── 📁 analyzers/               # AI/ML analysis modules
│   │   ├── ai_quality_analyzer.py  # TensorFlow quality analysis
│   │   ├── ai_defect_detector.py   # YOLO defect detection
│   │   ├── ai_similarity_finder.py # CLIP similarity detection
│   │   └── ai_compliance_checker.py # OCR compliance checking
│   ├── 📁 utils/
│   │   ├── file_integrity.py       # File protection mechanisms
│   │   ├── notification_manager.py # Console notifications
│   │   └── web_report_generator.py # Web-based reports
│   ├── 📁 database/
│   │   ├── models.py               # SQLAlchemy models
│   │   ├── operations.py           # Database operations
│   │   └── 📁 migrations/          # Alembic database migrations
│   ├── 📁 config/                  # Backend configuration
│   │   ├── settings.py             # Configuration management
│   │   └── config.json             # Default configuration
│   ├── 📁 data/                    # Data files (moved from root)
│   │   ├── 📁 input/               # Test input images
│   │   ├── 📁 output/              # Processing results
│   │   ├── 📁 temp/                # Temporary files
│   │   └── 📁 models/              # AI/ML model files
│   ├── 📁 tests/                   # Backend tests
│   │   ├── test_quality_analyzer.py
│   │   ├── test_batch_processor.py
│   │   └── test_api_endpoints.py
│   ├── main.py                     # FastAPI app + CLI entry point
│   └── requirements.txt            # Python dependencies
│
├── 📁 frontend/                    # Next.js Frontend
│   ├── 📁 src/
│   │   ├── 📁 pages/
│   │   │   ├── index.tsx           # Main dashboard
│   │   │   ├── 📁 projects/
│   │   │   │   ├── new.tsx         # Create new project
│   │   │   │   └── 📁 [id]/
│   │   │   │       ├── monitor.tsx # Real-time monitoring
│   │   │   │       ├── review.tsx  # Human review interface
│   │   │   │       └── reports.tsx # Reports & analytics
│   │   │   └── 📁 api/             # Next.js API routes (proxy)
│   │   ├── 📁 components/
│   │   │   ├── 📁 ui/              # Base UI components (Shadcn/UI)
│   │   │   ├── 📁 forms/           # Form components
│   │   │   ├── 📁 charts/          # Chart.js components
│   │   │   └── 📁 image-viewer/    # Image comparison viewer
│   │   ├── 📁 hooks/
│   │   │   ├── useSocket.ts        # Socket.IO React hook
│   │   │   ├── useImageReview.ts   # Review functionality
│   │   │   └── useProjectStatus.ts # Project monitoring
│   │   ├── 📁 utils/
│   │   │   ├── api.ts              # API client functions
│   │   │   └── image-utils.ts      # Image handling utilities
│   │   └── 📁 types/               # Frontend TypeScript types
│   ├── 📁 tests/                   # Frontend tests
│   │   ├── components.test.tsx
│   │   └── hooks.test.ts
│   ├── package.json                # Node.js dependencies
│   └── next.config.js              # Next.js configuration
│
├── 📁 scripts/                     # Build/deploy scripts and utilities
│   ├── build.py                    # Build automation
│   ├── deploy.py                   # Deployment scripts
│   ├── check_structure.py          # Project structure validator (moved from tools/)
│   └── fix_imports.py              # Import path fixer (moved from tools/)
│
├── 📁 demos/                       # Demo applications
│   ├── demo_quality_analyzer.py
│   ├── demo_batch_processor.py
│   └── create_test_images.py
│
├── 📁 docs/                        # Documentation
│   ├── api.md                      # API documentation
│   ├── setup.md                    # Setup instructions
│   └── architecture.md             # Architecture overview
│
├── README.md                       # Main project documentation
├── docker-compose.yml              # Docker setup (optional)
├── .gitignore                      # Git ignore rules
└── .env.example                    # Environment variables template
```

### Development Workflow

```bash
# Development Mode (2 terminals)
# Terminal 1: Start Python Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py web --start --port 8000

# Terminal 2: Start Next.js Frontend
cd frontend
npm install
npm run dev  # Runs on localhost:3000

# Production Mode (Single command)
python main.py web --start  # Starts both backend (8000) + frontend (3000)
```

### Migration Strategy from Current Structure

```python
# scripts/migrate_project_structure.py
class ProjectStructureMigrator:
    """Migrate from scattered files to organized structure"""

    def __init__(self):
        self.migration_plan = {
            # Python files → backend/
            "*.py": "backend/",
            "analyzers/*.py": "backend/analyzers/",
            "utils/*.py": "backend/utils/",
            "config/*.py": "backend/config/",

            # Test directories → backend/data/
            "test_input/*": "backend/data/input/",
            "test_output/*": "backend/data/output/",
            "temp/*": "backend/data/temp/",

            # Utility files → scripts/
            "check_structure.py": "scripts/",
            "fix_imports.py": "scripts/",

            # Reports → cleanup
            "report_*.html": "DELETE",  # Will be generated in web interface
            "*.xlsx": "DELETE"          # Will be generated in web interface
        }

    def execute_migration(self):
        """Execute the migration plan"""
        print("🚀 Starting project structure migration...")

        # 1. Create new directory structure
        self.create_directories()

        # 2. Move files according to plan
        self.move_files()

        # 3. Update import paths
        self.fix_import_paths()

        # 4. Create new configuration files
        self.create_config_files()

        # 5. Clean up root directory
        self.cleanup_root()

        print("✅ Migration completed successfully!")

    def create_directories(self):
        """Create the new directory structure"""
        directories = [
            "backend/api/routes", "backend/api/websocket",
            "backend/core", "backend/analyzers", "backend/utils", "backend/database",
            "backend/config", "backend/data/input", "backend/data/output", "backend/data/temp",
            "backend/tests",
            "frontend/src/pages/projects", "frontend/src/components/ui",
            "frontend/src/hooks", "frontend/src/utils", "frontend/tests",
            "scripts", "docs"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"📁 Created: {directory}")
```

### File Organization Benefits

```
✅ Clean Separation:
   - Backend: Pure Python FastAPI
   - Frontend: Pure Next.js TypeScript
   - Shared: Common types and constants

✅ Eliminated Redundancy:
   - Single backend/data/ directory (no more test_*)
   - Development utilities moved to scripts/
   - Clean root with only essentials

✅ Better Development Experience:
   - Clear module boundaries
   - No circular dependencies
   - Easy to navigate and maintain

✅ Production Ready:
   - Docker support
   - Environment configuration
   - Proper documentation structure
```

### Backend Dependencies (Python)

```txt
# Core Framework
fastapi>=0.104.0
uvicorn>=0.24.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
redis>=5.0.0

# Socket.IO Real-time Communication
python-socketio>=5.10.0
python-socketio[asyncio_client]>=5.10.0

# AI/ML Stack
tensorflow>=2.13.0
ultralytics>=8.0.0  # YOLO v8
opencv-python>=4.8.0
torch>=2.0.0
transformers>=4.35.0  # For CLIP
easyocr>=1.7.0       # OCR for compliance checking

# Image Processing
pillow>=10.0.0
imagehash>=4.3.1
numpy>=1.24.0
scikit-learn>=1.3.0

# Database & Background Tasks
alembic>=1.12.0      # Database migrations
celery>=5.3.0        # Background tasks
python-multipart>=0.0.6

# Utilities
pydantic>=2.4.0
python-dotenv>=1.0.0
tqdm>=4.65.0
click>=8.1.0         # CLI framework
```

### Frontend Dependencies (Node.js)

```json
{
  "dependencies": {
    "next": "14.0.0",
    "react": "18.2.0",
    "typescript": "5.2.0",
    "tailwindcss": "3.3.0",
    "@headlessui/react": "1.7.0",
    "@radix-ui/react-dialog": "1.0.5",
    "@radix-ui/react-dropdown-menu": "2.0.6",
    "@radix-ui/react-select": "2.0.0",
    "class-variance-authority": "0.7.0",
    "clsx": "2.0.0",
    "tailwind-merge": "2.0.0",
    "framer-motion": "10.16.0",
    "@tanstack/react-query": "5.0.0",
    "zustand": "4.4.0",
    "chart.js": "4.4.0",
    "react-chartjs-2": "5.2.0",
    "sweetalert2": "11.7.0",
    "@floating-ui/react": "0.26.0",
    "aos": "2.3.4",
    "luxon": "3.4.0",
    "socket.io-client": "4.7.0"
  },
  "devDependencies": {
    "@next/bundle-analyzer": "14.0.0",
    "eslint": "8.52.0",
    "prettier": "3.0.0"
  }
}
```

### System Requirements

- **OS:** Windows 10/11, macOS 12+, Ubuntu 20.04+
- **Python:** 3.9+ (3.11 recommended)
- **Node.js:** 18+ (20 LTS recommended)
- **RAM:** 16GB+ (32GB recommended for large batches)
- **GPU:** RTX2060 8GB or better (CUDA 11.8+)
- **Storage:** 100GB+ free space
- **Database:** PostgreSQL 14+ or Docker

### CLI Interface Design

```python
# backend/main.py - Comprehensive CLI
import click
from fastapi import FastAPI
import uvicorn
import subprocess
import os

@click.group()
def cli():
    """Adobe Stock Image Processor - AI-powered image analysis tool"""
    pass

@cli.command()
@click.option('--input', '-i', required=True, help='Input folder path')
@click.option('--output', '-o', required=True, help='Output folder path')
@click.option('--mode', '-m', default='balanced', help='Performance mode: speed/balanced/smart')
@click.option('--resume', is_flag=True, help='Resume from last checkpoint')
def process(input, output, mode, resume):
    """Process images in batch mode"""
    click.echo(f"🚀 Starting batch processing...")
    click.echo(f"📁 Input: {input}")
    click.echo(f"📁 Output: {output}")
    click.echo(f"⚡ Mode: {mode}")
    # Implementation here

@cli.command()
@click.option('--port', default=8000, help='Backend port (default: 8000)')
@click.option('--frontend-port', default=3000, help='Frontend port (default: 3000)')
@click.option('--dev', is_flag=True, help='Development mode')
def web(port, frontend_port, dev):
    """Start web interface (backend + frontend)"""
    click.echo(f"🌐 Starting web interface...")
    click.echo(f"🐍 Backend: http://localhost:{port}")
    click.echo(f"⚛️  Frontend: http://localhost:{frontend_port}")

    if dev:
        # Development mode - start both servers
        start_development_servers(port, frontend_port)
    else:
        # Production mode - start optimized servers
        start_production_servers(port, frontend_port)

@cli.command()
@click.argument('session_id')
def resume(session_id):
    """Resume processing from checkpoint"""
    click.echo(f"🔄 Resuming session: {session_id}")
    # Implementation here

@cli.command()
def status():
    """Show system status and active sessions"""
    click.echo("📊 System Status:")
    # Show active sessions, system resources, etc.

@cli.command()
@click.option('--older-than', default='30days', help='Clean files older than specified time')
def cleanup(older_than):
    """Clean up old files and temporary data"""
    click.echo(f"🧹 Cleaning up files older than {older_than}")
    # Implementation here

@cli.command()
def models():
    """Manage AI/ML models"""
    click.echo("🤖 AI/ML Models Management:")
    # List, download, update models

if __name__ == "__main__":
    cli()
```

### Installation and Setup Process

```bash
# 1. Clone repository
git clone <repository-url>
cd adobe-stock-processor

# 2. Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Database setup
createdb adobe_stock_processor
python -m alembic upgrade head

# 4. Frontend setup
cd ../frontend
npm install
npm run build

# 5. Environment configuration
cp .env.example .env
# Edit .env with your settings

# 6. Start application
python main.py web --start  # Starts both backend and frontend

# 7. CLI Help
python main.py --help      # Show all available commands
python main.py web --help  # Show web command options
```

### Production Deployment Options

```yaml
# Option 1: Docker Compose (Recommended)
version: "3.8"
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: adobe_stock_processor
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    environment:
      DATABASE_URL: postgresql://user:password@postgres:5432/adobe_stock_processor
      REDIS_URL: redis://redis:6379
    volumes:
      - ./data:/app/data # Mount data directory
      - /path/to/gpu:/dev/nvidia0 # GPU access

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
    environment:
      NEXT_PUBLIC_API_URL: http://backend:8000

volumes:
  postgres_data:
```

### System Requirements and Optimization

```python
# System Requirements Checker
class SystemRequirements:
    def __init__(self):
        self.requirements = {
            "python": ">=3.9",
            "node": ">=18.0",
            "ram": "16GB+",
            "gpu": "RTX2060 8GB+",
            "storage": "100GB+",
            "os": ["Windows 10+", "macOS 12+", "Ubuntu 20.04+"]
        }

    def check_system(self):
        """Check if system meets requirements"""
        checks = {
            "python_version": self.check_python(),
            "node_version": self.check_node(),
            "gpu_available": self.check_gpu(),
            "memory_available": self.check_memory(),
            "disk_space": self.check_disk_space()
        }

        return all(checks.values()), checks

    def optimize_for_hardware(self):
        """Auto-optimize settings based on hardware"""
        gpu_memory = self.get_gpu_memory()
        system_ram = self.get_system_ram()

        if gpu_memory >= 8:  # RTX2060 or better
            return {
                "batch_size": 50,
                "performance_mode": "speed",
                "ai_models": "all"
            }
        elif gpu_memory >= 4:  # Lower-end GPU
            return {
                "batch_size": 20,
                "performance_mode": "balanced",
                "ai_models": "essential"
            }
        else:  # CPU only
            return {
                "batch_size": 10,
                "performance_mode": "smart",
                "ai_models": "basic"
            }
```

### Docker Deployment (Optional)

```yaml
# docker-compose.yml
version: "3.8"
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: adobe_stock_processor
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    environment:
      DATABASE_URL: postgresql://user:password@postgres:5432/adobe_stock_processor
      REDIS_URL: redis://redis:6379

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

volumes:
  postgres_data:
```
