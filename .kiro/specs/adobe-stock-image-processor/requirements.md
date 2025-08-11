# Requirements Document

## Introduction

Adobe Stock Image Processor เป็นระบบประมวลผลและคัดกรองภาพแบบ Desktop Application ที่ใช้ Python Backend + Next.js Frontend พร้อม AI/ML enhancement เพื่อวิเคราะห์ภาพจำนวนมาก (25,000+ ภาพ) อย่างอัตโนมัติ โดยตรวจสอบคุณภาพ ความผิดปกติ และความคล้ายกัน พร้อมระบบ Real-time Web Monitoring และ Human Review System ที่ช่วยให้ผู้ใช้สามารถตรวจสอบและแก้ไขผลการวิเคราะห์ได้ ระบบออกแบบให้ทำงานบนเครื่องส่วนตัว (Local Desktop) โดยไม่ต้องพึ่งพา Cloud Services

## Requirements

### Requirement 1

**User Story:** As a stock photographer, I want to create processing projects through a web interface and run autonomous batch processing, so that I can analyze large image collections (25,000+) without constant supervision while monitoring progress in real-time.

#### Acceptance Criteria

1. WHEN creating a new project THEN the system SHALL provide a web interface to select input/output folders and configure processing settings
2. WHEN starting processing THEN the system SHALL run autonomously without requiring human intervention during analysis
3. WHEN processing large image batches THEN the system SHALL handle 25,000+ images with proper memory management and GPU acceleration (RTX2060)
4. WHEN processing is active THEN users SHALL be able to monitor real-time progress through web interface showing current file (e.g., 1002/20000) and percentage completion
5. WHEN processing completes THEN the system SHALL notify users via Python console with completion statistics

### Requirement 2

**User Story:** As a stock photographer, I want AI-enhanced quality detection using free ML models, so that I can achieve higher accuracy in identifying image issues compared to basic computer vision methods.

#### Acceptance Criteria

1. WHEN analyzing image quality THEN the system SHALL use TensorFlow models (ResNet50, VGG16) with GPU acceleration for advanced quality assessment
2. WHEN detecting defects THEN the system SHALL use YOLO v8 for object detection and anomaly identification
3. WHEN processing images for analysis THEN the system SHALL be allowed to resize images temporarily for faster processing while preserving original files
4. WHEN AI models are unavailable THEN the system SHALL fallback to OpenCV-based methods without crashing
5. WHEN analyzing quality THEN the system SHALL provide confidence scores and detailed reasoning for each decision

### Requirement 3

**User Story:** As a stock photographer, I want a comprehensive Human Review System through web interface, so that I can review rejected images with clear reasoning and override AI decisions when necessary.

#### Acceptance Criteria

1. WHEN viewing rejected images THEN the system SHALL display clear rejection reasons in Thai language (e.g., "คุณภาพภาพต่ำ ไม่เหมาะสำหรับขาย", "พบภาพที่คล้ายกัน อาจถูกปฏิเสธเป็น spam")
2. WHEN filtering review results THEN users SHALL be able to filter by rejection reasons, review status (All/Pending Review/Human Reviewed), and processing status
3. WHEN reviewing similar images THEN the system SHALL display the rejected image alongside all similar images found with visual comparison interface
4. WHEN making review decisions THEN users SHALL be able to approve rejected images, which will immediately copy them to the output folder
5. WHEN human override occurs THEN the system SHALL update the database and file system without interrupting ongoing processing of other projects

### Requirement 4

**User Story:** As a stock photographer, I want multi-session project management, so that I can run multiple processing projects simultaneously and view historical results from previous sessions.

#### Acceptance Criteria

1. WHEN creating projects THEN the system SHALL support multiple concurrent processing sessions without conflicts
2. WHEN organizing output THEN the system SHALL use input folder name with "_processed" suffix (e.g., "imageTest1" → "imageTest1_processed")
3. WHEN structuring output THEN the system SHALL mirror input folder structure, creating output subfolders only for folders containing approved images
4. WHEN managing sessions THEN users SHALL be able to view and review results from previous processing sessions while new sessions are running
5. WHEN accessing historical data THEN the system SHALL maintain separate database records and web interfaces for each processing session

### Requirement 5

**User Story:** As a stock photographer, I want strict file integrity protection, so that my original images are never modified, renamed, or corrupted during processing.

#### Acceptance Criteria

1. WHEN processing images THEN the system SHALL never modify, rename, or alter original source files under any circumstances
2. WHEN copying approved images THEN the system SHALL preserve exact original filenames, file sizes, and metadata
3. WHEN resizing images for analysis THEN the system SHALL only resize temporary copies in memory, never the original files
4. WHEN copying to output folders THEN the system SHALL use atomic copy operations with integrity verification
5. WHEN organizing output THEN the system SHALL create subfolders matching input structure (e.g., imageTest1/1 → imageTest1_processed/1) only if they contain approved images

### Requirement 6

**User Story:** As a stock photographer, I want advanced similarity detection and clustering, so that I can identify duplicate and similar images that might be rejected by Adobe Stock as spam.

#### Acceptance Criteria

1. WHEN comparing images THEN the system SHALL use both perceptual hashing and deep learning features (CLIP embeddings) for comprehensive similarity detection
2. WHEN grouping similar images THEN the system SHALL cluster related images and display them together in the review interface
3. WHEN showing similarity matches THEN the system SHALL display the rejected image alongside all similar images with similarity scores
4. WHEN evaluating similarity THEN the system SHALL apply configurable thresholds to prevent spam-like submissions while allowing reasonable variations

### Requirement 7

**User Story:** As a stock photographer, I want robust resume capability with multiple recovery options, so that I can continue processing after any interruption (power failure, system crash, blue screen).

#### Acceptance Criteria

1. WHEN processing is interrupted THEN the system SHALL save detailed checkpoints every 10 images with complete session state
2. WHEN restarting after interruption THEN the system SHALL detect previous incomplete sessions and offer resume options
3. WHEN resuming processing THEN users SHALL be able to choose between "Continue from last checkpoint", "Restart current batch", or "Start completely fresh"
4. WHEN resuming THEN the system SHALL verify data integrity and continue from the exact point of interruption without reprocessing completed images

### Requirement 8

**User Story:** As a stock photographer, I want web-based configuration and performance optimization, so that I can adjust processing settings through the web interface and optimize performance for my hardware.

#### Acceptance Criteria

1. WHEN configuring processing THEN users SHALL be able to select performance modes (Speed/Balanced/Smart) through the web interface
2. WHEN using different performance modes THEN the system SHALL use unified processing logic with different batch sizes and resource allocation
3. WHEN system resources are insufficient THEN the system SHALL automatically adjust batch sizes and provide performance recommendations
4. WHEN using CLI THEN the system SHALL provide comprehensive help (-h) showing all available commands and options

### Requirement 9

**User Story:** As a stock photographer, I want comprehensive compliance checking with AI assistance, so that I can ensure my images meet Adobe Stock guidelines for copyright, privacy, and content standards.

#### Acceptance Criteria

1. WHEN scanning image content THEN the system SHALL detect visible logos, trademarks, and brand elements using OCR and template matching
2. WHEN analyzing privacy concerns THEN the system SHALL identify faces, license plates, and other potentially sensitive information
3. WHEN checking metadata THEN the system SHALL verify EXIF data appropriateness and keyword relevance
4. WHEN evaluating content THEN the system SHALL flag potentially culturally insensitive or inappropriate material

### Requirement 10

**User Story:** As a stock photographer, I want web-based reports and analytics, so that I can view detailed processing results and statistics through an intuitive web interface without needing to export files.

#### Acceptance Criteria

1. WHEN processing completes THEN the system SHALL provide comprehensive web-based reports with interactive filtering and sorting
2. WHEN viewing reports THEN users SHALL see visual summaries, thumbnail previews, and detailed statistics through the web interface
3. WHEN tracking progress THEN the system SHALL provide real-time progress indicators, estimated completion times, and performance metrics
4. WHEN reviewing results THEN users SHALL be able to filter by approval status, rejection reasons, and processing dates without requiring file exports

### Requirement 11

**User Story:** As a developer, I want a clean and organized project structure, so that the codebase is maintainable, scalable, and follows modern development best practices.

#### Acceptance Criteria

1. WHEN organizing the project THEN the system SHALL separate backend (Python FastAPI) and frontend (Next.js) into distinct directories with clear boundaries
2. WHEN structuring backend code THEN the system SHALL organize code into logical modules (api/, core/, analyzers/, utils/, database/, config/)
3. WHEN managing data files THEN the system SHALL consolidate all data into backend/data/ directory (input/, output/, temp/) as data belongs to backend processing
4. WHEN handling shared code THEN the system SHALL move shared utilities into backend/ and expose functionality through API endpoints rather than shared libraries
5. WHEN deploying the application THEN the root directory SHALL contain only essential files (main.py, README.md, docker-compose.yml, .gitignore) with development tools moved to scripts/ or external repositories

### Requirement 12

**User Story:** As a developer, I want robust real-time communication using Socket.IO, so that the web interface can receive live updates with automatic reconnection and fallback mechanisms.

#### Acceptance Criteria

1. WHEN implementing real-time updates THEN the system SHALL use Socket.IO with python-socketio for robust communication
2. WHEN connecting to progress updates THEN Next.js frontend SHALL connect to Python backend via Socket.IO with automatic reconnection
3. WHEN broadcasting progress THEN the system SHALL use Socket.IO rooms to manage multiple concurrent sessions
4. WHEN connections fail THEN Socket.IO SHALL automatically handle reconnection with exponential backoff and fallback to HTTP polling
5. WHEN managing real-time state THEN the system SHALL cache progress data and sync state when clients reconnect