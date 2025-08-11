# Adobe Stock Image Processor

A modern desktop application for analyzing and filtering large collections of images for Adobe Stock submission. Built with Python FastAPI backend, Next.js frontend, and AI/ML enhancement for superior accuracy and user experience.

## Architecture

This application follows a clean, modern architecture:

```
adobe-stock-processor/
├── backend/                # Python FastAPI backend
│   ├── api/               # REST API endpoints
│   ├── core/              # Core processing logic
│   ├── analyzers/         # AI/ML image analysis modules
│   ├── utils/             # Backend utilities
│   ├── database/          # Database models
│   └── main.py           # Backend entry point
├── frontend/              # Next.js frontend (to be implemented)
├── infra/                 # Docker, environment, migrations
├── data/                  # Unified data directory
│   ├── input/            # Test input images
│   ├── output/           # Processing results
│   └── temp/             # Temporary files
├── tools/                 # Development and utility tools
├── scripts/               # Processing and report scripts
├── demos/                 # Demo applications
├── tests/                 # Test suite
├── docs/                  # Documentation
├── reports/               # Generated reports
├── logs/                  # Application logs
└── README.md             # This file
```

## Features

### Modern Tech Stack
- **Backend**: Python FastAPI + SQLAlchemy + Socket.IO
- **Frontend**: Next.js 14 + TypeScript + Tailwind CSS (planned)
- **Database**: PostgreSQL + Redis for caching
- **AI/ML**: TensorFlow + YOLO v8 + OpenCV with GPU acceleration
- **Real-time**: Socket.IO for live progress updates

### Core Capabilities
- **AI-Enhanced Analysis**: TensorFlow models with GPU acceleration (RTX2060)
- **Batch Processing**: Handle 25,000+ images with intelligent memory management
- **Real-time Monitoring**: Web-based progress tracking with Socket.IO
- **Human Review System**: Web interface for reviewing and overriding AI decisions
- **Multi-Session Management**: Concurrent processing projects with PostgreSQL
- **File Integrity Protection**: Strict protection of original files
- **Advanced Resume**: Multiple recovery options with detailed checkpoints

### Processing Features
- **Quality Analysis**: AI-enhanced sharpness, noise, exposure assessment
- **Defect Detection**: YOLO v8 for advanced object and anomaly detection
- **Similarity Detection**: CLIP embeddings + perceptual hashing
- **Compliance Checking**: OCR, face detection, privacy analysis
- **Performance Modes**: Speed/Balanced/Smart with unified processing logic

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+ (for frontend development)
- Docker & Docker Compose (recommended)
- GPU drivers for RTX2060 (optional, for AI acceleration)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd adobe-stock-processor
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment**
   ```bash
   cp infra/.env.sample .env
   # edit DATABASE_URL if your Postgres credentials differ
   ```

4. **Run with Docker (Recommended)**
   ```bash
   (cd infra && docker-compose up -d)
   ```

5. **Or run locally**
   ```bash
   # Start backend
   python backend/main.py server

   # Access web interface at http://localhost:3000
   ```

### Basic Usage

```bash
# Process images (legacy CLI)
python backend/main.py process input_folder output_folder

# Resume interrupted processing
python backend/main.py resume

# Start API server
python backend/main.py server

# Show all options
python backend/main.py --help
```

## Development

### Project Structure Philosophy

This project follows modern software architecture principles:

1. **Separation of Concerns**: Backend and frontend are clearly separated
2. **Clean Architecture**: Each layer has well-defined responsibilities
3. **Scalability**: Modular design allows for easy extension and maintenance
4. **Developer Experience**: Clear structure makes onboarding and development efficient

### Backend Development

The backend is organized into logical modules:

- **API Layer** (`backend/api/`): REST endpoints and WebSocket handlers
- **Core Logic** (`backend/core/`): Business logic and processing orchestration
- **Analyzers** (`backend/analyzers/`): AI/ML image analysis modules
- **Database** (`backend/database/`): Models, migrations, and data access
- **Utils** (`backend/utils/`): Backend-specific utilities

### Frontend Development (Planned)

The frontend will be built with:

- **Next.js 14**: React framework with app router
- **TypeScript**: Type safety and better developer experience
- **Tailwind CSS**: Utility-first CSS framework
- **Shadcn/UI**: Modern component library
- **Socket.IO Client**: Real-time communication with backend

### Development Tools

Located in the `tools/` directory:

- `check_structure.py`: Analyze project structure and test commands
- `fix_imports.py`: Fix import paths after reorganization

## Testing

```bash
# Run comprehensive tests
python scripts/run_comprehensive_tests.py

# Run simple tests
python scripts/run_simple_tests.py

# Test specific components
python tests/test_quality_analyzer.py
```

## Configuration

The system supports multiple configuration methods:

1. **JSON Configuration**: `backend/config/config.json`
2. **Environment Variables**: For Docker deployment
3. **Web Interface**: Runtime configuration through UI (planned)

## Performance

### Benchmarks
- **Processing Speed**: 2-10 images/second (depending on AI models and hardware)
- **Memory Management**: Adaptive batch sizing based on system resources
- **GPU Acceleration**: RTX2060 support for TensorFlow and YOLO models
- **Scalability**: Tested with 25,000+ image datasets

### Optimization
- Configurable performance modes (Speed/Balanced/Smart)
- Automatic hardware detection and optimization
- Memory cleanup and garbage collection
- Multi-threading for I/O operations

## Docker Deployment

The application includes a complete Docker setup:

```bash
# Start all services
(cd infra && docker-compose up -d)

# View logs
(cd infra && docker-compose logs -f)

# Stop services
(cd infra && docker-compose down)
```

Services included:
- **PostgreSQL**: Primary database
- **Redis**: Caching and Socket.IO adapter
- **Backend**: Python FastAPI application
- **Frontend**: Next.js application (when implemented)

## Migration from Legacy Structure

This version represents a complete restructuring from the previous scattered file organization to a modern, maintainable architecture. Key improvements:

1. **Organized Structure**: Clear separation of backend and frontend code
2. **Eliminated Redundancy**: Consolidated duplicate test directories
3. **Modern Stack**: Upgraded to FastAPI, PostgreSQL, and planned Next.js frontend
4. **Better Developer Experience**: Clear project structure and development tools
5. **Scalability**: Architecture supports future enhancements and team development

## Legacy Support

The system maintains backward compatibility with existing scripts and demos:

```bash
# Legacy processing (still works)
python backend/main.py process input_folder output_folder

# Demo scripts (moved to demos/)
python demos/demo_quality_analyzer.py
python demos/demo_defect_detector.py

# Utility scripts (moved to scripts/)
python scripts/create_excel_report.py
python scripts/view_database_report.py

# Development tools (moved to tools/)
python tools/check_structure.py
python tools/fix_imports.py
```

## Troubleshooting

### Common Issues

1. **Import Errors After Restructuring**
   ```bash
   python tools/fix_imports.py
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Database Connection Issues**
   ```bash
   (cd infra && docker-compose up postgres redis)
   ```

4. **Performance Issues**
   - Check GPU drivers for AI acceleration
   - Adjust batch sizes in configuration
   - Monitor system resources

### Development Tools

```bash
# Check project structure and test commands
python tools/check_structure.py

# Fix import paths after file moves
python tools/fix_imports.py

# Run comprehensive tests
python scripts/run_comprehensive_tests.py
```

## Contributing

1. Follow the established project structure
2. Use TypeScript for frontend development
3. Add tests for new functionality
4. Update documentation for changes
5. Follow conventional commit messages

## License

This project is for educational and research purposes. Ensure compliance with Adobe Stock guidelines for commercial use.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run `python tools/check_structure.py` for diagnostics
3. Review logs in the `logs/` directory
4. Test with smaller datasets first