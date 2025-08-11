# Adobe Stock Image Processor - Project Structure

This document describes the reorganized project structure after the cleanup and modernization effort.

## New Project Structure

```
adobe-stock-processor/
├── backend/                    # Python FastAPI backend
│   ├── api/                   # REST API endpoints and routes
│   ├── core/                  # Core processing logic and orchestration
│   ├── analyzers/             # AI/ML image analysis modules
│   ├── utils/                 # Backend-specific utilities
│   ├── database/              # Database models, migrations, and access
│   ├── config/                # Backend configuration files
│   └── main.py               # Backend application entry point
├── frontend/                   # Next.js frontend (to be implemented)
├── data/                       # Unified data directory
│   ├── input/                 # Test input images and datasets
│   ├── output/                # Processing results and approved images
│   └── temp/                  # Temporary files and processing cache
├── tools/                      # Development and utility tools
│   ├── check_structure.py     # Project structure analyzer
│   └── fix_imports.py         # Import path fixer
├── scripts/                    # Processing and report generation scripts
├── demos/                      # Demo applications and examples
├── tests/                      # Comprehensive test suite
├── docs/                       # Documentation files
├── reports/                    # Generated reports and dashboards
├── logs/                       # Application logs
├── main.py                     # Main entry point (delegates to backend)
├── docker-compose.yml          # Docker orchestration
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── PROJECT_STRUCTURE.md        # This file
├── requirements.txt            # Python dependencies
└── requirements-minimal.txt    # Minimal dependencies
```

## Key Changes from Legacy Structure

### 1. Backend Organization
- **Before**: Scattered modules in root (`analyzers/`, `core/`, `utils/`, `config/`)
- **After**: Organized under `backend/` with clear separation of concerns

### 2. Data Consolidation
- **Before**: Multiple test directories (test_input/, test_output/, test_output_resume/, temp/)
- **After**: Unified `data/` directory with logical subdirectories

### 3. Development Tools
- **Before**: Utility files scattered in root (`check_structure.py`, `fix_imports.py`)
- **After**: Organized in `tools/` directory

### 4. Clean Root Directory
- **Before**: Many files in root directory
- **After**: Only essential files (main.py, README.md, docker-compose.yml, .gitignore)

### 5. Modern Architecture Preparation
- **Backend**: Ready for FastAPI implementation
- **Frontend**: Directory prepared for Next.js
- **Docker**: Complete containerization setup

## Directory Purposes

### Backend (`backend/`)
Contains all Python backend code organized by function:

- **`api/`**: FastAPI routes, WebSocket handlers, request/response models
- **`core/`**: Business logic, batch processing, session management
- **`analyzers/`**: AI/ML modules for image analysis (quality, defects, similarity, compliance)
- **`utils/`**: Backend utilities (file management, error handling, logging)
- **`database/`**: Database models, migrations, connection management
- **`config/`**: Configuration files and validation

### Frontend (`frontend/`)
Prepared for Next.js implementation:
- Will contain React components, pages, hooks, and styles
- TypeScript + Tailwind CSS + Shadcn/UI
- Socket.IO client for real-time communication

### Data (`data/`)
Unified data management:
- **`input/`**: Test datasets and sample images
- **`output/`**: Processing results and approved images
- **`temp/`**: Temporary files and processing cache

### Tools (`tools/`)
Development and maintenance utilities:
- **`check_structure.py`**: Analyze project structure and test commands
- **`fix_imports.py`**: Fix import paths after file reorganization

## Migration Benefits

### 1. Improved Maintainability
- Clear separation of concerns
- Logical file organization
- Easier navigation and development

### 2. Better Scalability
- Modular architecture supports team development
- Clear boundaries between components
- Easy to add new features

### 3. Modern Development Practices
- Docker-ready setup
- Prepared for microservices architecture
- CI/CD friendly structure

### 4. Enhanced Developer Experience
- Clear project structure
- Consistent naming conventions
- Comprehensive documentation

## Import Path Changes

After the reorganization, import paths have been updated:

### Backend Modules
```python
# Old
from analyzers.quality_analyzer import QualityAnalyzer
from core.batch_processor import BatchProcessor
from utils.file_manager import FileManager

# New
from backend.analyzers.quality_analyzer import QualityAnalyzer
from backend.core.batch_processor import BatchProcessor
from backend.utils.file_manager import FileManager
```

### Scripts and Demos
Scripts and demos now include proper path setup:
```python
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

## Backward Compatibility

The new structure maintains backward compatibility:

1. **Main Entry Point**: `python main.py` still works (delegates to backend)
2. **Scripts**: All existing scripts work with updated import paths
3. **Demos**: Demo applications continue to function
4. **Tests**: Test suite runs with new structure

## Future Enhancements

The new structure prepares for:

1. **FastAPI Backend**: Modern API framework implementation
2. **Next.js Frontend**: React-based web interface
3. **PostgreSQL Database**: Scalable database solution
4. **Socket.IO**: Real-time communication
5. **Docker Deployment**: Containerized deployment
6. **Microservices**: Potential service separation

## Development Workflow

### Setting Up Development Environment
```bash
# Clone repository
git clone <repository-url>
cd adobe-stock-processor

# Install dependencies
pip install -r requirements.txt

# Run structure check
python tools/check_structure.py

# Fix any import issues
python tools/fix_imports.py
```

### Running the Application
```bash
# Main application (legacy CLI)
python main.py process input_folder output_folder

# Development tools
python tools/check_structure.py
python tools/fix_imports.py

# Demo applications
python demos/demo_quality_analyzer.py

# Test suite
python scripts/run_comprehensive_tests.py
```

## Conclusion

This restructuring transforms the Adobe Stock Image Processor from a collection of scattered files into a modern, maintainable application architecture. The new structure supports current functionality while preparing for future enhancements including web interface, real-time monitoring, and advanced AI/ML capabilities.