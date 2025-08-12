# Adobe Stock Image Processor - Backend

Modern FastAPI backend for the Adobe Stock Image Processor with PostgreSQL database support and modular analyzer system.

## Architecture

```
backend/
├── api/                    # FastAPI routes and dependencies
│   ├── main.py            # FastAPI application
│   ├── dependencies.py    # Dependency injection
│   ├── schemas.py         # Pydantic models
│   └── routes/            # API route modules
├── core/                  # Core business logic
│   ├── services.py        # Business services
│   └── dependency_injection.py
├── database/              # Database layer
│   ├── connection.py      # Database connection management
│   ├── models.py          # SQLAlchemy models
│   └── migrations.py      # Database migrations
├── analyzers/             # Image analysis modules
│   └── analyzer_factory.py
├── config/                # Configuration management
└── utils/                 # Utilities
```

## Features

- **FastAPI Backend**: Modern async web framework with automatic API documentation
- **PostgreSQL Database**: Enhanced schema for multi-session support
- **Modular Analyzers**: Factory pattern for image analysis components
- **Dependency Injection**: Clean separation of concerns
- **RESTful API**: Complete CRUD operations for projects and sessions
- **Real-time Updates**: WebSocket support for progress monitoring
- **Configuration Management**: JSON-based configuration with validation

## Quick Start

### 1. Install Dependencies

```bash
# Install backend dependencies
python backend/setup_backend.py

# Or manually install
pip install fastapi uvicorn sqlalchemy asyncpg pydantic
```

### 2. Set up Database

```bash
# Install PostgreSQL (example for Ubuntu)
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb stockdb

# Set environment variable
export DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/stockdb"
```

### 3. Start the Server

```bash
# Start FastAPI server
python backend/main.py server

# Or with custom options
python backend/main.py server --host 0.0.0.0 --port 8000 --reload
```

### 4. Access API Documentation

- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## API Endpoints

### Projects
- `POST /api/projects/` - Create new project
- `GET /api/projects/` - List projects
- `GET /api/projects/{id}` - Get project details
- `PUT /api/projects/{id}` - Update project
- `DELETE /api/projects/{id}` - Delete project
- `POST /api/projects/{id}/start` - Start processing

### Sessions
- `GET /api/sessions/` - List processing sessions
- `GET /api/sessions/{id}` - Get session details
- `GET /api/sessions/{id}/status` - Get session status
- `GET /api/sessions/{id}/results` - Get session results
- `POST /api/sessions/{id}/resume` - Resume session
- `POST /api/sessions/{id}/pause` - Pause session

### Analysis
- `POST /api/analysis/single` - Analyze single image
- `POST /api/analysis/upload` - Analyze uploaded image
- `POST /api/analysis/batch` - Analyze batch of images
- `GET /api/analysis/types` - Get available analysis types

### Health
- `GET /api/health/` - Basic health check
- `GET /api/health/detailed` - Detailed system status

## Configuration

Configuration is managed through JSON files in `backend/config/`:

```json
{
  "processing": {
    "batch_size": 20,
    "max_workers": 4,
    "checkpoint_interval": 50
  },
  "quality": {
    "min_sharpness": 100.0,
    "max_noise_level": 0.1,
    "min_resolution": [800, 600]
  },
  "database": {
    "url": "postgresql+asyncpg://postgres:postgres@localhost:5432/stockdb"
  }
}
```

## Database Schema

### Projects Table
- Multi-project support with settings and status tracking

### Processing Sessions Table
- Session-based processing with progress tracking
- Resume capability with checkpoints

### Image Results Table
- Detailed analysis results with JSON storage
- Human review support with override capabilities

### Enhanced Features
- Similarity grouping
- Processing logs
- System metrics
- User preferences

## Development

### Running Tests

```bash
# Test backend structure
python backend/test_backend_structure.py

# Run unit tests (when available)
pytest backend/tests/
```

### Database Migrations

```bash
# Run migrations
python -c "from backend.database.migrations import run_migrations; import asyncio; asyncio.run(run_migrations())"
```

### Development Server

```bash
# Start with auto-reload
python backend/main.py server --reload
```

## Environment Variables

- `DATABASE_URL`: PostgreSQL connection string
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `API_HOST`: API server host (default: 127.0.0.1)
- `API_PORT`: API server port (default: 8000)

## Integration with CLI

The backend maintains compatibility with the existing CLI interface:

```bash
# CLI processing (existing functionality)
python backend/main.py process --input /path/to/images --output /path/to/output

# API server mode (new functionality)
python backend/main.py server --port 8000
```

## Next Steps

1. **Frontend Integration**: Connect with Next.js frontend
2. **WebSocket Support**: Real-time progress updates
3. **AI/ML Enhancement**: Integrate TensorFlow and YOLO models
4. **Human Review System**: Web-based review interface
5. **Performance Optimization**: GPU acceleration and caching