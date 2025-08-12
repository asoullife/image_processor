# Architecture Overview

## Backend Entrypoints
- `backend/api/main.py` defines the FastAPI application `app` and wraps it in a Socket.IO `socket_app` ASGI application.
- `backend/main.py` exposes a CLI `start_api_server` that builds the FastAPI app and runs Uvicorn.

## API Routes
The FastAPI app mounts these route groups:
- `/api/health` – service and database health checks
- `/api/auth` – session-based user authentication
- `/api/projects` – create projects and start processing
- `/api/sessions` – query processing sessions
- `/api/analysis` – ad‑hoc analysis helpers
- `/api/review` – human review tooling
- `/api/thumbnails` – thumbnail retrieval
- `/api/recovery` – recover interrupted sessions
- `/api/settings` – system configuration
- `/api/reports` and `/api/mock-reports` – reporting endpoints
- `/api/websocket` – WebSocket handshake used by Socket.IO

## Services
- **Database** – PostgreSQL accessed via SQLAlchemy async engine
- **Realtime** – Socket.IO server with optional Redis adapter for horizontal scaling
- **Processors** – batch image processing pipeline with progress tracking

## Machine Learning
Analyzers use TensorFlow, PyTorch, OpenCV and related libraries for quality, defect, similarity and compliance checks.

## Decisions
- Socket.IO is the only realtime transport.
- PostgreSQL is the only supported database.
- Docker Compose will live at the repository root and the `/infra` folder will be removed.
