# AGENTS

This repository is under an incremental refactor. Follow these guardrails for any change.

## Development Setup
- Python 3.11+
- Node 18+
- Install backend dependencies: `pip install -r backend/requirements.txt`
- Install frontend dependencies: run `npm install` in `frontend`
- Copy `infra/.env.sample` to `.env` and adjust values as needed

### Environment Variables
From `infra/.env.sample`:
- `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/stockdb`
- `REDIS_URL=redis://localhost:6379`
- `NEXT_PUBLIC_API_URL=http://localhost:8000`
- `NEXT_PUBLIC_WS_URL=ws://localhost:8000`

## Commands
- Run tests before committing: `pytest`
- Start backend: `python backend/main.py server --reload`
- Start frontend: `(cd frontend && npm run dev)`
- Start services with Docker: `(cd infra && docker-compose up -d)`

## Database
- **PostgreSQL only.** The app must run against a local Docker PostgreSQL instance.
- Remove or migrate any SQLite or other database references.

## Contribution Guidelines
- Prefer refactoring over rewriting.
- Keep pull requests small (≤300 changed lines).
- Ensure tests pass and relevant docs are updated.
- Stay within the canonical repo layout:
  - `/backend` – FastAPI application
  - `/frontend` – Next.js web client
  - `/infra` – Docker, environment samples, and migrations
