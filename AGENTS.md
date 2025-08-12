# AGENTS

This repository is under an incremental refactor. Follow these guardrails for any change.

## Development Setup
- Python 3.11+
- Node 18+
- Install backend dependencies: `pip install -r backend/requirements.txt`
- Install frontend dependencies: run `npm install` in `frontend`
- Copy `backend/.env.example` and `frontend/.env.example` to `.env` in each app and adjust values as needed

## Commands
- Run tests before committing: `pytest`
- Start backend: `uvicorn backend.main:app --reload`
- Start frontend: `(cd frontend && npm run dev)`
- Start services with Docker: `docker-compose up -d`

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
  - `/docs` – architecture and process notes
