# Refactor Roadmap

## Inventory of Suspected Unused / Duplicate Files
- `demos/` – standalone demo scripts
- `scripts/` – multiple utility/report scripts (`create_csv_report.py`, `create_html_dashboard.py`, etc.)
- Root docs left from earlier tasks: `FINAL_INTEGRATION_SUMMARY.md`, `TASK_31_IMPLEMENTATION_SUMMARY.md`, `TASK_32_IMPLEMENTATION_SUMMARY.md`, `PROJECT_STRUCTURE.md`
- Windows helper scripts: `setup_postgres.bat`, `docker_commands.bat`
- Environment artifacts: `.kiro/`, `.venv/`, `test_results/`

## Database References
- PostgreSQL configs: `infra/docker-compose.yml`, `backend/database/connection.py`, `infra/migrations/*`
- Goal: consolidate on PostgreSQL via Docker and remove any legacy database references.

## Proposed Canonical Layout
```
/ backend  - FastAPI application
/ frontend - Next.js interface
/ infra    - docker-compose, environment files, migrations
```

## Step-by-Step PR Plan
1. **Remove legacy database helpers and tests**
   - Delete legacy helper scripts and related tests.
   - Update `backend/database/connection.py` to rely solely on PostgreSQL.
2. **Prune demo and report scripts**
   - Delete `demos/` contents and unused scripts in `scripts/`.
3. **Cleanup legacy docs and artifacts**
   - Remove task summary files, `.kiro/`, `test_results/`, and Windows `.bat` helpers.
4. **Introduce `/infra` directory**
   - Move `docker-compose.yml`, deployment scripts, and environment samples into `infra/`.
5. **Consolidate database configuration**
   - Create single DB config module; update tests and README to point to Postgres.
6. **Rationalize test suite**
   - Remove redundant tests and ensure remaining tests rely on Postgres fixtures.
7. **Document standard development workflow**
   - Update README and docs with canonical commands and layout.

Each PR should change ≤300 lines and include test runs.
