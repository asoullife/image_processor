# Refactor Roadmap

## Inventory of Suspected Unused / Duplicate Files
- `test_basic_functionality.py` – empty placeholder
- `run_final_integration_tests.py`, `validate_all_requirements.py` – standalone scripts duplicating pytest
- `requirements-minimal.txt` – merged into `backend/requirements.txt`
- `docs/FINAL_INTEGRATION_SUMMARY.md` and other legacy docs
- `backend/RESUME_RECOVERY_IMPLEMENTATION.md` and similar design notes
- `backend/start_server.py` – duplicates server startup logic in `backend/main.py`

## Step-by-Step PR Plan
1. **Cleanup root test helpers**
   - Remove `test_basic_functionality.py`, `run_final_integration_tests.py`, and `validate_all_requirements.py`.
2. **Consolidate dependency manifests**
   - Drop `requirements-minimal.txt` and ensure `backend/requirements.txt` covers development and runtime.
3. **Unify backend startup**
   - Merge `start_server.py` functionality into `backend/main.py` or use a single entry point; update docs accordingly.
4. **Prune outdated documentation**
   - Delete legacy design docs like `docs/FINAL_INTEGRATION_SUMMARY.md` and `backend/RESUME_RECOVERY_IMPLEMENTATION.md`.
5. **Organize tests**
   - Move remaining root-level tests into `backend/tests` and group by feature.
6. **Enforce PostgreSQL**
   - Audit for any SQLite or file-based database usage and migrate to PostgreSQL adapters.
7. **Document workflow**
   - Update README and docs with final run commands, environment variables, and architecture notes.

Each PR should change ≤300 lines and include test runs.
