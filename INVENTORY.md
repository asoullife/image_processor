# Inventory

## Suspected Trash / Duplicates
- `infra/` – contains compose and migrations slated for removal after moving compose to root
- `backend/start_server.py` – duplicates `start_api_server` logic in `backend/main.py`
- `backend/RESUME_RECOVERY_IMPLEMENTATION.md` – outdated design note
- `docs/COMPREHENSIVE_TEST_SUITE_IMPLEMENTATION.md` and `docs/FINAL_INTEGRATION_SUMMARY.md` – legacy documentation
- `backend/tests/run_final_integration_tests.py` and `backend/tests/validate_all_requirements.py` – ad-hoc test scripts overlapping with `pytest`

## Requirements Files
- `backend/requirements.txt` – unified backend dependencies **(keep)**
- `frontend/package.json` – frontend dependencies **(keep)**
