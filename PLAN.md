# Refactor Plan

- [ ] Move `docker-compose.yml` to repo root and drop the `infra/` folder
- [ ] Expose a single backend entrypoint `backend.main:app` and remove `start_server.py`
- [ ] Consolidate Python requirements and decide on `requirements-ai.txt`
- [ ] Prune outdated docs and ad-hoc scripts/tests listed in `INVENTORY.md`
- [ ] Group remaining tests under `backend/tests` with consistent naming
- [ ] Add `.env.example` files to `backend` and `frontend`
- [ ] Update README and developer guides for the new structure
