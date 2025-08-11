"""PostgreSQL-based progress tracker (simplified)."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from .base import ProgressTracker, ProcessingResult
from ..database.connection import DatabaseManager
from ..database.models import Project, ProcessingSession, ImageResult, Checkpoint
from sqlalchemy import select


class PostgresProgressTracker(ProgressTracker):
    """Minimal progress tracker that stores state in PostgreSQL."""

    def __init__(self, database_url: Optional[str] = None, checkpoint_interval: int = 50):
        self.db_manager = DatabaseManager(database_url)
        self.checkpoint_interval = checkpoint_interval
        self.logger = logging.getLogger(self.__class__.__name__)
        asyncio.run(self.db_manager.initialize())

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def create_session(
        self,
        input_folder: str,
        output_folder: str,
        total_images: int,
        config: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Create a new processing session."""
        session_id = session_id or str(uuid.uuid4())

        async def _create() -> None:
            async with self.db_manager.get_session() as session:
                project = Project(
                    name="default",
                    input_folder=input_folder,
                    output_folder=output_folder,
                    settings=config or {},
                )
                session.add(project)
                await session.flush()  # get project.id

                proc_session = ProcessingSession(
                    id=uuid.UUID(session_id),
                    project_id=project.id,
                    total_images=total_images,
                    processed_images=0,
                    approved_images=0,
                    rejected_images=0,
                    status="created",
                    session_config=config or {},
                )
                session.add(proc_session)

        asyncio.run(_create())
        return session_id

    # ------------------------------------------------------------------
    # Checkpoint handling
    # ------------------------------------------------------------------
    def save_checkpoint(
        self,
        session_id: str,
        processed_count: int,
        total_count: int,
        results: List[ProcessingResult],
    ) -> bool:
        """Save processing progress and checkpoint if needed."""

        async def _save() -> None:
            async with self.db_manager.get_session() as session:
                # ensure session exists
                stmt = select(ProcessingSession).where(ProcessingSession.id == uuid.UUID(session_id))
                res = await session.execute(stmt)
                proc_session = res.scalar_one()

                # store individual results
                approved = 0
                rejected = 0
                for r in results:
                    img = ImageResult(
                        session_id=proc_session.id,
                        image_path=r.image_path,
                        filename=r.filename,
                        final_decision=r.final_decision,
                        rejection_reasons=r.rejection_reasons,
                        processing_time=r.processing_time,
                    )
                    session.add(img)
                    if r.final_decision == "approved":
                        approved += 1
                    elif r.final_decision == "rejected":
                        rejected += 1

                proc_session.processed_images = processed_count
                proc_session.approved_images += approved
                proc_session.rejected_images += rejected

                if processed_count and processed_count % self.checkpoint_interval == 0:
                    checkpoint = Checkpoint(
                        session_id=proc_session.id,
                        checkpoint_type="batch",
                        processed_count=processed_count,
                        session_state={"progress": processed_count / max(total_count, 1)},
                    )
                    session.add(checkpoint)

        try:
            asyncio.run(_save())
            return True
        except Exception as exc:  # pragma: no cover - best effort logging
            self.logger.error(f"Failed to save checkpoint: {exc}")
            return False

    def load_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint for a session."""

        async def _load() -> Optional[Checkpoint]:
            async with self.db_manager.get_session() as session:
                stmt = (
                    select(Checkpoint)
                    .where(Checkpoint.session_id == uuid.UUID(session_id))
                    .order_by(Checkpoint.processed_count.desc())
                )
                res = await session.execute(stmt)
                return res.scalars().first()

        try:
            checkpoint = asyncio.run(_load())
        except Exception as exc:  # pragma: no cover - best effort logging
            self.logger.error(f"Failed to load checkpoint: {exc}")
            return None

        if checkpoint is None:
            return None

        return {
            "processed_count": checkpoint.processed_count,
            "checkpoint_type": checkpoint.checkpoint_type,
            "session_state": checkpoint.session_state,
        }

    # ------------------------------------------------------------------
    def get_session_status(self, session_id: str) -> Optional[str]:
        """Get processing status for a session."""

        async def _status() -> Optional[str]:
            async with self.db_manager.get_session() as session:
                stmt = select(ProcessingSession.status).where(ProcessingSession.id == uuid.UUID(session_id))
                res = await session.execute(stmt)
                return res.scalar_one_or_none()

        try:
            return asyncio.run(_status())
        except Exception as exc:  # pragma: no cover - best effort logging
            self.logger.error(f"Failed to get session status: {exc}")
            return None

    # Placeholder methods for legacy API ---------------------------------
    def complete_session(self, *args, **kwargs):  # pragma: no cover - legacy
        self.logger.debug("complete_session not implemented")

    def save_image_result(self, *args, **kwargs):  # pragma: no cover - legacy
        self.logger.debug("save_image_result not implemented")

    def update_session_progress(self, *args, **kwargs):  # pragma: no cover - legacy
        self.logger.debug("update_session_progress not implemented")

    def get_session_results(self, *args, **kwargs):  # pragma: no cover - legacy
        return []

    def get_resumable_sessions(self, *args, **kwargs):  # pragma: no cover - legacy
        return []

    def get_session_results_summary(self, *args, **kwargs):  # pragma: no cover - legacy
        return {}

    def get_recent_results(self, *args, **kwargs):  # pragma: no cover - legacy
        return []

    def list_sessions(self, *args, **kwargs):  # pragma: no cover - legacy
        return []
