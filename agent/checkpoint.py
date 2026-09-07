from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import time_ns
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from utils.config import agent_config
from utils.path_tool import get_project_root


class CheckpointProvider:
    """按操作提供 Checkpointer，避免异步 SQLite 连接跨事件循环复用。"""

    def __init__(
        self,
        backend: str,
        path: str | None = None,
        dsn: str | None = None,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        retention_days: int | None = None,
        max_threads: int | None = None,
    ) -> None:
        self.backend = backend.lower()
        self._memory = checkpointer if checkpointer is not None else None
        self.path = self._resolve_path(path) if self.backend == "sqlite" else None
        self.dsn = dsn if self.backend == "postgres" else None
        checkpoint_config = agent_config.get("checkpoint", {})
        self.retention_days = (
            int(checkpoint_config.get("retention_days", 30))
            if retention_days is None
            else retention_days
        )
        self.max_threads = (
            int(checkpoint_config.get("max_threads", 500)) if max_threads is None else max_threads
        )

        if self.backend not in {"memory", "sqlite", "postgres"}:
            raise ValueError(f"不支持的 checkpoint backend：{backend}")
        if self.backend == "postgres" and not self.dsn:
            raise ValueError("PostgreSQL checkpoint 缺少 DSN")

    @classmethod
    def from_config(
        cls,
        checkpointer: BaseCheckpointSaver[str] | None = None,
    ) -> "CheckpointProvider":
        if checkpointer is not None:
            return cls("memory", checkpointer=checkpointer)

        checkpoint_config = agent_config.get("checkpoint", {})
        backend = os.getenv(
            "AGENT_CHECKPOINT_BACKEND",
            checkpoint_config.get("backend", "sqlite"),
        )
        dsn_environment = checkpoint_config.get("dsn_env", "AGENT_CHECKPOINT_DSN")
        return cls(
            backend=backend,
            path=checkpoint_config.get("path", ".agent_data/checkpoints.sqlite3"),
            dsn=os.getenv(dsn_environment),
            retention_days=int(checkpoint_config.get("retention_days", 30)),
            max_threads=int(checkpoint_config.get("max_threads", 500)),
        )

    @property
    def reusable(self) -> bool:
        return self.backend == "memory"

    @asynccontextmanager
    async def open(self) -> AsyncIterator[BaseCheckpointSaver[str]]:
        if self.backend == "memory":
            if self._memory is None:
                self._memory = InMemorySaver()
            yield self._memory
            return

        if self.backend == "postgres":
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            assert self.dsn is not None
            async with AsyncPostgresSaver.from_conn_string(self.dsn) as saver:
                await saver.setup()
                yield saver
            return

        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(self.path)) as saver:
            await saver.setup()
            await self._setup_retention(saver)
            await self._prune(saver)
            yield saver

    async def touch_thread(
        self,
        saver: BaseCheckpointSaver[str],
        thread_id: str,
    ) -> None:
        if self.backend != "sqlite":
            return
        connection = self._sqlite_connection(saver)
        await connection.execute(
            """
            INSERT INTO agent_thread_activity(thread_id, updated_at)
            VALUES (?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET updated_at = excluded.updated_at
            """,
            (thread_id, time_ns() // 1_000_000),
        )
        await connection.commit()
        await self._prune(saver)

    async def forget_thread(
        self,
        saver: BaseCheckpointSaver[str],
        thread_id: str,
    ) -> None:
        if self.backend != "sqlite":
            return
        connection = self._sqlite_connection(saver)
        await connection.execute(
            "DELETE FROM agent_thread_activity WHERE thread_id = ?",
            (thread_id,),
        )
        await connection.commit()

    async def _setup_retention(self, saver: BaseCheckpointSaver[str]) -> None:
        connection = self._sqlite_connection(saver)
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_thread_activity (
                thread_id TEXT PRIMARY KEY,
                updated_at INTEGER NOT NULL
            )
            """
        )
        await connection.execute(
            """
            INSERT OR IGNORE INTO agent_thread_activity(thread_id, updated_at)
            SELECT DISTINCT thread_id, ? FROM checkpoints
            """,
            (time_ns() // 1_000_000,),
        )
        await connection.commit()

    async def _prune(self, saver: BaseCheckpointSaver[str]) -> None:
        connection = self._sqlite_connection(saver)
        expired_threads: set[str] = set()

        if self.retention_days > 0:
            cutoff = time_ns() // 1_000_000 - self.retention_days * 24 * 60 * 60 * 1000
            async with connection.execute(
                "SELECT thread_id FROM agent_thread_activity WHERE updated_at < ?",
                (cutoff,),
            ) as cursor:
                async for row in cursor:
                    expired_threads.add(row[0])

        if self.max_threads > 0:
            async with connection.execute(
                """
                SELECT thread_id FROM agent_thread_activity
                ORDER BY updated_at DESC, thread_id DESC
                LIMIT -1 OFFSET ?
                """,
                (self.max_threads,),
            ) as cursor:
                async for row in cursor:
                    expired_threads.add(row[0])

        if not expired_threads:
            return

        for thread_id in expired_threads:
            await saver.adelete_thread(thread_id)

        parameters = [(thread_id,) for thread_id in expired_threads]
        await connection.executemany(
            "DELETE FROM agent_thread_activity WHERE thread_id = ?",
            parameters,
        )
        await connection.commit()

    @staticmethod
    def _sqlite_connection(saver: BaseCheckpointSaver[str]) -> Any:
        connection = getattr(saver, "conn", None)
        if connection is None:
            raise RuntimeError("SQLite Checkpointer 缺少数据库连接")
        return connection

    @staticmethod
    def _resolve_path(path: str | None) -> Path:
        project_root = Path(get_project_root()).resolve()
        configured_path = Path(path or ".agent_data/checkpoints.sqlite3")
        candidate = (
            configured_path.resolve()
            if configured_path.is_absolute()
            else (project_root / configured_path).resolve()
        )
        if not candidate.is_relative_to(project_root):
            raise ValueError("checkpoint path 必须位于项目目录内")
        return candidate
