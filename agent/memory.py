from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import time_ns
from typing import Any

import aiosqlite

from utils.config import agent_config
from utils.logger_handler import logger
from utils.path_tool import get_project_root

_REMEMBER_PREFIXES = ("请记住", "记住", "帮我记住", "remember:", "remember ")
_FORGET_ALL = ("清除所有记忆", "忘记所有", "清空长期记忆", "clear all memories")
_FORGET_PREFIXES = ("请忘记", "忘记", "删除记忆", "forget:", "forget ")
_LIST_COMMANDS = ("查看记忆", "列出记忆", "你记得什么", "list memories")


@dataclass(frozen=True)
class MemoryRecord:
    memory_id: str
    content: str
    score: float
    created_at: int


class LongTermMemory:
    """仅保存用户明确要求记住的内容，并支持跨 thread 语义检索与删除。"""

    def __init__(
        self,
        path: str,
        embedder: Any | None = None,
        top_k: int = 3,
        min_score: float = 0.2,
        retention_days: int = 180,
        max_items_per_user: int = 200,
        max_content_length: int = 2000,
        embedding_timeout_seconds: float = 8.0,
    ) -> None:
        self.path = self._resolve_path(path)
        self.embedder = embedder
        self.top_k = max(top_k, 1)
        self.min_score = min_score
        self.retention_days = retention_days
        self.max_items_per_user = max(max_items_per_user, 1)
        self.max_content_length = max(max_content_length, 1)
        self.embedding_timeout_seconds = max(embedding_timeout_seconds, 0.1)
        self._setup_lock = asyncio.Lock()
        self._ready = False

    @classmethod
    def from_config(cls) -> "LongTermMemory":
        config = agent_config.get("memory", {})
        embedder = None
        if config.get("semantic_search", True):
            from model.factory import embedding_model

            embedder = embedding_model
        return cls(
            path=config.get("path", ".agent_data/long_term_memory.sqlite3"),
            embedder=embedder,
            top_k=int(config.get("top_k", 3)),
            min_score=float(config.get("min_score", 0.2)),
            retention_days=int(config.get("retention_days", 180)),
            max_items_per_user=int(config.get("max_items_per_user", 200)),
            max_content_length=int(config.get("max_content_length", 2000)),
            embedding_timeout_seconds=float(config.get("embedding_timeout_seconds", 8)),
        )

    async def remember(self, user_id: str, content: str) -> MemoryRecord:
        self._validate_user_id(user_id)
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("记忆内容不能为空")
        if len(clean_content) > self.max_content_length:
            raise ValueError(f"单条记忆不能超过 {self.max_content_length} 个字符")
        await self._setup()
        now = time_ns() // 1_000_000
        fingerprint = sha256(self._normalize(clean_content).encode("utf-8")).hexdigest()
        vector = await self._embed(clean_content)
        async with aiosqlite.connect(self.path, timeout=10.0) as connection:
            await connection.execute(
                """
                INSERT INTO long_term_memories(
                    memory_id, user_id, content, fingerprint, embedding,
                    created_at, updated_at, last_accessed_at, access_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(user_id, fingerprint) DO UPDATE SET
                    content = excluded.content,
                    embedding = excluded.embedding,
                    updated_at = excluded.updated_at
                """,
                (
                    fingerprint,
                    user_id,
                    clean_content,
                    fingerprint,
                    json.dumps(vector) if vector is not None else None,
                    now,
                    now,
                    now,
                ),
            )
            await connection.commit()
            await self._prune(connection, user_id)
        return MemoryRecord(fingerprint, clean_content, 1.0, now)

    async def search(self, user_id: str, query: str) -> list[MemoryRecord]:
        if not user_id or not query.strip():
            return []
        self._validate_user_id(user_id)
        await self._setup()
        async with aiosqlite.connect(self.path, timeout=10.0) as connection:
            connection.row_factory = aiosqlite.Row
            await self._prune(connection, user_id)
            cursor = await connection.execute(
                """
                SELECT memory_id, content, embedding, created_at
                FROM long_term_memories
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, self.max_items_per_user),
            )
            rows = await cursor.fetchall()
            if not rows:
                return []

            query_vector = await self._embed(query)

            scored = []
            for row in rows:
                try:
                    raw_vector = json.loads(row["embedding"]) if row["embedding"] else None
                    stored_vector = (
                        [float(value) for value in raw_vector]
                        if isinstance(raw_vector, list)
                        else None
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    stored_vector = None
                semantic_score = self._cosine(query_vector, stored_vector)
                lexical_score = self._lexical_score(query, row["content"])
                score = max(semantic_score, lexical_score)
                if score >= self.min_score:
                    scored.append(
                        MemoryRecord(
                            row["memory_id"],
                            row["content"],
                            round(score, 4),
                            row["created_at"],
                        )
                    )

            ranked = sorted(scored, key=lambda item: item.score, reverse=True)[: self.top_k]
            if ranked:
                now = time_ns() // 1_000_000
                await connection.executemany(
                    """
                    UPDATE long_term_memories
                    SET last_accessed_at = ?, access_count = access_count + 1
                    WHERE memory_id = ? AND user_id = ?
                    """,
                    [(now, item.memory_id, user_id) for item in ranked],
                )
                await connection.commit()
        return ranked

    async def list_memories(self, user_id: str) -> list[MemoryRecord]:
        self._validate_user_id(user_id)
        await self._setup()
        async with aiosqlite.connect(self.path, timeout=10.0) as connection:
            connection.row_factory = aiosqlite.Row
            await self._prune(connection, user_id)
            cursor = await connection.execute(
                """
                SELECT memory_id, content, created_at
                FROM long_term_memories WHERE user_id = ?
                ORDER BY updated_at DESC LIMIT ?
                """,
                (user_id, self.max_items_per_user),
            )
            rows = await cursor.fetchall()
        return [
            MemoryRecord(row["memory_id"], row["content"], 1.0, row["created_at"]) for row in rows
        ]

    async def forget(self, user_id: str, content: str) -> int:
        self._validate_user_id(user_id)
        needle = self._normalize(content)
        if not needle:
            return 0
        memories = await self.list_memories(user_id)
        targets = [
            item.memory_id
            for item in memories
            if needle in self._normalize(item.content) or self._normalize(item.content) in needle
        ]
        if not targets:
            return 0
        async with aiosqlite.connect(self.path, timeout=10.0) as connection:
            await connection.executemany(
                "DELETE FROM long_term_memories WHERE user_id = ? AND memory_id = ?",
                [(user_id, memory_id) for memory_id in targets],
            )
            await connection.commit()
        return len(targets)

    async def clear_user(self, user_id: str) -> int:
        self._validate_user_id(user_id)
        await self._setup()
        async with aiosqlite.connect(self.path, timeout=10.0) as connection:
            cursor = await connection.execute(
                "DELETE FROM long_term_memories WHERE user_id = ?",
                (user_id,),
            )
            await connection.commit()
        return max(cursor.rowcount, 0)

    async def handle_command(self, user_id: str, query: str) -> str:
        normalized = query.strip()
        lowered = normalized.lower()
        if lowered in _FORGET_ALL or normalized in _FORGET_ALL:
            count = await self.clear_user(user_id)
            return f"已清除 {count} 条长期记忆。"
        if lowered in _LIST_COMMANDS or normalized in _LIST_COMMANDS:
            memories = await self.list_memories(user_id)
            if not memories:
                return "目前没有长期记忆。只有你明确要求记住的内容才会保存。"
            return "当前长期记忆：\n" + "\n".join(
                f"{index}. {item.content}" for index, item in enumerate(memories, 1)
            )
        remembered = self._strip_prefix(normalized, _REMEMBER_PREFIXES)
        if remembered is not None:
            if not remembered:
                return "请在“记住”后面提供需要保存的内容。"
            await self.remember(user_id, remembered)
            return "已保存到长期记忆。你可以随时查看或删除。"
        forgotten = self._strip_prefix(normalized, _FORGET_PREFIXES)
        if forgotten is not None:
            if not forgotten:
                return "请在“忘记”后面提供需要删除的内容。"
            count = await self.forget(user_id, forgotten)
            return f"已删除 {count} 条匹配的长期记忆。"
        return "请明确使用“记住…、忘记…、查看记忆或清除所有记忆”。"

    async def _setup(self) -> None:
        if self._ready:
            return
        async with self._setup_lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self.path, timeout=10.0) as connection:
                await connection.execute("PRAGMA journal_mode=WAL")
                await connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS long_term_memories (
                        memory_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        fingerprint TEXT NOT NULL,
                        embedding TEXT,
                        created_at INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL,
                        last_accessed_at INTEGER NOT NULL,
                        access_count INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY(memory_id, user_id),
                        UNIQUE(user_id, fingerprint)
                    )
                    """
                )
                await connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_user_updated "
                    "ON long_term_memories(user_id, updated_at DESC)"
                )
                await connection.commit()
            self._ready = True

    async def _prune(self, connection: aiosqlite.Connection, user_id: str) -> None:
        if self.retention_days > 0:
            cutoff = time_ns() // 1_000_000 - self.retention_days * 24 * 60 * 60 * 1000
            await connection.execute(
                "DELETE FROM long_term_memories WHERE user_id = ? AND updated_at < ?",
                (user_id, cutoff),
            )
        await connection.execute(
            """
            DELETE FROM long_term_memories
            WHERE user_id = ? AND memory_id IN (
                SELECT memory_id FROM long_term_memories
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (user_id, user_id, self.max_items_per_user),
        )
        await connection.commit()

    async def _embed(self, text: str) -> list[float] | None:
        if self.embedder is None:
            return None
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self.embedder.embed_query, text),
                timeout=self.embedding_timeout_seconds,
            )
            return [float(value) for value in result]
        except Exception as exc:
            logger.warning("长期记忆语义向量生成失败（%s），降级为词面检索", type(exc).__name__)
            return None

    @staticmethod
    def _cosine(left: list[float] | None, right: list[float] | None) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right))
        if denominator == 0:
            return 0.0
        return max(sum(x * y for x, y in zip(left, right, strict=True)) / denominator, 0.0)

    @staticmethod
    def _lexical_score(query: str, content: str) -> float:
        left = set(LongTermMemory._tokens(query))
        right = set(LongTermMemory._tokens(content))
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    @staticmethod
    def _tokens(value: str) -> list[str]:
        normalized = LongTermMemory._normalize(value)
        latin = re.findall(r"[a-z0-9]+", normalized)
        chinese = [char for char in normalized if "\u4e00" <= char <= "\u9fff"]
        return latin + chinese

    @staticmethod
    def _strip_prefix(value: str, prefixes: tuple[str, ...]) -> str | None:
        lowered = value.lower()
        for prefix in prefixes:
            if lowered.startswith(prefix.lower()):
                return value[len(prefix) :].lstrip("：:，,。 ")
        return None

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())

    @staticmethod
    def _validate_user_id(user_id: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", user_id):
            raise ValueError("user_id 格式无效")

    @staticmethod
    def _resolve_path(path: str) -> Path:
        root = Path(get_project_root()).resolve()
        configured = Path(path)
        candidate = (
            configured.resolve() if configured.is_absolute() else (root / configured).resolve()
        )
        if not candidate.is_relative_to(root):
            raise ValueError("memory path 必须位于项目目录内")
        return candidate
