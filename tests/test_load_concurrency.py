from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agent.memory import LongTermMemory


@pytest.mark.load
class LongTermMemoryConcurrencyTest(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_writes_do_not_lose_or_cross_user_data(self) -> None:
        with TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            memory = LongTermMemory(
                str(Path(temporary_directory) / "concurrent-memory.sqlite3"),
                embedder=None,
                max_items_per_user=100,
            )
            writes = [
                memory.remember("load-user-a", f"A 用户偏好 {index}") for index in range(30)
            ] + [memory.remember("load-user-b", f"B 用户偏好 {index}") for index in range(30)]

            results = await asyncio.gather(*writes)
            user_a, user_b = await asyncio.gather(
                memory.list_memories("load-user-a"),
                memory.list_memories("load-user-b"),
            )

            self.assertEqual(len(results), 60)
            self.assertEqual(len(user_a), 30)
            self.assertEqual(len(user_b), 30)
            self.assertTrue(all(item.content.startswith("A 用户") for item in user_a))
            self.assertTrue(all(item.content.startswith("B 用户") for item in user_b))


if __name__ == "__main__":
    unittest.main()
