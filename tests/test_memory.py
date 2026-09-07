import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.memory import LongTermMemory
from agent.multi_agent import build_multi_agent_graph
from agent.router import RoutingPolicy


class FakeAgent:
    def __init__(self, name):
        self.name = name
        self.calls = []

    async def ainvoke(self, payload):
        messages = list(payload["messages"])
        self.calls.append(messages)
        last_user = next(
            message for message in reversed(messages) if isinstance(message, HumanMessage)
        )
        return {
            "messages": [
                *messages,
                AIMessage(content=f"{self.name}:{last_user.content}"),
            ]
        }


class FakeEmbeddings:
    def __init__(self):
        self.call_count = 0

    def embed_query(self, text: str) -> list[float]:
        self.call_count += 1
        normalized = text.lower()
        return [
            1.0 if "咖啡" in normalized or "coffee" in normalized else 0.0,
            1.0 if "上海" in normalized or "shanghai" in normalized else 0.0,
        ]


class LongTermMemoryTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary_directory = TemporaryDirectory(dir=Path.cwd())
        self.embedder = FakeEmbeddings()
        self.memory = LongTermMemory(
            str(Path(self.temporary_directory.name) / "memory.sqlite3"),
            embedder=self.embedder,
            min_score=0.1,
        )

    async def asyncTearDown(self):
        self.temporary_directory.cleanup()

    async def test_explicit_memory_can_be_recalled_and_forgotten(self):
        saved = await self.memory.handle_command("user-a", "请记住我喜欢手冲咖啡")
        recalled = await self.memory.search("user-a", "推荐一种咖啡")
        forgotten = await self.memory.handle_command("user-a", "忘记我喜欢手冲咖啡")

        self.assertIn("已保存", saved)
        self.assertEqual([item.content for item in recalled], ["我喜欢手冲咖啡"])
        self.assertIn("1 条", forgotten)
        self.assertEqual(await self.memory.list_memories("user-a"), [])

    async def test_memories_are_isolated_by_user(self):
        await self.memory.remember("user-a", "我常驻上海")

        self.assertEqual(len(await self.memory.search("user-a", "上海")), 1)
        self.assertEqual(await self.memory.search("user-b", "上海"), [])

    async def test_empty_memory_does_not_call_embedding_service(self):
        self.assertEqual(await self.memory.search("new-user", "任意问题"), [])
        self.assertEqual(self.embedder.call_count, 0)

    async def test_rejects_unbounded_content_and_invalid_user_id(self):
        limited = LongTermMemory(
            str(Path(self.temporary_directory.name) / "limited.sqlite3"),
            max_content_length=5,
        )
        with self.assertRaises(ValueError):
            await limited.remember("valid-user", "123456")
        with self.assertRaises(ValueError):
            await limited.remember("../invalid", "valid")

    async def test_graph_recalls_memory_across_threads_without_persisting_prompt(self):
        workers = {
            "knowledge": FakeAgent("knowledge"),
            "report": FakeAgent("report"),
            "utility": FakeAgent("utility"),
        }
        policy = RoutingPolicy(report_keywords=("报告",), utility_keywords=("天气",))
        graph = build_multi_agent_graph(
            workers,
            InMemorySaver(),
            policy,
            memory_store=self.memory,
        )
        await graph.ainvoke(
            {
                "messages": [{"role": "user", "content": "请记住我喜欢手冲咖啡"}],
                "user_id": "stable-user",
            },
            config={"configurable": {"thread_id": "thread-one"}},
        )
        result = await graph.ainvoke(
            {
                "messages": [{"role": "user", "content": "推荐一种咖啡"}],
                "user_id": "stable-user",
            },
            config={"configurable": {"thread_id": "thread-two"}},
        )

        call_messages = workers["knowledge"].calls[0]
        self.assertIsInstance(call_messages[0], SystemMessage)
        self.assertIn("我喜欢手冲咖啡", call_messages[0].content)
        self.assertIsInstance(call_messages[-1], HumanMessage)
        self.assertFalse(any(isinstance(message, SystemMessage) for message in result["messages"]))


if __name__ == "__main__":
    unittest.main()
