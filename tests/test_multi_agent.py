import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent.checkpoint import CheckpointProvider
from agent.multi_agent import MultiAgentSystem, build_multi_agent_graph
from agent.router import RoutingPolicy


class FakeAgent:
    def __init__(self, name):
        self.name = name
        self.calls = []

    async def ainvoke(self, payload):
        messages = list(payload["messages"])
        self.calls.append(messages)
        last_user_message = next(
            message for message in reversed(messages) if isinstance(message, HumanMessage)
        )
        return {
            "messages": [
                *messages,
                AIMessage(content=f"{self.name}:{last_user_message.content}"),
            ]
        }


class ToolUsingFakeAgent(FakeAgent):
    async def ainvoke(self, payload):
        messages = list(payload["messages"])
        self.calls.append(messages)
        return {
            "messages": [
                *messages,
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "private_tool",
                            "args": {"secret": "do-not-stream"},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(content="private tool output", tool_call_id="call-1"),
                AIMessage(content="safe final answer"),
            ]
        }


class FailsOnceAgent(FakeAgent):
    async def ainvoke(self, payload):
        if not self.calls:
            self.calls.append(list(payload["messages"]))
            raise RuntimeError("worker failure")
        return await super().ainvoke(payload)


def create_test_graph(checkpointer):
    workers = {
        "knowledge": FakeAgent("knowledge"),
        "report": FakeAgent("report"),
        "utility": FakeAgent("utility"),
    }
    policy = RoutingPolicy(
        report_keywords=("报告",),
        utility_keywords=("天气",),
    )
    graph = build_multi_agent_graph(workers, checkpointer, policy)
    return graph, workers


class MultiAgentMemoryTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.checkpointer = InMemorySaver()
        self.graph, self.workers = create_test_graph(self.checkpointer)

    async def test_routes_to_specialized_agents(self):
        report_result = await self.graph.ainvoke(
            {"messages": [{"role": "user", "content": "生成财务报告"}]},
            config={"configurable": thread_config("report-thread")},
        )
        utility_result = await self.graph.ainvoke(
            {"messages": [{"role": "user", "content": "查询天气"}]},
            config={"configurable": thread_config("utility-thread")},
        )

        self.assertEqual(report_result["last_agent"], "report")
        self.assertEqual(utility_result["last_agent"], "utility")
        self.assertEqual(len(self.workers["report"].calls), 1)
        self.assertEqual(len(self.workers["utility"].calls), 1)

    async def test_same_thread_remembers_history(self):
        config = {"configurable": thread_config("thread-a")}

        await self.graph.ainvoke(
            {"messages": [{"role": "user", "content": "生成财务报告"}]},
            config=config,
        )
        result = await self.graph.ainvoke(
            {"messages": [{"role": "user", "content": "继续"}]},
            config=config,
        )

        report_calls = self.workers["report"].calls
        self.assertEqual(len(report_calls), 2)
        self.assertEqual(len(report_calls[1]), 3)
        self.assertEqual(len(result["messages"]), 4)
        self.assertEqual(result["last_agent"], "report")

    async def test_different_threads_are_isolated(self):
        await self.graph.ainvoke(
            {"messages": [{"role": "user", "content": "公司制度是什么"}]},
            config={"configurable": thread_config("thread-a")},
        )
        await self.graph.ainvoke(
            {"messages": [{"role": "user", "content": "公司理念是什么"}]},
            config={"configurable": thread_config("thread-b")},
        )

        knowledge_calls = self.workers["knowledge"].calls
        self.assertEqual(len(knowledge_calls), 2)
        self.assertEqual(len(knowledge_calls[0]), 1)
        self.assertEqual(len(knowledge_calls[1]), 1)

    async def test_clear_thread_removes_checkpoint(self):
        config = {"configurable": thread_config("thread-to-clear")}
        await self.graph.ainvoke(
            {"messages": [{"role": "user", "content": "公司制度是什么"}]},
            config=config,
        )
        self.assertIsNotNone(await self.checkpointer.aget_tuple(config))

        system = MultiAgentSystem(checkpointer=self.checkpointer)
        await system.clear_thread("thread-to-clear")

        self.assertIsNone(await self.checkpointer.aget_tuple(config))

    async def test_execute_stream_returns_only_worker_answer(self):
        system = MultiAgentSystem(checkpointer=self.checkpointer)
        system.graph = self.graph
        system._initialized = True

        chunks = [
            chunk
            async for chunk in system.execute_stream(
                "公司制度是什么",
                thread_id="stream-thread",
            )
        ]

        self.assertEqual(chunks, ["knowledge:公司制度是什么"])

    async def test_execute_stream_requires_thread_id(self):
        system = MultiAgentSystem(checkpointer=self.checkpointer)
        system.graph = self.graph
        system._initialized = True

        with self.assertRaises(ValueError):
            async for _ in system.execute_stream("问题", thread_id=""):
                pass

    async def test_execute_stream_requires_query(self):
        system = MultiAgentSystem(checkpointer=self.checkpointer)
        system.graph = self.graph
        system._initialized = True

        with self.assertRaises(ValueError):
            async for _ in system.execute_stream("  ", thread_id="thread"):
                pass

    async def test_execute_stream_does_not_expose_tool_messages(self):
        workers = {
            "knowledge": ToolUsingFakeAgent("knowledge"),
            "report": FakeAgent("report"),
            "utility": FakeAgent("utility"),
        }
        graph = build_multi_agent_graph(
            workers,
            self.checkpointer,
            RoutingPolicy(report_keywords=(), utility_keywords=()),
        )
        system = MultiAgentSystem(checkpointer=self.checkpointer)
        system.graph = graph
        system._initialized = True

        chunks = [
            chunk
            async for chunk in system.execute_stream(
                "测试安全输出",
                thread_id="safe-stream",
            )
        ]

        self.assertEqual(chunks, ["safe final answer"])

    async def test_failure_response_is_persisted_and_completes_turn(self):
        workers = {
            "knowledge": FailsOnceAgent("knowledge"),
            "report": FakeAgent("report"),
            "utility": FakeAgent("utility"),
        }
        graph = build_multi_agent_graph(
            workers,
            self.checkpointer,
            RoutingPolicy(report_keywords=(), utility_keywords=()),
        )
        system = MultiAgentSystem(checkpointer=self.checkpointer)
        system.graph = graph
        system._initialized = True

        chunks = [
            chunk
            async for chunk in system.execute_stream(
                "first request",
                thread_id="failed-turn",
            )
        ]
        snapshot = await graph.aget_state({"configurable": thread_config("failed-turn")})

        self.assertEqual(chunks, ["系统错误，请稍后重试。"])
        self.assertEqual(snapshot.next, ())
        self.assertEqual(len(snapshot.values["messages"]), 2)
        self.assertEqual(snapshot.values["messages"][-1].content, chunks[0])


class PersistentCheckpointTest(unittest.IsolatedAsyncioTestCase):
    async def test_sqlite_recovers_state_after_provider_reopens(self):
        with TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            checkpoint_path = str(Path(temporary_directory) / "memory.sqlite3")
            provider = CheckpointProvider("sqlite", checkpoint_path)

            first_workers = {
                "knowledge": FakeAgent("knowledge"),
                "report": FakeAgent("report"),
                "utility": FakeAgent("utility"),
            }
            policy = RoutingPolicy(
                report_keywords=("报告",),
                utility_keywords=(),
            )
            async with provider.open() as saver:
                first_graph = build_multi_agent_graph(first_workers, saver, policy)
                await first_graph.ainvoke(
                    {"messages": [{"role": "user", "content": "生成报告"}]},
                    config={"configurable": thread_config("persistent-thread")},
                )

            second_workers = {
                "knowledge": FakeAgent("knowledge"),
                "report": FakeAgent("report"),
                "utility": FakeAgent("utility"),
            }
            async with CheckpointProvider("sqlite", checkpoint_path).open() as saver:
                second_graph = build_multi_agent_graph(second_workers, saver, policy)
                result = await second_graph.ainvoke(
                    {"messages": [{"role": "user", "content": "继续"}]},
                    config={"configurable": thread_config("persistent-thread")},
                )

            self.assertEqual(result["last_agent"], "report")
            self.assertEqual(len(result["messages"]), 4)
            self.assertEqual(len(second_workers["report"].calls[0]), 3)

            system = MultiAgentSystem(checkpointer=InMemorySaver())
            system.checkpoint_provider = CheckpointProvider("sqlite", checkpoint_path)
            system._workers = second_workers
            system._initialized = True
            saved_messages = await system.get_thread_messages("persistent-thread")
            self.assertEqual(len(saved_messages), 4)
            self.assertEqual(saved_messages[-1].content, "report:继续")

    async def test_sqlite_prunes_expired_thread(self):
        with TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            checkpoint_path = str(Path(temporary_directory) / "retention.sqlite3")
            provider = CheckpointProvider(
                "sqlite",
                checkpoint_path,
                retention_days=1,
                max_threads=10,
            )
            graph_config = {"configurable": thread_config("expired-thread")}
            workers = {
                "knowledge": FakeAgent("knowledge"),
                "report": FakeAgent("report"),
                "utility": FakeAgent("utility"),
            }
            policy = RoutingPolicy(report_keywords=(), utility_keywords=())

            async with provider.open() as saver:
                graph = build_multi_agent_graph(workers, saver, policy)
                await graph.ainvoke(
                    {"messages": [{"role": "user", "content": "old request"}]},
                    config=graph_config,
                )
                await provider.touch_thread(saver, "expired-thread")
                await saver.conn.execute(
                    "UPDATE agent_thread_activity SET updated_at = 0 WHERE thread_id = ?",
                    ("expired-thread",),
                )
                await saver.conn.commit()

            async with provider.open() as saver:
                self.assertIsNone(await saver.aget_tuple(graph_config))

    async def test_sqlite_limits_tracked_threads(self):
        with TemporaryDirectory(dir=Path.cwd()) as temporary_directory:
            provider = CheckpointProvider(
                "sqlite",
                str(Path(temporary_directory) / "capacity.sqlite3"),
                retention_days=0,
                max_threads=2,
            )

            async with provider.open() as saver:
                await provider.touch_thread(saver, "thread-a")
                await provider.touch_thread(saver, "thread-b")
                await provider.touch_thread(saver, "thread-c")
                async with saver.conn.execute(
                    "SELECT thread_id FROM agent_thread_activity"
                ) as cursor:
                    retained_threads = {row[0] async for row in cursor}

            self.assertEqual(retained_threads, {"thread-b", "thread-c"})


def thread_config(thread_id: str) -> dict[str, str]:
    return {"thread_id": thread_id}


if __name__ == "__main__":
    unittest.main()
