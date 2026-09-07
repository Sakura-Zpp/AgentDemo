import os
import unittest
from unittest.mock import patch

from langchain_core.tools import tool

from agent.mcp.mcp_tools import (
    McpConfigurationError,
    build_mcp_connection,
    load_mcp_tools,
)


@tool
def external_tool() -> str:
    """测试外部工具。"""

    return "ok"


class McpConfigurationTest(unittest.TestCase):
    def test_expands_header_from_environment(self):
        config = {
            "type": "sse",
            "baseUrl": "https://example.com/sse",
            "headers": {"Authorization": "Bearer ${TEST_MCP_KEY}"},
        }

        with patch.dict(os.environ, {"TEST_MCP_KEY": "secret-value"}):
            connection = build_mcp_connection("test", config)

        self.assertEqual(connection["transport"], "sse")
        self.assertEqual(connection["headers"]["Authorization"], "Bearer secret-value")

    def test_rejects_missing_environment_variable(self):
        config = {
            "type": "sse",
            "baseUrl": "https://example.com/sse",
            "headers": {"Authorization": "Bearer ${MISSING_MCP_KEY}"},
        }

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(McpConfigurationError):
                build_mcp_connection("test", config)


class McpIsolationTest(unittest.IsolatedAsyncioTestCase):
    async def test_one_server_failure_does_not_hide_healthy_tools(self):
        config = {
            "mcpServers": {
                "healthy": {"type": "sse", "baseUrl": "https://example.com/ok"},
                "broken": {"type": "sse", "baseUrl": "https://example.com/fail"},
            }
        }

        async def fake_load(server_name, server_config, timeout):
            del server_config, timeout
            if server_name == "broken":
                raise TimeoutError
            return [external_tool]

        with patch("agent.mcp.mcp_tools._load_server_tools", side_effect=fake_load):
            loaded = await load_mcp_tools(config=config, timeout=1)

        self.assertEqual([loaded_tool.name for loaded_tool in loaded], ["external_tool"])

    async def test_partial_failure_is_not_cached(self):
        config = {
            "mcpServers": {
                "healthy": {"type": "sse", "baseUrl": "https://example.com/ok"},
                "recovering": {"type": "sse", "baseUrl": "https://example.com/retry"},
            }
        }
        attempts = {"recovering": 0}

        async def fake_load(server_name, server_config, timeout):
            del server_config, timeout
            if server_name == "recovering":
                attempts["recovering"] += 1
                if attempts["recovering"] == 1:
                    raise TimeoutError
            return [external_tool]

        from agent.mcp import mcp_tools

        with (
            patch.object(mcp_tools, "mcp_config", config),
            patch("agent.mcp.mcp_tools._load_server_tools", side_effect=fake_load),
        ):
            mcp_tools.clear_mcp_tool_cache()
            await load_mcp_tools()
            await load_mcp_tools()
            mcp_tools.clear_mcp_tool_cache()

        self.assertEqual(attempts["recovering"], 2)
