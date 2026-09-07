from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.mcp.chart_tools import create_chart
from agent.tools.tools import fill_context_report, get_weather, rag_search
from utils.config import agent_config, mcp_config
from utils.logger_handler import logger

load_dotenv(
    Path(__file__).parent.parent.parent / ".env",
    override=False,
    encoding="utf-8-sig",
)

_ENV_REFERENCE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
_TRANSPORT_NAMES = {
    "sse": "sse",
    "streamableHttp": "streamable_http",
    "streamable_http": "streamable_http",
}
_cached_tools: tuple[BaseTool, ...] = ()
_cache_initialized = False


class McpConfigurationError(ValueError):
    pass


def _expand_environment(value: str) -> str:
    missing_variables: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        variable_name = match.group(1)
        environment_value = os.getenv(variable_name)
        if environment_value is None:
            missing_variables.add(variable_name)
            return ""
        return environment_value

    expanded = _ENV_REFERENCE.sub(replace, value)
    if missing_variables:
        names = ", ".join(sorted(missing_variables))
        raise McpConfigurationError(f"缺少环境变量：{names}")
    return expanded


def build_mcp_connection(server_name: str, server_config: Mapping[str, Any]) -> dict[str, Any]:
    transport_name = server_config.get("transport", server_config.get("type"))
    transport = _TRANSPORT_NAMES.get(str(transport_name))
    if transport is None:
        raise McpConfigurationError(f"{server_name} 使用了不支持的 transport")

    url = server_config.get("url", server_config.get("baseUrl"))
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise McpConfigurationError(f"{server_name} 缺少合法的 HTTP URL")

    raw_headers = server_config.get("headers", {})
    if not isinstance(raw_headers, Mapping):
        raise McpConfigurationError(f"{server_name} 的 headers 必须是对象")

    headers = {str(key): _expand_environment(str(value)) for key, value in raw_headers.items()}
    return {
        "transport": transport,
        "url": url,
        "headers": headers,
    }


async def _load_server_tools(
    server_name: str,
    server_config: Mapping[str, Any],
    timeout: int,
) -> list[BaseTool]:
    connection = build_mcp_connection(server_name, server_config)
    client = MultiServerMCPClient({server_name: connection})
    tools = await asyncio.wait_for(
        client.get_tools(server_name=server_name),
        timeout=timeout,
    )

    allowlist = server_config.get("allowedTools")
    if allowlist is not None:
        if not isinstance(allowlist, list) or not all(isinstance(name, str) for name in allowlist):
            raise McpConfigurationError(f"{server_name} 的 allowedTools 必须是字符串数组")
        allowed_names = set(allowlist)
        tools = [tool for tool in tools if tool.name in allowed_names]
    return tools


async def load_mcp_tools(
    config: Mapping[str, Any] | None = None,
    timeout: int | None = None,
    force_reload: bool = False,
) -> list[BaseTool]:
    """并行加载 MCP 服务；单个服务失败不会阻止其他服务。"""

    global _cache_initialized, _cached_tools

    use_cache = config is None
    if use_cache and _cache_initialized and not force_reload:
        return list(_cached_tools)

    source_config = mcp_config if config is None else config
    raw_servers = source_config.get("mcpServers", {})
    if not isinstance(raw_servers, Mapping):
        raise McpConfigurationError("mcpServers 必须是对象")

    server_timeout = timeout or int(agent_config.get("mcp_server_timeout", 12))
    enabled_servers = [
        (str(name), server)
        for name, server in raw_servers.items()
        if isinstance(server, Mapping) and server.get("isActive", True)
    ]

    tasks = [
        asyncio.create_task(_load_server_tools(name, server, server_timeout))
        for name, server in enabled_servers
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    loaded_tools: list[BaseTool] = []
    known_tool_names: set[str] = set()
    all_servers_loaded = True
    for (server_name, _), result in zip(enabled_servers, results, strict=True):
        if isinstance(result, BaseException):
            all_servers_loaded = False
            logger.warning(
                "MCP 服务 %s 加载失败（%s）",
                server_name,
                type(result).__name__,
            )
            continue
        for loaded_tool in result:
            if loaded_tool.name in known_tool_names:
                logger.warning("忽略重名 MCP 工具：%s", loaded_tool.name)
                continue
            known_tool_names.add(loaded_tool.name)
            loaded_tools.append(loaded_tool)
        logger.info("MCP 服务 %s 加载成功，共 %s 个工具", server_name, len(result))

    if use_cache and all_servers_loaded:
        _cached_tools = tuple(loaded_tools)
        _cache_initialized = True
    return loaded_tools


def clear_mcp_tool_cache() -> None:
    global _cache_initialized, _cached_tools
    _cached_tools = ()
    _cache_initialized = False


async def get_all_tools() -> list[BaseTool]:
    external_tools = await load_mcp_tools()
    return [rag_search, fill_context_report, get_weather, create_chart, *external_tools]


if __name__ == "__main__":
    loaded = asyncio.run(get_all_tools())
    print(f"成功加载 {len(loaded)} 个工具")
    for tool in loaded:
        print(f"  - {tool.name}")
