from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from langchain.agents import create_agent
from langchain_core.tools import BaseTool

from agent.mcp.chart_config import CHART_CONFIGS
from agent.mcp.chart_tools import create_chart
from agent.tools.middleware import log_before_model, monitor_tool
from agent.tools.tools import get_weather, rag_search
from model.factory import chat_model
from utils.prompt_load import load_report_prompts, load_system_prompts

UTILITY_SYSTEM_PROMPT = """
你是工具执行智能体，负责天气、地图、位置与外部服务查询。
优先调用最匹配的工具，并基于工具结果给出简洁、准确的回答。
如果工具不可用或结果不足，请明确说明，不要编造。
""".strip()


@dataclass(frozen=True)
class WorkerAgents:
    knowledge: Any
    report: Any
    utility: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "knowledge": self.knowledge,
            "report": self.report,
            "utility": self.utility,
        }


def build_worker_agents(mcp_tools: Sequence[BaseTool]) -> WorkerAgents:
    """创建三个职责隔离的工作智能体。"""

    chart_tool_names = {config["mcp_tool"] for config in CHART_CONFIGS.values()}
    utility_tools = [
        get_weather,
        *[tool for tool in mcp_tools if tool.name not in chart_tool_names],
    ]

    common_middleware = [monitor_tool, log_before_model]

    return WorkerAgents(
        knowledge=create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_search],
            middleware=common_middleware,
            name="knowledge_agent",
        ),
        report=create_agent(
            model=chat_model,
            system_prompt=load_report_prompts(),
            tools=[rag_search, create_chart],
            middleware=common_middleware,
            name="report_agent",
        ),
        utility=create_agent(
            model=chat_model,
            system_prompt=UTILITY_SYSTEM_PROMPT,
            tools=utility_tools,
            middleware=common_middleware,
            name="utility_agent",
        ),
    )
