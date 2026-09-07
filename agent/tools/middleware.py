from collections.abc import Callable

from langchain.agents.middleware import (
    ModelRequest,
    before_model,
    dynamic_prompt,
    wrap_tool_call,
)
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from utils.logger_handler import logger
from utils.observability import metrics, timed_metric
from utils.prompt_load import load_report_prompts, load_system_prompts


@wrap_tool_call
async def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    tool_name = request.tool_call["name"]
    arguments = request.tool_call.get("args", {})
    argument_fields = sorted(arguments) if isinstance(arguments, dict) else ["<非结构化>"]
    logger.info("执行工具：%s，参数字段：%s", tool_name, argument_fields)

    metrics.increment("tools.total")
    try:
        with timed_metric(f"tools.{tool_name}.latency"):
            result = await handler(request)
        metrics.increment("tools.success")
        metrics.increment(f"tools.{tool_name}.success")
        logger.info("工具 %s 调用成功", tool_name)

        if tool_name == "fill_context_report":
            request.runtime.context["report"] = True

        return result
    except Exception as exc:
        metrics.increment("tools.error")
        metrics.increment(f"tools.{tool_name}.error")
        logger.exception("工具 %s 调用失败（%s）", tool_name, type(exc).__name__)
        raise


@before_model
async def log_before_model(
    state: AgentState,
    runtime: Runtime,
) -> None:
    del runtime
    last_message_type = type(state["messages"][-1]).__name__ if state["messages"] else "None"
    logger.info(
        "即将调用模型，消息数：%s，末条消息类型：%s",
        len(state["messages"]),
        last_message_type,
    )
    return None


@dynamic_prompt
async def report_prompt_switch(requests: ModelRequest) -> str:
    is_report = requests.runtime.context.get("report", False)
    if is_report:
        return load_report_prompts()

    return load_system_prompts()
