from typing import Callable
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger
from utils.prompt_load import load_report_prompts, load_system_prompts


@wrap_tool_call
def monitor_tool(                       #工具执行监控
        #对请求的数据封装
        request: ToolCallRequest,
        #执行函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command :
    logger.info(f"执行工具：{request.tool_call['name']}")
    logger.info(f"传入参数：{request.tool_call['args']}")

    try:
        result = handler(request)
        logger.info(f"工具{request.tool_call['name']}调用成功")

        if request.tool_call['name'] =="fill_context_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因{str(e)}")
        raise e

@before_model
def log_before_model(
        #agent的状态记录
        state: AgentState,
        #上下文执行信息
        runtime: Runtime,
):                 #模型执行前输出日志
    logger.info(f"即将调用模型，带有{len(state['messages'])}条消息。")
    logger.debug(f"{type(state['messages'][-1])} | {state['messages'][-1].content.strip()}")

    return None

@dynamic_prompt
def report_prompt_switch(requests: ModelRequest):             #提示词生成前调用函数，动态切换提示词
    is_report = requests.runtime.context.get("report",False)
    if is_report:
        return load_report_prompts()

    return load_system_prompts()