import json
from typing import Any

from langchain_core.tools import tool

from agent.mcp.chart_config import CHART_CONFIGS
from agent.mcp.chart_converter import ChartDataConverter, ChartDataError
from utils.logger_handler import logger


@tool
async def create_chart(
    data: list[dict[str, Any]],
    chart_type: str = "column",
    title: str = "数据图表",
) -> str:
    """
    生成图表（支持column/bar/line/area/pie）
    :param data:图表数据
    :param chart_type:图表类型
    :param title:标题
    :return:图表渲染结果
    """

    if not data or not isinstance(data, list):
        return json.dumps({"error": "数据必须是非空列表"}, ensure_ascii=False)

    try:
        standardized_data = ChartDataConverter.normalize_data(data, chart_type)
    except ChartDataError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    logger.info("数据已标准化")

    config = CHART_CONFIGS.get(chart_type.lower(), CHART_CONFIGS["column"])
    mcp_tool_name = config["mcp_tool"]

    try:
        from agent.mcp.mcp_tools import load_mcp_tools

        mcp_tools = await load_mcp_tools()

        chart_tool = next(
            (tool for tool in mcp_tools if tool.name == mcp_tool_name),
            None,
        )

        if not chart_tool:
            return json.dumps({"error": f"未找到工具：{mcp_tool_name}"}, ensure_ascii=False)

        # 调用工具
        chart_params = {"data": standardized_data, "title": title}
        logger.info("调用图表工具：%s", chart_tool.name)

        result = await chart_tool.ainvoke(chart_params)
        return result

    except Exception as exc:
        logger.exception("图表生成失败（%s）", type(exc).__name__)
        return json.dumps({"error": "图表服务暂时不可用"}, ensure_ascii=False)
