from langchain.tools import tool
from typing import List, Dict, Any
import json
from utils.logger_handler import logger
from agent.mcp.chart_converter import ChartDataConverter
from agent.mcp.chart_config import CHART_CONFIGS

@tool
async def create_chart(
        data: List[Dict[str, Any]],
        chart_type: str = "column",
        title: str = "数据图表"
) -> str:
    """
    生成图表（支持column/bar/line/area/pie）
    :param data:图表数据
    :param chart_type:图表类型
    :param title:标题
    :return:图表渲染结果
    """

    if not data or not isinstance(data, list):
        return logger.error("数据必须是非空列表，此时为空")

    standardized_data = ChartDataConverter.normalize_data(data, chart_type)
    logger.info("数据已标准化")

    config = CHART_CONFIGS.get(chart_type.lower(), CHART_CONFIGS["column"])
    mcp_tool_name = config["mcp_tool"]

    try:
        from agent.mcp.mcp_tools import load_mcp_tools
        mcp_tools = await load_mcp_tools()

        # 查找对应工具
        chart_tool = None
        for tool in mcp_tools:
            if tool.name == mcp_tool_name:
                chart_tool = tool
                break

        if not chart_tool:
            return json.dumps({"error": f"未找到工具：{mcp_tool_name}"}, ensure_ascii=False)

        # 调用工具
        chart_params = {"data": standardized_data, "title": title}
        logger.info(f"调用工具：{chart_tool.name}")

        result = await chart_tool.ainvoke(chart_params)
        return result

    except Exception as e:
        logger.error(f"图表生成失败：{e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)