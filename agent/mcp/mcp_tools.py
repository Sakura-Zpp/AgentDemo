import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from agent.tools.tools import rag_search,fill_context_report,get_weather
import traceback
from agent.mcp.chart_tools import create_chart

load_dotenv(Path(__file__).parent.parent.parent / ".env", override=True, encoding='utf-8-sig')
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

async def load_mcp_tools():
   client = MultiServerMCPClient(
        {
            "amap-maps": {
                "transport": "streamable_http",
                "url": "https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/mcp",
                "headers": {
                    "Authorization": f"Bearer {DASHSCOPE_API_KEY}"
                }
            },
            "antv-visualization-chart": {
                "transport": "sse",
                "url": "https://dashscope.aliyuncs.com/api/v1/mcps/antv-visualization-chart/sse",
                "headers": {
                    "Authorization": f"Bearer {DASHSCOPE_API_KEY}"
                }
            },
        }
    )

   mcp_tools = await client.get_tools()
   return mcp_tools

async def get_all_tools()-> list[BaseTool]:
    mcp_tools = await load_mcp_tools()
    all_tools = [rag_search, fill_context_report, get_weather,create_chart] + mcp_tools
    return all_tools

if __name__ == '__main__':
    try:
        result = asyncio.run(get_all_tools())
        print(f"成功加载 {len(result)} 个工具")
        for tool in result:
            print(f"  - {tool.name}")
    except Exception as e:
        print(f"错误：{e}")
        traceback.print_exc()

