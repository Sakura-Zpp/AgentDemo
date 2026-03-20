from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_load import load_system_prompts
from agent.tools.tools import rag_search,fill_context_report,get_weather
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch
from agent.mcp.mcp_tools import get_all_tools
from utils.logger_handler import logger
import asyncio

class ReactAgent:

    def __init__(self):
        self.agent = None
        self.tools = None
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            logger.info("智能体已初始化，跳过")
            return
        try:
            self.tools = await get_all_tools()
            logger.info(f"工具加载，共有 {len(self.tools)} 个")
        except Exception as e:
            #防止调用MCP服务失败，进行工具降级
            logger.error(f"MCP工具加载失败: {e} ,工具降级，只使用基础工具")
            self.tools = [rag_search, fill_context_report, get_weather]
            logger.info(f"基础工具加载成功，共有 {len(self.tools)} 个")

        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=self.tools,
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

        self._initialized = True

#处理文本返回类型
    def extract_content(self,content):

        if not content:
            return None

        #内容为字符串
        if isinstance(content,str):
            return content

        #内容为列表
        if isinstance(content,list):
            full_parts = []
            for item in content:
                if isinstance(item,str):
                    full_parts.append(item)
                elif isinstance(item,dict):
                    if item.get("type") == "text":
                        full_parts.append(item.get("text",""))
                    elif "text" in item:
                        full_parts.append(item["text"])

            return "\n".join(full_parts)
        #其他类型转回为字符串
        return str(content)

    async def execute_stream(self,query: str):

        if not self._initialized or self.agent is None:
            raise RuntimeError("请先调用 await agent.initialize()")

        input_dict = {
            "messages":[
                {"role":"user","content":query},
            ]
        }

        async for chunk in self.agent.astream(input_dict,stream_mode="values",context={"report":False}):
            #确保消息不为空
            if not chunk.get("messages"):
                continue

            last_messages = chunk["messages"][-1]

            #获取last_messages的类型
            content = last_messages.content if hasattr(last_messages, 'content') else last_messages
            logger.info(f"获取到的类型为{type(content)}")
            # 使用辅助方法提取文本
            text = self.extract_content(content)

            if text:
               yield text.strip() + "\n"

if __name__ == '__main__':

    async def main():
        print("程序启动\n")
        agent = ReactAgent()
        await agent.initialize()
        print(f"工具数量 = {len(agent.tools) if agent.tools else 0}")
        query = "公司理念与行为总则第五条是什么，并且生成一份关于智源科技的全年财报报告，并生成一份图表"
        async for chunk in agent.execute_stream(query):
            print(chunk)
    asyncio.run(main())