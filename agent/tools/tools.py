from langchain_core.tools import tool
from rag.rag_service import RagSearch

@tool(description="检索参考资料")
def rag_search(query: str) -> str:
    rag = RagSearch()
    return rag.rag_search_doc(query)

@tool(description="返回天气")
def get_weather() -> str:
    return  "今天天气很好"

@tool(description="调用后触发中间件，为生成报告和切换提示词提供上下文信息")
def fill_context_report()-> str:
    return  "调用fill_context_report"