from functools import lru_cache

from langchain_core.tools import tool

from rag.lifecycle import get_index_revision
from rag.rag_service import RagSearch


@lru_cache(maxsize=2)
def _get_rag_search(index_revision: int) -> RagSearch:
    del index_revision
    return RagSearch()


def get_rag_search() -> RagSearch:
    """复用检索服务，并在知识库索引版本变化时自动重建。"""

    return _get_rag_search(get_index_revision())


def reset_rag_search() -> None:
    _get_rag_search.cache_clear()


@tool(description="检索参考资料")
def rag_search(query: str) -> str:
    return get_rag_search().rag_search_doc(query)


@tool(description="返回天气")
def get_weather() -> str:
    return "今天天气很好"


@tool(description="调用后触发中间件，为生成报告和切换提示词提供上下文信息")
def fill_context_report() -> str:
    return "调用fill_context_report"
