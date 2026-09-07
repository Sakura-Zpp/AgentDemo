"""
总结服务：用户提问，搜索参考资料，将提问和资料提交各模型，让模型总结回复
"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model, rerank_model
from rag.vector_store import VectorStore
from utils.config import rag_config
from utils.logger_handler import logger
from utils.prompt_load import load_rag_prompts


class RagSearch:
    def __init__(self, reranker: Any | None = None) -> None:
        self.vector = VectorStore()
        candidate_k = int(rag_config.get("rerank_candidate_k", 12))
        self.retriever = self.vector.get_retriever(k=candidate_k)
        self.reranker = rerank_model if reranker is None else reranker
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self) -> Any:
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retriever_doc(self, query: str) -> list[Document]:
        candidates = list(self.retriever.invoke(query))
        final_k = int(rag_config.get("rerank_top_n", 5))
        if not self.reranker or not candidates:
            return candidates[:final_k]
        try:
            ranked = list(self.reranker.compress_documents(candidates, query))
            return ranked[:final_k] if ranked else candidates[:final_k]
        except Exception as exc:
            logger.warning("Reranker 调用失败（%s），保留混合检索顺序", type(exc).__name__)
            return candidates[:final_k]

    def rag_search_doc(self, query: str) -> str:
        context_docs = self.retriever_doc(query)
        context = "\n\n".join(
            f"[参考资料{index}]\n{document.page_content}"
            for index, document in enumerate(context_docs, start=1)
        )

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


if __name__ == "__main__":
    rag = RagSearch()
    print(rag.rag_search_doc("公司理念与行为总则"))
