from __future__ import annotations

import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Generic, TypeVar

from dashscope import TextReRank
from langchain_community.chat_models import ChatTongyi
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from utils.config import rag_config

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

ModelOutput = TypeVar("ModelOutput")


class BaseModel(ABC, Generic[ModelOutput]):
    @abstractmethod
    def generator(self) -> ModelOutput:
        """创建模型适配器。"""


class ChatModel(BaseModel[BaseChatModel]):
    def generator(self) -> BaseChatModel:
        return ChatTongyi(model=rag_config["chat_model_name"], api_key=DASHSCOPE_API_KEY)


class EmbeddingsModel(BaseModel[Embeddings]):
    def generator(self) -> Embeddings:
        return DashScopeEmbeddings(
            model=rag_config["embedding_model_name"],
            dashscope_api_key=DASHSCOPE_API_KEY,
        )


class RerankModel(BaseModel["DashScopeDocumentReranker | None"]):
    def generator(self) -> DashScopeDocumentReranker | None:
        if not DASHSCOPE_API_KEY or not rag_config.get("rerank_enabled", True):
            return None
        return DashScopeDocumentReranker(
            model_name=rag_config.get("rerank_model_name", "gte-rerank"),
            top_n=int(rag_config.get("rerank_top_n", 5)),
            api_key=DASHSCOPE_API_KEY,
        )


class DashScopeDocumentReranker:
    def __init__(self, model_name: str, top_n: int, api_key: str) -> None:
        self.model_name = model_name
        self.top_n = top_n
        self.api_key = api_key

    def compress_documents(
        self,
        documents: list[Document],
        query: str,
    ) -> list[Document]:
        response = TextReRank.call(
            model=self.model_name,
            query=query,
            documents=[document.page_content for document in documents],
            return_documents=False,
            top_n=self.top_n,
            api_key=self.api_key,
        )
        if getattr(response, "status_code", 500) != 200:
            raise RuntimeError("DashScope Rerank 请求失败")
        ranked: list[Document] = []
        for result in response.output.results:
            source = documents[result.index]
            metadata = deepcopy(source.metadata)
            metadata["relevance_score"] = float(result.relevance_score)
            ranked.append(Document(page_content=source.page_content, metadata=metadata))
        return ranked


chat_model = ChatModel().generator()
embedding_model = EmbeddingsModel().generator()
rerank_model = RerankModel().generator()

if __name__ == "__main__":
    print("模型配置已加载" if DASHSCOPE_API_KEY else "缺少 DASHSCOPE_API_KEY")
