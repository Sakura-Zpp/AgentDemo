import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from utils.config import rag_config

load_dotenv(Path(__file__).parent.parent / ".env", override=True, encoding='utf-8-sig')
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

class BaseModel(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass

class ChatModel(BaseModel):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_config["chat_model_name"],api_key=DASHSCOPE_API_KEY)

class EmbeddingsModel(BaseModel):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_config["embedding_model_name"],dashscope_api_key=DASHSCOPE_API_KEY)

class RerankModel(BaseModel):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return

chat_model = ChatModel().generator()
embedding_model = EmbeddingsModel().generator()

if __name__ == '__main__':
    print(DASHSCOPE_API_KEY)