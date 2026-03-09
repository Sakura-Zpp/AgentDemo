"""
总结服务：用户提问，搜索参考资料，将提问和资料提交各模型，让模型总结回复
"""
from langchain_core.documents import Document
from rag.vector_store import VectorStore
from utils.prompt_load import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser


class RagSearch(object):
    def __init__(self):
        self.vector = VectorStore()
        self.retriever = self.vector.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retriever_doc(self,query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_search_doc(self,query: str) -> str:
        context_docs = self.retriever_doc(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"[参考资料{counter}]:内容:{doc.page_content}"

        return self.chain.invoke(
            {
                "input":query,
                "context":context,
            }
        )

if __name__ == '__main__':
    rag = RagSearch()
    print(rag.rag_search_doc("公司理念与行为总则"))