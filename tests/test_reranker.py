import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from model.factory import DashScopeDocumentReranker
from rag.rag_service import RagSearch


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, query):
        del query
        return self.documents


class FakeReranker:
    def compress_documents(self, documents, query):
        del query
        return list(reversed(documents))


class RerankerTest(unittest.TestCase):
    def test_rag_uses_reranker_order(self):
        rag = RagSearch.__new__(RagSearch)
        documents = [Document(page_content="first"), Document(page_content="second")]
        rag.retriever = FakeRetriever(documents)
        rag.reranker = FakeReranker()

        with patch.dict("rag.rag_service.rag_config", {"rerank_top_n": 1}):
            result = rag.retriever_doc("query")

        self.assertEqual([document.page_content for document in result], ["second"])

    def test_rag_falls_back_when_reranker_fails(self):
        rag = RagSearch.__new__(RagSearch)
        documents = [Document(page_content="first"), Document(page_content="second")]
        rag.retriever = FakeRetriever(documents)
        rag.reranker = Mock()
        rag.reranker.compress_documents.side_effect = RuntimeError("unavailable")

        with patch.dict("rag.rag_service.rag_config", {"rerank_top_n": 1}):
            result = rag.retriever_doc("query")

        self.assertEqual([document.page_content for document in result], ["first"])

    def test_dashscope_response_is_converted_to_scored_documents(self):
        reranker = DashScopeDocumentReranker("gte-rerank", 1, "test-key")
        documents = [
            Document(page_content="first", metadata={"source": "a"}),
            Document(page_content="second", metadata={"source": "b"}),
        ]
        response = SimpleNamespace(
            status_code=200,
            output=SimpleNamespace(results=[SimpleNamespace(index=1, relevance_score=0.91)]),
        )
        with patch("model.factory.TextReRank.call", return_value=response):
            ranked = reranker.compress_documents(documents, "query")

        self.assertEqual(ranked[0].page_content, "second")
        self.assertEqual(ranked[0].metadata["relevance_score"], 0.91)
        self.assertNotIn("relevance_score", documents[1].metadata)


if __name__ == "__main__":
    unittest.main()
