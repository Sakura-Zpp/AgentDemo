from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embedding_model
from rag.lifecycle import mark_index_updated
from utils.config import chroma_config
from utils.file_handler import get_file_sha256_hex, listdir_with_allowed_type, pdf_load, txt_load
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


class VectorStore:
    """管理 Chroma 向量索引和本地 BM25 混合检索索引。"""

    def __init__(self) -> None:
        self.vectors = Chroma(
            collection_name=str(chroma_config["collection_name"]),
            embedding_function=embedding_model,
            persist_directory=get_abs_path(str(chroma_config["persist_directory"])),
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(chroma_config["chunk_size"]),
            chunk_overlap=int(chroma_config["chunk_overlap"]),
            separators=list(chroma_config["separators"]),
            length_function=len,
        )
        # 兼容历史拼写，外部代码不应依赖该属性。
        self.spliter = self.splitter
        self.bm25_documents: list[Document] = []
        persist_directory = Path(get_abs_path(str(chroma_config["persist_directory"])))
        self.bm25_persist_path = persist_directory / "bm25_index.json"
        self.hash_store_path = Path(get_abs_path(str(chroma_config["sha256_hex_store"])))
        self._load_bm25_index()

    def _load_bm25_index(self) -> None:
        """从 JSON 加载 BM25 文档；首次升级时可由 Chroma 安全重建。"""
        if not self.bm25_persist_path.exists():
            self._rebuild_bm25_from_chroma()
            return
        try:
            loaded: Any = json.loads(self.bm25_persist_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                raise ValueError("BM25 索引格式无效")
            documents: list[Document] = []
            for item in loaded:
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("page_content"), str)
                    or not isinstance(item.get("metadata", {}), dict)
                ):
                    raise ValueError("BM25 索引条目格式无效")
                documents.append(
                    Document(
                        page_content=item["page_content"],
                        metadata=item.get("metadata", {}),
                    )
                )
            self.bm25_documents = documents
            logger.info("已加载 %s 个 BM25 文档", len(self.bm25_documents))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("加载 BM25 索引失败（%s），仅使用向量检索", type(exc).__name__)
            self.bm25_documents = []

    def _save_bm25_index(self) -> None:
        """以替换写入方式保存 BM25 文档，避免留下半写文件。"""
        self.bm25_persist_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.bm25_persist_path.with_suffix(".tmp")
        try:
            serialized = [
                {"page_content": document.page_content, "metadata": document.metadata}
                for document in self.bm25_documents
            ]
            temporary_path.write_text(
                json.dumps(serialized, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary_path.replace(self.bm25_persist_path)
            logger.info("已保存 %s 个 BM25 文档", len(self.bm25_documents))
        except OSError as exc:
            logger.error("保存 BM25 索引失败（%s）", type(exc).__name__)
            temporary_path.unlink(missing_ok=True)
            raise

    def _rebuild_bm25_from_chroma(self) -> None:
        """从现有向量库重建旧 Pickle 索引，不反序列化历史 Pickle 文件。"""
        try:
            stored = self.vectors.get(include=["documents", "metadatas"])
            contents = stored.get("documents") or []
            metadatas = stored.get("metadatas") or []
            self.bm25_documents = [
                Document(
                    page_content=content,
                    metadata=(
                        metadatas[index]
                        if index < len(metadatas) and isinstance(metadatas[index], dict)
                        else {}
                    ),
                )
                for index, content in enumerate(contents)
                if isinstance(content, str)
            ]
            if self.bm25_documents:
                self._save_bm25_index()
                logger.info("已从 Chroma 重建 %s 个 BM25 文档", len(self.bm25_documents))
            else:
                logger.info("未找到可重建的 BM25 文档")
        except Exception as exc:
            logger.warning("从 Chroma 重建 BM25 索引失败（%s）", type(exc).__name__)
            self.bm25_documents = []

    def get_retriever(self, k: int | None = None) -> BaseRetriever:
        result_count = int(k or chroma_config["k"])
        vector_retriever = self.vectors.as_retriever(search_kwargs={"k": result_count})
        if not self.bm25_documents:
            logger.warning("BM25 索引为空，仅使用向量检索")
            return vector_retriever

        bm25_retriever = BM25Retriever.from_documents(
            documents=self.bm25_documents,
            k=result_count,
        )
        return EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=list(chroma_config["weights"]),
        )

    def load_document(self) -> None:
        """导入配置目录中的新文档，并在内容变化后更新索引版本。"""
        index_changed = False
        known_hashes = self._load_known_hashes()
        allowed_file_paths = listdir_with_allowed_type(
            get_abs_path(str(chroma_config["data_path"])),
            tuple(str(item) for item in chroma_config["allowed_file_type"]),
        )

        for file_path in allowed_file_paths:
            digest = get_file_sha256_hex(file_path)
            if digest is None:
                continue
            if digest in known_hashes:
                logger.info("知识库文件已存在，跳过")
                continue

            try:
                documents = self._load_file_documents(file_path)
                split_documents = self.splitter.split_documents(documents)
                if not split_documents:
                    logger.warning("知识库文件无可索引内容，跳过")
                    continue

                self.vectors.add_documents(split_documents)
                self.bm25_documents.extend(split_documents)
                self._save_bm25_index()
                self._append_known_hash(digest)
                known_hashes.add(digest)
                index_changed = True
                logger.info("知识库文件加载成功")
            except (OSError, ValueError, RuntimeError) as exc:
                logger.exception("知识库文件加载失败（%s）", type(exc).__name__)

        if index_changed:
            mark_index_updated()

    def _load_known_hashes(self) -> set[str]:
        try:
            return {
                line.strip()
                for line in self.hash_store_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        except FileNotFoundError:
            return set()
        except OSError as exc:
            logger.error("读取知识库摘要清单失败（%s）", type(exc).__name__)
            raise

    def _append_known_hash(self, digest: str) -> None:
        self.hash_store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.hash_store_path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(f"{digest}\n")

    @staticmethod
    def _load_file_documents(file_path: str) -> list[Document]:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".txt":
            return txt_load(file_path)
        if suffix == ".pdf":
            return pdf_load(file_path)
        raise ValueError(f"不支持的知识库文件类型：{suffix}")


if __name__ == "__main__":
    vector_store = VectorStore()
    vector_store.load_document()
