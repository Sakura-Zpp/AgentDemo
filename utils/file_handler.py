from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from utils.logger_handler import logger


def get_file_sha256_hex(filepath: str | Path) -> str | None:
    """返回文件 SHA-256；无效路径或读取失败时返回 ``None``。"""
    path = Path(filepath)
    if not path.is_file():
        logger.error("待计算摘要的路径不是有效文件")
        return None

    digest = hashlib.sha256()
    try:
        with path.open("rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(64 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        logger.error("计算文件 SHA-256 失败（%s）", type(exc).__name__)
        return None
    return digest.hexdigest()


def get_file_SHA256_hex(filepath: str | Path) -> str | None:
    """兼容旧调用方；新代码应使用 :func:`get_file_sha256_hex`。"""
    return get_file_sha256_hex(filepath)


def listdir_with_allowed_type(
    path: str | Path,
    allowed_types: tuple[str, ...],
) -> tuple[str, ...]:
    """列出目录下允许扩展名的普通文件，结果按名称排序。"""
    directory = Path(path)
    if not directory.is_dir():
        logger.error("知识库数据目录不存在或不是目录")
        return ()
    normalized_types = tuple(
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in allowed_types
    )
    return tuple(
        str(candidate.resolve())
        for candidate in sorted(directory.iterdir(), key=lambda item: item.name.lower())
        if candidate.is_file() and candidate.suffix.lower() in normalized_types
    )


def pdf_load(filepath: str | Path, password: str | bytes | None = None) -> list[Document]:
    return PyPDFLoader(str(filepath), password).load()


def txt_load(filepath: str | Path) -> list[Document]:
    return TextLoader(str(filepath), encoding="utf-8").load()
