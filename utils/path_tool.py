"""为项目提供统一、可测试的绝对路径。"""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> str:
    """返回工程根目录。"""
    return str(Path(__file__).resolve().parent.parent)


def get_abs_path(relative_path: str | Path) -> str:
    """将工程内相对路径转换为绝对路径；绝对路径保持不变。"""
    path = Path(relative_path)
    return str(path if path.is_absolute() else Path(get_project_root()) / path)
