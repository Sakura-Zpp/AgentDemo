from __future__ import annotations

from pathlib import Path

from utils.config import prompts_config
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


def _load_prompt(config_key: str, label: str) -> str:
    relative_path = prompts_config.get(config_key)
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"缺少有效提示词配置：{config_key}")
    prompt_path = Path(get_abs_path(relative_path))
    try:
        return prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("读取%s失败（%s）", label, type(exc).__name__)
        raise RuntimeError(f"无法读取{label}") from exc


def load_system_prompts() -> str:
    return _load_prompt("main_prompt_path", "系统提示词")


def load_rag_prompts() -> str:
    return _load_prompt("rag_search_prompt_path", "RAG 总结提示词")


def load_report_prompts() -> str:
    return _load_prompt("report_prompt_path", "报告提示词")
