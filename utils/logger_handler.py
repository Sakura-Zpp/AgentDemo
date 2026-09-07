from __future__ import annotations

import logging
import os
import re
from logging.handlers import TimedRotatingFileHandler

from utils.observability import request_id_var
from utils.path_tool import get_abs_path

_SECRET_PATTERNS = (
    (re.compile(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer ***"),
    (re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{8,}\b"), "sk-***"),
    (
        re.compile(r"(?i)(authorization)(\s*[:=]\s*)(?:Bearer\s+)?([^\s,;]+)"),
        r"\1\2***",
    ),
    (
        re.compile(r"(?i)(api[_-]?key|access[_-]?token)(\s*[:=]\s*)([^\s,;]+)"),
        r"\1\2***",
    ),
)


def redact_sensitive(value: str) -> str:
    redacted = value
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact_sensitive(super().format(record))


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def _log_level(environment_name: str, default: int) -> int:
    configured = os.getenv(environment_name)
    if not configured:
        return default
    return logging.getLevelNamesMapping().get(configured.upper(), default)


log_root = get_abs_path("log")
os.makedirs(log_root, exist_ok=True)

default_log_format = RedactingFormatter(
    "%(asctime)s - %(name)s - %(levelname)s - request_id=%(request_id)s - "
    "%(filename)s:%(lineno)d - %(message)s"
)


def get_logger(
    logger_name: str = "agent",
    console_level: int | None = None,
    file_level: int | None = None,
    log_file: str | None = None,
) -> logging.Logger:
    configured_logger = logging.getLogger(logger_name)
    configured_logger.setLevel(logging.DEBUG)
    configured_logger.propagate = False

    if configured_logger.handlers:
        return configured_logger

    console_handler = logging.StreamHandler()
    console_handler.addFilter(RequestContextFilter())
    console_handler.setLevel(
        console_level
        if console_level is not None
        else _log_level("AGENT_CONSOLE_LOG_LEVEL", logging.INFO)
    )
    console_handler.setFormatter(default_log_format)
    configured_logger.addHandler(console_handler)

    target_log_file = log_file or os.path.join(log_root, f"{logger_name}.log")
    file_handler = TimedRotatingFileHandler(
        target_log_file,
        when="midnight",
        backupCount=max(int(os.getenv("AGENT_LOG_RETENTION_DAYS", "14")), 1),
        encoding="utf-8",
    )
    file_handler.addFilter(RequestContextFilter())
    file_handler.setLevel(
        file_level if file_level is not None else _log_level("AGENT_FILE_LOG_LEVEL", logging.INFO)
    )
    file_handler.setFormatter(default_log_format)
    configured_logger.addHandler(file_handler)

    return configured_logger


logger = get_logger()


if __name__ == "__main__":
    logger.info("信息日志")
    logger.error("错误日志")
    logger.warning("警告日志")
