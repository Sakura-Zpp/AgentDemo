from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict, cast

import yaml
from dotenv import load_dotenv

from utils.path_tool import get_abs_path


class CheckpointConfig(TypedDict):
    backend: str
    path: str
    dsn_env: str
    retention_days: int
    max_threads: int


class MemoryConfig(TypedDict):
    enabled: bool
    path: str
    semantic_search: bool
    top_k: int
    min_score: float
    retention_days: int
    max_items_per_user: int
    max_content_length: int
    embedding_timeout_seconds: int


class RoutingConfig(TypedDict):
    report_keywords: list[str]
    utility_keywords: list[str]
    memory_keywords: list[str]
    follow_up_phrases: list[str]


class AgentConfig(TypedDict):
    init_timeout: int
    execute_timeout: int
    default_agent: str
    checkpoint: CheckpointConfig
    mcp_server_timeout: int
    memory: MemoryConfig
    routing: RoutingConfig


class RagConfig(TypedDict):
    chat_model_name: str
    embedding_model_name: str
    rerank_enabled: bool
    rerank_model_name: str
    rerank_candidate_k: int
    rerank_top_n: int


class ChromaConfig(TypedDict):
    collection_name: str
    persist_directory: str
    data_path: str
    sha256_hex_store: str
    allowed_file_type: list[str]
    chunk_size: int
    chunk_overlap: int
    k: int
    separators: list[str]
    weights: list[float]


class PromptsConfig(TypedDict):
    main_prompt_path: str
    rag_search_prompt_path: str
    report_prompt_path: str


load_dotenv(Path(get_abs_path(".env")), override=False, encoding="utf-8-sig")


def _load_yaml(config_path: str | Path, encoding: str) -> dict[str, Any]:
    path = Path(config_path)
    with path.open(encoding=encoding) as config_file:
        loaded = yaml.safe_load(config_file)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"配置文件根节点必须是对象：{path.name}")
    return cast(dict[str, Any], loaded)


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"配置项 {key} 必须是对象")
    return cast(dict[str, Any], value)


def _require_string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"配置项 {key} 必须是非空字符串")
    return value


def _require_bool(config: dict[str, Any], key: str) -> bool:
    value = config.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"配置项 {key} 必须是布尔值")
    return value


def _require_int(config: dict[str, Any], key: str, *, minimum: int = 0) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"配置项 {key} 必须是不小于 {minimum} 的整数")
    return value


def _require_float(config: dict[str, Any], key: str, *, minimum: float = 0) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
        raise ValueError(f"配置项 {key} 必须是不小于 {minimum} 的数值")
    return float(value)


def _require_string_list(config: dict[str, Any], key: str) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"配置项 {key} 必须是非空字符串列表")
    return cast(list[str], value)


def load_rag_config(
    config_path: str | Path = get_abs_path("config/rag.yml"),
    encoding: str = "utf-8",
) -> RagConfig:
    config = _load_yaml(config_path, encoding)
    candidate_k = _require_int(config, "rerank_candidate_k", minimum=1)
    top_n = _require_int(config, "rerank_top_n", minimum=1)
    if top_n > candidate_k:
        raise ValueError("rerank_top_n 不能大于 rerank_candidate_k")
    return {
        "chat_model_name": _require_string(config, "chat_model_name"),
        "embedding_model_name": _require_string(config, "embedding_model_name"),
        "rerank_enabled": _require_bool(config, "rerank_enabled"),
        "rerank_model_name": _require_string(config, "rerank_model_name"),
        "rerank_candidate_k": candidate_k,
        "rerank_top_n": top_n,
    }


def load_chroma_config(
    config_path: str | Path = get_abs_path("config/chroma.yml"),
    encoding: str = "utf-8",
) -> ChromaConfig:
    config = _load_yaml(config_path, encoding)
    weights = config.get("weights")
    if (
        not isinstance(weights, list)
        or len(weights) != 2
        or not all(
            isinstance(item, (int, float)) and not isinstance(item, bool) for item in weights
        )
    ):
        raise ValueError("配置项 weights 必须包含两个数值")
    chunk_size = _require_int(config, "chunk_size", minimum=1)
    chunk_overlap = _require_int(config, "chunk_overlap")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    numeric_weights = [float(item) for item in weights]
    if sum(numeric_weights) <= 0:
        raise ValueError("weights 总和必须大于 0")
    return {
        "collection_name": _require_string(config, "collection_name"),
        "persist_directory": _require_string(config, "persist_directory"),
        "data_path": _require_string(config, "data_path"),
        "sha256_hex_store": _require_string(config, "sha256_hex_store"),
        "allowed_file_type": _require_string_list(config, "allowed_file_type"),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "k": _require_int(config, "k", minimum=1),
        "separators": _require_string_list(config, "separators"),
        "weights": numeric_weights,
    }


def load_prompts_config(
    config_path: str | Path = get_abs_path("config/prompts.yml"),
    encoding: str = "utf-8",
) -> PromptsConfig:
    config = _load_yaml(config_path, encoding)
    return {
        "main_prompt_path": _require_string(config, "main_prompt_path"),
        "rag_search_prompt_path": _require_string(config, "rag_search_prompt_path"),
        "report_prompt_path": _require_string(config, "report_prompt_path"),
    }


def load_agent_config(
    config_path: str | Path = get_abs_path("config/agent.yml"),
    encoding: str = "utf-8",
) -> AgentConfig:
    config = _load_yaml(config_path, encoding)
    checkpoint = _require_mapping(config, "checkpoint")
    memory = _require_mapping(config, "memory")
    routing = _require_mapping(config, "routing")
    min_score = _require_float(memory, "min_score")
    if min_score > 1:
        raise ValueError("memory.min_score 不能大于 1")
    checkpoint_backend = _require_string(checkpoint, "backend").lower()
    if checkpoint_backend not in {"memory", "sqlite", "postgres"}:
        raise ValueError("checkpoint.backend 仅支持 memory、sqlite 或 postgres")
    default_agent = _require_string(config, "default_agent")
    if default_agent not in {"knowledge", "report", "utility", "memory"}:
        raise ValueError("default_agent 必须是已注册的智能体名称")
    return {
        "init_timeout": _require_int(config, "init_timeout", minimum=1),
        "execute_timeout": _require_int(config, "execute_timeout", minimum=1),
        "default_agent": default_agent,
        "checkpoint": {
            "backend": checkpoint_backend,
            "path": _require_string(checkpoint, "path"),
            "dsn_env": _require_string(checkpoint, "dsn_env"),
            "retention_days": _require_int(checkpoint, "retention_days"),
            "max_threads": _require_int(checkpoint, "max_threads", minimum=1),
        },
        "mcp_server_timeout": _require_int(config, "mcp_server_timeout", minimum=1),
        "memory": {
            "enabled": _require_bool(memory, "enabled"),
            "path": _require_string(memory, "path"),
            "semantic_search": _require_bool(memory, "semantic_search"),
            "top_k": _require_int(memory, "top_k", minimum=1),
            "min_score": min_score,
            "retention_days": _require_int(memory, "retention_days"),
            "max_items_per_user": _require_int(memory, "max_items_per_user", minimum=1),
            "max_content_length": _require_int(memory, "max_content_length", minimum=1),
            "embedding_timeout_seconds": _require_int(
                memory, "embedding_timeout_seconds", minimum=1
            ),
        },
        "routing": {
            "report_keywords": _require_string_list(routing, "report_keywords"),
            "utility_keywords": _require_string_list(routing, "utility_keywords"),
            "memory_keywords": _require_string_list(routing, "memory_keywords"),
            "follow_up_phrases": _require_string_list(routing, "follow_up_phrases"),
        },
    }


def load_mcp_config(
    config_path: str | Path = get_abs_path("mcp_config.json"),
    encoding: str = "utf-8",
) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {"mcpServers": {}}
    config = _load_yaml(path, encoding)
    servers = config.get("mcpServers")
    if not isinstance(servers, dict):
        raise ValueError("配置项 mcpServers 必须是对象")
    return config


rag_config = load_rag_config()
chroma_config = load_chroma_config()
prompts_config = load_prompts_config()
agent_config = load_agent_config()
mcp_config = load_mcp_config()
