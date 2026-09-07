from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AgentName = Literal["knowledge", "report", "utility", "memory"]


@dataclass(frozen=True)
class RoutingPolicy:
    """按照现有业务范围，将请求分配给职责单一的智能体。"""

    report_keywords: tuple[str, ...]
    utility_keywords: tuple[str, ...]
    memory_keywords: tuple[str, ...] = (
        "请记住",
        "记住",
        "帮我记住",
        "请忘记",
        "忘记",
        "删除记忆",
        "查看记忆",
        "列出记忆",
        "你记得什么",
        "清除所有记忆",
        "清空长期记忆",
        "remember:",
        "forget:",
        "list memories",
        "clear all memories",
    )
    follow_up_phrases: tuple[str, ...] = (
        "继续",
        "继续说",
        "请继续",
        "详细一点",
        "再详细一点",
        "展开说说",
        "为什么",
        "还有吗",
        "然后呢",
        "再说说",
    )
    default_agent: AgentName = "knowledge"

    def route(self, query: str, last_agent: AgentName | None = None) -> AgentName:
        normalized_query = query.strip().lower()

        if self._starts_with_keyword(normalized_query, self.memory_keywords):
            return "memory"

        if self._contains_keyword(normalized_query, self.report_keywords):
            return "report"

        if self._contains_keyword(normalized_query, self.utility_keywords):
            return "utility"

        # 只有明确的追问表达才沿用上一轮智能体，普通短问题仍重新路由。
        if last_agent and normalized_query in self.follow_up_phrases:
            return last_agent

        return self.default_agent

    @staticmethod
    def _contains_keyword(query: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword.lower() in query for keyword in keywords)

    @staticmethod
    def _starts_with_keyword(query: str, keywords: tuple[str, ...]) -> bool:
        return any(query.startswith(keyword.lower()) for keyword in keywords)
