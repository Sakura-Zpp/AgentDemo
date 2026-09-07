from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent.mcp.chart_config import CHART_CONFIGS
from utils.logger_handler import logger


class ChartDataError(ValueError):
    """图表输入无法安全转换时抛出的校验错误。"""


class ChartDataConverter:
    """将常见字段名转换为图表 MCP 接受的统一格式。"""

    @staticmethod
    def normalize_data(
        data: Sequence[Mapping[str, Any]],
        chart_type: str,
    ) -> list[dict[str, Any]]:
        """标准化数据；拒绝未知图表类型和可能造成数据失真的输入。"""
        normalized_type = chart_type.strip().lower()
        if normalized_type not in CHART_CONFIGS:
            supported = "、".join(sorted(CHART_CONFIGS))
            raise ChartDataError(f"不支持的图表类型：{chart_type}；支持：{supported}")
        if not data:
            raise ChartDataError("图表数据不能为空")

        config = CHART_CONFIGS[normalized_type]
        standardized_data: list[dict[str, Any]] = []
        for row_number, item in enumerate(data, start=1):
            if not isinstance(item, Mapping):
                raise ChartDataError(f"第 {row_number} 行必须是对象")
            x_value = ChartDataConverter._find_field(item, config["x_fields"])
            y_value = ChartDataConverter._find_numeric(item, config["y_fields"])
            if x_value is None:
                raise ChartDataError(f"第 {row_number} 行缺少横轴字段")
            if y_value is None:
                raise ChartDataError(f"第 {row_number} 行缺少有效数值字段")
            standardized_data.append(ChartDataConverter._build_item(x_value, y_value))

        logger.info("数据标准化完成：%s 条", len(standardized_data))
        return standardized_data

    @staticmethod
    def _find_field(item: Mapping[str, Any], field_names: Sequence[str]) -> Any | None:
        """按优先级查找横轴字段，保留 0 和 False 等合法值。"""
        for field in field_names:
            if field in item and item[field] is not None:
                return item[field]
        return None

    @staticmethod
    def _find_numeric(
        item: Mapping[str, Any],
        field_names: Sequence[str],
    ) -> float | None:
        """只从声明的纵轴候选字段中读取有限数值。"""
        for field in field_names:
            if field not in item:
                continue
            try:
                value = float(item[field])
            except (TypeError, ValueError):
                continue
            if value == float("inf") or value == float("-inf") or value != value:
                continue
            return value
        return None

    @staticmethod
    def _build_item(x_value: Any, y_value: float) -> dict[str, Any]:
        """构建兼容不同图表 MCP 参数命名的数据项。"""
        item: dict[str, Any] = {}
        x_text = str(x_value)
        for field in ("category", "time", "text", "x", "name", "year", "date", "month"):
            item[field] = x_text
        for field in ("value", "y", "amount", "count", "num", "revenue", "profit"):
            item[field] = y_value
        return item
