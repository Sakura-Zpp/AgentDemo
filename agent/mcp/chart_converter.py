from typing import List, Dict, Any
from utils.logger_handler import logger
from agent.mcp.chart_config import CHART_CONFIGS

class ChartDataConverter:
    """基础图表数据转换器"""

    @staticmethod
    def normalize_data(data: List[Dict[str, Any]], chart_type: str) -> List[Dict[str, Any]]:
        """标准化数据格式"""
        if not data or not isinstance(data, list):
            return []

        config = CHART_CONFIGS.get(chart_type.lower(), CHART_CONFIGS["column"])
        standardized_data = []

        for item in data:
            # 查找 X 轴字段
            x_value = ChartDataConverter._find_field(item, config["x_fields"])

            # 查找 Y 轴字段（确保是数字）
            y_value = ChartDataConverter._find_numeric(item, config["y_fields"])

            # 构建兼容数据（同时填充所有可能的字段名）
            std_item = ChartDataConverter._build_item(x_value, y_value)
            standardized_data.append(std_item)

        logger.info(f"📊 数据标准化：{len(standardized_data)} 条")
        return standardized_data

    @staticmethod
    def _find_field(item: Dict, field_names: List[str]) -> Any:
        """查找字段值"""
        for field in field_names:
            if field in item:
                return item[field]
        return str(list(item.values())[0]) if item else ""

    @staticmethod
    def _find_numeric(item: Dict, field_names: List[str]) -> float:
        """查找数值字段"""
        for field in field_names:
            if field in item:
                try:
                    return float(item[field])
                except (ValueError, TypeError):
                    pass
        #  fallback：找第一个能转数字的值
        for v in item.values():
            try:
                return float(v)
            except:
                pass
        return 0.0

    @staticmethod
    def _build_item(x_value: Any, y_value: float) -> Dict[str, Any]:
        """
        构建兼容数据项 - ⚠️ 关键：同时填充所有字段名
        """
        item = {}

        # X 轴字段（全部填充）
        x_str = str(x_value) if x_value else ""
        for field in ["category", "time", "text", "x", "name", "year", "date", "month"]:
            item[field] = x_str

        # Y 轴字段（全部填充）
        for field in ["value", "y", "amount", "count", "num", "revenue", "profit"]:
            item[field] = y_value

        return item
