import json
import unittest

from agent.mcp.chart_converter import ChartDataConverter, ChartDataError
from agent.mcp.chart_tools import create_chart


class ChartDataConverterTest(unittest.IsolatedAsyncioTestCase):
    def test_preserves_zero_as_valid_x_axis_value(self) -> None:
        result = ChartDataConverter.normalize_data([{"category": 0, "value": 8}], "column")

        self.assertEqual(result[0]["category"], "0")
        self.assertEqual(result[0]["value"], 8.0)

    def test_rejects_invalid_numeric_value_instead_of_silently_using_zero(self) -> None:
        with self.assertRaisesRegex(ChartDataError, "缺少有效数值字段"):
            ChartDataConverter.normalize_data(
                [{"category": "A", "value": "not-a-number"}], "column"
            )

    def test_rejects_unknown_chart_type(self) -> None:
        with self.assertRaisesRegex(ChartDataError, "不支持的图表类型"):
            ChartDataConverter.normalize_data([{"category": "A", "value": 1}], "radar")

    def test_all_supported_chart_types_accept_category_value_shape(self) -> None:
        for chart_type in ("column", "bar", "line", "area", "pie"):
            with self.subTest(chart_type=chart_type):
                result = ChartDataConverter.normalize_data(
                    [{"category": "A", "value": 1}],
                    chart_type,
                )
                self.assertEqual(result[0]["category"], "A")
                self.assertEqual(result[0]["value"], 1.0)

    async def test_tool_returns_safe_validation_error(self) -> None:
        result = await create_chart.ainvoke(
            {"data": [{"category": "A", "value": "invalid"}], "chart_type": "column"}
        )

        self.assertIn("缺少有效数值字段", json.loads(result)["error"])


if __name__ == "__main__":
    unittest.main()
