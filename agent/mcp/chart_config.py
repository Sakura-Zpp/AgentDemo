"""
图表所需字段配置

"""

CHART_CONFIGS = {
    "column": {
        "x_fields": ["category", "time", "year", "date", "month", "name", "text", "x"],
        "y_fields": ["value", "amount", "revenue", "profit", "count", "num", "y"],
        "required": ["category", "value"],
        "mcp_tool": "generate_column_chart",
    },
    "bar": {
        "x_fields": ["category", "time", "year", "date", "month", "name", "text", "x"],
        "y_fields": ["value", "amount", "revenue", "profit", "count", "num", "y"],
        "required": ["category", "value"],
        "mcp_tool": "generate_bar_chart",
    },
    "line": {
        "x_fields": ["time", "date", "year", "month", "category", "x"],
        "y_fields": ["value", "amount", "revenue", "profit", "count", "num", "y"],
        "required": ["time", "value"],
        "mcp_tool": "generate_line_chart",
    },
    "area": {
        "x_fields": ["time", "date", "year", "month", "category", "x"],
        "y_fields": ["value", "amount", "revenue", "profit", "count", "num", "y"],
        "required": ["time", "value"],
        "mcp_tool": "generate_area_chart",
    },
    "pie": {
        "x_fields": ["category", "name", "type", "label", "text"],
        "y_fields": ["value", "amount", "revenue", "profit", "count", "num"],
        "required": ["category", "value"],
        "mcp_tool": "generate_pie_chart",
    },
}


FIELD_MAPPING = {
    "category": ["category", "time", "text", "year", "date", "month", "name", "label", "x"],
    "value": ["value", "amount", "revenue", "profit", "count", "num", "y"],
}
