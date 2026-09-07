from __future__ import annotations

import os
import re
from typing import Any

import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not os.getenv("E2E_BASE_URL"), reason="需要运行中的 Streamlit 服务"),
]


def test_streamlit_shell_is_ready_and_has_stable_identity(page: Any) -> None:
    base_url = os.environ["E2E_BASE_URL"]
    page.goto(base_url, wait_until="networkidle")

    page.get_by_role("heading", name=re.compile("LangGraph 多智能体助手")).wait_for()
    chat_input = page.get_by_placeholder("输入您的问题，例如：公司理念与行为总则是什么？")
    chat_input.wait_for(state="visible")
    assert chat_input.is_enabled()
    page.get_by_text("运行状态", exact=True).wait_for(state="visible")
    assert re.search(r"[?&]thread=[0-9a-f-]{36}", page.url)
    assert re.search(r"[?&]user=[0-9a-f-]{36}", page.url)
