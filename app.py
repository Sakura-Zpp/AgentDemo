import re

import streamlit as st
from agent.react_agent import ReactAgent
from utils.logger_handler import logger
import nest_asyncio
nest_asyncio.apply()

st.set_page_config(
    page_title="React Agent 智能助手",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 React Agent 智能助手")
st.divider()

# 初始化 Agent
if "agent" not in st.session_state:
    with st.spinner("🔄 正在初始化智能体..."):
        try:
            import asyncio

            agent = ReactAgent()
            asyncio.run(agent.initialize())
            st.session_state["agent"] = agent
            st.session_state["agent_initialized"] = True
            st.success("✅ 智能体就绪")
        except Exception as e:
            st.error(f"❌ 初始化失败：{str(e)}")
            st.session_state["agent_initialized"] = False

# 初始化消息历史
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 侧边栏 - 删除聊天
with st.sidebar:
    st.title("⚙️ 设置")
    st.divider()

    if st.button("🗑️ 删除当前聊天", use_container_width=True):
        st.session_state["messages"] = []
        st.success("✅ 聊天已清空")
        st.rerun()

    st.divider()
    st.info("💡 支持多工具协作和图表生成")

# 显示聊天历史
for i, message in enumerate(st.session_state["messages"]):
    with st.chat_message(message["role"]):
        content = message["content"]
        if "https://mdn.alipayobjects.com" in content:
            chart_pattern = r'!\[.*?\]\((https://[^)]+)\)'
            matches = re.findall(chart_pattern, content)

            if matches:
                for url in matches:
                    st.image(url, caption="生成的图表", width=400)  # 400 像素宽

            content = re.sub(chart_pattern, '', content)

        st.write(content)

# 用户输入
st.divider()

agent_ready = st.session_state.get("agent_initialized", False)

prompt = st.chat_input(
    "输入您的问题，例如：公司理念与行为总则是什么？",
    disabled=not agent_ready
)

if prompt:
    # 显示用户消息
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # 生成回复
    with st.chat_message("assistant"):
        if not agent_ready:
            st.error("⚠️ 智能体未初始化，请稍后重试")
        else:
            with st.spinner("🤔 思考中..."):
                try:
                    agent = st.session_state["agent"]

                    response = st.write_stream(agent.execute_stream(prompt))

                    # 保存回复
                    if response:
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": response
                        })
                    else:
                        st.session_state["messages"].append({
                            "role": "assistant",
                            "content": "没有回复产生"
                        })

                except Exception as e:
                    st.error(f"❌ 生成回复失败：{logger.error(str(e))}")
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": f"抱歉，处理您的请求时出现错误：{logger.error(str(e))}"
                    })

# 页脚
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <p>💡 复杂问题会自动调用多个工具 | 📊 数据类问题会自动生成可视化图表</p>
</div>
""", unsafe_allow_html=True)
