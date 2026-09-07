import asyncio
import re
from uuid import UUID, uuid4

import nest_asyncio
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent.multi_agent import MultiAgentSystem
from utils.logger_handler import logger

nest_asyncio.apply()

st.set_page_config(page_title="LangGraph 多智能体助手", page_icon="🤖", layout="wide")

st.title("🤖 LangGraph 多智能体助手")
st.divider()

# 初始化 Agent
if "agent" not in st.session_state:
    with st.spinner("🔄 正在初始化智能体..."):
        try:
            agent = MultiAgentSystem()
            asyncio.run(agent.initialize())
            st.session_state["agent"] = agent
            st.session_state["agent_initialized"] = True
            st.success("✅ 智能体就绪")
        except Exception as exc:
            logger.exception("页面初始化失败（%s）", type(exc).__name__)
            st.error("❌ 初始化失败，请检查服务配置后重试")
            st.session_state["agent_initialized"] = False

# URL 保存随机 thread_id，使浏览器刷新或服务重启后仍能恢复同一会话。
if "thread_id" not in st.session_state:
    query_thread_id = st.query_params.get("thread")
    try:
        st.session_state["thread_id"] = str(UUID(query_thread_id))
    except (TypeError, ValueError, AttributeError):
        st.session_state["thread_id"] = str(uuid4())
    st.query_params["thread"] = st.session_state["thread_id"]

# user_id 独立于 thread_id，用于跨会话长期记忆；两者均为不可预测随机 UUID。
if "user_id" not in st.session_state:
    query_user_id = st.query_params.get("user")
    try:
        st.session_state["user_id"] = str(UUID(query_user_id))
    except (TypeError, ValueError, AttributeError):
        st.session_state["user_id"] = str(uuid4())
    st.query_params["user"] = st.session_state["user_id"]

# 从持久化 Checkpoint 恢复页面消息。
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    agent = st.session_state.get("agent")
    if agent and st.session_state.get("agent_initialized"):
        try:
            saved_messages = asyncio.run(agent.get_thread_messages(st.session_state["thread_id"]))
            for saved_message in saved_messages:
                if isinstance(saved_message, HumanMessage):
                    st.session_state["messages"].append(
                        {"role": "user", "content": str(saved_message.content)}
                    )
                elif isinstance(saved_message, AIMessage):
                    content = agent.extract_content(saved_message.content)
                    if content:
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": content}
                        )
        except Exception as exc:
            logger.exception("恢复历史消息失败（%s）", type(exc).__name__)

# 侧边栏 - 删除聊天
with st.sidebar:
    st.title("⚙️ 设置")
    st.divider()

    if st.button("🗑️ 删除当前聊天", use_container_width=True):
        agent = st.session_state.get("agent")
        thread_id = st.session_state.get("thread_id")
        if agent and thread_id:
            asyncio.run(agent.clear_thread(thread_id))
        st.session_state["messages"] = []
        st.session_state["thread_id"] = str(uuid4())
        st.query_params["thread"] = st.session_state["thread_id"]
        st.success("✅ 聊天已清空")
        st.rerun()

    confirm_memory_clear = st.checkbox("确认清除全部长期记忆")
    if st.button(
        "🧠 清除长期记忆",
        use_container_width=True,
        disabled=not confirm_memory_clear,
    ):
        agent = st.session_state.get("agent")
        user_id = st.session_state.get("user_id")
        if agent and user_id:
            deleted = asyncio.run(agent.clear_user_memory(user_id))
            st.success(f"✅ 已清除 {deleted} 条长期记忆")

    agent = st.session_state.get("agent")
    if agent:
        with st.expander("📈 运行状态"):
            st.json(agent.health_snapshot())

    st.divider()
    st.info("💡 支持多工具协作和图表生成")

# 显示聊天历史
for i, message in enumerate(st.session_state["messages"]):
    with st.chat_message(message["role"]):
        content = message["content"]
        if not isinstance(content, str):
            content = str(content or "")
        if "https://mdn.alipayobjects.com" in content:
            chart_pattern = r"!\[.*?\]\((https://[^)]+)\)"
            matches = re.findall(chart_pattern, content)

            if matches:
                for url in matches:
                    st.image(url, caption="生成的图表", width=400)  # 400 像素宽

            content = re.sub(chart_pattern, "", content)

        st.write(content)

# 用户输入
st.divider()

agent_ready = st.session_state.get("agent_initialized", False)

prompt = st.chat_input("输入您的问题，例如：公司理念与行为总则是什么？", disabled=not agent_ready)

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

                    response = st.write_stream(
                        agent.execute_stream(
                            prompt,
                            thread_id=st.session_state["thread_id"],
                            user_id=st.session_state["user_id"],
                        )
                    )

                    # 保存回复
                    if response:
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": response}
                        )
                    else:
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": "没有回复产生"}
                        )

                except Exception as exc:
                    logger.exception("页面生成回复失败（%s）", type(exc).__name__)
                    error_message = "抱歉，处理您的请求时出现错误，请稍后重试。"
                    st.error(f"❌ {error_message}")
                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": error_message,
                        }
                    )

# 页脚
st.divider()
st.markdown(
    """
<div style="text-align: center; color: #666; font-size: 12px;">
    <p>💡 复杂问题会自动调用多个工具 | 📊 数据类问题会自动生成可视化图表</p>
</div>
""",
    unsafe_allow_html=True,
)
