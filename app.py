import streamlit as st
from agent.react_agent import ReactAgent

st.title("React Agent")
st.divider()

#初始化agent
if "agent" not  in st.session_state:
    st.session_state["agent"] = ReactAgent()

#初始化消息
if "message" not in st.session_state:
    st.session_state["message"] = []

#初始化历史消息
for message in st.session_state["message"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

#用户输入
prompts = st.chat_input()

if prompts:
    #显示用户消息
    with st.chat_message("user"):
        st.write(prompts)

    #保存用户消息到状态
    st.session_state["message"].append({"role":"user","content":prompts})

    response_messages = []
    with st.spinner("思考中..."):
        res_stream = st.session_state["agent"].execute_stream(prompts)

        def capture(generator,cache_list):
            for chunk in generator:
                if isinstance(chunk,str):
                    cache_list.append(chunk)
                    yield chunk
                else:
                    text = str(chunk)
                    cache_list.append(chunk)
                    yield text

        with st.chat_message("assistant"):
            response = st.write_stream(capture(res_stream,response_messages))

        if response:
            full_content = "".join(response_messages)
        else:
            full_content = response if response else "没有回复产生"

        st.session_state["message"].append({"role":"assistant","content":full_content})
        st.rerun()