from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_load import load_system_prompts
from agent.tools.tools import rag_search,fill_context_report,get_weather
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch

class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model = chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_search,fill_context_report,get_weather],
            middleware=[monitor_tool,log_before_model,report_prompt_switch],
        )

    def execute_stream(self,query: str):
        input_dict = {
            "messages":[
                {"role":"user","content":query},
            ]
        }

        for chunk in self.agent.stream(input_dict,stream_mode="values",context={"report":False}):
           last_messages = chunk["messages"][-1]
           if last_messages.content:
               yield last_messages.content.strip() + "\n"

if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("公司理念与行为总则第五条是什么，并且生成一份关于智源科技的全年财报报告"):
        print(chunk)