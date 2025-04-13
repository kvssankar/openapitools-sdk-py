from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from reacter_openapitools import LangChainAdapter
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

memory = MemorySaver()
model = ChatAnthropic(model_name="claude-3-sonnet-20240229",
                      api_key="your-anthropic-key")

adapter = LangChainAdapter(
    api_key="your-openapitools-key", verbose=True)

tools = adapter.get_langchain_tools()

agent_executor = create_react_agent(model, tools)

response = agent_executor.invoke(
    {"messages": [HumanMessage(content="can u generate an OTP for 98989898981?")]})

print(response["messages"])
