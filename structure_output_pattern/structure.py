from langgraph.graph import StateGraph,MessagesState,START,END
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

class CityDetail(BaseModel):
    state_name:str = Field(description="state name of the city")
    state_capital:str = Field(description="State capital name of the city")
    country_name:str = Field(description="Country name of the city")
    country_capital:str = Field(description="Country capital name of the city")
    
class Agentstate(MessagesState):
    Final_response : CityDetail

class chat:
    
    def __init__(self):
        self.llm = ChatGroq(model_name="llama-3.3-70b-specdec")

    def tool_call(self,prompt):
        tool = TavilySearchResults(max_results=2)
        tools = [tool]
        self.model_with_tool = self.llm.bind_tools(tools)
        return self.model_with_tool
    
    def call_model(self, state: AgentState):
        agent_state = AgentState(messages=state.messages, final_response=state.final_response)
        messages = state['messages'][-1]
        response = self.model_with_tool.invoke(messages)
        return {'messages':[response]}