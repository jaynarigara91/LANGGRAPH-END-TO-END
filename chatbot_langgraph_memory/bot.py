from langgraph.graph import StateGraph,MessagesState,START,END
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv 
import os
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

class chatbot:
    
    def __init__(self):
        self.llm = ChatGroq(model_name = "llama-3.3-70b-specdec")

    def tool_call(self):
        tool = TavilySearchResults(max_results=2)
        tools = [tool]
        self.tool_node = ToolNode(tools=tools)
        self.llm_with_tool = self.llm.bind_tools(tools)
        
    def call_model(self, state: MessagesState):
        message = state['messages']
        response = self.llm_with_tool.invoke(message)
        return {"messages": [response]}
    
    def route_function(self,state:MessagesState):
        message = state['messages'][-1]
        if message.tool_calls:
            return "tools"
        return END
    
    def __call__(self):
        self.tool_call()
        workflow = StateGraph(MessagesState)
        workflow.add_node("Agent",self.call_model)
        workflow.add_node("tools",self.tool_node)
        workflow.add_edge(START,"Agent")
        workflow.add_conditional_edges(
            'Agent',
            self.route_function,
            {
                "tools":"tools",
                END:END
            }
        )
        workflow.add_edge("tools","Agent")
        self.app = workflow.compile(checkpointer=MemorySaver())
        return self.app
    
        
if __name__ == "__main__":
    bot = chatbot()
    app = bot()
    while True:
        user_input = input("you : ")
        if "exit" in user_input:
            break
        config = {"configurable":{"thread_id":'2'}}
        response = app.invoke({'messages':[user_input]},config=config)
        print("AI : ",response['messages'][-1].content)