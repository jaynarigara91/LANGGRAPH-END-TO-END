from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.pydantic_v1 import BaseModel, Field
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools.retriever import create_retriever_tool
from langchain.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize LLM and Embeddings
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Load and process documents
urls = [    'https://dahfood.com/DAHFOOD-Sultana-raisin/',
                    'https://dahfood.com/DAHFOOD-Red-pistachio-kernel/',
                    'https://dahfood.com/DAHFOOD-Media/',
                    'https://dahfood.com/DAHFOOD-Dates-Catalogues/',
                    'https://dahfood.com/Red-Apple/',
                    'https://dahfood.com/DAHFOOD-Kabkab-dates/',
                    'https://dahfood.com/DAHFOOD-Fandoghi/',
                    'https://dahfood.com/DAHFOOD-Golden-Raisin/',
                    'https://dahfood.com/DAHFOOD-Contact-us/',
                    'https://dahfood.com/DAHFOOD-Moringa-leaf/',
                    'https://dahfood.com/documents/',
                    'https://dahfood.com/Date-seed-kernel/',
                    'https://dahfood.com/',
                    'https://dahfood.com/DAHFOOD-Moringa-powder/',
                    'https://dahfood.com/DAHFOOD-About-us/',
                    'https://dahfood.com/Date-syrup/',
                    'https://dahfood.com/DAHFOOD-Pistachio-Catalogues/',
                    'https://dahfood.com/DAHFOOD-Dried-fruit-Catalogues/',
                    'https://dahfood.com/DAHFOOD-Golden-Kashmari-raisin/',
                    'https://dahfood.com/DAHFOOD-Sun-dried-raisin/',
                    'https://dahfood.com/DAHFOOD-Sliced-dates/',
                    'https://dahfood.com/DAHFOOD-Packing-Label/',
                    'https://dahfood.com/DAHFOOD-Mavis-raisins/',
                    'https://dahfood.com/DAHFOOD-Quality-Control-Guarantee/',
                    'https://dahfood.com/DAHFOOD-Moringa/',
                    'https://dahfood.com/DAHFOOD-Laboratory-and-Technical-Data/',
                    'https://dahfood.com/DAHFOOD-Green-kashmiri-raisin/',
                    'https://dahfood.com/DAHFOOD-Jumbo/',
                    'https://dahfood.com/DAHFOOD-Piarom-dates/',
                    'https://dahfood.com/DAHFOOD-Pistachio/',
                    'https://dahfood.com/#wz-section-wzs297',
                    'https://dahfood.com/DAHFOOD-Dates/',
                    'https://dahfood.com/DAHFOOD-Raisin-Catalogues/',
                    'https://dahfood.com/DAHFOOD-Moringa-oil/',
                    'https://dahfood.com/DAHFOOD-Moringa-Catalogues/',
                    'https://dahfood.com/DAHFOOD-Date-paste/',
                    'https://dahfood.com/DAHFOOD-Mazafati-dates/',
                    'https://dahfood.com/DAHFOOD-Risins/',
                    'https://dahfood.com/DAHFOOD-Rabbi-dates/',
                    'https://dahfood.com/DAHFOOD-Sayer-dates/',
                    'https://dahfood.com/DAHFOOD-Zahedi-Dates/',
                    'https://dahfood.com/DAHFOOD-Green-Peeled-Pistachio-Kernel/',
                    'https://dahfood.com/DAHFOOD-FAQ/',
                    'https://dahfood.com/DAHFOOD-Ahmad-Aghaei/',
                    'https://dahfood.com/DAHFOOD-Akbari/',
                ]
docs = [WebBaseLoader(url).load() for url in urls]
doc_slid = [item for doc in docs for item in doc]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200, chunk_overlap=50
)

text_spliter = text_splitter.split_documents(doc_slid)

# Initialize Vectorstore and Retriever
vector_store = Chroma.from_documents(
    documents=text_spliter, embedding=embedding
)
retriever = vector_store.as_retriever()

# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define GradScore Model
class GradScore(BaseModel):
    binary_score: str = Field(description="Give 'yes' or 'no' if the document is relevant to the question.")

# Create Retriever Tool
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about relevant documents. Use this tool only when the query explicitly relates to the provided data. For other queries, respond directly. For simple queries, provide a normal response.",
)
tool_node = ToolNode([retriever_tool])

# Define Node Functions
def grade_documents(state: AgentState) -> Literal["Output_Generator", "Query_Rewriter"]:
    llm_with_structure_op = llm.with_structured_output(GradScore)

    prompt = PromptTemplate(
        template="""You are a grader deciding if a document is relevant to a user’s question.
            Here is the document: {context}
            Here is the user’s question: {question}
            If the document contains information related to the user’s question, mark it as relevant. 
            Give a 'yes' or 'no' answer.""",
        input_variables=["context", "question"],
    )
    chain = prompt | llm_with_structure_op

    question = state["messages"][-1].content  # Get the latest message (user query)
    docs_string = "\n".join([doc.page_content for doc in text_spliter]) # Joining the context
    scored_result = chain.invoke({"question": question, "context": docs_string})
    score = scored_result.binary_score

    if score.lower() == "yes":  # Handle case sensitivity
        print("---DECISION: DOCS RELEVANT---")
        return "Output_Generator"
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        return "Query_Rewriter"


def generator(state: AgentState):
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[-2].content  # User's question (second to last message)
    docs = messages[-1].content # Retrieved documents

    prompt = hub.pull("rlm/rag-prompt")  # Or a custom prompt
    rag_chain = prompt | llm

    response = rag_chain.invoke({"context": docs, "question": question})
    print(f"this is my response:{response}")

    return {"messages": [AIMessage(content=response)]}  # Return AIMessage


def rewrite(state: AgentState):
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[-1].content # last message
    
    message = [
        HumanMessage(
            content=f"""Look at the input and try to reason about the underlying semantic intent or meaning. 
            Here is the initial question: {question} 
            Formulate an improved question: """
        )
    ]
    response = llm.invoke(message)
    return {"messages": [HumanMessage(content=response)]} # Return HumanMessage


def research_agent(state: AgentState):
    question = state['messages'][-1].content # last message

    tools = [TavilySearchResults()]

    prompt = hub.pull("hwchase17/react")  # or custom prompt

    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke({"input": question})

    return {'messages': [AIMessage(content=response['output'])]} # Return AIMessage



# Build Workflow
workflow = StateGraph(AgentState)
workflow.add_node("Vector_Retriever", tool_node)
workflow.add_node("Query_Rewriter", rewrite)
workflow.add_node("Output_Generator", generator)
workflow.add_node("research_agent", research_agent)

workflow.add_edge(START, "Vector_Retriever")
workflow.add_conditional_edges(
    "Vector_Retriever",
    grade_documents,
    {"Output_Generator": "Output_Generator", "Query_Rewriter": "Query_Rewriter"},
)
workflow.add_edge("Output_Generator", END)
workflow.add_edge("Query_Rewriter", "research_agent")
workflow.add_edge("research_agent", END)

app = workflow.compile()

# Example usage (in your agent loop):
# ... (rest of your agent loop code) ...

input = {'messages':["what is DAHfood"]}
print(app.invoke(input))