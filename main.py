import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from sub_agents import hybrid_research_worker

load_dotenv()

# 1. Define the Shared State with continuous message memory
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Initialize the Groq LLM engine for the Main Manager
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

AGENT_RULEBOOK_PATH = "./projects/agent.md"

# 3. Define the Core Manager Logic Node
def main_orchestrator_node(state: AgentState):
    # Get the latest message the user sent
    last_user_message = state["messages"][-1].content
    
    # Load your global rules from step 2
    rules = ""
    if os.path.exists(AGENT_RULEBOOK_PATH):
        with open(AGENT_RULEBOOK_PATH, "r", encoding="utf-8") as f:
            rules = f.read()

    print(f"\n[Main Agent] Analyzing request and managing context...")
    
    # Delegate factual exploration to your sub-agent
    fact_sheet = hybrid_research_worker(last_user_message)
    
    # Standardize the context format returned by the researcher
    research_context = (
        f"Research Summary: {fact_sheet.summary}\n"
        f"Extracted Facts: {', '.join(fact_sheet.verified_facts)}\n"
        f"Source Links/Papers: {', '.join(fact_sheet.sources_used)}\n"
    )
    
    # Combine the rulebook directives with the extracted facts
    system_instruction = (
        f"{rules}\n\n"
        f"CONTEXT FACTS PROVIDED BY SUB-AGENT:\n{research_context}\n\n"
        f"CRITICAL: Base your answer strictly on the provided context facts. "
        f"Include clear source citations at the end of your response."
    )
    
    # Run the full chat history through the orchestrator model
    response = llm.invoke([
        {"role": "system", "content": system_instruction},
        *state["messages"]
    ])
    
    return {"messages": [response]}

# 4. Build and Compile the LangGraph Workflow
workflow = StateGraph(AgentState)
workflow.add_node("orchestrator", main_orchestrator_node)
workflow.add_edge(START, "orchestrator")
workflow.add_edge("orchestrator", END)

# Attach the live memory checkpointer
chat_memory = MemorySaver()
agent_system = workflow.compile(checkpointer=chat_memory)

# 5. Interactive Chat Interface Loop
if __name__ == "__main__":
    print("\n==================================================")
    print("🤖 Agentic RAG System with Conversational Memory Active")
    print("==================================================")
    print("Type 'exit' or 'quit' to close the session.\n")
    
    # Establish a unique thread ID for tracking this specific chat session
    session_config = {"configurable": {"thread_id": "active_developer_session"}}
    
    while True:
        try:
            user_query = input("You: ")
            if user_query.strip().lower() in ["exit", "quit"]:
                print("Powering down orchestrator. System offline.")
                break
                
            if not user_query.strip():
                continue
                
            # Stream input data directly through the graph state loop
            state_output = agent_system.invoke(
                {"messages": [{"role": "user", "content": user_query}]},
                config=session_config
            )
            
            # Fetch the final structural AI message from the conversation graph
            final_response = state_output["messages"][-1].content
            print(f"\nAgent: {final_response}\n")
            print("-" * 50)
            
        except Exception as e:
            print(f"\nAn error occurred in the execution loop: {str(e)}\n")