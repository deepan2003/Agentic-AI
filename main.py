import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from sub_agents import hybrid_research_worker
from llm_gateway import smart_gateway
# Import your new Guardrails security filters
from gateway import verify_user_input, verify_agent_output

load_dotenv()

# 1. Define the Shared State with continuous message memory
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Initialize the Groq LLM engine for the Main Manager
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

AGENT_RULEBOOK_PATH = "./projects/agent.md"

# 3. Define the Core Manager Logic Node
def main_orchestrator_node(state: AgentState):
    # Get the latest message the user sent
    last_user_message = state["messages"][-1].content
    
    # Load your global rules
    rules = ""
    if os.path.exists(AGENT_RULEBOOK_PATH):
        with open(AGENT_RULEBOOK_PATH, "r", encoding="utf-8") as f:
            rules = f.read()

    print(f"\n[Main Agent] Analyzing request and managing context...")
    
    # --- INTENT CLASSIFIER / ROUTER (Updated to 3 Routes) ---
    router_prompt = (
        "Classify the user's prompt into one of three categories: 'CHITCHAT', 'SAVE_HISTORY', or 'RESEARCH'.\n"
        "- Use 'CHITCHAT' for greetings, casual conversation, jokes, expressions of emotion, or simple pleasantries.\n"
        "- Use 'SAVE_HISTORY' ONLY if the user explicitly asks to save, export, or write the PREVIOUS conversation text, previous response, or chat history into a file/document.\n"
        "- Use 'RESEARCH' if the user asks for new news, technical documentation, coding, calculations, general questions, or hard factual data.\n\n"
        "Return ONLY the word 'CHITCHAT', 'SAVE_HISTORY', or 'RESEARCH' and absolutely nothing else.\n\n"
        f"User Prompt: {last_user_message}"
    )
    
    # Instead of calling llm.invoke directly, route through the Unified Proxy API
    # Force the small model choice since routing is a lightweight, low-cost task
    intent_response = smart_gateway.completion(
        messages=[{"role": "user", "content": router_prompt}], 
        override_model="llama-3.1-8b-instant"
    )
    intent = intent_response.content.strip().upper()
    
    # Route A: Conversational Path
    if "CHITCHAT" in intent:
        print("[Main Agent] Intent identified: Conversational Chit-Chat. Bypassing Sub-Agent RAG.")
        
        system_instruction = (
            f"{rules}\n\n"
            "You are a helpful AI assistant. Respond to the user's greeting or statement naturally, "
            "warmly, and conversationally. Do not include fact sheets or source links."
        )
        
        response = llm.invoke([
            {"role": "system", "content": system_instruction},
            *state["messages"]
        ])
        
        return {"messages": [response]}
        
    # Route B: Save History Context Path (NEWLY ADDED FIX)
    elif "SAVE_HISTORY" in intent:
        print("[Main Agent] Intent identified: Save History Context. Packaging conversation history for Sub-Agent...")
        
        # Grab the exact message the AI just generated in the previous turn
        previous_agent_text = "No previous content found to save."
        if len(state["messages"]) > 1:
            previous_agent_text = state["messages"][-2].content
            
        # Build a complete command structure so the sub-agent doesn't go to the web
        history_payload = (
            f"The user wants to save the previous text into a file. Directive: {last_user_message}\n\n"
            f"CRITICAL: Do NOT search the web for documentation. Execute the save_file_to_disk tool immediately using this exact text content:\n"
            f"{previous_agent_text}"
        )
        
        # Delegate to sub-agent with the history text explicitly injected
        fact_sheet = hybrid_research_worker(history_payload)
        
        research_context = (
            f"Research Summary: {fact_sheet.summary}\n"
            f"Extracted Facts: {', '.join(fact_sheet.verified_facts)}\n"
            f"Source Links/Papers: {', '.join(fact_sheet.sources_used)}\n"
        )
        
        system_instruction = (
            f"{rules}\n\n"
            f"CONTEXT FACTS PROVIDED BY SUB-AGENT:\n{research_context}\n\n"
            f"CRITICAL: Confirm clearly to the user that the previous content has been successfully saved to disk as requested."
        )
        
        response = llm.invoke([
            {"role": "system", "content": system_instruction},
            *state["messages"]
        ])
        
        # Cross-check via Gateway Guard
        validated_content = verify_agent_output(response.content, fact_sheet.verified_facts)
        response.content = validated_content
        
        return {"messages": [response]}
        
    # Route C: Tool and Document Research Path
    else:
        print("[Main Agent] Intent identified: Fact/Tool Dependent. Activating Research Sub-Agent...")
        
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
            f"NOTE: If the user asked to save or create a file, the Sub-Agent has already successfully executed that tool and written the document to disk. "
            f"Acknowledge and confirm the successful file save to the user dynamically instead of saying you cannot do it."
        )
        
        # Run the full chat history through the orchestrator model
        response = llm.invoke([
            {"role": "system", "content": system_instruction},
            *state["messages"]
        ])
        
        # --- FIXED GATEWAY OUTPUT GUARD (With Tool Pass-Through for combined requests) ---
        if "save" in last_user_message.lower() or "file" in last_user_message.lower():
            # Let the tool confirmation response pass through cleanly
            pass 
        else:
            # Run factual verification only for standard text questions
            validated_content = verify_agent_output(response.content, fact_sheet.verified_facts)
            response.content = validated_content
        
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
            
            # --- GATEWAY INPUT GUARD (Prompt Injection Check) ---
            secure_query = verify_user_input(user_query)
            if secure_query == "SECURITY_BLOCK":
                print("\nAgent: Access Denied: Malicious request or safety violation detected.\n")
                print("-" * 50)
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