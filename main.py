import os
import re
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from sub_agents import hybrid_research_worker
from llm_gateway import smart_gateway
from tools.file_system import save_file_to_disk, clean_document_content
from gateway import verify_user_input, verify_agent_output

load_dotenv()

# 1. Define the Shared State with continuous message memory
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

AGENT_RULEBOOK_PATH = "./agent.md"

def invoke_orchestrator_llm(system_instruction: str, history_messages: list):
    """
    Routes orchestrator completions through SmartLLMGateway with automatic
    retry, caching, and cross-provider fallback (Groq -> Gemini).
    """
    formatted = [{"role": "system", "content": system_instruction}]
    for m in history_messages:
        role = getattr(m, "type", None) or getattr(m, "role", "user")
        if role in ["human", "user"]:
            r = "user"
        else:
            r = "assistant"
        content = getattr(m, "content", "")
        if content:
            formatted.append({"role": r, "content": str(content)})
            
    return smart_gateway.completion(
        messages=formatted,
        override_model="openai/gpt-oss-20b"
    )

# 3. Define the Core Manager Logic Node
def main_orchestrator_node(state: AgentState):
    # Get the latest message the user sent
    last_user_message = state["messages"][-1].content
    
    # Load global rules
    rules = ""
    if os.path.exists(AGENT_RULEBOOK_PATH):
        with open(AGENT_RULEBOOK_PATH, "r", encoding="utf-8") as f:
            rules = f.read()

    print(f"\n[Main Agent] Analyzing request and managing context...")
    
    # --- INTENT CLASSIFIER / ROUTER ---
    router_prompt = (
        "Classify the user's prompt into one of three categories: 'CHITCHAT', 'SAVE_HISTORY', or 'RESEARCH'.\n"
        "- Use 'CHITCHAT' for greetings, casual conversation, jokes, expressions of emotion, or simple pleasantries.\n"
        "- Use 'SAVE_HISTORY' ONLY if the user explicitly asks to save, export, or write the PREVIOUS conversation text, previous response, or chat history into a file/document.\n"
        "- Use 'RESEARCH' if the user asks for new news, technical documentation, coding, calculations, general questions, or hard factual data.\n\n"
        "Return ONLY the word 'CHITCHAT', 'SAVE_HISTORY', or 'RESEARCH' and absolutely nothing else.\n\n"
        f"User Prompt: {last_user_message}"
    )
    
    intent_response = smart_gateway.completion(
        messages=[{"role": "user", "content": router_prompt}], 
        override_model="openai/gpt-oss-20b"
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
        
        response = invoke_orchestrator_llm(system_instruction, state["messages"])
        return {"messages": [response]}
        
    # Route B: Save History Context Path
    elif "SAVE_HISTORY" in intent:
        print("[Main Agent] Intent identified: Save History Context. Packaging conversation history for File Saver...")
        
        # Grab the exact message the AI generated in the previous turn
        previous_agent_text = "No previous content found to save."
        if len(state["messages"]) > 1:
            previous_agent_text = state["messages"][-2].content
            
        # Determine target filename from prompt
        filename_match = re.search(r'[\w\-]+\.(txt|md|docx?|csv|json|py|doc)', last_user_message, re.IGNORECASE)
        if filename_match:
            filename = filename_match.group(0)
        else:
            words = [w for w in re.findall(r'[a-zA-Z0-9]+', last_user_message.lower()) if w not in ["save", "the", "previous", "into", "file", "to", "a", "as", "format", "doc", "docx", "txt"]]
            base_name = "_".join(words[:4]) if words else "saved_document"
            ext_match = re.search(r'\b(txt|md|docx?|csv|json|doc)\b', last_user_message.lower())
            ext = ext_match.group(1) if ext_match else "txt"
            filename = f"{base_name}.{ext}"

        # Clean content so that raw code blocks/Python boilerplate are not saved into text/word documents
        clean_text = clean_document_content(previous_agent_text)

        # Physically execute the save tool directly to write file to projects/ folder!
        save_result = save_file_to_disk.invoke({"filename": filename, "content": clean_text})
        print(f"[Main Agent] Physical file saved to disk: {save_result}")

        system_instruction = (
            f"{rules}\n\n"
            f"FILE SAVE EXECUTION RESULT:\n{save_result}\n\n"
            f"CRITICAL: Inform the user that their request has been executed and the file '{filename}' was successfully saved. "
            f"Explicitly inform them that they can view, download, or delete it from the 'Saved Documents' section in the left sidebar."
        )
        
        response = invoke_orchestrator_llm(system_instruction, state["messages"])
        return {"messages": [response]}
        
    # Route C: Tool and Document Research Path
    else:
        print("[Main Agent] Intent identified: Fact/Tool Dependent. Activating Research Sub-Agent...")
        
        # Delegate factual exploration to your sub-agent
        fact_sheet = hybrid_research_worker(last_user_message)
        
        # If user asked to save research findings to a file, ensure it is written to disk!
        if any(w in last_user_message.lower() for w in ["save", "create a file", "write to file", ".txt", ".md", ".doc", ".docx"]):
            filename_match = re.search(r'[\w\-]+\.(txt|md|docx?|csv|json|py|doc)', last_user_message, re.IGNORECASE)
            if filename_match:
                target_filename = filename_match.group(0)
            else:
                words = [w for w in re.findall(r'[a-zA-Z0-9]+', last_user_message.lower()) if w not in ["save", "the", "into", "file", "to", "a", "as", "format", "news", "give", "me", "what", "is"]]
                base_name = "_".join(words[:4]) if words else "research_report"
                target_filename = f"{base_name}.txt"
            
            target_path = os.path.join("./projects", os.path.basename(target_filename))
            if not os.path.exists(target_path):
                file_content = f"Research Summary:\n{fact_sheet.summary}\n\nKey Verified Facts:\n" + "\n".join(f"- {f}" for f in fact_sheet.verified_facts)
                save_file_to_disk.invoke({"filename": target_filename, "content": file_content})
                print(f"[Main Agent] Physical file saved to disk: {target_filename}")

        # Standardize context format returned by researcher
        research_context = (
            f"Research Summary: {fact_sheet.summary}\n"
            f"Extracted Facts: {', '.join(fact_sheet.verified_facts)}\n"
            f"Source Links/Papers: {', '.join(fact_sheet.sources_used)}\n"
        )
        
        system_instruction = (
            f"{rules}\n\n"
            f"CONTEXT FACTS PROVIDED BY SUB-AGENT:\n{research_context}\n\n"
            f"CRITICAL RULES:\n"
            f"1. Base your answer strictly on the provided context facts. Do NOT hallucinate or mix unrelated facts.\n"
            f"2. Never use raw HTML tags such as <br>, <br/>, or <div>. Use Markdown bullet points (- ) and paragraphs instead.\n"
            f"3. Directly address the user's specific inquiry. If the facts do not answer the question, state clearly that the information is unavailable in the retrieved sources.\n"
            f"4. If the user asked to save or create a file, confirm the filename saved and remind them they can download it directly from the 'Saved Documents' sidebar."
        )
        
        response = invoke_orchestrator_llm(system_instruction, state["messages"])
        return {"messages": [response]}

# 4. Build and Compile the LangGraph Workflow
workflow = StateGraph(AgentState)
workflow.add_node("orchestrator", main_orchestrator_node)
workflow.add_edge(START, "orchestrator")
workflow.add_edge("orchestrator", END)

# Attach live memory checkpointer
chat_memory = MemorySaver()
agent_system = workflow.compile(checkpointer=chat_memory)

# 5. Interactive Chat Interface Loop
if __name__ == "__main__":
    print("\n==================================================")
    print("🤖 Agentic RAG System with Conversational Memory Active")
    print("==================================================")
    print("Type 'exit' or 'quit' to close the session.\n")
    
    session_config = {"configurable": {"thread_id": "active_developer_session"}}
    
    while True:
        try:
            user_query = input("You: ")
            if user_query.strip().lower() in ["exit", "quit"]:
                print("Powering down orchestrator. System offline.")
                break
                
            if not user_query.strip():
                continue
            
            secure_query = verify_user_input(user_query)
            if secure_query == "SECURITY_BLOCK":
                print("\nAgent: Access Denied: Malicious request or safety violation detected.\n")
                print("-" * 50)
                continue
                
            state_output = agent_system.invoke(
                {"messages": [{"role": "user", "content": user_query}]},
                config=session_config
            )
            
            final_response = state_output["messages"][-1].content
            print(f"\nAgent: {final_response}\n")
            print("-" * 50)
            
        except Exception as e:
            print(f"\nAn error occurred in the execution loop: {str(e)}\n")
