import os
import re
from typing import Annotated, TypedDict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from sub_agents import hybrid_research_worker
from llm_gateway import smart_gateway
from tools.file_system import save_file_to_disk, clean_document_content
from gateway import verify_user_input, verify_agent_output

load_dotenv()

# 1. Define the Shared State with continuous message memory AND thread_id preservation
class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    thread_id: Optional[str]

AGENT_RULEBOOK_PATH = "./agent.md"

def invoke_orchestrator_llm(system_instruction: str, history_messages: list):
    """
    Routes orchestrator completions through SmartLLMGateway with automatic
    retry, caching, and cross-provider fallback (Groq -> Gemini).
    """
    formatted = [{"role": "system", "content": system_instruction}]
    # Only send the last 6 messages to keep token latency low
    recent_messages = history_messages[-6:] if len(history_messages) > 6 else history_messages
    for m in recent_messages:
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

def resolve_Requested_filename(user_message: str, default_stem: str = "research_report") -> str:
    """
    Extracts an explicit filename or infers the proper file extension (.docx for 'word document',
    .md for 'markdown', .csv for 'csv', .txt otherwise) and builds a clean filename.
    """
    explicit_match = re.search(r'[\w\-]+\.(txt|md|docx?|csv|json|py)', user_message, re.IGNORECASE)
    if explicit_match:
        filename = explicit_match.group(0)
        if filename.lower().endswith(".doc"):
            filename = filename[:-4] + ".docx"
        return filename

    msg_lower = user_message.lower()
    if any(k in msg_lower for k in ["word document", "word file", "ms word", "microsoft word", "docx", "as word", "in word", ".doc"]):
        ext = "docx"
    elif any(k in msg_lower for k in ["markdown", ".md"]):
        ext = "md"
    elif "csv" in msg_lower:
        ext = "csv"
    elif "json" in msg_lower:
        ext = "json"
    else:
        ext = "txt"

    stop_words = {
        "save", "the", "previous", "into", "file", "to", "a", "an", "as", "format",
        "doc", "docx", "txt", "md", "word", "document", "also", "that", "this",
        "in", "short", "brief", "please", "and", "write", "create", "export",
        "what", "is", "are", "give", "me", "tell", "about", "current", "latest"
    }
    words = [w for w in re.findall(r'[a-zA-Z0-9]+', msg_lower) if w not in stop_words and len(w) > 1]
    base_name = "_".join(words[:4]) if words else default_stem
    return f"{base_name}.{ext}"

def strip_save_instructions_for_research(user_message: str) -> str:
    """
    Removes file-saving phrasing (like 'also save that news as word document in short')
    before passing the query to hybrid_research_worker when the user is NOT asking about
    an uploaded PDF/resume.
    """
    msg_lower = user_message.lower()
    if any(k in msg_lower for k in ["resume", "cv", "uploaded", "pdf", "deepan", "my document", "the document"]):
        return user_message

    cleaned = re.sub(
        r'\b(and\s+also|also|and|then|please)?\s*(save|write|export|store|create)\s+(that|this|the|it|those|these)?\s*(news|result|results|answer|response|information|summary|report|data)?\s*(as|into|in|to)\s+(a\s+|an\s+)?(word\s+document|word\s+file|docx|doc|text\s+file|txt|markdown|md|file|document)(\s+in\s+short|\s+briefly)?\b',
        '',
        user_message,
        flags=re.IGNORECASE
    ).strip()

    if "word document" in cleaned.lower() and any(w in msg_lower for w in ["save", "write", "export"]):
        cleaned = re.sub(r'\bword\s+document\b', 'file', cleaned, flags=re.IGNORECASE).strip()

    return cleaned if len(cleaned) >= 3 else user_message

# 3. Define the Core Manager Logic Node
def main_orchestrator_node(state: AgentState, config: dict = None):
    last_user_message = state["messages"][-1].content

    # Reliably extract thread_id from state OR LangGraph config so user 'carbon' never falls back to 'default'
    thread_id = state.get("thread_id")
    if not thread_id and config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
    thread_id = (thread_id or "default").strip() or "default"
    
    # Load global rules
    rules = ""
    if os.path.exists(AGENT_RULEBOOK_PATH):
        with open(AGENT_RULEBOOK_PATH, "r", encoding="utf-8") as f:
            rules = f.read()

    print(f"\n[Main Agent] Analyzing request and managing context (User: '{thread_id}')...", flush=True)
    
    # --- FAST-PATH & INTENT CLASSIFIER / ROUTER ---
    clean_msg = last_user_message.strip().lower()
    if clean_msg in ["hi", "hello", "hey", "good morning", "good evening", "how are you", "yo", "sup", "greetings"]:
        intent = "CHITCHAT"
    elif any(kw in clean_msg for kw in ["what did we discuss", "summarize our chat", "chat history", "previous message", "what was my last question", "conversation so far"]):
        intent = "CHAT_HISTORY"
    elif any(kw in clean_msg for kw in ["save previous", "save the chat", "save our conversation", "export chat", "save history"]):
        intent = "SAVE_HISTORY"
    elif any(kw in clean_msg for kw in ["resume", "cv", "pdf", "deepan", "document", "uploaded", "code", "python", "function", "who is", "what is", "how to", "search", "explain", "tell me", "say about", "summary", "summarize", "skills", "experience", "project", "education"]):
        intent = "RESEARCH"
    else:
        router_prompt = (
            "Classify the user's prompt into one of four categories: 'CHITCHAT', 'CHAT_HISTORY', 'SAVE_HISTORY', or 'RESEARCH'.\n"
            "- Use 'CHITCHAT' for greetings, casual conversation, jokes, expressions of emotion, or simple pleasantries.\n"
            "- Use 'CHAT_HISTORY' if the user asks what we discussed, asks to summarize our previous conversation, asks what their previous question was, or references past interactions between you and them.\n"
            "- Use 'SAVE_HISTORY' ONLY if the user explicitly asks to save, export, or write the PREVIOUS conversation text, previous response, or chat history into a file/document.\n"
            "- Use 'RESEARCH' if the user asks for new news, facts, uploaded PDF context, technical documentation, coding, calculations, or general questions.\n\n"
            "Return ONLY the word 'CHITCHAT', 'CHAT_HISTORY', 'SAVE_HISTORY', or 'RESEARCH' and absolutely nothing else.\n\n"
            f"User Prompt: {last_user_message}"
        )
        
        intent_response = smart_gateway.completion(
            messages=[{"role": "user", "content": router_prompt}], 
            override_model="openai/gpt-oss-20b"
        )
        intent = intent_response.content.strip().upper()
    
    # Route A: Conversational Path
    if "CHITCHAT" in intent:
        print("[Main Agent] Intent identified: Conversational Chit-Chat. Bypassing Sub-Agent RAG.", flush=True)
        
        system_instruction = (
            f"{rules}\n\n"
            "You are a helpful AI assistant. Respond to the user's greeting or statement naturally, "
            "warmly, and conversationally. Do not include fact sheets or source links."
        )
        
        response = invoke_orchestrator_llm(system_instruction, state["messages"])
        return {"messages": [response], "thread_id": thread_id}

    # Route A2: Conversation Memory & Chat History Path
    elif "CHAT_HISTORY" in intent:
        print("[Main Agent] Intent identified: Chat History / Memory Summary. Reading directly from conversation state...", flush=True)
        
        history_transcript = []
        past_turns = state["messages"][:-1]
        
        for m in past_turns:
            role = getattr(m, "type", None) or getattr(m, "role", "user")
            speaker = "User" if role in ["human", "user"] else "Assistant"
            content = getattr(m, "content", "")
            if content:
                history_transcript.append(f"{speaker}: {content}")

        if not history_transcript:
            return {"messages": [invoke_orchestrator_llm(
                "Inform the user politely that this is the beginning of the chat session, so there are no earlier conversation topics logged yet.",
                state["messages"]
            )], "thread_id": thread_id}

        past_context = "\n\n".join(history_transcript)
        system_instruction = (
            f"{rules}\n\n"
            f"ACTUAL CONVERSATION LOG BETWEEN USER AND ASSISTANT:\n"
            f"{past_context}\n\n"
            f"CRITICAL INSTRUCTIONS:\n"
            f"1. Summarize ONLY the actual questions and answers discussed above between you and the user.\n"
            f"2. DO NOT perform web searches, and do NOT talk about third-party books or external articles.\n"
            f"3. Present the summary clearly using clean Markdown bullet points (- ) and concise points."
        )
        
        response = invoke_orchestrator_llm(system_instruction, state["messages"])
        return {"messages": [response], "thread_id": thread_id}
        
    # Route B: Save History Context Path
    elif "SAVE_HISTORY" in intent:
        print("[Main Agent] Intent identified: Save History Context. Packaging conversation history for File Saver...", flush=True)
        
        previous_agent_text = "No previous content found to save."
        if len(state["messages"]) > 1:
            previous_agent_text = state["messages"][-2].content
            
        filename = resolve_Requested_filename(last_user_message, default_stem="saved_document")
        clean_text = clean_document_content(previous_agent_text)

        save_result = save_file_to_disk.invoke({"filename": filename, "content": clean_text})
        print(f"[Main Agent] Physical file saved to disk: {save_result}", flush=True)

        system_instruction = (
            f"{rules}\n\n"
            f"FILE SAVE EXECUTION RESULT:\n{save_result}\n\n"
            f"CRITICAL: Inform the user that their request has been executed and the file '{filename}' was successfully saved. "
            f"Explicitly inform them that they can view, download, or delete it from the 'Saved Documents' section in the left sidebar."
        )
        
        response = invoke_orchestrator_llm(system_instruction, state["messages"])
        return {"messages": [response], "thread_id": thread_id}
        
    # Route C: Tool and Document Research Path
    else:
        print(f"[Main Agent] Intent identified: Fact/Tool Dependent. Activating Research Sub-Agent for user '{thread_id}'...", flush=True)
        
        research_query = strip_save_instructions_for_research(last_user_message)
        fact_sheet = hybrid_research_worker(research_query, thread_id=thread_id)
        
        saved_filename_note = ""
        if any(w in last_user_message.lower() for w in ["save", "create a file", "write to file", "word document", "word file", ".txt", ".md", ".doc", ".docx"]):
            target_filename = resolve_Requested_filename(last_user_message, default_stem="research_report")
            file_content = f"Research Summary:\n{fact_sheet.summary}\n\nKey Verified Facts:\n" + "\n".join(f"- {f}" for f in fact_sheet.verified_facts)
            save_file_to_disk.invoke({"filename": target_filename, "content": file_content})
            print(f"[Main Agent] Physical file saved to disk: {target_filename}", flush=True)
            saved_filename_note = f"File Saved to Disk: {target_filename}\n"

        research_context = (
            f"{saved_filename_note}"
            f"Research Summary: {fact_sheet.summary}\n"
            f"Extracted Facts: {', '.join(fact_sheet.verified_facts)}\n"
            f"Source Links/Papers: {', '.join(fact_sheet.sources_used)}\n"
        )
        
        system_instruction = (
            f"{rules}\n\n"
            f"CONTEXT FACTS PROVIDED BY SUB-AGENT:\n{research_context}\n\n"
            f"CRITICAL RULES:\n"
            f"1. Base your answer strictly on the provided context facts. Do NOT hallucinate or mix unrelated facts.\n"
            f"2. Do NOT call any tools or output JSON function calls; the sub-agent has already run.\n"
            f"3. Never use raw HTML tags such as <br>, <br/>, or <div>. Use Markdown bullet points (- ) and paragraphs instead.\n"
            f"4. Directly address the user's specific inquiry. If the facts do not answer the question, state clearly that the information is unavailable in the retrieved sources.\n"
            f"5. If a file was saved to disk, confirm the exact filename saved and remind the user they can download it from the 'Saved Documents' sidebar."
        )
        
        raw_response = invoke_orchestrator_llm(system_instruction, state["messages"])
        
        if fact_sheet.verified_facts:
            verified_content = verify_agent_output(raw_response.content, fact_sheet.verified_facts)
            raw_response.content = verified_content
            
        return {"messages": [raw_response], "thread_id": thread_id}

# 4. Build and Compile the LangGraph Workflow
workflow = StateGraph(AgentState)
workflow.add_node("orchestrator", main_orchestrator_node)
workflow.add_edge(START, "orchestrator")
workflow.add_edge("orchestrator", END)

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
                {"messages": [{"role": "user", "content": user_query}], "thread_id": "active_developer_session"},
                config=session_config
            )
            
            final_response = state_output["messages"][-1].content
            print(f"\nAgent: {final_response}\n")
            print("-" * 50)
            
        except Exception as e:
            print(f"\nAn error occurred in the execution loop: {str(e)}\n")