import os
import re
import json
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
    recent_messages = history_messages[-6:] if len(history_messages) > 6 else history_messages
    for m in recent_messages:
        role = getattr(m, "type", None) or getattr(m, "role", "user")
        r = "user" if role in ["human", "user"] else "assistant"
        content = getattr(m, "content", "")
        if content:
            formatted.append({"role": r, "content": str(content)})

    return smart_gateway.completion(
        messages=formatted,
        override_model="openai/gpt-oss-20b"
    )


def build_recent_conversation_digest(messages: list, max_turns: int = 6) -> str:
    """
    Builds a compact digest of recent conversation turns so the LLM Router can
    semantically understand what 'this', 'that', 'it', 'the previous news', or
    'our conversation' refers to without relying on hardcoded keywords.
    """
    past_messages = messages[:-1]
    if not past_messages:
        return "No prior conversation history in this session."

    recent = past_messages[-max_turns:]
    digest_lines = []
    for idx, m in enumerate(recent, 1):
        role = getattr(m, "type", None) or getattr(m, "role", "user")
        speaker = "User" if role in ["human", "user"] else "Assistant"
        content = str(getattr(m, "content", "") or "").strip()
        # Keep a concise preview (first 350 chars) so the router sees the topic of each turn
        preview = re.sub(r'\s+', ' ', content)[:350]
        if len(content) > 350:
            preview += "..."
        digest_lines.append(f"[Turn {idx} - {speaker}]: {preview}")

    return "\n".join(digest_lines)


def analyze_user_intent_with_llm(user_message: str, all_messages: list) -> dict:
    """
    100% LLM-Driven Semantic Intent & Action Analyzer (Zero Keyword Lists).
    The LLM reads both the recent conversation history and the user's new prompt
    to understand the true semantic intent and extract structured execution parameters:
    - intent:
        * 'CHITCHAT'     -> Casual greeting, small talk, pleasantries, thank you, etc.
        * 'CHAT_HISTORY' -> User is asking about earlier messages in the conversation
                            (e.g., summarizing chat, asking what they asked earlier, recalling past discussion).
        * 'SAVE_HISTORY' -> User wants to save/export/convert an ALREADY generated response or
                            earlier topic from the conversation history into a file/document.
        * 'RESEARCH'     -> User is asking for new information, news, uploaded PDF/RAG questions,
                            coding, analysis, or asking a new question (with or without saving).
    - also_save_to_file: True if the user wants the output saved to a file/document.
    - file_extension: 'docx' (if Word/doc/document requested), 'txt', 'md', 'csv', or 'json'.
    - suggested_filename: Clean, meaningful filename inferred by the LLM based on the actual topic
                          (e.g. 'karur_news_highlights.docx'), never conversational words.
    - standalone_search_query: If intent is 'RESEARCH', a clean search/research query stripped of
                               any file-saving instructions.
    - target_history_topic: If intent is 'SAVE_HISTORY', a brief description of which prior assistant
                            response the user wants saved (e.g. 'Karur news highlights').
    """
    conversation_digest = build_recent_conversation_digest(all_messages, max_turns=6)

    analyzer_prompt = (
        "You are the Semantic Intent & Context Analyzer for an intelligent Agentic RAG system.\n"
        "Carefully read the RECENT CONVERSATION HISTORY and the CURRENT USER MESSAGE.\n"
        "Do NOT rely on surface keywords — understand the user's true meaning in context.\n\n"
        "Classify the user's request and extract parameters into a single valid JSON object with these exact keys:\n"
        "{\n"
        '  "intent": "CHITCHAT" | "CHAT_HISTORY" | "SAVE_HISTORY" | "RESEARCH",\n'
        '  "also_save_to_file": true | false,\n'
        '  "file_extension": "docx" | "txt" | "md" | "csv" | "json",\n'
        '  "suggested_filename": "clean_descriptive_name_with_extension",\n'
        '  "standalone_search_query": "clean research question without save instructions",\n'
        '  "target_history_topic": "topic of the prior response the user wants to save"\n'
        "}\n\n"
        "SEMANTIC ROUTING RULES:\n"
        "1. 'SAVE_HISTORY': Choose this whenever the user is asking to save, export, put, write, or convert "
        "information that was ALREADY provided in an earlier Assistant turn into a file/document/Word/txt "
        "(for example: 'save this current news into document', 'no I am asking you save the previous news into word document', "
        "'make a word file of that', 'put the above points in a txt file', 'export what you just told me'). "
        "Even if the user says 'current news' or 'document', if that news was ALREADY fetched in the conversation history "
        "and the user is now asking to save it, the intent MUST be 'SAVE_HISTORY' (NOT 'RESEARCH').\n"
        "2. 'CHAT_HISTORY': Choose this whenever the user asks about the conversation itself — e.g., what we talked about, "
        "what their previous question was, to recap/summarize our chat history, or what happened earlier in the session.\n"
        "3. 'CHITCHAT': Choose this for greetings, casual pleasantries, thank-yous, or small talk that need no research.\n"
        "4. 'RESEARCH': Choose this when the user is asking for NEW information, asking about their uploaded PDF/resume/document, "
        "asking for live news on a new topic, coding, or factual questions not yet answered in the chat.\n"
        "   - If the user asks a new research question AND asks to save it in the same message (e.g. 'What is the news in Chennai and save it as a Word doc'), "
        "set 'intent' to 'RESEARCH', 'also_save_to_file' to true, and put ONLY the pure research question ('What is the current news in Chennai?') "
        "in 'standalone_search_query' so the search engine never searches for 'how to save a document'.\n"
        "5. FILENAME & FORMAT SELECTION:\n"
        "   - If the user mentions 'word', 'docx', 'doc', or generic 'document', set 'file_extension' to 'docx'.\n"
        "   - If the user mentions 'text file' or 'txt', set 'file_extension' to 'txt'.\n"
        "   - If the user mentions 'markdown' or 'md', set 'file_extension' to 'md'.\n"
        "   - Generate 'suggested_filename' based on the ACTUAL SUBJECT MATTER (e.g. 'karur_current_news.docx' or 'resume_summary.docx'), "
        "NEVER from conversational phrasing like 'no_am_asking_you.docx'.\n\n"
        f"RECENT CONVERSATION HISTORY:\n{conversation_digest}\n\n"
        f"CURRENT USER MESSAGE:\n{user_message}\n\n"
        "Respond ONLY with the raw JSON object and nothing else."
    )

    try:
        res = smart_gateway.completion(
            messages=[{"role": "user", "content": analyzer_prompt}],
            override_model="llama-3.3-70b-versatile"
        )
        raw_text = res.content.strip()
        raw_text = re.sub(r'^```(?:json)?\s*', '', raw_text)
        raw_text = re.sub(r'\s*```$', '', raw_text).strip()

        start_idx = raw_text.find("{")
        end_idx = raw_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            parsed = json.loads(raw_text[start_idx:end_idx + 1])
            intent = str(parsed.get("intent", "RESEARCH")).strip().upper()
            if intent not in {"CHITCHAT", "CHAT_HISTORY", "SAVE_HISTORY", "RESEARCH"}:
                intent = "RESEARCH"

            ext = str(parsed.get("file_extension", "docx")).strip().lower().lstrip(".")
            if ext in ("doc", "word", "document"):
                ext = "docx"
            elif ext not in ("docx", "txt", "md", "csv", "json"):
                ext = "docx"

            raw_filename = str(parsed.get("suggested_filename") or "saved_report").strip()
            safe_stem = re.sub(r'[^a-zA-Z0-9_\-]', '_', os.path.splitext(os.path.basename(raw_filename))[0]).strip('_').lower()
            if not safe_stem:
                safe_stem = "saved_report"
            final_filename = f"{safe_stem}.{ext}"

            return {
                "intent": intent,
                "also_save_to_file": bool(parsed.get("also_save_to_file", intent == "SAVE_HISTORY")),
                "file_extension": ext,
                "suggested_filename": final_filename,
                "standalone_search_query": str(parsed.get("standalone_search_query") or user_message).strip(),
                "target_history_topic": str(parsed.get("target_history_topic") or "").strip(),
            }
    except Exception as e:
        print(f"[Main Agent Warning] Semantic analyzer fallback notice: {e}", flush=True)

    return {
        "intent": "RESEARCH",
        "also_save_to_file": False,
        "file_extension": "docx",
        "suggested_filename": "research_report.docx",
        "standalone_search_query": user_message,
        "target_history_topic": "",
    }


def select_relevant_history_content_with_llm(messages: list, user_message: str, target_topic: str = "") -> str:
    """
    Uses the LLM to pick or extract the exact substantive Assistant response from the conversation
    history that the user wants saved — even if there were intermediate clarification turns or errors
    between the original answer and the save request.
    """
    assistant_turns = []
    for m in messages[:-1]:
        role = getattr(m, "type", None) or getattr(m, "role", "")
        if role in ["ai", "assistant"]:
            content = str(getattr(m, "content", "") or "").strip()
            if content:
                assistant_turns.append(content)

    if not assistant_turns:
        return "No previous assistant response found in the conversation history."

    if len(assistant_turns) == 1:
        return assistant_turns[0]

    # Build a numbered list of prior assistant responses so the LLM selects the right one
    numbered_candidates = []
    for idx, text in enumerate(assistant_turns):
        numbered_candidates.append(f"=== CANDIDATE [{idx}] ===\n{text[:1500]}\n")

    selector_prompt = (
        "The user wants to save a previous assistant response from the conversation into a document.\n"
        f"User Request: {user_message}\n"
        f"Target Topic: {target_topic or 'Substantive answer requested by the user'}\n\n"
        "Below are the prior Assistant responses from the conversation.\n"
        "Select the index number (e.g. 0, 1, 2...) of the candidate that contains the ACTUAL substantive content "
        "(such as the news highlights, research report, or answer) that the user wants saved.\n"
        "NEVER select a candidate that only gives generic instructions on how to press Ctrl+A / Ctrl+C or says a file was saved.\n"
        "Reply with ONLY the integer index number and nothing else.\n\n"
        + "\n".join(numbered_candidates)
    )

    try:
        sel_res = smart_gateway.completion(
            messages=[{"role": "user", "content": selector_prompt}],
            override_model="openai/gpt-oss-20b"
        )
        match = re.search(r'\d+', sel_res.content.strip())
        if match:
            chosen_idx = int(match.group(0))
            if 0 <= chosen_idx < len(assistant_turns):
                return assistant_turns[chosen_idx]
    except Exception as e:
        print(f"[Main Agent Warning] History selector fallback: {e}", flush=True)

    return assistant_turns[-1]


# 3. Define the Core Manager Logic Node
def main_orchestrator_node(state: AgentState, config: dict = None):
    last_user_message = state["messages"][-1].content

    # Reliably extract thread_id from state OR LangGraph config so users are strictly isolated
    thread_id = state.get("thread_id")
    if not thread_id and config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
    thread_id = (thread_id or "default").strip() or "default"

    # Load global rules
    rules = ""
    if os.path.exists(AGENT_RULEBOOK_PATH):
        with open(AGENT_RULEBOOK_PATH, "r", encoding="utf-8") as f:
            rules = f.read()

    print(f"\n[Main Agent] Semantic LLM Analyzer evaluating request in context (User: '{thread_id}')...", flush=True)

    # --- 100% LLM-DRIVEN SEMANTIC INTENT & PARAMETER ANALYSIS ---
    analysis = analyze_user_intent_with_llm(last_user_message, state["messages"])
    intent = analysis["intent"]
    print(
        f"[Main Agent] LLM Decision -> Intent: {intent} | Save File: {analysis['also_save_to_file']} "
        f"| Filename: {analysis['suggested_filename']}",
        flush=True
    )

    # Route A: Conversational Path
    if intent == "CHITCHAT":
        print("[Main Agent] Intent identified: Conversational Chit-Chat. Bypassing Sub-Agent RAG.", flush=True)

        system_instruction = (
            f"{rules}\n\n"
            "You are a helpful AI assistant. Respond to the user's greeting or statement naturally, "
            "warmly, and conversationally. Do not include fact sheets or source links."
        )

        response = invoke_orchestrator_llm(system_instruction, state["messages"])
        return {"messages": [response], "thread_id": thread_id}

    # Route A2: Conversation Memory & Chat History Path
    elif intent == "CHAT_HISTORY":
        print("[Main Agent] Intent identified: Chat History / Memory Query. Reading full conversation state...", flush=True)

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

        saved_note = ""
        if analysis["also_save_to_file"]:
            filename = analysis["suggested_filename"]
            save_res = save_file_to_disk.invoke({
                "filename": filename,
                "content": past_context,
                "thread_id": thread_id
            })
            print(f"[Main Agent] Physical file saved to disk for user '{thread_id}': {save_res}", flush=True)
            saved_note = f"\nFILE SAVED TO DISK: '{filename}' has been saved to the user's Saved Documents sidebar.\n"

        system_instruction = (
            f"{rules}\n\n"
            f"ACTUAL CONVERSATION LOG BETWEEN USER AND ASSISTANT:\n"
            f"{past_context}\n"
            f"{saved_note}\n"
            f"CRITICAL INSTRUCTIONS:\n"
            f"1. Answer the user's question about the conversation history accurately based ONLY on the log above.\n"
            f"2. DO NOT perform web searches, and do NOT talk about third-party books or external articles.\n"
            f"3. Present the answer clearly using clean Markdown bullet points (- ) and concise points."
        )

        response = invoke_orchestrator_llm(system_instruction, state["messages"])
        return {"messages": [response], "thread_id": thread_id}

    # Route B: Save History Context Path
    elif intent == "SAVE_HISTORY":
        print(f"[Main Agent] Intent identified: Save Prior Response for user '{thread_id}'. Selecting relevant turn via LLM...", flush=True)

        previous_agent_text = select_relevant_history_content_with_llm(
            state["messages"],
            user_message=last_user_message,
            target_topic=analysis["target_history_topic"]
        )
        filename = analysis["suggested_filename"]
        clean_text = clean_document_content(previous_agent_text)

        save_result = save_file_to_disk.invoke({
            "filename": filename,
            "content": clean_text,
            "thread_id": thread_id
        })
        print(f"[Main Agent] Physical file saved to disk for user '{thread_id}': {save_result}", flush=True)

        system_instruction = (
            f"{rules}\n\n"
            f"FILE SAVE EXECUTION RESULT:\n{save_result}\n\n"
            f"SAVED CONTENT PREVIEW:\n{clean_text[:1500]}\n\n"
            f"CRITICAL: Confirm clearly to the user that the content has been saved into '{filename}' "
            f"and is now ready to download in the 'Saved Documents' section in the left sidebar. "
            f"Do NOT give manual copy-paste (Ctrl+A / Ctrl+C) instructions."
        )

        response = invoke_orchestrator_llm(system_instruction, state["messages"])
        return {"messages": [response], "thread_id": thread_id}

    # Route C: Tool and Document Research Path
    else:
        research_query = analysis["standalone_search_query"] or last_user_message
        print(f"[Main Agent] Intent identified: Research. Querying Sub-Agent with '{research_query}' (User: '{thread_id}')...", flush=True)

        fact_sheet = hybrid_research_worker(research_query, thread_id=thread_id)

        saved_filename_note = ""
        if analysis["also_save_to_file"]:
            target_filename = analysis["suggested_filename"]
            title_heading = target_filename.rsplit(".", 1)[0].replace("_", " ").title()
            file_content = (
                f"# {title_heading}\n\n"
                f"{fact_sheet.summary}\n\n"
                f"## Key Verified Highlights\n"
                + "\n".join(f"- {f}" for f in fact_sheet.verified_facts)
            )
            if fact_sheet.sources_used:
                file_content += "\n\n## Sources\n" + "\n".join(f"- {s}" for s in fact_sheet.sources_used)

            save_file_to_disk.invoke({
                "filename": target_filename,
                "content": file_content,
                "thread_id": thread_id
            })
            print(f"[Main Agent] Physical file saved to disk for user '{thread_id}': {target_filename}", flush=True)
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