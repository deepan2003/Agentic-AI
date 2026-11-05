import os
import re
import json
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from tools.file_system import save_file_to_disk
from tools.code_executor import execute_python_code
from tools.research import query_local_research_papers
from tools.web_search import custom_web_search, custom_web_fetch

load_dotenv()

# Define the strict output format the Sub-Agent must use
class FactSheet(BaseModel):
    summary: str = Field(description="A concise summary of the core answer found.")
    verified_facts: List[str] = Field(description="List of hard factual points extracted from sources.")
    sources_used: List[str] = Field(description="List of document names or URLs actually utilized.")

def parse_fact_sheet(raw_content: str, combined_context: str) -> FactSheet:
    """
    Safely parses the FactSheet JSON output from the LLM, avoiding fragile
    tool-call JSON parsing errors on Groq.
    """
    # 1. Attempt regex extraction of the JSON object
    match = re.search(r'\{[\s\S]*\}', raw_content)
    if match:
        json_candidate = match.group(0)
        try:
            data = json.loads(json_candidate)
            summary = str(data.get("summary", "")).strip() or "Summary of gathered research."
            raw_facts = data.get("verified_facts", [])
            if not isinstance(raw_facts, list):
                raw_facts = [str(raw_facts)]
            raw_sources = data.get("sources_used", [])
            if not isinstance(raw_sources, list):
                raw_sources = [str(raw_sources)]
                
            clean_facts = [str(f).strip() for f in raw_facts if str(f).strip()]
            clean_sources = [str(s).strip() for s in raw_sources if str(s).strip()]
            
            if clean_facts:
                return FactSheet(
                    summary=summary,
                    verified_facts=clean_facts,
                    sources_used=clean_sources
                )
        except Exception:
            pass

    # 2. Resilient fallback parsing: extract bullet points or sentences
    lines = []
    for line in raw_content.splitlines():
        cleaned = line.strip().lstrip("-*•0123456789. ")
        if len(cleaned) > 12 and not cleaned.startswith("{") and not cleaned.startswith("}"):
            lines.append(cleaned)
            
    summary = lines[0] if lines else "Research findings from retrieved context."
    facts = lines[:15] if lines else ["Verified factual data retrieved from live research."]
    
    return FactSheet(
        summary=summary,
        verified_facts=facts,
        sources_used=[]
    )

def hybrid_research_worker(query: str, thread_id: str = "default") -> FactSheet:
    print(f"\n[Sub-Agent] Starting research for: '{query}' (User: '{thread_id}')", flush=True)
    
    # 1. Initialize Groq LLM early for internal evaluations
    llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0
    )
    
    # ====================================================================
    # 2. 📚 STEP 1: ALWAYS SEARCH LOCAL RAG FIRST
    # ====================================================================
    print(f"[Sub-Agent] Checking user '{thread_id}' RAG database first...", flush=True)
    query_lower = query.lower()
    is_doc_query = any(ind in query_lower for ind in [
        "resume", "cv", "pdf", "paper", "document", "uploaded", "rag", "system", 
        "profile", "candidate", "deepan", "experience", "education", "skill", "who is", "role", "fit"
    ])

    local_data = query_local_research_papers(query, top_k=5, thread_id=thread_id)

    # If document query and local data is still empty, auto-build from Cloudinary/uploads immediately
    if is_doc_query and not local_data:
        print(f"[Sub-Agent] Document query detected with empty index. Running auto-build for '{thread_id}'...", flush=True)
        from ingestion import build_local_vector_db
        build_local_vector_db(thread_id=thread_id)
        local_data = query_local_research_papers(query, top_k=5, thread_id=thread_id)
    
    # If user asks for general document overview, expand search with profile keywords
    if is_doc_query and len(local_data) < 3:
        broad_data = query_local_research_papers("Deepan AI Engineer Computer Vision GenAI experience skills education projects", top_k=5, thread_id=thread_id)
        seen_snippets = {d.get("content_snippet", "") for d in local_data}
        for bd in broad_data:
            if bd.get("content_snippet", "") not in seen_snippets and "error" not in bd:
                local_data.append(bd)
                seen_snippets.add(bd.get("content_snippet", ""))

    local_text_snippet = ""
    for doc in local_data:
        if "error" not in doc:
            local_text_snippet += f"{doc.get('content_snippet', '')}\n"

    # Smart Relevancy Check: Does the local PDF answer or relate to the query?
    is_local_relevant = False
    if local_text_snippet.strip():
        if is_doc_query:
            # User specifically asked about the document / resume / RAG
            is_local_relevant = True
        else:
            relevancy_prompt = (
                f"User Question: {query}\n"
                f"Local Document Snippets:\n{local_text_snippet[:2000]}\n\n"
                f"Does the Local Document contain information related to or answering the user's question? "
                f"Answer 'YES' if it relates or can answer; answer 'NO' ONLY if completely unrelated. Answer ONLY 'YES' or 'NO'."
            )
            try:
                check_response = llm.invoke(relevancy_prompt).content.strip().upper()
                if "YES" in check_response:
                    is_local_relevant = True
            except Exception:
                is_local_relevant = True

    # Containers for search findings
    web_data = []
    combined_context = ""

    # ====================================================================
    # 3. 🌐 STEP 2: RAG MATCH vs LIVE TAVILY SEARCH FALLBACK
    # ====================================================================
    if is_local_relevant:
        print("[Sub-Agent] ✅ RAG Match: Context found in uploaded PDF(s). Bypassing live web crawling.", flush=True)
        combined_context += "--- LOCAL KNOWLEDGE BASE (PDF) EXTRACTS ---\n"
        for doc in local_data:
            if "error" not in doc:
                combined_context += f"Source: {doc.get('source_file', 'Uploaded PDF')}\nContent: {doc.get('content_snippet', '')}\n\n"
    elif is_doc_query:
        # STRICT RULE: NEVER search the public web for a private user's resume or CV!
        print("[Sub-Agent] 🛡️ Resume/Document query with no indexed text. Banning live web search to prevent public confusion.", flush=True)
        combined_context += (
            "--- LOCAL KNOWLEDGE BASE (PDF) ---\n"
            "The requested information is regarding the user's private uploaded document/resume. "
            "However, the document index is currently empty. Please ensure the PDF is indexed via 'Sync Cloud Assets to RAG System'. "
            "Do NOT fabricate information or search the public web for other people.\n\n"
        )
    else:
        print("[Sub-Agent] 🌐 Non-document query with no local RAG match. Falling back to Tavily live web search...", flush=True)
        web_links = custom_web_search(query, max_results=1)
        for link in web_links:
            if "url" in link:
                content = custom_web_fetch(link["url"])
                web_data.append(content)

        combined_context += "--- LIVE WEB EXTRACTS ---\n"
        for page in web_data:
            combined_context += f"Source: {page['url']}\nContent: {page['content']}\n\n"

        from datetime import datetime
        current_year = datetime.now().year
        optimizer_prompt = f"""
        You are an expert search query generator. CURRENT SYSTEM YEAR: {current_year}.
        Convert the user's conversational question into a strict, highly targeted search engine keyword string.
        User Question: {query}
        """
        try:
            optimized_query = llm.invoke(optimizer_prompt).content.strip()
            tavily_search = TavilySearch(max_results=1, search_depth="basic", include_raw_content=False)
            tavily_data = tavily_search.invoke({"query": optimized_query})
            combined_context += f"--- TAVILY LIVE NEWS ---\n{tavily_data}\n\n"
        except Exception as e:
            print(f"[Sub-Agent Warning] Tavily search error: {e}", flush=True)

    # ====================================================================
    # 4. 💻 STEP 3: CODING TASK PRIORITY (CHECK RAG FIRST -> RUN CODE TOOL)
    # ====================================================================
    is_coding_task = any(kw in query_lower for kw in ["code", "python", "script", "function", "algorithm", "calculate", "math", "run "])
    if is_coding_task:
        has_code_in_rag = is_local_relevant and any(marker in local_text_snippet.lower() for marker in [
            "def ", "class ", "import ", "return ", "for ", "while ", "function", "lambda", "const "
        ])
        user_wants_execution = any(act in query_lower for act in ["execute", "run ", "test the code", "calculate", "compute", "solve"])

        if has_code_in_rag and not user_wants_execution:
            print("[Sub-Agent] Code found in local PDF RAG context. Using document code.", flush=True)
        else:
            print("[Sub-Agent] Code not in PDF or execution requested. Running Python sandbox tool...", flush=True)
            tool_agent = create_react_agent(llm, tools=[execute_python_code, save_file_to_disk])
            tool_instructions = (
                f"You are a coding and execution agent.\n"
                f"- If math calculation, code generation, or execution is needed, run it using 'execute_python_code'.\n"
                f"- If the user asks to save/create a file, use 'save_file_to_disk'.\n"
                f"CONTEXT:\n{combined_context}\n\n"
                f"User Query: {query}"
            )
            try:
                tool_results = tool_agent.invoke(
                    {"messages": [("user", tool_instructions)]},
                    config={"recursion_limit": 5}
                )
                final_tool_answer = tool_results["messages"][-1].content
                combined_context += f"--- PYTHON CODE EXECUTION RESULTS ---\n{final_tool_answer}\n\n"
            except Exception as te:
                print(f"[Sub-Agent Warning] Tool execution note: {te}", flush=True)

    # File save tool trigger if requested and not yet run
    if any(kw in query_lower for kw in ["save", "write to", "create a file", "export"]) and "save_file_to_disk" not in combined_context:
        print("[Sub-Agent] File creation requested. Invoking save_file_to_disk tool...", flush=True)
        tool_agent = create_react_agent(llm, tools=[save_file_to_disk])
        try:
            tool_results = tool_agent.invoke(
                {"messages": [("user", f"Save the required content using save_file_to_disk.\nContext: {combined_context}\nQuery: {query}")]},
                config={"recursion_limit": 3}
            )
            combined_context += f"--- FILE SAVE RESULT ---\n{tool_results['messages'][-1].content}\n\n"
        except Exception as fe:
            print(f"[Sub-Agent Warning] File save note: {fe}", flush=True)

    # ====================================================================
    # 5. 📋 STEP 4: FINAL OUTPUT STRUCTURING (FACT SHEET)
    # ====================================================================
    system_prompt = (
        "You are an elite research assistant. Analyze the provided context extracts thoroughly.\n\n"
        "Generate a strictly formatted JSON object matching this structure:\n"
        "{\n"
        '  "summary": "A concise summary of the core answer found.",\n'
        '  "verified_facts": [\n'
        '    "Detailed factual headline or extract point 1",\n'
        '    "Detailed factual headline or extract point 2"\n'
        '  ],\n'
        '  "sources_used": ["source 1", "source 2"]\n'
        "}\n\n"
        "CRITICAL RULES:\n"
        "1. List EVERY SINGLE factual finding, skill, project, experience, or headline as an individual string in 'verified_facts'.\n"
        "2. Keep the JSON 100% syntactically valid. Escape internal quotes with \\\". Do not include markdown code fences.\n"
        "3. Output ONLY the JSON object and nothing else."
    )

    print("[Sub-Agent] Compiling facts into structured schema...", flush=True)
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{combined_context}\n\nQuery: {query}"}
    ])
    
    fact_sheet_object = parse_fact_sheet(response.content, combined_context)
    
    # Ensure source documents are listed
    if is_local_relevant:
        for doc in local_data:
            sf = doc.get("source_file", "")
            if sf and sf != "Unknown":
                base_sf = os.path.basename(sf)
                if base_sf not in fact_sheet_object.sources_used:
                    fact_sheet_object.sources_used.append(base_sf)

    print(f"[Sub-Agent] Research complete. Extracted {len(fact_sheet_object.verified_facts)} verified facts.", flush=True)
    return fact_sheet_object
