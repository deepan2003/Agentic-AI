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

def hybrid_research_worker(query: str) -> FactSheet:
    print(f"\n[Sub-Agent] Starting research for: '{query}'", flush=True)
    
    # 1. Initialize Groq LLM early for internal evaluations
    llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0
    )
    
    # 2. Gather data from your local PDF Vector DB
    print("[Sub-Agent] Searching local research database...", flush=True)
    local_data = query_local_research_papers(query, top_k=2)
    
    # Extract raw text snippet from local files to verify content relevance
    local_text_snippet = ""
    for doc in local_data:
        if "error" not in doc:
            local_text_snippet += f"{doc['content_snippet']}\n"
            
    # 3. 🧠 Smart Relevancy Check: Does the PDF actually answer the question?
    is_local_relevant = "NO"
    if local_text_snippet.strip():
        relevancy_prompt = (
            f"User Question: {query}\n"
            f"Local PDF Context:\n{local_text_snippet}\n\n"
            f"Does the Local PDF Context explicitly contain the factual answer to the User Question? "
            f"Answer ONLY with 'YES' or 'NO' and absolutely nothing else."
        )
        check_response = llm.invoke(relevancy_prompt).content.strip().upper()
        if "YES" in check_response:
            is_local_relevant = "YES"

    # Containers for search findings
    web_data = []
    tavily_data = ""
    combined_context = ""

    # 4. Routing Decision Branch
    if is_local_relevant == "YES":
        print("[Sub-Agent] Relevant match discovered in local PDFs. Bypassing live web crawling.", flush=True)
        
        combined_context += "--- LOCAL RESEARCH PAPERS EXTRACTS ---\n"
        for doc in local_data:
            if "error" not in doc:
                combined_context += f"Source: {doc['source_file']}\nContent: {doc['content_snippet']}\n\n"
    else:
        print("[Sub-Agent] Local PDFs do not contain the answer. Clear fallback to live web search...", flush=True)
        
        # Gather data from live web links
        print("[Sub-Agent] Searching the live web...", flush=True)
        web_links = custom_web_search(query, max_results=1)
        
        for link in web_links:
            if "url" in link:
                content = custom_web_fetch(link["url"])
                web_data.append(content)

        # Build dynamic context string for web fetches
        combined_context += "--- LIVE WEB EXTRACTS ---\n"
        for page in web_data:
            combined_context += f"Source: {page['url']}\nContent: {page['content']}\n\n"

        # Query Optimizer for Tavily Engine
        print("[Sub-Agent] Optimizing search query for Tavily...", flush=True)
        from datetime import datetime
        current_year = datetime.now().year
        
        optimizer_prompt = f"""
        You are an expert search query generator. 
        CURRENT SYSTEM YEAR: {current_year}.
        
        Convert the user's conversational question into a strict, highly targeted search engine keyword string.

        CRITICAL RULES:
        1. If the user asks about "current", "now", "present", or today's facts, you MUST append the year "{current_year}" to the search keywords to avoid outdated results.
        2. If the user explicitly asks about a specific past or future year (e.g., 2021, 2015), use their requested year INSTEAD of {current_year}.
        3. Do NOT include conversational words like "who is", "what is", "say in short", or "tell me".
        4. Only output the raw keywords.

        Examples: 
        - "who is current cm of tamilnadu" -> "current Chief Minister Tamil Nadu {current_year}"
        - "who was cm of tamilnadu in 2021" -> "Chief Minister Tamil Nadu 2021"

        User Question: {query}
        """
        optimized_query = llm.invoke(optimizer_prompt).content.strip()
        
        print("[Sub-Agent] Gathering live data from Tavily...", flush=True)
        tavily_search = TavilySearch(
            max_results=1,              # Only grab the single best search result
            search_depth="basic",        # Prevents slow, deep-web crawling
            include_raw_content=False    # Blocks massive HTML blocks from bloating the tokens
        )
        tavily_data = tavily_search.invoke({"query": optimized_query})
        combined_context += f"--- TAVILY LIVE NEWS ---\n{tavily_data}\n\n"

    # ====================================================================
    # 5. ACTIVE TOOL EXECUTION LOOP (Coding + File System)
    # ====================================================================
    needs_tools = any(kw in query.lower() for kw in ["code", "python", "calculate", "math", "save", "write to", "create a file", "export"])
    if needs_tools:
        print("[Sub-Agent] Active tool required. Invoking tool execution agent...", flush=True)
        tool_agent = create_react_agent(llm, tools=[execute_python_code, save_file_to_disk])
        
        tool_instructions = (
            f"You are a tool execution agent.\n"
            f"- If the user asks for math/coding, use 'execute_python_code'.\n"
            f"- If the user asks to save/create a file, use 'save_file_to_disk'.\n"
            f"- CRITICAL JSON RULE: When using 'save_file_to_disk', you MUST pass the content to the tool as ONE single, continuous paragraph without unescaped newlines or quotes.\n\n"
            f"RESEARCH CONTEXT:\n{combined_context}\n\n"
            f"User Query: {query}"
        )

        tool_results = tool_agent.invoke(
            {"messages": [("user", tool_instructions)]},
            config={"recursion_limit": 5}
        )
        final_tool_answer = tool_results["messages"][-1].content
        combined_context += f"--- ACTIVE TOOL RESULTS ---\n{final_tool_answer}\n\n"
    else:
        print("[Sub-Agent] Standard informational query. Skipping redundant tool agent to optimize latency and memory.", flush=True)
    # ====================================================================

    # 6. Final Output Structuring Block (Robust JSON Generation without fragile tool-calling)
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
        "1. List EVERY SINGLE news headline, statistical figure, or factual finding as an individual string in 'verified_facts'.\n"
        "2. Keep the JSON 100% syntactically valid. Escape any internal quotes with \\\". Do not include markdown code fences or backticks.\n"
        "3. Output ONLY the JSON object and nothing else."
    )

    print("[Sub-Agent] Compiling facts into structured schema...", flush=True)
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{combined_context}\n\nQuery: {query}"}
    ])
    
    fact_sheet_object = parse_fact_sheet(response.content, combined_context)
    print(f"[Sub-Agent] Research complete. Extracted {len(fact_sheet_object.verified_facts)} verified facts.", flush=True)
    return fact_sheet_object
