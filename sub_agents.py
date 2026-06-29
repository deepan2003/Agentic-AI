import os
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

def hybrid_research_worker(query: str) -> FactSheet:
    print(f"\n[Sub-Agent] Starting research for: '{query}'", flush=True)
    
    # 1. Initialize Groq LLM early for internal evaluations
    llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
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

        # Query Optimizer for Tavily Engine (Converted to real f-string format)
        # Query Optimizer for Tavily Engine
        print("[Sub-Agent] Optimizing search query for Tavily...", flush=True)
        
        # 1. Dynamically grab the real-world current year
        from datetime import datetime
        current_year = datetime.now().year
        
        # 2. Pass the dynamic year into the prompt with smart conditions
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
    print("[Sub-Agent] Checking if tool execution is required...", flush=True)
    tool_agent = create_react_agent(llm, tools=[execute_python_code, save_file_to_disk])
    
    tool_instructions = (
        f"You are a core research extractor and tool router.\n\n"
        f"TASK 1: FACT EXTRACTION (ALWAYS DO THIS)\n"
        f"- Extract ALL relevant factual news points (up to 10 points) from the context so nothing gets missed.\n\n"
        f"TASK 2: TOOL ROUTING\n"
        f"- If the user asks for math/coding, use 'execute_python_code'.\n"
        f"- If the user asks to save/create a file, use 'save_file_to_disk'.\n"
        f"- CRITICAL JSON RULE: When using 'save_file_to_disk', you MUST pass the content to the tool as ONE single, continuous paragraph. DO NOT use newlines (\\n), asterisks, bullet points, or unescaped quotes in the tool argument! It will crash the API.\n"
        f"- If the user just asks a question, do NOT use any tools. Just return your extracted facts.\n\n"
        f"RESEARCH CONTEXT:\n{combined_context}\n\n"
        f"User Query: {query}"
    )

    tool_results = tool_agent.invoke(
        {"messages": [("user", tool_instructions)]},
        config={"recursion_limit": 10}
    )
    final_tool_answer = tool_results["messages"][-1].content
    combined_context += f"--- ACTIVE TOOL RESULTS ---\n{final_tool_answer}\n\n"
    # ====================================================================

    # 6. Final Output Structuring Block
    structured_llm = llm.with_structured_output(FactSheet, method="function_calling")

    system_prompt = (
        "You are an elite research assistant. Analyze the provided context extracts thoroughly.\n\n"
        "CRITICAL GATEWAY VERIFICATION RULE:\n"
        "The Main Agent will be completely BLOCKED if it mentions anything that is not explicitly written in your 'verified_facts' array.\n"
        "Therefore, you must extract and list EVERY SINGLE factual component separately:\n"
        "1. List the exact filename and path that was saved (e.g., 'projects/ai_news_report.md').\n"
        "2. Explicitly note that the user requested a Word document, but it was saved as a Markdown/Text file due to tool limitations.\n"
        "3. Read through the live news extracts and list EVERY SINGLE news headline/bullet point as its own separate string item in the 'verified_facts' list. Do not combine or summarize them into a single bullet point. List them all individually so the Main Agent can safely discuss them.\n\n"
        "Do not leave any headline out, or the gateway filter will drop the response."
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Context:\n{context}\n\nQuery: {user_query}")
    ])

    print("[Sub-Agent] Compiling facts into structured schema using LangChain parsing...", flush=True)
    chain = prompt_template | structured_llm
    
    fact_sheet_object = chain.invoke({
        "context": combined_context,
        "user_query": query
    })
    
    print("[Sub-Agent] Research complete. Handing clean data to Main Agent.", flush=True)
    return fact_sheet_object