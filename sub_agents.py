import os
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

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
    print(f"\n[Sub-Agent] Starting research for: '{query}'")
    
    # 1. Gather data from your local PDF Vector DB
    print("[Sub-Agent] Searching local research database...")
    local_data = query_local_research_papers(query, top_k=2)
    
    # 2. Gather data from the live internet
    print("[Sub-Agent] Searching the live web...")
    web_links = custom_web_search(query, max_results=2)
    
    web_data = []
    for link in web_links:
        if "url" in link:
            content = custom_web_fetch(link["url"])
            web_data.append(content)

    # 3. Combine raw research text
    combined_context = "--- LOCAL RESEARCH PAPERS EXTRACTS ---\n"
    for doc in local_data:
        if "error" not in doc:
            combined_context += f"Source: {doc['source_file']}\nContent: {doc['content_snippet']}\n\n"
            
    combined_context += "--- LIVE WEB EXTRACTS ---\n"
    for page in web_data:
        combined_context += f"Source: {page['url']}\nContent: {page['content']}\n\n"

    # 4. Initialize Groq LLM
    llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0
    )

    # 5. Query Optimizer for DuckDuckGo
    print("[Sub-Agent] Optimizing search query for DuckDuckGo...")
    optimizer_prompt = f"Extract only the core search keywords from this prompt. Return ONLY the keywords. Prompt: {query}"
    optimized_query = llm.invoke(optimizer_prompt).content.strip()
    
    print("[Sub-Agent] Gathering live news from DuckDuckGo...")
    ddg_data = DuckDuckGoSearchRun().run(optimized_query)
    combined_context += f"--- DUCKDUCKGO LIVE NEWS ---\n{ddg_data}\n\n"
    
    # ====================================================================
    # 6. NEW: ACTIVE TOOL EXECUTION LOOP (Coding + File System)
    # ====================================================================
    print("[Sub-Agent] Checking if tool execution is required...")
    
    # We hand BOTH tools to the sub-agent
    tool_agent = create_react_agent(llm, tools=[execute_python_code, save_file_to_disk])
    
    # Tell the agent how to use them
    tool_instructions = (
        f"Analyze the user query. "
        f"If they need math or logic, use execute_python_code. "
        f"If they ask to save, write, or store data, use save_file_to_disk to create a file. "
        f"If neither is needed, reply 'No tools needed.'\n\nQuery: {query}"
    )
    
    # Run the loop and append the final tool output to our context
    tool_results = tool_agent.invoke({"messages": [("user", tool_instructions)]})
    final_tool_answer = tool_results["messages"][-1].content
    
    combined_context += f"--- ACTIVE TOOL RESULTS ---\n{final_tool_answer}\n\n"
    # ====================================================================

    # 7. Final Output Structuring
    structured_llm = llm.with_structured_output(FactSheet, method="function_calling")

    system_prompt = (
        "You are an elite research assistant. Analyze the provided context extracts thoroughly. "
        "Extract only hard facts that directly answer the query. Do not assume or extrapolate. "
        "If information is missing, state it clearly in the summary."
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Context:\n{context}\n\nQuery: {user_query}")
    ])

    print("[Sub-Agent] Compiling facts into structured schema using LangChain parsing...")
    chain = prompt_template | structured_llm
    
    fact_sheet_object = chain.invoke({
        "context": combined_context,
        "user_query": query
    })
    
    print("[Sub-Agent] Research complete. Handing clean data to Main Agent.")
    return fact_sheet_object