import os
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tools.research import query_local_research_papers
from tools.web_search import custom_web_search, custom_web_fetch

load_dotenv()

# Define the strict output format the Sub-Agent must use
class FactSheet(BaseModel):
    summary: str = Field(description="A concise summary of the core answer found.")
    verified_facts: List[str] = Field(description="List of hard factual points extracted from sources.")
    sources_used: List[str] = Field(description="List of document names or URLs actually utilized.")

def hybrid_research_worker(query: str) -> FactSheet:
    """
    The Research Sub-Agent execution loop. 
    It collects local and web data, then uses Groq to compile a clean FactSheet.
    """
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
            print(f"[Sub-Agent] Extracting content from: {link['url']}")
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

    # 4. Initialize Groq LLM using LangChain's OpenAI wrapper pointing to Groq's endpoint
    # This uses your GROQ_API_KEY and avoids OpenAI subscription issues
    llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0
    )

    # Force the LLM to output exactly matching our structured Pydantic schema
    structured_llm = llm.with_structured_output(FactSheet, method="json_mode")

    system_prompt = (
        "You are an elite research assistant. Analyze the provided context extracts thoroughly. "
        "Extract only hard facts that directly answer the query. Do not assume or extrapolate. "
        "If information is missing, state it clearly in the summary. "
        "You must respond in valid JSON using EXACTLY these three keys: "
        "'summary' (a string), 'verified_facts' (a list of strings), and 'sources_used' (a list of strings)."
    )

    user_prompt = f"Context:\n{combined_context}\n\nQuery: {query}"

    print("[Sub-Agent] Compiling facts into structured schema using Groq...")
    
    # Run the model to get structured Pydantic output
    fact_sheet = structured_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    
    print("[Sub-Agent] Research complete. Handing clean data to Main Agent.")
    return fact_sheet