import os
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def custom_web_search(query_str: str, max_results: int = 3) -> list:
    """
    Searches the live internet to find high-quality URLs related to the query.
    Use this when looking for real-time news, current events, or modern software updates.
    """
    try:
        response = tavily_client.search(
            query=query_str, 
            max_results=max_results, 
            include_raw_content=False
        )
        return [{"title": r["title"], "url": r["url"]} for r in response.get("results", [])]
    except Exception as e:
        return [{"error": f"Web search failed: {str(e)}"}]

def custom_web_fetch(url: str) -> dict:
    """
    Fetches the full text from a specific URL. 
    Always use this after custom_web_search to actually read the website content.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        
        if res.status_code != 200:
            return {"url": url, "content": f"Error: HTTP {res.status_code}"}
            
        soup = BeautifulSoup(res.text, "html.parser")
        
        # Strip away messy website code
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.extract()
            
        # Get clean text and limit size to prevent AI memory overload
        clean_text = " ".join(soup.get_text().split())[:5000] 
        return {"url": url, "content": clean_text}
    except Exception as e:
        return {"url": url, "content": f"Failed to fetch website: {str(e)}"}