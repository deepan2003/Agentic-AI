import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

VECTOR_DB_PATH = "./vector_store/research_papers"
_embeddings = None

def get_embeddings():
    """
    Returns lightweight embeddings:
    Uses Gemini embeddings (0 MB RAM) to avoid Render 512MB limit.
    """
    global _embeddings
    if _embeddings is None:
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        if gemini_key:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                print("[System] Using Google Gemini Cloud Embeddings for research lookup (0MB RAM)...", flush=True)
                _embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=gemini_key
                )
                return _embeddings
            except Exception as e:
                print(f"[System Warning] Gemini embeddings initialization error: {e}")
                
        print("[System] Loading HuggingFace embedding weights into RAM...", flush=True)
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings

def query_local_research_papers(query_str: str, top_k: int = 3) -> list:
    """
    Searches the internal local vector database for research papers and academic context.
    If no vector index exists or index is empty, returns empty list without loading PyTorch/embeddings!
    """
    index_file = os.path.join(VECTOR_DB_PATH, "index.faiss")
    if not os.path.exists(index_file):
        # DO NOT load HuggingFace embeddings weights into RAM if no PDF index exists!
        return []
        
    try:
        # Load the local database using lazy-loaded embeddings
        embeddings = get_embeddings()
        vector_db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Search for the best matches
        docs = vector_db.similarity_search_with_score(query_str, k=top_k)
        
        results = []
        for doc, score in docs:
            results.append({
                "source_file": doc.metadata.get("source", "Unknown"),
                "content_snippet": doc.page_content,
                "relevance_score": float(score)
            })
        return results
    except Exception as e:
        return [{"error": f"Failed to search local database: {str(e)}"}]
