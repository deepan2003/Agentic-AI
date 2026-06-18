import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DB_PATH = "./vector_store/research_papers"
print("[System] Pre-loading HuggingFace embedding weights into RAM...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
def query_local_research_papers(query_str: str, top_k: int = 3) -> list:
    """
    Searches the internal local vector database for research papers and academic context.
    """
    if not os.path.exists(VECTOR_DB_PATH):
        return [{"error": "Local vector database not found."}]
        
    try:
        # Load the local database using the globally cached embeddings
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