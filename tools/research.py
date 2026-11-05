import os
from langchain_community.vectorstores import FAISS

VECTOR_DB_PATH = "./vector_store/research_papers"

def get_embeddings():
    """
    Returns unified FastEmbed CPU ONNX quantized model (~30MB RAM, zero PyTorch, 100% reliable).
    """
    from ingestion import get_embedding_model
    return get_embedding_model()

def query_local_research_papers(query_str: str, top_k: int = 3, thread_id: str = "default") -> list:
    """
    Searches the user's isolated local vector database for uploaded PDFs.
    If no vector index exists for this user, returns empty list without loading embeddings.
    """
    from ingestion import get_user_vector_path
    user_vector_path = get_user_vector_path(thread_id)
    index_file = os.path.join(user_vector_path, "index.faiss")
    if not os.path.exists(index_file):
        return []
        
    try:
        # Load the user's isolated vector database
        embeddings = get_embeddings()
        vector_db = FAISS.load_local(user_vector_path, embeddings, allow_dangerous_deserialization=True)
        
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
        return [{"error": f"Failed to search user database: {str(e)}"}]
