import os
from langchain_community.vectorstores import FAISS

VECTOR_DB_PATH = "./vector_store/research_papers"

def get_embeddings():
    """Returns the unified fast embedding model from ingestion.py."""
    from ingestion import get_embedding_model
    return get_embedding_model()

def query_local_research_papers(query_str: str, top_k: int = 5, thread_id: str = "default") -> list:
    """
    Searches the user's isolated local vector database for uploaded PDFs.
    If no vector index exists, automatically builds it from user's Cloudinary/local files.
    """
    from ingestion import get_user_vector_path, build_local_vector_db, list_cloud_storage_files, list_uploaded_pdfs
    user_vector_path = get_user_vector_path(thread_id)
    index_file = os.path.join(user_vector_path, "index.faiss")
    
    # Auto-heal: If index doesn't exist, check if there are files in uploads or Cloudinary
    if not os.path.exists(index_file):
        has_files = list_uploaded_pdfs(thread_id=thread_id) or list_cloud_storage_files(thread_id=thread_id)
        if has_files:
            print(f"[RAG Engine] Index missing for '{thread_id}'. Building vector store now...", flush=True)
            build_local_vector_db(thread_id=thread_id)

    # Check fallback to default vector path if user index is empty
    if not os.path.exists(index_file) and thread_id not in ("default", "web_default_session"):
        default_vector_path = get_user_vector_path("default")
        if os.path.exists(os.path.join(default_vector_path, "index.faiss")):
            user_vector_path = default_vector_path
            index_file = os.path.join(user_vector_path, "index.faiss")

    if not os.path.exists(index_file):
        print(f"[RAG Engine] No vector index available for '{thread_id}'.", flush=True)
        return []
        
    try:
        embeddings = get_embeddings()
        vector_db = FAISS.load_local(user_vector_path, embeddings, allow_dangerous_deserialization=True)
        docs = vector_db.similarity_search_with_score(query_str, k=top_k)
        
        results = []
        for doc, score in docs:
            results.append({
                "source_file": os.path.basename(doc.metadata.get("source", "Uploaded PDF")),
                "content_snippet": doc.page_content,
                "relevance_score": float(score)
            })
        print(f"[RAG Engine] Successfully retrieved {len(results)} chunks for query '{query_str}'", flush=True)
        return results
    except Exception as e:
        print(f"[RAG Engine Warning] Rebuilding index for '{thread_id}' due to load notice: {e}", flush=True)
        try:
            build_local_vector_db(thread_id=thread_id, force_sync=False)
            embeddings = get_embeddings()
            vector_db = FAISS.load_local(user_vector_path, embeddings, allow_dangerous_deserialization=True)
            docs = vector_db.similarity_search_with_score(query_str, k=top_k)
            return [
                {
                    "source_file": os.path.basename(doc.metadata.get("source", "Uploaded PDF")),
                    "content_snippet": doc.page_content,
                    "relevance_score": float(score)
                }
                for doc, score in docs
            ]
        except Exception as e2:
            print(f"[RAG Engine Error] Search failed: {e2}", flush=True)
            return []