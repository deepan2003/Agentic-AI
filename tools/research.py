import os
from langchain_community.vectorstores import FAISS
from ingestion import (
    sanitize_thread_id,
    get_user_vector_path,
    get_user_upload_dir,
    get_embedding_model,
    build_local_vector_db,
    list_cloud_storage_files,
)


def query_local_research_papers(query: str, top_k: int = 5, thread_id: str = "default") -> list:
    """
    Strictly queries ONLY the active user's (`thread_id`) isolated FAISS vector store (`./vector_store/<thread_id>/`).
    - If User B logs in and has NOT uploaded any PDF, returns `[]` immediately (never leaks User A's PDFs or root store).
    - If User A logs back in later, automatically loads User A's existing index (or restores it from User A's private
      Cloudinary backup `agentic-rag-data/<user_a>/` if local storage was cleared).
    """
    safe_thread = sanitize_thread_id(thread_id)
    user_upload_dir = get_user_upload_dir(safe_thread)
    user_vector_path = get_user_vector_path(safe_thread)
    index_file = os.path.join(user_vector_path, "index.faiss")

    # 1. Check if this user has local PDFs or private cloud PDFs
    local_pdfs = [
        f for f in os.listdir(user_upload_dir)
        if f.lower().endswith(".pdf")
    ] if os.path.exists(user_upload_dir) else []

    if not os.path.exists(index_file):
        # Check if returning user has PDFs locally or in their private Cloudinary folder
        cloud_pdfs = list_cloud_storage_files(thread_id=safe_thread, use_cache=True)
        if local_pdfs or cloud_pdfs:
            print(f"[RAG Engine] Restoring private FAISS index for returning user '{safe_thread}'...", flush=True)
            build_local_vector_db(thread_id=safe_thread, force_sync=False)
        else:
            # Brand-new user with no uploads: strictly return empty!
            print(f"[RAG Engine] User '{safe_thread}' has no uploaded PDFs. Returning empty RAG context.", flush=True)
            return []

    if not os.path.exists(index_file):
        return []

    try:
        embeddings = get_embedding_model()
        vector_db = FAISS.load_local(
            user_vector_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        docs = vector_db.similarity_search(query, k=top_k)
        results = []
        for d in docs:
            src = os.path.basename(d.metadata.get("source", "Uploaded PDF"))
            page = d.metadata.get("page", 0)
            results.append({
                "source_file": f"{src} (Page {page + 1})" if isinstance(page, int) else src,
                "content_snippet": d.page_content
            })
        print(f"[RAG Engine] Retrieved {len(results)} chunk(s) from private store of user '{safe_thread}'.", flush=True)
        return results
    except Exception as e:
        print(f"[RAG Engine Error] Failed to query vector store for user '{safe_thread}': {e}", flush=True)
        return []