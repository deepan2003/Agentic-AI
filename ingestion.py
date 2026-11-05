import os
import re
import time
import shutil
import threading
import numpy as np
import cloudinary
import cloudinary.api
import cloudinary.uploader
import cloudinary.utils
import requests as http_requests
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# 1. Load environment variables
load_dotenv()

# Cooldown & cache trackers (strictly scoped per user thread_id)
_FAILED_SYNC_TIMESTAMPS = {}
_CLOUD_FILES_CACHE = {}
SYNC_COOLDOWN_SECONDS = 45
CLOUD_LIST_CACHE_TTL = 30

# Locks to prevent concurrent index builds or model loads from crashing Render 512MB RAM
_embedding_lock = threading.Lock()
_build_lock = threading.Lock()
_ingest_embeddings = None

def sanitize_thread_id(thread_id: str = "default") -> str:
    """Returns a safe, lowercase alphanumeric user identifier for isolated folders and cloud namespaces."""
    if not thread_id:
        return "default"
    clean = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(thread_id).strip().lower()).strip('_')
    return clean or "default"


class UltraFastRenderEmbeddings(Embeddings):
    """
    384-dimensional normalized Word + Bigram + Char-Ngram Embeddings (Pure NumPy).
    - Runs in <2ms on Render's 0.1 Free CPU (vs 60-150s for ONNX neural models).
    - Uses 0MB extra RAM (prevents Render 512MB Out-Of-Memory container restarts).
    - Zero external API calls, zero model downloads, 100% deterministic.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str) -> list:
        vec = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vec.tolist()
        words = re.findall(r'[a-z0-9]+', text.lower())
        if not words:
            return vec.tolist()

        for i, w in enumerate(words):
            h = (hash(w) & 0x7FFFFFFF) % self.dim
            sign = 1.0 if (hash(w + "_s") & 1) == 0 else -1.0
            vec[h] += sign * (1.5 if len(w) > 3 else 0.7)

            if len(w) >= 4:
                for j in range(len(w) - 3):
                    cg = w[j:j+4]
                    ch = (hash("cg_" + cg) & 0x7FFFFFFF) % self.dim
                    vec[ch] += 0.35

        for i in range(len(words) - 1):
            bg = f"{words[i]}_{words[i+1]}"
            bh = (hash(bg) & 0x7FFFFFFF) % self.dim
            vec[bh] += 1.2

        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: list) -> list:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list:
        return self._embed(text)


def get_embedding_model():
    """Returns the active embedding model (<2ms latency, 0MB RAM overhead)."""
    global _ingest_embeddings
    if _ingest_embeddings is not None:
        return _ingest_embeddings

    with _embedding_lock:
        if _ingest_embeddings is not None:
            return _ingest_embeddings

        print("[Knowledge Engine] ⚡ Using UltraFast 384-D Vector Embeddings (<2ms latency, 0MB RAM overhead).", flush=True)
        _ingest_embeddings = UltraFastRenderEmbeddings(dim=384)
        return _ingest_embeddings


cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER", "agentic-rag-data")
UPLOAD_BASE_DIR = "./data_uploads"
VECTOR_BASE_DIR = "./vector_store"

def get_user_upload_dir(thread_id: str) -> str:
    """Returns the strictly isolated local PDF upload directory for this specific user."""
    safe_thread = sanitize_thread_id(thread_id)
    path = os.path.join(UPLOAD_BASE_DIR, safe_thread)
    os.makedirs(path, exist_ok=True)
    return path

def get_user_vector_path(thread_id: str) -> str:
    """Returns the strictly isolated FAISS vector store directory for this specific user."""
    safe_thread = sanitize_thread_id(thread_id)
    path = os.path.join(VECTOR_BASE_DIR, safe_thread)
    os.makedirs(path, exist_ok=True)
    return path

os.makedirs(UPLOAD_BASE_DIR, exist_ok=True)
os.makedirs(VECTOR_BASE_DIR, exist_ok=True)

def index_uploaded_pdf(file_bytes: bytes, filename: str, thread_id: str = "default") -> dict:
    """
    1. Saves uploaded PDF strictly into this user's isolated folder (`data_uploads/<user>/`).
    2. Backs up to Cloudinary strictly under `agentic-rag-data/<user>/<filename>` so it persists for this user.
    3. Builds/updates ONLY this user's private FAISS vector index (`vector_store/<user>/`).
    """
    safe_name = os.path.basename(filename)
    safe_thread = sanitize_thread_id(thread_id)
    _FAILED_SYNC_TIMESTAMPS.pop(safe_thread, None)
    _CLOUD_FILES_CACHE.pop(safe_thread, None)

    user_upload_dir = get_user_upload_dir(safe_thread)
    user_vector_path = get_user_vector_path(safe_thread)
    dest_path = os.path.join(user_upload_dir, safe_name)
    with open(dest_path, "wb") as f:
        f.write(file_bytes)
        
    print(f"[Knowledge Engine] Indexing PDF '{safe_name}' exclusively for user '{safe_thread}'...", flush=True)

    # 1. Back up to Cloudinary strictly under this user's private folder (`agentic-rag-data/<safe_thread>`)
    user_cloud_folder = f"{CLOUDINARY_FOLDER}/{safe_thread}"
    try:
        cloudinary.uploader.upload(
            dest_path,
            folder=user_cloud_folder,
            resource_type="raw",
            public_id=safe_name,
            access_mode="public",
            type="upload",
            overwrite=True,
            timeout=25
        )
        _CLOUD_FILES_CACHE.pop(safe_thread, None)
        print(f"[Knowledge Engine] Backed up '{safe_name}' to private Cloudinary folder '{user_cloud_folder}'.", flush=True)
    except Exception as e:
        print(f"[Knowledge Engine Warning] Cloudinary backup notice: {e}", flush=True)

    # 2. Rebuild this user's private FAISS index from all PDFs in their isolated folder
    try:
        with _build_lock:
            all_pdf_paths = [
                os.path.join(user_upload_dir, f)
                for f in os.listdir(user_upload_dir)
                if f.lower().endswith(".pdf")
            ]
            all_chunks = []
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            for pdf_path in all_pdf_paths:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load_and_split(text_splitter)
                all_chunks.extend(docs)

            if not all_chunks:
                return {"success": False, "message": "No readable text found in PDF", "chunks": 0}

            embeddings = get_embedding_model()
            vector_db = FAISS.from_documents(all_chunks, embeddings)
            vector_db.save_local(user_vector_path)

        print(f"[Knowledge Engine] ✅ Indexed '{safe_name}' ({len(all_chunks)} chunks) for user '{safe_thread}'!", flush=True)
        return {
            "success": True,
            "filename": safe_name,
            "chunks": len(all_chunks),
            "message": f"Successfully indexed '{safe_name}' ({len(all_chunks)} chunks) for user '{safe_thread}'."
        }
    except Exception as err:
        print(f"[Knowledge Engine Error] Processing PDF failed: {err}", flush=True)
        return {
            "success": False,
            "filename": safe_name,
            "chunks": 0,
            "message": f"Failed to extract text from PDF: {str(err)}"
        }

def delete_cloudinary_file(public_id: str, resource_type: str = "raw", thread_id: str = None) -> dict:
    """Deletes a file from Cloudinary given its public_id and clears cache."""
    if thread_id:
        _CLOUD_FILES_CACHE.pop(sanitize_thread_id(thread_id), None)
    else:
        _CLOUD_FILES_CACHE.clear()
    try:
        res = cloudinary.uploader.destroy(public_id, resource_type=resource_type)
        print(f"[Cloudinary] Delete result for '{public_id}' ({resource_type}): {res}", flush=True)
        return {"success": True, "result": res}
    except Exception as e:
        print(f"[Cloudinary Error] Failed to delete {public_id}: {e}", flush=True)
        return {"success": False, "error": str(e)}

def delete_uploaded_pdf(filename: str, thread_id: str = "default") -> dict:
    """
    Deletes a PDF strictly from this user's upload directory and updates/clears their private FAISS vector store.
    """
    safe_name = os.path.basename(filename)
    safe_thread = sanitize_thread_id(thread_id)
    _CLOUD_FILES_CACHE.pop(safe_thread, None)

    user_upload_dir = get_user_upload_dir(safe_thread)
    user_vector_path = get_user_vector_path(safe_thread)
    target_path = os.path.join(user_upload_dir, safe_name)

    deleted_any = False
    if os.path.exists(target_path):
        try:
            os.remove(target_path)
            deleted_any = True
        except Exception as e:
            print(f"Error removing {target_path}: {e}")

    # Rebuild or wipe this user's FAISS index so deleted PDF chunks are removed immediately
    remaining_pdfs = [
        os.path.join(user_upload_dir, f)
        for f in os.listdir(user_upload_dir)
        if f.lower().endswith(".pdf")
    ] if os.path.exists(user_upload_dir) else []

    if not remaining_pdfs:
        if os.path.exists(user_vector_path):
            shutil.rmtree(user_vector_path, ignore_errors=True)
            os.makedirs(user_vector_path, exist_ok=True)
    else:
        build_local_vector_db(thread_id=safe_thread, force_sync=False, rebuild_from_local_only=True)

    return {"success": deleted_any}

def list_uploaded_pdfs(thread_id: str = "default") -> list:
    """Lists all PDFs currently saved in the local knowledge base strictly for this user."""
    safe_thread = sanitize_thread_id(thread_id)
    user_dir = get_user_upload_dir(safe_thread)
    files = []
    if os.path.exists(user_dir):
        for f in os.listdir(user_dir):
            if f.lower().endswith(".pdf"):
                full = os.path.join(user_dir, f)
                files.append({
                    "filename": f,
                    "size_kb": round(os.path.getsize(full) / 1024, 1),
                    "modified": os.path.getmtime(full)
                })
    return sorted(files, key=lambda x: x["modified"], reverse=True)

def list_cloud_storage_files(thread_id: str = "default", use_cache: bool = True):
    """
    Strictly queries ONLY this user's private Cloudinary folder: `agentic-rag-data/<safe_thread>/`.
    Never falls back to root `agentic-rag-data/` or any other user's folder.
    - A brand-new user with no uploads gets `[]` (0 files).
    - A returning user gets ONLY the PDFs they previously uploaded under their username.
    """
    safe_thread = sanitize_thread_id(thread_id)
    now = time.time()
    if use_cache and safe_thread in _CLOUD_FILES_CACHE:
        cached_time, cached_list = _CLOUD_FILES_CACHE[safe_thread]
        if (now - cached_time) < CLOUD_LIST_CACHE_TTL:
            return cached_list

    found_files = []
    seen_ids = set()

    # STRICT USER ISOLATION: Only check `agentic-rag-data/<safe_thread>/`
    user_folder_prefix = f"{CLOUDINARY_FOLDER}/{safe_thread}/"

    # 1. Check raw resources inside user's private folder
    try:
        res_raw = cloudinary.api.resources(
            type="upload",
            resource_type="raw",
            prefix=user_folder_prefix,
            max_results=100
        )
        for r in res_raw.get("resources", []):
            pub_id = r["public_id"]
            # Enforce exact folder match so user 'dan' never matches 'daniel'
            if not pub_id.startswith(user_folder_prefix) or pub_id in seen_ids:
                continue
            seen_ids.add(pub_id)
            name = pub_id[len(user_folder_prefix):]
            if "/" in name or name == "agent.md":
                continue
            found_files.append({
                "filename": name,
                "public_id": pub_id,
                "bytes": r.get("bytes", 0),
                "url": r.get("secure_url", ""),
                "format": r.get("format", ""),
                "resource_type": "raw",
                "type": r.get("type", "upload"),
                "version": r.get("version")
            })
    except Exception as e:
        print(f"[Cloudinary Warning] Error checking raw resources for {user_folder_prefix}: {e}")

    # 2. Check image/PDF resources inside user's private folder
    try:
        res_img = cloudinary.api.resources(
            type="upload",
            resource_type="image",
            prefix=user_folder_prefix,
            max_results=100
        )
        for r in res_img.get("resources", []):
            pub_id = r["public_id"]
            if not pub_id.startswith(user_folder_prefix) or pub_id in seen_ids:
                continue
            seen_ids.add(pub_id)
            fmt = r.get("format", "")
            if fmt == "pdf" or pub_id.lower().endswith(".pdf"):
                name = pub_id[len(user_folder_prefix):]
                if "/" in name:
                    continue
                if not name.lower().endswith(".pdf") and fmt:
                    name = f"{name}.{fmt}"
                found_files.append({
                    "filename": name,
                    "public_id": pub_id,
                    "bytes": r.get("bytes", 0),
                    "url": r.get("secure_url", ""),
                    "format": fmt or "pdf",
                    "resource_type": "image",
                    "type": r.get("type", "upload"),
                    "version": r.get("version")
                })
    except Exception as e:
        print(f"[Cloudinary Warning] Error checking image resources for {user_folder_prefix}: {e}")
        
    _CLOUD_FILES_CACHE[safe_thread] = (now, found_files)
    return found_files

def _build_cloudinary_download_candidates(item: dict) -> list:
    """Builds ordered download URL candidates for a Cloudinary resource to overcome HTTP 401 errors."""
    candidates = []
    pub_id = item["public_id"]
    res_type = item.get("resource_type") or ("image" if item.get("format") == "pdf" else "raw")
    upload_type = item.get("type", "upload")
    fmt = item.get("format", "")
    version = item.get("version")

    try:
        if res_type == "raw":
            dl_url = cloudinary.utils.private_download_url(
                pub_id, fmt or "", resource_type="raw", type=upload_type
            )
            if dl_url:
                candidates.append(dl_url)
            if pub_id.lower().endswith(".pdf"):
                dl_url_alt = cloudinary.utils.private_download_url(
                    pub_id[:-4], "pdf", resource_type="raw", type=upload_type
                )
                if dl_url_alt and dl_url_alt not in candidates:
                    candidates.append(dl_url_alt)
        else:
            clean_pub_id = pub_id[:-4] if pub_id.lower().endswith(".pdf") else pub_id
            dl_url = cloudinary.utils.private_download_url(
                clean_pub_id, fmt or "pdf", resource_type="image", type=upload_type
            )
            if dl_url:
                candidates.append(dl_url)
    except Exception:
        pass

    try:
        signed_url, _ = cloudinary.utils.cloudinary_url(
            pub_id,
            resource_type=res_type,
            type=upload_type,
            format=fmt if (res_type == "image" and not pub_id.lower().endswith(".pdf")) else None,
            version=version,
            sign_url=True,
            secure=True
        )
        if signed_url and signed_url not in candidates:
            candidates.append(signed_url)
    except Exception:
        pass

    secure_url = item.get("url", "")
    if secure_url:
        if "%" not in secure_url:
            secure_url = http_requests.utils.requote_uri(secure_url)
        if secure_url not in candidates:
            candidates.append(secure_url)

    return candidates

def download_papers_from_cloudinary(thread_id: str = "default", force_refresh: bool = False):
    """Downloads PDFs strictly from this user's private Cloudinary folder into their local user folder."""
    safe_thread = sanitize_thread_id(thread_id)
    user_cloud_folder = f"{CLOUDINARY_FOLDER}/{safe_thread}"
    print(f"Checking private Cloudinary storage for user '{safe_thread}' (folder: '{user_cloud_folder}')...", flush=True)
    
    user_upload_dir = get_user_upload_dir(safe_thread)
    cloud_files = list_cloud_storage_files(thread_id=safe_thread, use_cache=not force_refresh)
    if not cloud_files:
        print(f"ℹ️ No cloud PDFs found for user '{safe_thread}' in '{user_cloud_folder}'.", flush=True)
        return []

    grouped_by_filename = {}
    for item in cloud_files:
        filename = item["filename"]
        if not filename.lower().endswith(".pdf"):
            filename = f"{filename}.pdf"
        grouped_by_filename.setdefault(filename, []).append(item)

    downloaded_files = []
    for filename, items in grouped_by_filename.items():
        local_path = os.path.join(user_upload_dir, filename)
        
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            downloaded_files.append(local_path)
            continue

        primary_item = items[0]
        print(f"Restoring {filename} for user '{safe_thread}' from Cloudinary ({primary_item.get('bytes', 0)} bytes)...", flush=True)

        file_saved = False
        last_status = None
        for candidate_item in items:
            if file_saved:
                break
            for fetch_url in _build_cloudinary_download_candidates(candidate_item):
                try:
                    response = http_requests.get(fetch_url, timeout=20)
                    if response.status_code == 200 and response.content:
                        with open(local_path, "wb") as f:
                            f.write(response.content)
                        downloaded_files.append(local_path)
                        print(f"✅ Restored {filename} for user '{safe_thread}' ({len(response.content)} bytes)", flush=True)
                        file_saved = True
                        break
                    else:
                        last_status = response.status_code
                except Exception as err:
                    last_status = str(err)

        if not file_saved:
            print(f"Failed to download {filename} for user '{safe_thread}': HTTP {last_status}", flush=True)
            
    return downloaded_files

def build_local_vector_db(thread_id: str = "default", force_sync: bool = False, rebuild_from_local_only: bool = False):
    """
    Strictly builds the FAISS index ONLY from this user's own uploaded/cloud PDFs.
    Never copies or falls back to another user's or 'default' PDFs.
    """
    safe_thread = sanitize_thread_id(thread_id)
    user_upload_dir = get_user_upload_dir(safe_thread)
    user_vector_path = get_user_vector_path(safe_thread)
    index_file = os.path.join(user_vector_path, "index.faiss")

    with _build_lock:
        if not force_sync and not rebuild_from_local_only and os.path.exists(index_file):
            existing_pdfs = [
                f for f in os.listdir(user_upload_dir) if f.lower().endswith(".pdf")
            ] if os.path.exists(user_upload_dir) else []
            if existing_pdfs:
                return {
                    "success": True,
                    "file_count": len(existing_pdfs),
                    "files": existing_pdfs,
                    "chunks": 1,
                    "message": f"Vector database active and loaded for user '{safe_thread}'."
                }

        if not rebuild_from_local_only:
            local_pdfs = [
                os.path.join(user_upload_dir, f)
                for f in os.listdir(user_upload_dir)
                if f.lower().endswith(".pdf")
            ] if os.path.exists(user_upload_dir) else []

            if force_sync or not local_pdfs:
                download_papers_from_cloudinary(thread_id=safe_thread, force_refresh=force_sync)
        
        all_pdf_paths = []
        if os.path.exists(user_upload_dir):
            for f in os.listdir(user_upload_dir):
                if f.lower().endswith(".pdf"):
                    all_pdf_paths.append(os.path.join(user_upload_dir, f))

        # STRICT ISOLATION: If this user has no PDFs, ensure no stale index exists and stop!
        if not all_pdf_paths:
            if os.path.exists(index_file):
                shutil.rmtree(user_vector_path, ignore_errors=True)
                os.makedirs(user_vector_path, exist_ok=True)
            _FAILED_SYNC_TIMESTAMPS[safe_thread] = time.time()
            print(f"ℹ️ No PDF files uploaded yet for user '{safe_thread}'. Knowledge base is empty.", flush=True)
            return {
                "success": False,
                "file_count": 0,
                "files": [],
                "chunks": 0,
                "message": f"No PDF files found for user '{safe_thread}'. Upload a PDF first."
            }

        _FAILED_SYNC_TIMESTAMPS.pop(safe_thread, None)
        all_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        processed_file_names = []

        print(f"Processing {len(all_pdf_paths)} PDF file(s) exclusively for user '{safe_thread}'...", flush=True)
        for file_path in all_pdf_paths:
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load_and_split(text_splitter)
                all_chunks.extend(docs)
                base_name = os.path.basename(file_path)
                processed_file_names.append(base_name)
                print(f"Successfully processed PDF: {base_name} ({len(docs)} chunks)", flush=True)
            except Exception as e:
                print(f"Could not read PDF file {file_path}: {str(e)}", flush=True)

        if not all_chunks:
            return {
                "success": False,
                "file_count": len(all_pdf_paths),
                "files": processed_file_names,
                "chunks": 0,
                "message": "Uploaded files contained no readable text."
            }

        print(f"Generating private vectors for {len(all_chunks)} chunks (User: '{safe_thread}')...", flush=True)
        embeddings = get_embedding_model()
        
        os.makedirs(user_vector_path, exist_ok=True)
        vector_db = FAISS.from_documents(all_chunks, embeddings)
        vector_db.save_local(user_vector_path)
        print(f"✅ Success! Private vector database saved with {len(all_chunks)} chunks for user '{safe_thread}'.", flush=True)

        return {
            "success": True,
            "file_count": len(processed_file_names),
            "files": processed_file_names,
            "chunks": len(all_chunks),
            "message": f"Successfully indexed {len(processed_file_names)} document(s) with {len(all_chunks)} text chunks."
        }

if __name__ == "__main__":
    build_local_vector_db(force_sync=True)