import os
import re
import time
import threading
import cloudinary
import cloudinary.api
import cloudinary.uploader
import cloudinary.utils
import requests as http_requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# 1. Load environment variables
load_dotenv()

# Limit ONNX / OpenMP CPU threads on Render Free Tier (0.1 vCPU) to prevent thread thrashing & 60s hangs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("ONNXRUNTIME_INTRA_OP_NUM_THREADS", "1")
os.environ.setdefault("ONNXRUNTIME_INTER_OP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Cooldown & cache trackers to eliminate redundant Cloudinary network calls
_FAILED_SYNC_TIMESTAMPS = {}
_CLOUD_FILES_CACHE = {}
SYNC_COOLDOWN_SECONDS = 45
CLOUD_LIST_CACHE_TTL = 30

_embedding_lock = threading.Lock()
_ingest_embeddings = None

def sanitize_thread_id(thread_id: str = "default") -> str:
    """Returns a safe, alphanumeric thread identifier for folders and cloud namespaces."""
    if not thread_id:
        return "default"
    clean = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(thread_id).strip()).strip('_')
    return clean or "default"

def get_embedding_model():
    """
    Thread-safe singleton embedding initializer tuned for Render Free Tier (0.1 vCPU / 512MB RAM).
    - Prevents race conditions between background startup pre-warm and incoming requests.
    - Restricts ONNX threads=1 so it doesn't stall for 60+ seconds on shared CPUs.
    """
    global _ingest_embeddings
    if _ingest_embeddings is not None:
        return _ingest_embeddings

    with _embedding_lock:
        if _ingest_embeddings is not None:
            return _ingest_embeddings

        # 1. Try direct native fastembed TextEmbedding with threads=1 for fast Render CPU execution
        try:
            from fastembed import TextEmbedding
            from langchain_core.embeddings import Embeddings

            class NativeFastEmbed(Embeddings):
                def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
                    print(f"[Knowledge Engine] Initializing FastEmbed ONNX ({model_name}, threads=1)...", flush=True)
                    self.model = TextEmbedding(model_name=model_name, threads=1)

                def embed_documents(self, texts: list) -> list:
                    return [list(vec) for vec in self.model.embed(texts, batch_size=16)]

                def embed_query(self, text: str) -> list:
                    return list(next(iter(self.model.embed([text]))))

            _ingest_embeddings = NativeFastEmbed()
            return _ingest_embeddings
        except Exception as nfe_err:
            print(f"[Knowledge Engine Warning] Native FastEmbed note: {nfe_err}", flush=True)

        # 2. Try langchain_community FastEmbedEmbeddings with threads=1
        try:
            from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
            print("[Knowledge Engine] Initializing langchain FastEmbed CPU ONNX model (threads=1)...", flush=True)
            _ingest_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=1)
            return _ingest_embeddings
        except Exception as fe_err:
            print(f"[Knowledge Engine Warning] langchain_community FastEmbed note: {fe_err}", flush=True)

        # 3. Lightweight instant fallback (0MB RAM, <1ms latency, zero PyTorch)
        from langchain_core.embeddings import Embeddings
        import numpy as np

        class LightweightDeterministicEmbeddings(Embeddings):
            """Fast 384-dimensional deterministic hash-based embedding (0MB RAM, zero crash)."""
            def _embed(self, text: str) -> list:
                vec = np.zeros(384, dtype=np.float32)
                words = re.findall(r'\w+', text.lower())
                for i, word in enumerate(words):
                    idx = abs(hash(word)) % 384
                    vec[idx] += 1.0 / (1.0 + i * 0.05)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                return vec.tolist()

            def embed_documents(self, texts: list) -> list:
                return [self._embed(t) for t in texts]

            def embed_query(self, text: str) -> list:
                return self._embed(text)

        print("[Knowledge Engine Warning] Using lightweight fallback embeddings (0MB RAM, zero PyTorch).", flush=True)
        _ingest_embeddings = LightweightDeterministicEmbeddings()
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
TMP_DIR = "./tmp_downloads"

def get_user_upload_dir(thread_id: str = "default") -> str:
    safe_thread = sanitize_thread_id(thread_id)
    path = os.path.join(UPLOAD_BASE_DIR, safe_thread)
    os.makedirs(path, exist_ok=True)
    return path

def get_user_vector_path(thread_id: str = "default") -> str:
    safe_thread = sanitize_thread_id(thread_id)
    path = os.path.join(VECTOR_BASE_DIR, safe_thread)
    os.makedirs(path, exist_ok=True)
    return path

os.makedirs(UPLOAD_BASE_DIR, exist_ok=True)
os.makedirs(VECTOR_BASE_DIR, exist_ok=True)

def index_uploaded_pdf(file_bytes: bytes, filename: str, thread_id: str = "default") -> dict:
    """
    1. Saves uploaded PDF into the isolated directory for this user thread_id.
    2. Backs up directly to Cloudinary under the user's specific folder path.
    3. Extracts chunks and builds/updates the user's private FAISS vector index.
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
        
    print(f"[Knowledge Engine] Indexing PDF '{safe_name}' for user '{safe_thread}'...", flush=True)

    # 1. Back up to Cloudinary under user's namespace as raw resource
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
            timeout=20
        )
        print(f"[Knowledge Engine] Successfully backed up {safe_name} to Cloudinary folder '{user_cloud_folder}'.", flush=True)
    except Exception as e:
        print(f"[Knowledge Engine Warning] Cloudinary upload notice: {e}", flush=True)

    # 2. Process text chunks and build user's private FAISS vector index
    try:
        loader = PyPDFLoader(dest_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = loader.load_and_split(text_splitter)
        
        if not docs:
            return {"success": False, "message": "No readable text found in PDF", "chunks": 0}

        embeddings = get_embedding_model()
        index_file = os.path.join(user_vector_path, "index.faiss")
        
        if os.path.exists(index_file):
            try:
                vector_db = FAISS.load_local(user_vector_path, embeddings, allow_dangerous_deserialization=True)
                vector_db.add_documents(docs)
                vector_db.save_local(user_vector_path)
            except Exception:
                vector_db = FAISS.from_documents(docs, embeddings)
                vector_db.save_local(user_vector_path)
        else:
            vector_db = FAISS.from_documents(docs, embeddings)
            vector_db.save_local(user_vector_path)

        return {
            "success": True,
            "filename": safe_name,
            "chunks": len(docs),
            "message": f"Successfully indexed '{safe_name}' with {len(docs)} text chunks."
        }
    except Exception as err:
        print(f"[Knowledge Engine Error] Processing PDF failed: {err}", flush=True)
        return {
            "success": False,
            "filename": safe_name,
            "chunks": 0,
            "message": f"Failed to extract text from PDF: {str(err)}"
        }

def delete_cloudinary_file(public_id: str, resource_type: str = "raw") -> dict:
    """Deletes a file from Cloudinary given its public_id."""
    _CLOUD_FILES_CACHE.clear()
    try:
        res = cloudinary.uploader.destroy(public_id, resource_type=resource_type)
        print(f"[Cloudinary] Delete result for '{public_id}' ({resource_type}): {res}", flush=True)
        return {"success": True, "result": res}
    except Exception as e:
        print(f"[Cloudinary Error] Failed to delete {public_id}: {e}", flush=True)
        return {"success": False, "error": str(e)}

def delete_uploaded_pdf(filename: str, thread_id: str = "default") -> dict:
    """Deletes a PDF from user directory and tmp_downloads."""
    safe_name = os.path.basename(filename)
    safe_thread = sanitize_thread_id(thread_id)
    _CLOUD_FILES_CACHE.pop(safe_thread, None)
    deleted_any = False
    dirs_to_check = [get_user_upload_dir(safe_thread), TMP_DIR, UPLOAD_BASE_DIR]
    for d in dirs_to_check:
        path = os.path.join(d, safe_name)
        if os.path.exists(path):
            try:
                os.remove(path)
                deleted_any = True
            except Exception as e:
                print(f"Error removing {path}: {e}")
    return {"success": deleted_any}

def list_uploaded_pdfs(thread_id: str = "default") -> list:
    """Lists all PDFs currently saved and indexed in the knowledge base for this user."""
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
    Queries Cloudinary and returns a deduplicated list of files available in this user's storage folder.
    Uses a 30-second in-memory cache so Streamlit sidebar refreshes don't stall the backend.
    """
    safe_thread = sanitize_thread_id(thread_id)
    now = time.time()
    if use_cache and safe_thread in _CLOUD_FILES_CACHE:
        cached_time, cached_list = _CLOUD_FILES_CACHE[safe_thread]
        if (now - cached_time) < CLOUD_LIST_CACHE_TTL:
            return cached_list

    found_files = []
    seen_ids = set()

    prefixes = [f"{CLOUDINARY_FOLDER}/{safe_thread}/"]
    if safe_thread in ("default", "web_default_session"):
        prefixes.append(f"{CLOUDINARY_FOLDER}/")

    for folder_prefix in prefixes:
        # 1. Check raw resources
        try:
            res_raw = cloudinary.api.resources(
                type="upload",
                resource_type="raw",
                prefix=folder_prefix,
                max_results=500
            )
            for r in res_raw.get("resources", []):
                pub_id = r["public_id"]
                if pub_id in seen_ids:
                    continue
                seen_ids.add(pub_id)
                name = pub_id.split("/")[-1]
                if name != "agent.md" and "skills/" not in pub_id:
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
            print(f"[Cloudinary Warning] Error checking raw resources for {folder_prefix}: {e}")

        # 2. Check image resources
        try:
            res_img = cloudinary.api.resources(
                type="upload",
                resource_type="image",
                prefix=folder_prefix,
                max_results=500
            )
            for r in res_img.get("resources", []):
                pub_id = r["public_id"]
                if pub_id in seen_ids:
                    continue
                seen_ids.add(pub_id)
                fmt = r.get("format", "")
                if fmt == "pdf" or pub_id.lower().endswith(".pdf"):
                    name = pub_id.split("/")[-1]
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
            print(f"[Cloudinary Warning] Error checking image resources for {folder_prefix}: {e}")
        
    _CLOUD_FILES_CACHE[safe_thread] = (now, found_files)
    return found_files

def _build_cloudinary_download_candidates(item: dict) -> list:
    """
    Builds ordered download URL candidates for a Cloudinary resource to overcome HTTP 401 errors.
    """
    candidates = []
    pub_id = item["public_id"]
    res_type = item.get("resource_type") or ("image" if item.get("format") == "pdf" else "raw")
    upload_type = item.get("type", "upload")
    fmt = item.get("format", "")
    version = item.get("version")

    # 1. Authenticated API download URLs
    try:
        if res_type == "raw":
            dl_url = cloudinary.utils.private_download_url(
                pub_id,
                fmt or "",
                resource_type="raw",
                type=upload_type
            )
            if dl_url:
                candidates.append(dl_url)
            if pub_id.lower().endswith(".pdf"):
                dl_url_alt = cloudinary.utils.private_download_url(
                    pub_id[:-4],
                    "pdf",
                    resource_type="raw",
                    type=upload_type
                )
                if dl_url_alt and dl_url_alt not in candidates:
                    candidates.append(dl_url_alt)
        else:
            clean_pub_id = pub_id[:-4] if pub_id.lower().endswith(".pdf") else pub_id
            dl_url = cloudinary.utils.private_download_url(
                clean_pub_id,
                fmt or "pdf",
                resource_type="image",
                type=upload_type
            )
            if dl_url:
                candidates.append(dl_url)
    except Exception:
        pass

    # 2. Signed CDN URL
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

    # 3. Direct secure_url fallback
    secure_url = item.get("url", "")
    if secure_url:
        if "%" not in secure_url:
            secure_url = http_requests.utils.requote_uri(secure_url)
        if secure_url not in candidates:
            candidates.append(secure_url)

    return candidates

def download_papers_from_cloudinary(thread_id: str = "default"):
    """Downloads all raw and PDF files from Cloudinary user folder into local user uploads."""
    safe_thread = sanitize_thread_id(thread_id)
    user_cloud_folder = f"{CLOUDINARY_FOLDER}/{safe_thread}"
    print(f"Connecting to Cloudinary Cloud Storage (folder: '{user_cloud_folder}')...", flush=True)
    
    user_upload_dir = get_user_upload_dir(safe_thread)
    cloud_files = list_cloud_storage_files(thread_id=safe_thread, use_cache=True)
    if not cloud_files:
        print(f"⚠️ No files found in Cloudinary folder '{user_cloud_folder}'.", flush=True)
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
        print(f"Downloading {filename} from Cloudinary ({primary_item.get('bytes', 0)} bytes)...", flush=True)

        file_saved = False
        last_status = None
        for candidate_item in items:
            if file_saved:
                break
            for fetch_url in _build_cloudinary_download_candidates(candidate_item):
                try:
                    response = http_requests.get(fetch_url, timeout=30)
                    if response.status_code == 200 and response.content:
                        with open(local_path, "wb") as f:
                            f.write(response.content)
                        downloaded_files.append(local_path)
                        print(f"✅ Successfully downloaded {filename} ({len(response.content)} bytes)", flush=True)
                        file_saved = True
                        break
                    else:
                        last_status = response.status_code
                except Exception as err:
                    last_status = str(err)

        if not file_saved:
            print(
                f"Failed to download {filename}: HTTP {last_status}. "
                f"(Tip: Enable 'Allow delivery of PDF and ZIP files' in Cloudinary Settings -> Security, or re-upload the PDF.)",
                flush=True
            )
            
    return downloaded_files

def build_local_vector_db(thread_id: str = "default", force_sync: bool = False):
    """Reads all PDFs for this specific user, chunks text, and saves isolated FAISS index."""
    safe_thread = sanitize_thread_id(thread_id)
    user_upload_dir = get_user_upload_dir(safe_thread)
    user_vector_path = get_user_vector_path(safe_thread)
    index_file = os.path.join(user_vector_path, "index.faiss")

    # Fast-path: if index already exists on disk and force_sync is False, skip rebuilding!
    if not force_sync and os.path.exists(index_file):
        existing_pdfs = [
            f for f in os.listdir(user_upload_dir) if f.lower().endswith(".pdf")
        ] if os.path.exists(user_upload_dir) else []
        return {
            "success": True,
            "file_count": len(existing_pdfs),
            "files": existing_pdfs,
            "chunks": 1,
            "message": "Vector database already active and loaded."
        }

    # Check if we recently failed syncing for this thread_id to avoid retry loops
    now = time.time()
    last_failed = _FAILED_SYNC_TIMESTAMPS.get(safe_thread, 0)
    if not force_sync and (now - last_failed) < SYNC_COOLDOWN_SECONDS:
        existing_pdfs = [
            os.path.join(user_upload_dir, f)
            for f in os.listdir(user_upload_dir)
            if f.lower().endswith(".pdf")
        ] if os.path.exists(user_upload_dir) else []
        if not existing_pdfs:
            print(f"[RAG Engine] Skipping redundant cloud retry for '{safe_thread}' (cooldown active).", flush=True)
            return {
                "success": False,
                "file_count": 0,
                "files": [],
                "chunks": 0,
                "message": "No local PDF files available to index."
            }

    # Only query Cloudinary if we don't already have local PDFs (or if force_sync=True)
    local_pdfs = [
        os.path.join(user_upload_dir, f)
        for f in os.listdir(user_upload_dir)
        if f.lower().endswith(".pdf")
    ] if os.path.exists(user_upload_dir) else []

    if force_sync or not local_pdfs:
        download_papers_from_cloudinary(thread_id=safe_thread)
    
    all_pdf_paths = []
    if os.path.exists(user_upload_dir):
        for f in os.listdir(user_upload_dir):
            if f.lower().endswith(".pdf"):
                all_pdf_paths.append(os.path.join(user_upload_dir, f))

    if not all_pdf_paths:
        _FAILED_SYNC_TIMESTAMPS[safe_thread] = time.time()
        print(f"❌ Pipeline stopped: No PDF files available to index for user '{safe_thread}'.", flush=True)
        return {
            "success": False,
            "file_count": 0,
            "files": [],
            "chunks": 0,
            "message": "No PDF files found to index. Upload a PDF first."
        }

    _FAILED_SYNC_TIMESTAMPS.pop(safe_thread, None)
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    processed_file_names = []

    print(f"Processing {len(all_pdf_paths)} PDF file(s) for user '{safe_thread}'...", flush=True)
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
            "message": "Downloaded files contained no readable text."
        }

    print(f"Generating FastEmbed ONNX vectors for {len(all_chunks)} chunks (~30MB RAM)...", flush=True)
    embeddings = get_embedding_model()
    
    os.makedirs(user_vector_path, exist_ok=True)
    vector_db = FAISS.from_documents(all_chunks, embeddings)
    vector_db.save_local(user_vector_path)
    print(f"✅ Success! Local vector database saved with {len(all_chunks)} chunks for user '{safe_thread}'.", flush=True)

    return {
        "success": True,
        "file_count": len(processed_file_names),
        "files": processed_file_names,
        "chunks": len(all_chunks),
        "message": f"Successfully indexed {len(processed_file_names)} document(s) with {len(all_chunks)} text chunks."
    }

if __name__ == "__main__":
    build_local_vector_db(force_sync=True)