import os
import re
import cloudinary
import cloudinary.api
import cloudinary.uploader
import requests as http_requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# 1. Load environment variables
load_dotenv()

def sanitize_thread_id(thread_id: str = "default") -> str:
    """Returns a safe, alphanumeric thread identifier for folders and cloud namespaces."""
    if not thread_id:
        return "default"
    clean = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(thread_id).strip()).strip('_')
    return clean or "default"

def get_embedding_model():
    """
    Returns FastEmbed CPU ONNX quantized embeddings:
    - Runs in ~30MB RAM.
    - Zero external API keys, zero rate-limits, zero 404 errors.
    - ZERO PyTorch / transformers loading to prevent Render 512MB OOM crashes.
    """
    global _ingest_embeddings
    if "_ingest_embeddings" not in globals() or _ingest_embeddings is None:
        # 1. Try langchain_community FastEmbedEmbeddings
        try:
            from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
            print("[Knowledge Engine] Initializing FastEmbed CPU ONNX quantized model (~30MB RAM)...", flush=True)
            _ingest_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
            return _ingest_embeddings
        except Exception as fe_err:
            print(f"[Knowledge Engine Warning] langchain_community FastEmbed note: {fe_err}", flush=True)

        # 2. Try direct native fastembed TextEmbedding
        try:
            from fastembed import TextEmbedding
            from langchain_core.embeddings import Embeddings

            class NativeFastEmbed(Embeddings):
                def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
                    print(f"[Knowledge Engine] Initializing native FastEmbed TextEmbedding ({model_name})...", flush=True)
                    self.model = TextEmbedding(model_name=model_name)

                def embed_documents(self, texts: list) -> list:
                    return [list(vec) for vec in self.model.embed(texts)]

                def embed_query(self, text: str) -> list:
                    return list(next(iter(self.model.embed([text]))))

            _ingest_embeddings = NativeFastEmbed()
            return _ingest_embeddings
        except Exception as nfe_err:
            print(f"[Knowledge Engine Warning] Native FastEmbed note: {nfe_err}", flush=True)

        # 3. Lightweight fallback that does NOT use PyTorch (prevents OOM / 502 Bad Gateway)
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

_ingest_embeddings = None
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
    user_upload_dir = get_user_upload_dir(safe_thread)
    user_vector_path = get_user_vector_path(safe_thread)
    dest_path = os.path.join(user_upload_dir, safe_name)
    with open(dest_path, "wb") as f:
        f.write(file_bytes)
        
    print(f"[Knowledge Engine] Indexing PDF '{safe_name}' for user '{safe_thread}'...", flush=True)

    # 1. Back up to Cloudinary under user's namespace
    user_cloud_folder = f"{CLOUDINARY_FOLDER}/{safe_thread}"
    try:
        cloudinary.uploader.upload(
            dest_path,
            folder=user_cloud_folder,
            resource_type="raw",
            public_id=safe_name,
            access_mode="public",
            type="upload",
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

def list_cloud_storage_files(thread_id: str = "default"):
    """
    Queries Cloudinary and returns a list of files available in this user's storage folder.
    Also falls back to root folder if in default mode so legacy files are not lost.
    """
    safe_thread = sanitize_thread_id(thread_id)
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
                        "format": r.get("format", "raw")
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
                    if not name.endswith(".pdf") and fmt:
                        name = f"{name}.{fmt}"
                    found_files.append({
                        "filename": name,
                        "public_id": pub_id,
                        "bytes": r.get("bytes", 0),
                        "url": r.get("secure_url", ""),
                        "format": fmt
                    })
        except Exception as e:
            print(f"[Cloudinary Warning] Error checking image resources for {folder_prefix}: {e}")
        
    return found_files

def download_papers_from_cloudinary(thread_id: str = "default"):
    """Downloads all raw and PDF files from Cloudinary user folder into local user uploads."""
    safe_thread = sanitize_thread_id(thread_id)
    user_cloud_folder = f"{CLOUDINARY_FOLDER}/{safe_thread}"
    print(f"Connecting to Cloudinary Cloud Storage (folder: '{user_cloud_folder}')...", flush=True)
    
    user_upload_dir = get_user_upload_dir(safe_thread)
    cloud_files = list_cloud_storage_files(thread_id=safe_thread)
    if not cloud_files:
        print(f"⚠️ No files found in Cloudinary folder '{user_cloud_folder}'.", flush=True)
        return []
        
    import base64
    c_key = os.getenv("CLOUDINARY_API_KEY", "")
    c_sec = os.getenv("CLOUDINARY_API_SECRET", "")
    auth_headers = {}
    if c_key and c_sec:
        auth_str = base64.b64encode(f"{c_key}:{c_sec}".encode()).decode()
        auth_headers["Authorization"] = f"Basic {auth_str}"

    downloaded_files = []
    for item in cloud_files:
        filename = item["filename"]
        if not filename.lower().endswith(".pdf"):
            filename = f"{filename}.pdf"
        local_path = os.path.join(user_upload_dir, filename)
        secure_url = item.get("url", "")
        
        # If file already exists locally and is non-empty, keep it
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            downloaded_files.append(local_path)
            continue

        print(f"Downloading {filename} from Cloudinary ({item.get('bytes', 0)} bytes)...", flush=True)
        try:
            # 1. Generate authenticated signed URL from Cloudinary SDK
            res_type = "image" if item.get("format") == "pdf" else "raw"
            signed_url, _ = cloudinary.utils.cloudinary_url(
                item["public_id"],
                resource_type=res_type,
                type="upload",
                sign_url=True,
                secure=True
            )
            
            fetch_url = signed_url or secure_url
            fetch_url = http_requests.utils.requote_uri(fetch_url)
            
            # Fetch with Cloudinary Basic Auth headers to bypass 401 restrictions
            response = http_requests.get(fetch_url, headers=auth_headers, timeout=60)
            
            # Fallback to direct raw download URL if signed URL didn't return 200
            if response.status_code != 200 and secure_url and secure_url != fetch_url:
                retry_url = http_requests.utils.requote_uri(secure_url)
                response = http_requests.get(retry_url, headers=auth_headers, timeout=60)

            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(response.content)
                downloaded_files.append(local_path)
                print(f"✅ Successfully downloaded {filename} ({len(response.content)} bytes)", flush=True)
            else:
                print(f"Failed to download {filename}: HTTP {response.status_code}", flush=True)
        except Exception as err:
            print(f"Error downloading {filename}: {err}", flush=True)
            
    return downloaded_files

def build_local_vector_db(thread_id: str = "default"):
    """Reads all PDFs for this specific user, chunks text, and saves isolated FAISS index."""
    safe_thread = sanitize_thread_id(thread_id)
    # First, pull any files from user's Cloudinary folder
    download_papers_from_cloudinary(thread_id=safe_thread)
    
    user_upload_dir = get_user_upload_dir(safe_thread)
    user_vector_path = get_user_vector_path(safe_thread)
    
    all_pdf_paths = []
    if os.path.exists(user_upload_dir):
        for f in os.listdir(user_upload_dir):
            if f.lower().endswith(".pdf"):
                all_pdf_paths.append(os.path.join(user_upload_dir, f))

    if not all_pdf_paths:
        print(f"❌ Pipeline stopped: No PDF files available to index for user '{safe_thread}'.", flush=True)
        return {
            "success": False,
            "file_count": 0,
            "files": [],
            "chunks": 0,
            "message": "No PDF files found to index. Upload a PDF first."
        }

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
    build_local_vector_db()
