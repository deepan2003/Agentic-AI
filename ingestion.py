import os
import cloudinary
import cloudinary.api
import cloudinary.uploader
import requests as http_requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings

# 1. Load environment variables
load_dotenv()

def get_embedding_model():
    """
    Returns lightweight embeddings:
    1. Gemini embeddings (free tier, 0 MB RAM) if GEMINI_API_KEY is available.
    2. HuggingFace as fallback.
    """
    global _ingest_embeddings
    if "_ingest_embeddings" not in globals() or _ingest_embeddings is None:
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        if gemini_key:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                print("[Knowledge Engine] Using Google Gemini Cloud Embeddings (0MB RAM)...", flush=True)
                _ingest_embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=gemini_key
                )
                return _ingest_embeddings
            except Exception as e:
                print(f"[Knowledge Engine Warning] Gemini embeddings initialization error: {e}")
                
        print("[Knowledge Engine] Using local HuggingFace embeddings...", flush=True)
        _ingest_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _ingest_embeddings

_ingest_embeddings = None
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER", "agentic-rag-data")
VECTOR_DB_PATH = "./vector_store/research_papers"
TMP_DIR = "./tmp_downloads"
UPLOAD_DIR = "./data_uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

def index_uploaded_pdf(file_bytes: bytes, filename: str) -> dict:
    """
    1. Saves uploaded PDF to data_uploads.
    2. Backs up directly to Cloudinary so it shows in Cloud Storage.
    3. Extracts chunks and updates FAISS vector index immediately.
    """
    safe_name = os.path.basename(filename)
    dest_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(dest_path, "wb") as f:
        f.write(file_bytes)
        
    print(f"[Knowledge Engine] Indexing newly uploaded PDF: {safe_name}...", flush=True)

    # 1. First: Always back up to Cloudinary
    try:
        cloudinary.uploader.upload(
            dest_path,
            folder=CLOUDINARY_FOLDER,
            resource_type="raw",
            public_id=safe_name,
            timeout=15
        )
        print(f"[Knowledge Engine] Successfully uploaded {safe_name} to Cloudinary folder '{CLOUDINARY_FOLDER}'.", flush=True)
    except Exception as e:
        print(f"[Knowledge Engine Warning] Cloudinary upload notice: {e}", flush=True)

    # 2. Second: Process text chunks and build FAISS vector index
    try:
        loader = PyPDFLoader(dest_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = loader.load_and_split(text_splitter)
        
        if not docs:
            return {"success": False, "message": "No readable text found in PDF", "chunks": 0}

        embeddings = get_embedding_model()
        index_file = os.path.join(VECTOR_DB_PATH, "index.faiss")
        
        if os.path.exists(index_file):
            try:
                vector_db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
                vector_db.add_documents(docs)
                vector_db.save_local(VECTOR_DB_PATH)
            except Exception:
                vector_db = FAISS.from_documents(docs, embeddings)
                vector_db.save_local(VECTOR_DB_PATH)
        else:
            vector_db = FAISS.from_documents(docs, embeddings)
            vector_db.save_local(VECTOR_DB_PATH)

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

def delete_uploaded_pdf(filename: str) -> dict:
    """Deletes a PDF from local data_uploads and tmp_downloads."""
    safe_name = os.path.basename(filename)
    deleted_any = False
    for d in [UPLOAD_DIR, TMP_DIR]:
        path = os.path.join(d, safe_name)
        if os.path.exists(path):
            try:
                os.remove(path)
                deleted_any = True
            except Exception as e:
                print(f"Error removing {path}: {e}")
    return {"success": deleted_any}

def list_uploaded_pdfs() -> list:
    """Lists all PDFs currently saved and indexed in the knowledge base."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    files = []
    for f in os.listdir(UPLOAD_DIR):
        if f.lower().endswith(".pdf"):
            full = os.path.join(UPLOAD_DIR, f)
            files.append({
                "filename": f,
                "size_kb": round(os.path.getsize(full) / 1024, 1),
                "modified": os.path.getmtime(full)
            })
    return sorted(files, key=lambda x: x["modified"], reverse=True)

def list_cloud_storage_files():
    """
    Queries Cloudinary and returns a list of files available in the storage folder.
    """
    found_files = []
    
    # 1. Check raw resources
    try:
        res_raw = cloudinary.api.resources(
            type="upload",
            resource_type="raw",
            prefix=CLOUDINARY_FOLDER + "/",
            max_results=500
        )
        for r in res_raw.get("resources", []):
            pub_id = r["public_id"]
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
        print(f"[Cloudinary Warning] Error checking raw resources: {e}")

    # 2. Check image resources
    try:
        res_img = cloudinary.api.resources(
            type="upload",
            resource_type="image",
            prefix=CLOUDINARY_FOLDER + "/",
            max_results=500
        )
        for r in res_img.get("resources", []):
            pub_id = r["public_id"]
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
        print(f"[Cloudinary Warning] Error checking image resources: {e}")
        
    return found_files

def download_papers_from_cloudinary():
    """Downloads all raw and PDF files from Cloudinary folder into local uploads."""
    print(f"Connecting to Cloudinary Cloud Storage (folder: '{CLOUDINARY_FOLDER}')...", flush=True)
    
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR, exist_ok=True)
        
    cloud_files = list_cloud_storage_files()
    if not cloud_files:
        print(f"⚠️ No files found in Cloudinary folder '{CLOUDINARY_FOLDER}'.", flush=True)
        return []
        
    downloaded_files = []
    for item in cloud_files:
        filename = item["filename"]
        if not filename.lower().endswith(".pdf"):
            filename = f"{filename}.pdf"
        local_path = os.path.join(TMP_DIR, filename)
        secure_url = item["url"]
        
        print(f"Downloading {filename} from Cloudinary ({item.get('bytes', 0)} bytes)...", flush=True)
        try:
            # If secure_url is empty, fetch using cloudinary image/raw URL format
            fetch_url = secure_url
            if not fetch_url:
                cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME", "")
                res_type = "image" if item.get("format") == "pdf" else "raw"
                fetch_url = f"https://res.cloudinary.com/{cloud_name}/{res_type}/upload/{item['public_id']}"
                
            response = http_requests.get(fetch_url, timeout=60)
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

def build_local_vector_db():
    """Reads all PDFs from uploads and Cloudinary, chunks text, and saves FAISS index."""
    # First, pull any files from Cloudinary into TMP_DIR
    download_papers_from_cloudinary()
    
    # Collect all PDFs from TMP_DIR and UPLOAD_DIR
    all_pdf_paths = []
    if os.path.exists(TMP_DIR):
        for f in os.listdir(TMP_DIR):
            if f.lower().endswith(".pdf"):
                all_pdf_paths.append(os.path.join(TMP_DIR, f))
    if os.path.exists(UPLOAD_DIR):
        for f in os.listdir(UPLOAD_DIR):
            if f.lower().endswith(".pdf"):
                path = os.path.join(UPLOAD_DIR, f)
                if path not in all_pdf_paths:
                    all_pdf_paths.append(path)

    if not all_pdf_paths:
        print("❌ Pipeline stopped: No PDF files available to index.", flush=True)
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

    print(f"Processing {len(all_pdf_paths)} PDF file(s) and splitting text...", flush=True)
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

    print(f"Generating embeddings for {len(all_chunks)} chunks using Hugging Face (all-MiniLM-L6-v2)...", flush=True)
    embeddings = get_embedding_model()
    
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    vector_db = FAISS.from_documents(all_chunks, embeddings)
    vector_db.save_local(VECTOR_DB_PATH)
    print(f"✅ Success! Local vector database saved with {len(all_chunks)} chunks from {len(processed_file_names)} file(s).", flush=True)

    return {
        "success": True,
        "file_count": len(processed_file_names),
        "files": processed_file_names,
        "chunks": len(all_chunks),
        "message": f"Successfully indexed {len(processed_file_names)} document(s) with {len(all_chunks)} text chunks."
    }

if __name__ == "__main__":
    build_local_vector_db()
