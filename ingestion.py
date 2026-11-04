import os
import cloudinary
import cloudinary.api
import requests as http_requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# 1. Load environment variables
load_dotenv()

# Configure Cloudinary with environment variables
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

CLOUDINARY_FOLDER = os.getenv("CLOUDINARY_FOLDER", "agentic-rag-data")
VECTOR_DB_PATH = "./vector_store/research_papers"
TMP_DIR = "./tmp_downloads"

def list_cloud_storage_files():
    """
    Queries Cloudinary and returns a list of files available in the storage folder.
    Checks both 'raw' and 'image' resource types since Cloudinary sometimes classifies PDFs as images.
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

    # 2. Check image resources (for PDFs uploaded as document/image)
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
    """Downloads all raw and PDF files from Cloudinary folder into a temporary local folder."""
    print(f"Connecting to Cloudinary Cloud Storage (folder: '{CLOUDINARY_FOLDER}')...", flush=True)
    
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR, exist_ok=True)
        
    cloud_files = list_cloud_storage_files()
    if not cloud_files:
        print(f"⚠️ No files found in Cloudinary folder '{CLOUDINARY_FOLDER}'! Please upload PDF files first.", flush=True)
        return []
        
    downloaded_files = []
    for item in cloud_files:
        filename = item["filename"]
        local_path = os.path.join(TMP_DIR, filename)
        secure_url = item["url"]
        
        print(f"Downloading {filename} from Cloudinary ({item.get('bytes', 0)} bytes)...", flush=True)
        try:
            response = http_requests.get(secure_url, timeout=60)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(response.content)
                downloaded_files.append(local_path)
            else:
                print(f"Failed to download {filename}: HTTP {response.status_code}", flush=True)
        except Exception as err:
            print(f"Error downloading {filename}: {err}", flush=True)
            
    return downloaded_files

def build_local_vector_db():
    """Reads downloaded PDFs, chunks text, creates embeddings, and saves FAISS index."""
    raw_files = download_papers_from_cloudinary()
    if not raw_files:
        print("❌ Pipeline stopped: 0 research files were found in Cloudinary.", flush=True)
        return {
            "success": False,
            "file_count": 0,
            "files": [],
            "chunks": 0,
            "message": f"No files found in Cloudinary folder '{CLOUDINARY_FOLDER}'. Vector database was NOT updated."
        }

    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    processed_file_names = []

    print(f"Processing {len(raw_files)} PDF file(s) and splitting text...", flush=True)
    for file_path in raw_files:
        if file_path.endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load_and_split(text_splitter)
                all_chunks.extend(docs)
                base_name = os.path.basename(file_path)
                processed_file_names.append(base_name)
                print(f"Successfully processed PDF: {base_name} ({len(docs)} chunks)", flush=True)
            except Exception as e:
                print(f"Could not read PDF file {file_path}: {str(e)}", flush=True)
        else:
            print(f"Skipping non-PDF file: {file_path}", flush=True)

    if not all_chunks:
        print("❌ No text chunks generated from the downloaded files.", flush=True)
        return {
            "success": False,
            "file_count": len(raw_files),
            "files": processed_file_names,
            "chunks": 0,
            "message": "Downloaded files contained no readable text."
        }

    print(f"Generating embeddings for {len(all_chunks)} chunks using Hugging Face (all-MiniLM-L6-v2)...", flush=True)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    vector_db = FAISS.from_documents(all_chunks, embeddings)
    vector_db.save_local(VECTOR_DB_PATH)
    print(f"✅ Success! Local vector database saved with {len(all_chunks)} chunks from {len(processed_file_names)} file(s).", flush=True)

    # Clean up temporary downloads
    for file in raw_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception:
                pass
    if os.path.exists(TMP_DIR):
        try:
            os.rmdir(TMP_DIR)
        except Exception:
            pass

    return {
        "success": True,
        "file_count": len(processed_file_names),
        "files": processed_file_names,
        "chunks": len(all_chunks),
        "message": f"Successfully indexed {len(processed_file_names)} document(s) with {len(all_chunks)} text chunks."
    }

if __name__ == "__main__":
    build_local_vector_db()
