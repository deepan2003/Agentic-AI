import os
import boto3
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# 1. Load environment variables
load_dotenv()

BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
VECTOR_DB_PATH = "./vector_store/research_papers"
TMP_DIR = "./tmp_downloads"

def download_papers_from_s3():
    """Downloads all files from S3 bucket into a temporary local folder."""
    print(f"Connecting to AWS S3 Bucket: {BUCKET_NAME}...")
    s3 = boto3.client('s3')
    
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)
        
    try:
        # List all files in the bucket
        response = s3.list_objects_v2(Bucket=BUCKET_NAME)
        if 'Contents' not in response:
            print("No files found in your S3 bucket! Please upload a text or PDF file first.")
            return []
            
        downloaded_files = []
        for obj in response['Contents']:
            filename = obj['Key']
            
            # Skip the agent.md rulebook file; we only want research papers here
            if filename == "agent.md" or filename.startswith("skills/"):
                continue
                
            local_path = os.path.join(TMP_DIR, filename)
            print(f"Downloading {filename} from S3...")
            s3.download_file(BUCKET_NAME, filename, local_path)
            downloaded_files.append(local_path)
            
        return downloaded_files
    except Exception as e:
        print(f"Error downloading from S3: {str(e)}")
        return []

def build_local_vector_db():
    """Reads downloaded PDFs, chunks text, creates embeddings, and saves FAISS index."""
    raw_files = download_papers_from_s3()
    if not raw_files:
        print("No research files to process.")
        return

    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    print("Processing PDFs and splitting text...")
    for file_path in raw_files:
        # Check if the file is a PDF
        if file_path.endswith(".pdf"):
            try:
                # Use PyPDFLoader to cleanly read pages and retain metadata
                loader = PyPDFLoader(file_path)
                docs = loader.load_and_split(text_splitter)
                all_chunks.extend(docs)
                print(f"Successfully processed PDF: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Could not read PDF file {file_path}: {str(e)}")
        else:
            print(f"Skipping non-PDF file: {file_path}")

    if not all_chunks:
        print("No text chunks generated from PDFs.")
        return

    print(f"Generating embeddings for {len(all_chunks)} chunks using Hugging Face...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = FAISS.from_documents(all_chunks, embeddings)
    vector_db.save_local(VECTOR_DB_PATH)
    print(f"Success! Your local vector database is saved at: {VECTOR_DB_PATH}")

    # Clean up temporary downloads
    for file in raw_files:
        if os.path.exists(file):
            os.remove(file)
    if os.path.exists(TMP_DIR):
        os.rmdir(TMP_DIR)

if __name__ == "__main__":
    build_local_vector_db()