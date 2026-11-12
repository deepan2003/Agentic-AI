import os
import io
import sys
import time
import threading
import contextlib
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from ingestion import (
    sanitize_thread_id,
    build_local_vector_db,
    list_cloud_storage_files,
    index_uploaded_pdf,
    list_uploaded_pdfs,
    delete_cloudinary_file,
    delete_uploaded_pdf,
    get_embedding_model,
    get_user_vector_path,
)

load_dotenv()

# Live per-user log store so Streamlit can display real-time backend progress while a query is running
LIVE_LOGS = {}
LIVE_LOGS_LOCK = threading.Lock()

def append_live_log(thread_id: str, line: str):
    clean = line.strip()
    if not clean:
        return
    safe_thread = sanitize_thread_id(thread_id)
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {clean}"
    with LIVE_LOGS_LOCK:
        LIVE_LOGS.setdefault(safe_thread, []).append(entry)

# Dual stdout writer that outputs to Render console immediately AND streams to LIVE_LOGS for frontend polling
class DualStream(io.StringIO):
    def __init__(self, thread_id: str = "default"):
        super().__init__()
        self.thread_id = sanitize_thread_id(thread_id)
        self._buf = ""

    def write(self, s):
        sys.__stdout__.write(s)
        sys.__stdout__.flush()
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                append_live_log(self.thread_id, line.strip())
        return super().write(s)

# Pre-warm embeddings on startup
def _bg_prewarm():
    try:
        get_embedding_model()
    except Exception as e:
        print(f"[Startup Warning] Pre-warm notice: {e}", flush=True)

threading.Thread(target=_bg_prewarm, daemon=True).start()

import requests as http_requests  # aliased to avoid clash with flask.request

from llm_gateway import smart_gateway
from gateway import verify_user_input
from main import agent_system

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

app = Flask(__name__)
CORS(app) 

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "Agentic RAG Backend"}), 200

@app.route("/api/live_logs", methods=["GET"])
def live_logs_endpoint():
    """Returns real-time backend execution logs for the currently running request."""
    thread_id = sanitize_thread_id(request.args.get("thread_id") or "default")
    with LIVE_LOGS_LOCK:
        logs = list(LIVE_LOGS.get(thread_id, []))
    return jsonify({"success": True, "thread_id": thread_id, "logs": logs}), 200

@app.route("/api/upload_pdf", methods=["POST"])
def upload_pdf_endpoint():
    """Receives an uploaded PDF file, indexes it into this user's private FAISS index, and backs it up."""
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file uploaded in request."}), 400
        
    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"success": False, "message": "Empty filename."}), 400

    filename = uploaded_file.filename
    if not filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "message": "Only PDF files are supported."}), 400

    thread_id = sanitize_thread_id(request.form.get("thread_id", "default"))
    file_bytes = uploaded_file.read()
    print(f"\n====== 📄 PDF UPLOAD RECEIVED: {filename} ({len(file_bytes)} bytes) FOR USER '{thread_id}' ======", flush=True)

    result = index_uploaded_pdf(file_bytes, filename, thread_id=thread_id)
    status_code = 200 if result.get("success") else 500
    return jsonify(result), status_code

@app.route("/api/delete_cloud_file", methods=["POST"])
def delete_cloud_file_endpoint():
    """Deletes a file from this user's Cloudinary folder and cleans their local copy & vector index."""
    data = request.json or {}
    public_id = data.get("public_id")
    filename = data.get("filename", "")
    resource_type = data.get("resource_type", "raw")
    thread_id = sanitize_thread_id(data.get("thread_id", "default"))
    
    if not public_id and not filename:
        return jsonify({"success": False, "message": "Missing public_id or filename"}), 400
        
    print(f"\n====== 🗑️ DELETE REQUEST: {public_id or filename} FOR USER '{thread_id}' ======", flush=True)
    res = {}
    if public_id:
        res = delete_cloudinary_file(public_id, resource_type=resource_type, thread_id=thread_id)
    if filename:
        delete_uploaded_pdf(filename, thread_id=thread_id)
        
    return jsonify({"success": True, "result": res}), 200

@app.route("/api/storage_status", methods=["GET"])
def storage_status_endpoint():
    """
    Returns ONLY the files belonging to `thread_id`.
    - If `thread_id` is a brand-new user, returns 0 files and `vector_index_ready: False`.
    - If `thread_id` is a returning user who previously uploaded PDFs (stored in `agentic-rag-data/<thread_id>/`),
      automatically restores their private FAISS index if local disk was reset.
    """
    thread_id = sanitize_thread_id(request.args.get("thread_id", "default"))
    try:
        uploaded_pdfs = list_uploaded_pdfs(thread_id=thread_id)
        cloud_files = list_cloud_storage_files(thread_id=thread_id, use_cache=True)
        user_vector_path = get_user_vector_path(thread_id)
        index_file = os.path.join(user_vector_path, "index.faiss")
        has_index = os.path.exists(index_file)

        # Returning user auto-restore: if this user has their own PDFs in cloud/local but index is not built yet
        if not has_index and (uploaded_pdfs or cloud_files):
            print(f"[Storage Status] Returning user '{thread_id}' detected with {len(uploaded_pdfs) or len(cloud_files)} PDF(s). Restoring private index...", flush=True)
            build_local_vector_db(thread_id=thread_id, force_sync=False)
            uploaded_pdfs = list_uploaded_pdfs(thread_id=thread_id)
            has_index = os.path.exists(index_file)

        # If user has no PDFs at all, ensure vector_index_ready is False
        if not uploaded_pdfs and not cloud_files:
            has_index = False
        
        return jsonify({
            "success": True,
            "thread_id": thread_id,
            "cloud_files": cloud_files,
            "cloud_file_count": len(cloud_files),
            "uploaded_pdfs": uploaded_pdfs,
            "uploaded_pdf_count": len(uploaded_pdfs),
            "vector_index_ready": has_index
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "cloud_files": [],
            "cloud_file_count": 0,
            "uploaded_pdfs": [],
            "uploaded_pdf_count": 0,
            "vector_index_ready": False
        }), 500

@app.route("/api/update_knowledge", methods=["POST"])
def update_knowledge_endpoint():
    print("\n====== 🔄 KNOWLEDGE BASE UPDATE REQUESTED ======", flush=True)
    data = request.json or {}
    thread_id = sanitize_thread_id(data.get("thread_id") or request.args.get("thread_id", "default"))
    print(f"[Knowledge Engine] Syncing private cloud assets for user: '{thread_id}'...", flush=True)
    
    captured_stdout = DualStream(thread_id=thread_id)
    try:
        with contextlib.redirect_stdout(captured_stdout):
            print(f"⏳ [Pipeline] Checking private storage and vector index for '{thread_id}'...", flush=True)
            result = build_local_vector_db(thread_id=thread_id, force_sync=True)

        printed_logs = [line.strip() for line in captured_stdout.getvalue().splitlines() if line.strip()]
        
        is_success = result.get("success", False) if isinstance(result, dict) else False
        msg = result.get("message", "Processed") if isinstance(result, dict) else "Done"
        file_count = result.get("file_count", 0) if isinstance(result, dict) else 0
        
        return jsonify({
            "success": is_success,
            "message": msg,
            "file_count": file_count,
            "logs": printed_logs
        }), 200

    except Exception as e:
        print(f"❌ PIPELINE ERROR: {str(e)}", flush=True)
        return jsonify({
            "success": False,
            "message": f"Pipeline failure: {str(e)}",
            "file_count": 0,
            "logs": [f"[CRITICAL ERROR] {str(e)}"]
        }), 500

@app.route("/api/transcribe", methods=["POST"])
def transcribe_audio():
    """Receives audio blob from frontend, sends to Deepgram Nova-3 REST API, returns transcript."""
    if "audio" not in request.files:
        return jsonify({"success": False, "error": "No audio file provided"}), 400

    if not DEEPGRAM_API_KEY:
        return jsonify({"success": False, "error": "Deepgram API key not configured on server."}), 500

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()

    if len(audio_bytes) < 100:
        return jsonify({"success": False, "error": "Audio too short. Please speak clearly and try again."}), 400

    print(f"[Voice] Received {len(audio_bytes)} bytes of audio for transcription.", flush=True)

    try:
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/wav",
        }
        dg_response = http_requests.post(
            "https://api.deepgram.com/v1/listen",
            headers=headers,
            params={
                "model": "nova-3",
                "smart_format": "true",
                "language": "en",
                "punctuate": "true",
            },
            data=audio_bytes,
            timeout=30,
        )

        if dg_response.status_code == 200:
            result = dg_response.json()
            transcript = (
                result.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
            )
            print(f"[Voice] Deepgram transcript: '{transcript}'", flush=True)
            return jsonify({"success": True, "transcript": transcript}), 200
        else:
            err_detail = dg_response.text[:200]
            print(f"[Voice] Deepgram error {dg_response.status_code}: {err_detail}", flush=True)
            return jsonify({"success": False, "error": f"Deepgram API error ({dg_response.status_code})"}), 502

    except http_requests.exceptions.Timeout:
        return jsonify({"success": False, "error": "Deepgram request timed out. Try again."}), 504
    except Exception as e:
        print(f"[Voice] Transcription error: {e}", flush=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    t_req_start = time.time()
    print("\n====== 🟢 NEW REQUEST RECEIVED ======", flush=True)
    data = request.json or {}
    user_query = data.get("message", "").strip()
    
    thread_id = sanitize_thread_id(data.get("thread_id") or "default")
    session_config = {"configurable": {"thread_id": thread_id}}
    
    if not user_query:
        return jsonify({"response": "Empty prompt received.", "logs": []}), 400

    with LIVE_LOGS_LOCK:
        LIVE_LOGS[thread_id] = []

    smart_gateway.execution_logs = []
    captured_stdout = DualStream(thread_id=thread_id)

    try:
        with contextlib.redirect_stdout(captured_stdout):
            print(f"⏳ [Gateway] Scanning user input for malicious intent (User: '{thread_id}')...", flush=True)
            secure_prompt = verify_user_input(user_query)
            
            if secure_prompt == "SECURITY_BLOCK":
                print("❌ [Security Block] Input failed security guardrail verification.", flush=True)
                return jsonify({
                    "response": "Access Denied: Malicious request or safety violation detected.", 
                    "logs": ["[Security Block] Input blocked by infrastructure guardrails."]
                }), 200

            print(f"⏳ [Engine] Activating Agentic Orchestrator Workflow (User: '{thread_id}')...", flush=True)
            
            state_output = agent_system.invoke(
                {"messages": [{"role": "user", "content": secure_prompt}], "thread_id": thread_id},
                config=session_config
            )
            
            final_response = state_output["messages"][-1].content
            total_s = time.time() - t_req_start
            print(f"✅ [System] Request completed in {total_s:.2f}s.", flush=True)

        with LIVE_LOGS_LOCK:
            timestamped_logs = list(LIVE_LOGS.get(thread_id, []))
        combined_logs = timestamped_logs + smart_gateway.execution_logs

        print(f"✅ 4. Success! Sending answer back to frontend ({len(combined_logs)} log lines).", flush=True)
        return jsonify({
            "response": final_response, 
            "logs": combined_logs
        })

    except Exception as e:
        err_msg = str(e)
        append_live_log(thread_id, f"❌ [CRITICAL SERVER ERROR] {err_msg}")
        print(f"❌ CRITICAL SERVER ERROR: {err_msg}", flush=True)
        with LIVE_LOGS_LOCK:
            timestamped_logs = list(LIVE_LOGS.get(thread_id, []))
        return jsonify({
            "response": f"An execution error occurred inside the agent engine: {err_msg}",
            "logs": timestamped_logs + smart_gateway.execution_logs + [f"[CRITICAL] {err_msg}"]
        }), 500

if __name__ == "__main__":
    flask_port = int(os.environ.get("FLASK_PORT", 5060))
    print(f"Starting Flask backend on 127.0.0.1:{flask_port}...", flush=True)
    app.run(host="127.0.0.1", port=flask_port, debug=False, threaded=True)