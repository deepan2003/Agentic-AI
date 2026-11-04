import os
import io
import contextlib
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from ingestion import build_local_vector_db, list_cloud_storage_files

load_dotenv()

from llm_gateway import smart_gateway
from gateway import verify_user_input
from main import agent_system

app = Flask(__name__)
CORS(app) 

@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "Agentic RAG Backend"}), 200

@app.route("/api/storage_status", methods=["GET"])
def storage_status_endpoint():
    """Returns actual files present in Cloudinary and the local FAISS index status."""
    try:
        cloud_files = list_cloud_storage_files()
        vector_db_path = "./vector_store/research_papers"
        has_index = os.path.exists(vector_db_path) and os.path.exists(os.path.join(vector_db_path, "index.faiss"))
        
        return jsonify({
            "success": True,
            "cloud_files": cloud_files,
            "cloud_file_count": len(cloud_files),
            "vector_index_ready": has_index
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "cloud_files": [],
            "cloud_file_count": 0,
            "vector_index_ready": False
        }), 500

@app.route("/api/update_knowledge", methods=["POST"])
def update_knowledge_endpoint():
    print("\n====== 🔄 KNOWLEDGE BASE UPDATE REQUESTED ======", flush=True)
    
    captured_stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured_stdout):
            print("⏳ [Pipeline] Checking Cloudinary storage and vector index...", flush=True)
            result = build_local_vector_db()

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

@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    print("\n====== 🟢 NEW REQUEST RECEIVED ======", flush=True)
    data = request.json or {}
    user_query = data.get("message", "").strip()
    
    thread_id = data.get("thread_id", "web_default_session")
    session_config = {"configurable": {"thread_id": thread_id}}
    
    if not user_query:
        return jsonify({"response": "Empty prompt received.", "logs": []}), 400

    smart_gateway.execution_logs = []
    captured_stdout = io.StringIO()

    try:
        with contextlib.redirect_stdout(captured_stdout):
            print("⏳ [Gateway] Scanning user input for malicious intent...", flush=True)
            secure_prompt = verify_user_input(user_query)
            
            if secure_prompt == "SECURITY_BLOCK":
                print("❌ [Security Block] Input failed security guardrail verification.", flush=True)
                return jsonify({
                    "response": "Access Denied: Malicious request or safety violation detected.", 
                    "logs": ["[Security Block] Input blocked by infrastructure guardrails."]
                }), 200

            print(f"⏳ [Engine] Activating Agentic Orchestrator Workflow...", flush=True)
            
            # The agent_system automatically runs the output guardrail internally!
            state_output = agent_system.invoke(
                {"messages": [{"role": "user", "content": secure_prompt}]},
                config=session_config
            )
            
            # Grab the final verified answer
            final_response = state_output["messages"][-1].content
            print("✅ [System] Gateway approval verified successfully.", flush=True)

        printed_logs = [line.strip() for line in captured_stdout.getvalue().splitlines() if line.strip()]
        combined_logs = printed_logs + smart_gateway.execution_logs

        print("✅ 4. Success! Sending answer back to frontend.", flush=True)
        return jsonify({
            "response": final_response, 
            "logs": combined_logs
        })

    except Exception as e:
        print(f"❌ CRITICAL SERVER ERROR: {str(e)}", flush=True)
        return jsonify({
            "response": f"An execution error occurred inside the agent engine: {str(e)}",
            "logs": smart_gateway.execution_logs + [f"[CRITICAL] {str(e)}"]
        }), 500

if __name__ == "__main__":
    flask_port = int(os.environ.get("FLASK_PORT", 5060))
    print(f"Starting Flask backend on 127.0.0.1:{flask_port}...", flush=True)
    app.run(host="127.0.0.1", port=flask_port, debug=False)
