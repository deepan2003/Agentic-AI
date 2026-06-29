import os
import io
import contextlib
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from ingestion import build_local_vector_db

load_dotenv()

from llm_gateway import smart_gateway
from gateway import verify_user_input
from main import agent_system

app = Flask(__name__)
CORS(app) 

@app.route("/api/update_knowledge", methods=["POST"])
def update_knowledge_endpoint():
    print("\n====== 🔄 KNOWLEDGE BASE UPDATE REQUESTED ======", flush=True)
    
    captured_stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured_stdout):
            print("⏳ [Pipeline] Initializing S3 connection parameters...", flush=True)
            build_local_vector_db()
            print("✅ [Pipeline] Vector index hot-swapped and live in active memory.", flush=True)

        printed_logs = [line.strip() for line in captured_stdout.getvalue().splitlines() if line.strip()]
        return jsonify({
            "success": True,
            "logs": printed_logs
        })

    except Exception as e:
        print(f"❌ PIPELINE ERROR: {str(e)}", flush=True)
        return jsonify({
            "success": False,
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
            
            # Just grab the final, fully-verified answer directly from the graph!
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
    app.run(host="0.0.0.0", port=5050, debug=False)