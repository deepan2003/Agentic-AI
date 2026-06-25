import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from llm_gateway import smart_gateway
# Import your guardrail functions from gateway.py
from gateway import verify_user_input, verify_agent_output

app = Flask(__name__)
CORS(app) # Allows Streamlit to communicate across different ports on EC2

@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    data = request.json or {}
    user_query = data.get("message", "").strip()
    
    if not user_query:
        return jsonify({"response": "Empty prompt received.", "logs": []}), 400

    # Clear logs before processing
    smart_gateway.execution_logs = []
    smart_gateway.execution_logs.append(f"[System] Initializing request validation for prompt...")

    # 1. Run Input Security Guardrail
    secure_prompt = verify_user_input(user_query)
    
    # Sync logs from security gateway processing
    current_logs = list(smart_gateway.execution_logs)
    
    if secure_prompt == "SECURITY_BLOCK":
        current_logs.append("[Security Guardrail] Request blocked permanently.")
        return jsonify({
            "response": "Access Denied: Malicious request or safety violation detected.",
            "logs": current_logs
        }), 200

    try:
        # 2. Process Core Prompt / Agentic Network
        # (If your main graph calls smart_gateway.completion internally, it automatically fills execution_logs)
        response_obj = smart_gateway.completion(messages=[{"role": "user", "content": secure_prompt}])
        raw_response = response_obj.content
        
        # 3. Output Guardrail Validation
        final_response = verify_agent_output(raw_response, verified_facts=[])
        
        return jsonify({
            "response": final_response,
            "logs": smart_gateway.execution_logs
        })

    except Exception as e:
        return jsonify({
            "response": f"An execution error occurred: {str(e)}",
            "logs": smart_gateway.execution_logs + [f"[CRITICAL] {str(e)}"]
        }), 500

if __name__ == "__main__":
    # Host on 0.0.0.0 to make it accessible outside localhost within EC2
    app.run(host="0.0.0.0", port=5000, debug=True)