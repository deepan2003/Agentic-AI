import streamlit as st
import requests

# Page Layout configuration mimicking modern clean look
st.set_page_config(page_title="Agentic RAG Gateway", page_icon="🤖", layout="wide")
st.title("🤖 Intelligent Enterprise RAG Gateway")
st.caption("Powered by Smart LLM Routing Engine & Cross-Provider Fallbacks")

# Define backend target API running on EC2
FLASK_API_URL = "http://localhost:5000/api/chat"

# Initialize conversation states
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If there are system routing logs available for this message iteration, show them
        if "logs" in msg and msg["logs"]:
            with st.expander("🔍 View Gateway Execution Trace & Performance Metrics"):
                for log in msg["logs"]:
                    if "[Gateway Warning]" in log or "[CRITICAL]" in log:
                        st.warning(log)
                    elif "Hit!" in log:
                        st.success(log)
                    else:
                        st.code(log, language="bash")

# Capture new conversation prompt
if user_input := st.chat_input("Ask a question or request a technical function..."):
    
    # Render user prompt immediately 
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Process request with visual placeholders
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("Gateway evaluating infrastructure security and optimal routing parameters..."):
            try:
                # Post transaction to local Flask server pipeline
                payload = {"message": user_input}
                api_response = requests.post(FLASK_API_URL, json=payload, timeout=45)
                
                if api_response.status_code == 200:
                    result = api_response.json()
                    agent_answer = result.get("response", "")
                    execution_logs = result.get("logs", [])
                    
                    # Output final clean text answer
                    response_placeholder.markdown(agent_answer)
                    
                    # Render the explicit routing steps visually right beneath the text block
                    with st.expander("🔍 View Gateway Execution Trace & Performance Metrics", expanded=True):
                        for log in execution_logs:
                            if "[Gateway Warning]" in log or "[CRITICAL]" in log:
                                st.warning(log)
                            elif "Hit!" in log:
                                st.success(log)
                            else:
                                st.code(log, language="bash")
                    
                    # Commit state parameters to history object
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": agent_answer, 
                        "logs": execution_logs
                    })
                else:
                    response_placeholder.error("Error communicating with backend processing nodes.")
                    
            except requests.exceptions.ConnectionError:
                response_placeholder.error("Critical Connection Error: Ensure Flask server is active on Port 5000.")