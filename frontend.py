import streamlit as st
import requests
import os
import re
import time
from datetime import datetime

st.set_page_config(page_title="Agentic RAG Gateway", page_icon="🤖", layout="wide")
st.title("🤖 Intelligent Enterprise RAG Gateway")
st.caption("Powered by Smart LLM Routing Engine & Cross-Provider Fallbacks")

FLASK_API_URL = "http://127.0.0.1:5050/api/chat"
UPDATE_API_URL = "http://127.0.0.1:5050/api/update_knowledge"

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- NEW DAILY RESET LOGIC ---
# 1. Grab the real-world current date (e.g., "2026-06-27")
current_date = datetime.now().strftime("%Y-%m-%d")

# 2. If it's a brand new session, OR if the calendar day has changed, wipe the history!
if "cost_date" not in st.session_state or st.session_state.cost_date != current_date:
    st.session_state.cost_history = []
    st.session_state.cost_date = current_date

# --- Sidebar Document Explorer ---
st.sidebar.title("📂 Saved Documents")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")

if not os.path.exists(PROJECTS_DIR):
    os.makedirs(PROJECTS_DIR)

saved_files = [f for f in os.listdir(PROJECTS_DIR) if os.path.isfile(os.path.join(PROJECTS_DIR, f))]

# Track which file is pending deletion across refreshes
if "file_to_delete" not in st.session_state:
    st.session_state.file_to_delete = None

if saved_files:
    st.sidebar.write("📥 Click to download or manage reports:")
    for file_name in saved_files:
        file_path = os.path.join(PROJECTS_DIR, file_name)
        
        # 1. Create side-by-side columns: 80% width for document link, 20% width for 'X' mark
        col_doc, col_del = st.sidebar.columns([4, 1])
        
        with open(file_path, "rb") as file_data:
            col_doc.download_button(
                label=f"📄 {file_name}",
                data=file_data,
                file_name=file_name,
                key=f"download_{file_name}",
                use_container_width=True
            )
        
        # 2. Place the small inline 'X' button in the second column
        if col_del.button("❌", key=f"del_btn_{file_name}", help=f"Delete {file_name}"):
            st.session_state.file_to_delete = file_name
            st.rerun()

    # 3. Handle the Confirmation Step directly beneath the file layout
    if st.session_state.file_to_delete:
        target_file = st.session_state.file_to_delete
        st.sidebar.markdown("---")
        st.sidebar.warning(f"Are you sure you want to delete **{target_file}**?")
        
        col_yes, col_no = st.sidebar.columns(2)
        if col_yes.button("✅ Yes", key="confirm_delete_yes", use_container_width=True):
            try:
                os.remove(os.path.join(PROJECTS_DIR, target_file))
                st.sidebar.success(f"Deleted {target_file}!")
                st.session_state.file_to_delete = None
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error deleting file: {e}")
                
        if col_no.button("🛑 Cancel", key="confirm_delete_no", use_container_width=True):
            st.session_state.file_to_delete = None
            st.rerun()
else:
    st.sidebar.info("No documents saved yet.")

# --- Sidebar Knowledge Base Management ---
st.sidebar.markdown("---")
st.sidebar.title("🧠 Core Knowledge Management")
st.sidebar.caption("Sync external documents from AWS S3 into your active vector indexes.")

if st.sidebar.button("🔄 Sync S3 Assets to RAG System"):
    pipeline_container = st.empty()
    
    with pipeline_container.status("🛸 Contacting secure cluster cloud storage...", expanded=True) as pipe_status:
        try:
            response = requests.post(UPDATE_API_URL, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                pipeline_logs = result.get("logs", [])
                
                for log_line in pipeline_logs:
                    pipe_status.update(label=f"🔄 Processing: {log_line}")
                    st.write(f"✔️ {log_line}")
                    time.sleep(0.3)
                
                pipe_status.update(label="✅ Knowledge Base Update Complete!", state="complete")
                time.sleep(1.5)
                pipeline_container.empty()
                st.sidebar.success("RAG System updated successfully!")
            else:
                pipeline_container.empty()
                st.sidebar.error("Failed to build vector representations from S3 resources.")
                
        except requests.exceptions.ConnectionError:
            pipeline_container.empty()
            st.sidebar.error("Connection Error: Check if Flask backend is alive on port 5050.")

# --- Sidebar Cost Tracker Panel ---
st.sidebar.markdown("---")
st.sidebar.title("💰 System Cost Analytics")

# Changed to expander to ensure metrics persist across chat interactions
with st.sidebar.expander("📊 View Query Cost History Log", expanded=False):
    if st.session_state.cost_history:
        st.markdown("### Transaction Ledger")
        for idx, item in enumerate(st.session_state.cost_history[::-1]):
            st.markdown(f"**Query {len(st.session_state.cost_history) - idx}:** `{item['query']}`")
            st.metric(label="Execution Cost", value=item['cost'])
            st.markdown("---")
    else:
        st.info("No cost transactions logged yet.")

# --- Main Chat Interface Display ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Capture New Conversation Prompts ---
if user_input := st.chat_input("Ask a question or request a technical function..."):
    
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        loading_container = st.empty()
        
        with loading_container.status("🤖 Orchestrator initializing environment infrastructure...", expanded=True) as status:
            try:
                payload = {"message": user_input, "thread_id": "production_session_002"}
                api_response = requests.post(FLASK_API_URL, json=payload, timeout=180)
                
                if api_response.status_code == 200:
                    result = api_response.json()
                    agent_answer = result.get("response", "")
                    execution_logs = result.get("logs", [])
                    
                    for log_line in execution_logs:
                        if "Query Cost:" not in log_line:
                            status.update(label=f"✨ Processing: {log_line}")
                            st.write(f"✔️ {log_line}")
                            time.sleep(0.3)
                    
                    current_cost = "$0.000000"
                    for log in execution_logs:
                        match = re.search(r"Query Cost:\s*\$(\d+\.\d+)", log)
                        if match:
                            current_cost = f"${match.group(1)}"
                    
                    st.session_state.cost_history.append({"query": user_input, "cost": current_cost})
                    
                    loading_container.empty()
                    response_placeholder.markdown(agent_answer)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": agent_answer
                    })
                    st.rerun()
                else:
                    loading_container.empty()
                    response_placeholder.error("Error communicating with backend processing nodes.")
                    
            except requests.exceptions.ConnectionError:
                loading_container.empty()
                response_placeholder.error("Critical Connection Error: Ensure Flask server is active on Port 5050.")