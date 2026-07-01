import streamlit as st
import requests
import os
import re
import time
from datetime import datetime
import uuid

st.set_page_config(page_title="Agentic RAG Gateway", page_icon="🤖", layout="wide")
st.title("🤖 Intelligent Enterprise RAG Gateway")
st.caption("Powered by Smart LLM Routing Engine & Cross-Provider Fallbacks")

FLASK_API_URL = "http://127.0.0.1:5050/api/chat"
UPDATE_API_URL = "http://127.0.0.1:5050/api/update_knowledge"

# --- USER IDENTIFICATION SYSTEM (LOCKED STATE) ---
st.sidebar.title("👤 User Profile")

# Initialize the thread_id state if it doesn't exist yet
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

# If the user hasn't logged in yet, show the input box
if not st.session_state.thread_id:
    user_id_input = st.sidebar.text_input("Enter your Username / Employee ID:")
    
    if user_id_input:
        # Once they press Enter, save the ID, clear history, and reload!
        st.session_state.thread_id = user_id_input
        st.session_state.messages = [] 
        st.rerun() 
    else:
        # Halt execution until they provide an ID
        st.warning("👈 Please enter your Username in the sidebar to securely connect to your personal chat history thread.")
        st.stop()
else:
    # If they ARE logged in, hide the input box and show uneditable text
    st.sidebar.success(f"Verified User: **{st.session_state.thread_id}**")


if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DAILY RESET LOGIC ---
# Grab the real-world current date string (YYYY-MM-DD)
current_date = datetime.now().strftime("%Y-%m-%d")

# If calendar date rolls over into a new day, clear out prior execution history
if "cost_date" not in st.session_state or st.session_state.cost_date != current_date:
    st.session_state.cost_history = []
    st.session_state.cost_date = current_date


# --- Sidebar Document Explorer ---
st.sidebar.markdown("---")
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
        
        # Create side-by-side columns: 80% width for document link, 20% width for 'X' mark
        col_doc, col_del = st.sidebar.columns([4, 1])
        
        with open(file_path, "rb") as file_data:
            col_doc.download_button(
                label=f"📄 {file_name}",
                data=file_data,
                file_name=file_name,
                key=f"download_{file_name}",
                use_container_width=True
            )
        
        # Place the small inline 'X' button in the second column
        if col_del.button("❌", key=f"del_btn_{file_name}", help=f"Delete {file_name}"):
            st.session_state.file_to_delete = file_name
            st.rerun()

    # Handle the Confirmation Step directly beneath the file layout
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
                # Dynamic thread extraction ensuring multi-tenant network safety
                payload = {"message": user_input, "thread_id": st.session_state.thread_id}
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