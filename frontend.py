import streamlit as st
import requests
import os
import re
import time
import threading
from datetime import datetime
import uuid

st.set_page_config(page_title="Agentic RAG Gateway", page_icon="🤖", layout="wide")
st.title("🤖 Intelligent Enterprise RAG Gateway")
st.caption("Powered by Smart LLM Routing Engine & Cross-Provider Fallbacks")

# Detect backend URL from environment (for Render deployment) or fallback to localhost
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5060")
FLASK_API_URL = f"{BACKEND_URL}/api/chat"
LIVE_LOGS_URL = f"{BACKEND_URL}/api/live_logs"
UPDATE_API_URL = f"{BACKEND_URL}/api/update_knowledge"
STORAGE_STATUS_URL = f"{BACKEND_URL}/api/storage_status"
UPLOAD_PDF_URL = f"{BACKEND_URL}/api/upload_pdf"
DELETE_CLOUD_FILE_URL = f"{BACKEND_URL}/api/delete_cloud_file"

# --- USER IDENTIFICATION SYSTEM (LOCKED STATE) ---
st.sidebar.title("👤 User Profile")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if not st.session_state.thread_id:
    with st.sidebar.form(key="login_form"):
        user_id_input = st.text_input("Enter your Username / Employee ID:")
        login_submitted = st.form_submit_button("🚀 Connect / Login", use_container_width=True)
        
    if (login_submitted or user_id_input) and user_id_input.strip():
        st.session_state.thread_id = user_id_input.strip()
        st.session_state.messages = [] 
        st.rerun() 
    else:
        st.warning("👈 Please enter your Username in the sidebar to securely connect to your personal chat history thread.")
        st.stop()
else:
    st.sidebar.success(f"Verified User: **{st.session_state.thread_id}**")


if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DAILY RESET LOGIC ---
current_date = datetime.now().strftime("%Y-%m-%d")

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

saved_files = [f for f in os.listdir(PROJECTS_DIR) if os.path.isfile(os.path.join(PROJECTS_DIR, f)) and f != "agent.md"]

if "file_to_delete" not in st.session_state:
    st.session_state.file_to_delete = None

if saved_files:
    st.sidebar.write("📥 Click to download or manage reports:")
    for file_name in saved_files:
        file_path = os.path.join(PROJECTS_DIR, file_name)
        
        col_doc, col_del = st.sidebar.columns([4, 1])
        
        with open(file_path, "rb") as file_data:
            col_doc.download_button(
                label=f"📄 {file_name}",
                data=file_data,
                file_name=file_name,
                key=f"download_{file_name}",
                use_container_width=True
            )
        
        if col_del.button("❌", key=f"del_btn_{file_name}", help=f"Delete {file_name}"):
            st.session_state.file_to_delete = file_name
            st.rerun()

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
st.sidebar.caption("Upload or sync PDF documents directly into active vector indexes.")

uploaded_pdf = st.sidebar.file_uploader(
    "📤 Upload PDF to Knowledge Base", 
    type=["pdf"], 
    key="knowledge_pdf_uploader",
    help="Upload research papers or documents to immediately ground agent responses."
)

if uploaded_pdf is not None:
    if "last_indexed_pdf" not in st.session_state or st.session_state.last_indexed_pdf != uploaded_pdf.name:
        with st.sidebar.status(f"⚙️ Indexing '{uploaded_pdf.name}' into RAG...", expanded=True):
            try:
                files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                data = {"thread_id": st.session_state.thread_id}
                up_res = requests.post(UPLOAD_PDF_URL, files=files, data=data, timeout=60)
                if up_res.status_code == 200:
                    res_data = up_res.json()
                    st.sidebar.success(f"✅ Indexed '{uploaded_pdf.name}' ({res_data.get('chunks', 0)} chunks)!")
                    st.session_state.last_indexed_pdf = uploaded_pdf.name
                    st.rerun()
                else:
                    st.sidebar.error("Failed to index PDF.")
            except Exception as e:
                st.sidebar.error(f"Error processing PDF upload: {e}")

# Fetch real-time storage and index status for this user
try:
    status_res = requests.get(STORAGE_STATUS_URL, params={"thread_id": st.session_state.thread_id}, timeout=5)
    if status_res.status_code == 200:
        storage_data = status_res.json()
        uploaded_pdfs = storage_data.get("uploaded_pdfs", [])
        cloud_files = storage_data.get("cloud_files", [])
        index_ready = storage_data.get("vector_index_ready", False)

        st.sidebar.markdown("#### 📚 Active Knowledge Base Files")
        if uploaded_pdfs:
            for up in uploaded_pdfs:
                st.sidebar.markdown(f"- 📄 **`{up['filename']}`** ({up['size_kb']} KB)")
        elif cloud_files:
            for cf in cloud_files:
                size_kb = cf.get("bytes", 0) / 1024
                st.sidebar.markdown(f"- 📄 **`{cf['filename']}`** ({size_kb:.1f} KB)")
        else:
            st.sidebar.warning("⚠️ No documents indexed yet. Use the upload box above to add your first PDF!")

        st.sidebar.markdown(f"**Vector Store Status:** {'🟢 Active & Loaded' if index_ready else '🔴 Empty / Waiting for PDF'}")

        with st.sidebar.expander("☁️ Cloud Storage (Cloudinary)", expanded=False):
            if cloud_files:
                st.write(f"**{len(cloud_files)} file(s) in Cloudinary:**")
                for cf in cloud_files:
                    c1, c2 = st.columns([3, 1])
                    c1.write(f"📄 `{cf['filename']}` ({cf.get('bytes', 0)/1024:.1f} KB)")
                    res_type = "image" if cf.get("format") == "pdf" else "raw"
                    if c2.button("🗑️", key=f"del_c_{cf['public_id']}", help=f"Delete {cf['filename']} from Cloudinary"):
                        try:
                            del_res = requests.post(
                                DELETE_CLOUD_FILE_URL, 
                                json={"public_id": cf["public_id"], "filename": cf["filename"], "resource_type": res_type, "thread_id": st.session_state.thread_id},
                                timeout=15
                            )
                            if del_res.status_code == 200:
                                st.success(f"Deleted {cf['filename']}!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
            else:
                st.caption("No files currently detected in Cloudinary folder `agentic-rag-data`.")
except Exception:
    st.sidebar.caption("Backend offline: cannot check knowledge status.")

# 3. Cloudinary Sync Button
if st.sidebar.button("🔄 Sync Cloud Assets to RAG System"):
    pipeline_container = st.empty()
    
    with pipeline_container.status("🛸 Syncing cloud storage & building index...", expanded=True) as pipe_status:
        try:
            payload = {"thread_id": st.session_state.thread_id}
            response = requests.post(UPDATE_API_URL, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                pipeline_logs = result.get("logs", [])
                is_success = result.get("success", False)
                message = result.get("message", "")
                file_count = result.get("file_count", 0)
                
                for log_line in pipeline_logs:
                    st.write(f"✔️ {log_line}")
                
                if is_success and file_count > 0:
                    pipe_status.update(label="✅ Knowledge Base Update Complete!", state="complete")
                    st.sidebar.success(f"Success! {file_count} document(s) indexed into RAG.")
                    st.rerun()
                else:
                    pipe_status.update(label="⚠️ Sync Finished: No Files Found", state="complete")
                    st.sidebar.warning(f"⚠️ {message}")
            else:
                st.sidebar.error("Failed to sync cloud resources.")
                
        except requests.exceptions.ConnectionError:
            st.sidebar.error("Connection Error: Check if Flask backend is alive.")


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
        if msg.get("logs"):
            with st.expander("📋 View Backend Execution Logs", expanded=False):
                for l in msg["logs"]:
                    st.code(l, language="text")
        st.markdown(msg["content"])

# --- Capture New Conversation Prompts with LIVE Real-Time Backend Log Streaming ---
if user_input := st.chat_input("Ask a question or request a technical function..."):
    
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        log_status = st.status("🤖 Orchestrator running — Live Backend Logs:", expanded=True)
        live_log_box = log_status.empty()
        response_placeholder = st.empty()

        # Run /api/chat in a background worker thread so we can poll /api/live_logs every 0.4s in real time!
        chat_result_holder = {"response": None, "error": None, "done": False}
        active_thread_id = st.session_state.thread_id

        def _call_chat_backend():
            try:
                payload = {"message": user_input, "thread_id": active_thread_id}
                res = requests.post(FLASK_API_URL, json=payload, timeout=180)
                chat_result_holder["response"] = res
            except Exception as ex:
                chat_result_holder["error"] = str(ex)
            finally:
                chat_result_holder["done"] = True

        worker = threading.Thread(target=_call_chat_backend, daemon=True)
        worker.start()

        start_ts = time.time()
        last_seen_logs = []
        while not chat_result_holder["done"]:
            elapsed = time.time() - start_ts
            try:
                poll_res = requests.get(LIVE_LOGS_URL, params={"thread_id": active_thread_id}, timeout=2)
                if poll_res.status_code == 200:
                    live_lines = poll_res.json().get("logs", [])
                    if live_lines != last_seen_logs:
                        last_seen_logs = live_lines
                        latest_line = live_lines[-1] if live_lines else "Initializing..."
                        log_status.update(label=f"⚙️ [{elapsed:.1f}s] {latest_line}", state="running", expanded=True)
                        live_log_box.code("\n".join(live_lines), language="text")
            except Exception:
                pass
            time.sleep(0.4)

        total_elapsed = time.time() - start_ts

        if chat_result_holder["error"]:
            log_status.update(label=f"❌ Connection Error after {total_elapsed:.1f}s", state="error", expanded=True)
            response_placeholder.error(f"Backend connection error: {chat_result_holder['error']}")
        else:
            api_response = chat_result_holder["response"]
            if api_response is not None and api_response.status_code == 200:
                result = api_response.json()
                agent_answer = result.get("response", "")
                execution_logs = result.get("logs", []) or last_seen_logs
                
                live_log_box.code("\n".join(execution_logs), language="text")
                log_status.update(label=f"✅ Completed in {total_elapsed:.1f}s (Click to inspect backend logs)", state="complete", expanded=False)
                
                current_cost = "$0.000000"
                for log in execution_logs:
                    match = re.search(r"Query Cost:\s*\$(\d+\.\d+)", log)
                    if match:
                        current_cost = f"${match.group(1)}"
                
                st.session_state.cost_history.append({"query": user_input, "cost": current_cost})
                response_placeholder.markdown(agent_answer)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": agent_answer,
                    "logs": execution_logs
                })
            else:
                err_msg = "Error communicating with backend processing nodes."
                err_logs = last_seen_logs
                try:
                    err_json = api_response.json()
                    if "response" in err_json:
                        err_msg = err_json["response"]
                    if "logs" in err_json:
                        err_logs = err_json["logs"]
                except Exception:
                    pass
                if err_logs:
                    live_log_box.code("\n".join(err_logs), language="text")
                log_status.update(label=f"❌ Backend Error after {total_elapsed:.1f}s", state="error", expanded=True)
                response_placeholder.error(err_msg)