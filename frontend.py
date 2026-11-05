import streamlit as st
import requests
import os
import re
import time
import threading
from datetime import datetime

st.set_page_config(page_title="Agentic RAG Gateway", page_icon="🤖", layout="wide")
st.title("🤖 Intelligent Enterprise RAG Gateway")
st.caption("Powered by Smart LLM Routing Engine & Cross-Provider Fallbacks • *Due to cost efficiency, this Agent is hosted on Render rather than AWS.*")

# Detect backend URL from environment (for Render deployment) or fallback to localhost
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5060")
FLASK_API_URL = f"{BACKEND_URL}/api/chat"
LIVE_LOGS_URL = f"{BACKEND_URL}/api/live_logs"
UPDATE_API_URL = f"{BACKEND_URL}/api/update_knowledge"
STORAGE_STATUS_URL = f"{BACKEND_URL}/api/storage_status"
UPLOAD_PDF_URL = f"{BACKEND_URL}/api/upload_pdf"
DELETE_CLOUD_FILE_URL = f"{BACKEND_URL}/api/delete_cloud_file"


def sanitize_user_id(user_id: str) -> str:
    """Normalizes username into a safe lowercase identifier so User A and User B are strictly isolated."""
    if not user_id:
        return ""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', str(user_id).strip().lower()).strip('_')


# --- USER IDENTIFICATION SYSTEM (WITH SWITCH USER / LOGOUT) ---
st.sidebar.title("👤 User Profile")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "user_messages_map" not in st.session_state:
    # Keeps per-user UI chat history in browser session so returning User A sees their messages too
    st.session_state.user_messages_map = {}

if not st.session_state.thread_id:
    with st.sidebar.form(key="login_form"):
        user_id_input = st.text_input("Enter your Username / Employee ID:")
        login_submitted = st.form_submit_button("🚀 Connect / Login", use_container_width=True)
        
    if (login_submitted or user_id_input) and user_id_input.strip():
        clean_uid = sanitize_user_id(user_id_input)
        st.session_state.thread_id = clean_uid
        # Restore this user's UI messages if they previously logged in during this browser session
        st.session_state.messages = list(st.session_state.user_messages_map.get(clean_uid, []))
        st.session_state.last_indexed_pdf = None
        st.session_state.file_to_delete = None
        st.rerun() 
    else:
        st.warning("👈 Please enter your Username in the sidebar to securely connect to your personal RAG knowledge base and chat thread.")
        st.stop()
else:
    col_user, col_logout = st.sidebar.columns([3, 1])
    col_user.success(f"Verified User: **{st.session_state.thread_id}**")
    if col_logout.button("🚪", help="Logout / Switch User", use_container_width=True):
        # Save current user's UI messages before logging out so when User A comes back, their chat is still there
        st.session_state.user_messages_map[st.session_state.thread_id] = list(st.session_state.get("messages", []))
        st.session_state.thread_id = None
        st.session_state.messages = []
        st.session_state.last_indexed_pdf = None
        st.session_state.file_to_delete = None
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DAILY RESET LOGIC ---
current_date = datetime.now().strftime("%Y-%m-%d")

if "cost_date" not in st.session_state or st.session_state.cost_date != current_date:
    st.session_state.cost_history = []
    st.session_state.cost_date = current_date


# --- Sidebar Document Explorer (STRICTLY ISOLATED PER USER) ---
st.sidebar.markdown("---")
st.sidebar.title("📂 Saved Documents")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_PROJECTS_DIR = os.path.join(BASE_DIR, "projects", sanitize_user_id(st.session_state.thread_id))

if not os.path.exists(USER_PROJECTS_DIR):
    os.makedirs(USER_PROJECTS_DIR, exist_ok=True)

saved_files = sorted([
    f for f in os.listdir(USER_PROJECTS_DIR)
    if os.path.isfile(os.path.join(USER_PROJECTS_DIR, f)) and f != "agent.md"
])

if "file_to_delete" not in st.session_state:
    st.session_state.file_to_delete = None

if saved_files:
    st.sidebar.write("📥 Click to download or manage your reports:")
    for file_name in saved_files:
        file_path = os.path.join(USER_PROJECTS_DIR, file_name)
        
        col_doc, col_del = st.sidebar.columns([4, 1])
        
        with open(file_path, "rb") as file_data:
            col_doc.download_button(
                label=f"📄 {file_name}",
                data=file_data,
                file_name=file_name,
                key=f"download_{st.session_state.thread_id}_{file_name}",
                use_container_width=True
            )
        
        if col_del.button("❌", key=f"del_btn_{st.session_state.thread_id}_{file_name}", help=f"Delete {file_name}"):
            st.session_state.file_to_delete = file_name
            st.rerun()

    if st.session_state.file_to_delete:
        target_file = st.session_state.file_to_delete
        st.sidebar.markdown("---")
        st.sidebar.warning(f"Are you sure you want to delete **{target_file}**?")
        
        col_yes, col_no = st.sidebar.columns(2)
        if col_yes.button("✅ Yes", key="confirm_delete_yes", use_container_width=True):
            try:
                os.remove(os.path.join(USER_PROJECTS_DIR, target_file))
                st.sidebar.success(f"Deleted {target_file}!")
                st.session_state.file_to_delete = None
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error deleting file: {e}")
                
        if col_no.button("🛑 Cancel", key="confirm_delete_no", use_container_width=True):
            st.session_state.file_to_delete = None
            st.rerun()
else:
    st.sidebar.info(f"No documents saved yet for **{st.session_state.thread_id}**.")


# --- Sidebar Knowledge Base Management (STRICTLY ISOLATED PER USER) ---
st.sidebar.markdown("---")
st.sidebar.title("🧠 Core Knowledge Management")
st.sidebar.caption(f"Private RAG Knowledge Base for **{st.session_state.thread_id}**")

# Scope the uploader key to the active user so User B never sees User A's uploaded file widget state
uploader_key = f"knowledge_pdf_uploader_{st.session_state.thread_id}"
st.sidebar.caption("📄 *Note: PDF file should be a maximum of 3 pages.*")
uploaded_pdf = st.sidebar.file_uploader(
    "📤 Upload PDF to Your Private Knowledge Base", 
    type=["pdf"], 
    key=uploader_key,
    help="Uploaded PDFs are strictly isolated to your username and automatically restored when you return (Max 3 pages)."
)

if uploaded_pdf is not None:
    last_indexed_key = f"{st.session_state.thread_id}:{uploaded_pdf.name}"
    if st.session_state.get("last_indexed_pdf") != last_indexed_key:
        with st.sidebar.status(f"⚙️ Indexing '{uploaded_pdf.name}' into {st.session_state.thread_id}'s RAG...", expanded=True):
            try:
                # Page count advisory check
                import pypdf
                import io
                try:
                    pdf_reader = pypdf.PdfReader(io.BytesIO(uploaded_pdf.getvalue()))
                    page_count = len(pdf_reader.pages)
                    if page_count > 3:
                        st.sidebar.warning(f"⚠️ Notice: Document has {page_count} pages (Recommended: maximum 3 pages). Processing first chunks...")
                except Exception:
                    pass

                files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                data = {"thread_id": st.session_state.thread_id}
                up_res = requests.post(UPLOAD_PDF_URL, files=files, data=data, timeout=60)
                if up_res.status_code == 200:
                    res_data = up_res.json()
                    st.sidebar.success(f"✅ Indexed '{uploaded_pdf.name}' ({res_data.get('chunks', 0)} chunks)!")
                    st.session_state.last_indexed_pdf = last_indexed_key
                    st.rerun()
                else:
                    st.sidebar.error("Failed to index PDF.")
            except Exception as e:
                st.sidebar.error(f"Error processing PDF upload: {e}")

# Fetch real-time storage and index status strictly for this user
try:
    status_res = requests.get(STORAGE_STATUS_URL, params={"thread_id": st.session_state.thread_id}, timeout=5)
    if status_res.status_code == 200:
        storage_data = status_res.json()
        uploaded_pdfs = storage_data.get("uploaded_pdfs", [])
        cloud_files = storage_data.get("cloud_files", [])
        index_ready = storage_data.get("vector_index_ready", False)

        st.sidebar.markdown("#### 📚 Your Active Knowledge Base Files")
        if uploaded_pdfs:
            for up in uploaded_pdfs:
                c_pdf, c_del_pdf = st.sidebar.columns([4, 1])
                c_pdf.markdown(f"- 📄 **`{up['filename']}`** ({up['size_kb']} KB)")
                if c_del_pdf.button("🗑️", key=f"del_local_pdf_{st.session_state.thread_id}_{up['filename']}", help=f"Remove {up['filename']}"):
                    try:
                        # Also find matching cloud public_id if backed up
                        matching_pub_id = next((cf["public_id"] for cf in cloud_files if cf["filename"] == up["filename"]), None)
                        requests.post(
                            DELETE_CLOUD_FILE_URL,
                            json={
                                "public_id": matching_pub_id,
                                "filename": up["filename"],
                                "resource_type": "raw",
                                "thread_id": st.session_state.thread_id
                            },
                            timeout=15
                        )
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Delete failed: {e}")
        elif cloud_files:
            for cf in cloud_files:
                size_kb = cf.get("bytes", 0) / 1024
                st.sidebar.markdown(f"- 📄 **`{cf['filename']}`** ({size_kb:.1f} KB)")
        else:
            st.sidebar.warning(f"⚠️ Knowledge base for **{st.session_state.thread_id}** is empty. Upload a PDF above to enable RAG!")

        st.sidebar.markdown(f"**Vector Store Status:** {'🟢 Active & Loaded' if index_ready else '🔴 Empty / Waiting for PDF'}")

        with st.sidebar.expander("☁️ Your Cloud Backup (Cloudinary)", expanded=False):
            if cloud_files:
                st.write(f"**{len(cloud_files)} file(s) backed up for `{st.session_state.thread_id}`:**")
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
                st.caption(f"No cloud files found for user `{st.session_state.thread_id}`.")
except Exception:
    st.sidebar.caption("Backend offline: cannot check knowledge status.")

# 3. Cloudinary Sync Button
if st.sidebar.button("🔄 Sync My Cloud Assets to RAG"):
    pipeline_container = st.empty()
    
    with pipeline_container.status("🛸 Syncing your private cloud storage & building index...", expanded=True) as pipe_status:
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
                    st.sidebar.success(f"Success! {file_count} document(s) indexed into your private RAG.")
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

# Currency Toggle (USD <-> INR)
if "cost_currency" not in st.session_state:
    st.session_state.cost_currency = "USD"

USD_TO_INR_RATE = 86.50  # 1 USD ≈ 86.50 INR

is_inr = st.session_state.cost_currency == "INR"
btn_label = "🇮🇳 Convert to INR (₹)" if not is_inr else "💵 Convert to USD ($)"
if st.sidebar.button(btn_label, key="currency_toggle_btn", use_container_width=True):
    st.session_state.cost_currency = "INR" if not is_inr else "USD"
    st.rerun()

curr_symbol = "₹" if is_inr else "$"
curr_rate = USD_TO_INR_RATE if is_inr else 1.0

with st.sidebar.expander("📊 View Query Cost History Log", expanded=False):
    if st.session_state.cost_history:
        # Calculate session total
        total_usd = 0.0
        for it in st.session_state.cost_history:
            try:
                raw_val = float(str(it.get('cost', '0')).replace('$', '').replace('₹', '').strip())
                total_usd += raw_val
            except Exception:
                pass
        
        total_disp = total_usd * curr_rate
        st.metric(label=f"Total Session Cost ({'INR' if is_inr else 'USD'})", value=f"{curr_symbol}{total_disp:.4f}")
        st.caption("Conversion rate: 1 USD ≈ ₹86.50" if is_inr else "Native API currency: USD ($)")
        st.markdown("---")
        st.markdown("### Transaction Ledger")
        
        for idx, item in enumerate(st.session_state.cost_history[::-1]):
            st.markdown(f"**Query {len(st.session_state.cost_history) - idx}:** `{item['query']}`")
            try:
                cost_num = float(str(item['cost']).replace('$', '').replace('₹', '').strip())
                display_cost = cost_num * curr_rate
                val_str = f"{curr_symbol}{display_cost:.6f}" if display_cost < 0.01 else f"{curr_symbol}{display_cost:.4f}"
            except Exception:
                val_str = item['cost']
                
            st.metric(label=f"Execution Cost ({'INR' if is_inr else 'USD'})", value=val_str)
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
    st.session_state.user_messages_map[st.session_state.thread_id] = list(st.session_state.messages)
    
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
                st.session_state.user_messages_map[st.session_state.thread_id] = list(st.session_state.messages)
                # If a document was saved during this turn, rerun so the sidebar immediately displays the new file button
                if any("Physical file saved to disk" in l for l in execution_logs):
                    st.rerun()
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