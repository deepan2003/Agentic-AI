# 🤖 Production Agentic RAG System

A full-stack, production-grade **Multi-Agent RAG (Retrieval-Augmented Generation)** system with an intelligent LLM gateway, dual-layer security guardrails, hallucination filtering, and live deployment on AWS EC2.

🔗 **Live Demo:** [http://51.20.249.245:8501](http://51.20.249.245:8501)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Core Components](#core-components)
- [LLM Gateway](#llm-gateway)
- [Security Guardrails](#security-guardrails)
- [Hybrid Research Sub-Agent](#hybrid-research-sub-agent)
- [Evaluation System](#evaluation-system)
- [Setup & Installation](#setup--installation)
- [Environment Variables](#environment-variables)
- [Running the System](#running-the-system)
- [API Endpoints](#api-endpoints)

---

## Overview

This system goes beyond a simple RAG chatbot. It is a **multi-agent orchestration pipeline** where:

- A **LangGraph** main orchestrator routes every query through an intent classifier before deciding which agent path to activate
- A **SmartLLMGateway** abstracts all LLM calls with caching, smart routing, load balancing, provider fallback, and cost tracking
- A **Hybrid Research Sub-Agent** intelligently decides between a local FAISS vector database and live web search depending on relevancy
- A **dual-layer security guardrail** blocks prompt injection and credential leaks before any LLM call is made
- A **hallucination guardrail** cross-checks every final response against verified facts before delivery

---

## Architecture

### Full System Flow

```
[ USER PROMPT ]
       │
       ▼
┌─────────────────────────┐
│  1. INPUT GATEWAY GUARD │ ◄── gateway.py → verify_user_input()
│  (Regex + LLM Layer)    │     - Regex: blocks PIN/credential patterns
└────────────┬────────────┘     - LLM: detects prompt injection (fail-secure)
             │ (If Secure)
             ▼
┌─────────────────────────┐
│  2. MAIN ORCHESTRATOR   │ ◄── main.py → main_orchestrator_node()
│  (LangGraph StateGraph) │     - Groq Llama-3.1-8b-instant (intent routing)
└────────────┬────────────┘     - LangGraph MemorySaver (session memory)
             │
             ▼
    [ INTENT CLASSIFIER ]
             │
    ┌────────┼────────────────┐
    │        │                │
    ▼        ▼                ▼
CHITCHAT  SAVE_HISTORY    RESEARCH
    │        │                │
    │        └────────┬───────┘
    │                 ▼
    │   ┌─────────────────────────┐
    │   │  3. HYBRID RESEARCH     │ ◄── sub_agents.py
    │   │     SUB-AGENT           │     - Query local FAISS vector DB
    │   │                         │     - LLM relevancy gate
    │   │  ┌─────────────────┐    │     - Tavily web search fallback
    │   │  │ Local PDF FAISS │    │     - BeautifulSoup page fetch
    │   │  │ (AWS S3 source) │    │     - Query optimizer (year injection)
    │   │  └─────────────────┘    │
    │   │  ┌─────────────────┐    │
    │   │  │  Tavily Search  │    │
    │   │  │  + Web Fetch    │    │
    │   │  └─────────────────┘    │
    │   └────────────┬────────────┘
    │                │
    │                ▼
    │   ┌─────────────────────────┐
    │   │  4. ReAct TOOL LOOP     │ ◄── create_react_agent()
    │   │                         │     - execute_python_code (code tasks)
    │   │  [Tool] Code Executor   │     - save_file_to_disk (file tasks)
    │   │  [Tool] File Saver      │
    │   └────────────┬────────────┘
    │                │
    │                ▼
    │   ┌─────────────────────────┐
    │   │  5. Pydantic FactSheet  │ ◄── Structured output (LangChain)
    │   │  (Strict Fact Extract)  │     - summary, verified_facts, sources_used
    │   └────────────┬────────────┘
    │                │
    └────────┬───────┘
             ▼
┌─────────────────────────┐
│  6. MAIN AGENT ANSWER   │ ◄── Groq Llama-3.1-8b-instant
│     COMPOSITION         │     - Combines rules + research context
└────────────┬────────────┘     - Generates final response
             │
             ▼
┌─────────────────────────┐
│  7. OUTPUT GATEWAY GUARD│ ◄── gateway.py → verify_agent_output()
│  (Hallucination Check)  │     - Cross-checks against verified_facts
└────────────┬────────────┘     - Rewrites if hallucination detected
             │ (If Validated)
             ▼
     [ AGENT RESPONSE ]
```

---

### LLM Gateway Architecture

```
[ Any LLM Call ]
       │
       ▼
┌───────────────────┐
│   SmartLLMGateway │ ──► [ Check MD5 Cache ] ── (Hit: Return instantly, 0 tokens)
└───────────────────┘
       │ (Miss)
       ▼
[ Smart Router ]
       ├──► Simple task (greeting, routing, safety)?  ──► Llama-3.1-8b-instant
       │         └──► Load-balanced Groq API key rotation
       │
       └──► Complex task (code, math, deep research)?  ──► Llama-3.3-70b-versatile
                 └──► Primary Fails? ──► Auto-fallback to Gemini API
       │
       ▼
[ Cost Tracker ] ──► Logs input/output tokens + USD cost per query
       │
       ▼
[ Final Response ]
```

---

## Folder Structure

```
AGENTIC AI/
│
├── projects/
│   └── agent.md                  # Global system rulebook for the orchestrator
│
├── tools/
│   ├── __init__.py
│   ├── file_system.py            # save_file_to_disk tool (LangChain @tool)
│   ├── research.py               # query_local_research_papers (FAISS lookup)
│   └── web_search.py             # custom_web_search + custom_web_fetch (Tavily + BS4)
│
├── vector_store/
│   └── research_papers/          # FAISS index (auto-built from S3 PDFs via ingestion.py)
│
├── generating dataset for langsmith/  # Synthetic Q&A data scripts for LangSmith eval
│
├── __pycache__/
│
├── .env                          # API keys (Groq, Gemini, Tavily, AWS, LangSmith)
├── .gitignore
├── agent.md                      # Orchestrator rules (also served from projects/)
├── app.py                        # Flask REST API server (port 5050)
├── evaluate_system.py            # LangSmith evaluation runner (4 LLM-as-judge metrics)
├── frontend.py                   # Streamlit UI (chat + file manager + cost tracker)
├── gateway.py                    # Input/output security guardrails
├── ingestion.py                  # S3 → PDF → FAISS ingestion pipeline
├── llm_gateway.py                # SmartLLMGateway (routing, cache, fallback, cost)
├── main.py                       # LangGraph orchestrator + intent classifier
├── rag-agent-user-accessKeys.csv # AWS IAM access keys (keep private, never commit)
├── rag-key.pem                   # EC2 SSH key (keep private, never commit)
├── README.md
├── requirements.txt
├── sub_agents.py                 # Hybrid Research Sub-Agent + Pydantic FactSheet
├── synthetic_data.json           # Generated Q&A pairs for LangSmith benchmark
└── Workflow_image.png            # Architecture diagram
```

---

## Core Components

### `main.py` — LangGraph Orchestrator

The central brain. Defines a `StateGraph` with a single `main_orchestrator_node` that:

1. Loads the global rulebook from `projects/agent.md`
2. Runs the **Intent Classifier** (Llama-3.1-8b-instant via SmartLLMGateway) to classify the query as `CHITCHAT`, `SAVE_HISTORY`, or `RESEARCH`
3. Routes to the correct path and composes the final answer
4. Uses `MemorySaver` as the LangGraph checkpointer for persistent multi-turn memory per `thread_id`

### `app.py` — Flask REST API

Exposes two endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/chat` | POST | Accepts `{message, thread_id}`, runs the full agent pipeline, returns `{response, logs}` |
| `/api/update_knowledge` | POST | Triggers `ingestion.py` to pull PDFs from S3 and hot-swap the FAISS index |

### `frontend.py` — Streamlit UI

- Chat interface with real-time execution log display (streamed from Flask logs)
- **Sidebar Document Explorer**: download or delete files saved by the agent to `./projects/`
- **Knowledge Sync Button**: triggers `/api/update_knowledge` with a live progress display
- **Cost Analytics Panel**: per-query USD cost history with daily reset

---

## LLM Gateway

**File:** `llm_gateway.py` — `SmartLLMGateway` class

The gateway is the single point of contact for all LLM calls in the system. It has 5 pillars:

| Pillar | Mechanism |
|---|---|
| **Unified API** | Single `.completion()` call regardless of provider |
| **Smart Routing** | Keyword-based complexity detection → 8b (simple) or 70b (code/math) |
| **MD5 Prompt Cache** | Exact prompt+model hash → instant return, zero tokens spent |
| **Load Balancing** | Cycles through `GROQ_API_KEY` + `GROQ_API_KEY_BACKUP` on each request |
| **Auto Fallback** | If Groq fails → silently retries on Gemini API |

Cost tracking logs `input_tokens`, `output_tokens`, and USD cost per query using per-model pricing tiers.

---

## Security Guardrails

**File:** `gateway.py`

### Input Guard — `verify_user_input()`

Two-layer check before any agent processing:

**Layer 1 — Regex (deterministic, zero latency):**
- Blocks patterns like `pin: 1234`, `account number: 12345678`, `cvv: 123`

**Layer 2 — LLM Cognitive (Groq Llama-3.1-8b-instant):**
- Detects prompt injection attempts (`ignore previous instructions`, `override system rules`, etc.)
- Returns `SECURITY_BLOCK` on any failure — **fail-secure by design**

### Output Guard — `verify_agent_output()`

After the agent composes its answer:
- Receives the `verified_facts` list from the Pydantic `FactSheet`
- Runs a factual cross-check via Groq Llama-3.1-8b-instant
- Rewrites the response to remove any hallucinated claims not present in the verified facts
- Falls back to raw response if the verification LLM call itself fails

---

## Hybrid Research Sub-Agent

**File:** `sub_agents.py` — `hybrid_research_worker()`

A 6-stage pipeline that produces a structured `FactSheet`:

**Stage 1 — Local FAISS Search**
Queries the local vector DB (`all-MiniLM-L6-v2` embeddings, top-2 results) built from PDFs in AWS S3.

**Stage 2 — LLM Relevancy Gate**
Asks Llama-3.3-70b-versatile: *"Does this local PDF context actually answer the question?"* Returns `YES` or `NO`.

**Stage 3 — Routing Decision**
- `YES` → use local PDF extracts, skip web entirely
- `NO` → fall through to live web search

**Stage 4 — Live Web Search (fallback)**
- `custom_web_search()` via Tavily API (top 1 result URL)
- `custom_web_fetch()` via BeautifulSoup (clean text, 5000 char limit)
- **Query Optimizer**: LLM rewrites the user's question into tight search keywords; dynamically appends current year for time-sensitive queries

**Stage 5 — ReAct Tool Loop**
`create_react_agent()` with two tools:
- `execute_python_code` — runs Python in a sandbox for math/coding tasks
- `save_file_to_disk` — writes content to `./projects/<filename>` for file save requests

**Stage 6 — Pydantic FactSheet**
Structured output extraction via `llm.with_structured_output(FactSheet)`:
```python
class FactSheet(BaseModel):
    summary: str
    verified_facts: List[str]
    sources_used: List[str]
```

---

## Evaluation System

**File:** `evaluate_system.py`

Uses **LangSmith** to run automated evals against a synthetic Q&A benchmark (`synthetic_data.json`).

| Metric | What It Measures |
|---|---|
| `correctness` | Does the answer match the ground truth? |
| `concision` | Is the response direct or bloated? |
| `groundedness` | Does it hallucinate or stay within facts? |
| `retrieval_relevance` | Did the system actually address the query intent? |

Each metric is scored 1–5 by **Groq Llama-3.3-70b-versatile** as the judge LLM, normalized to 0.0–1.0. Results are visible in the LangSmith Web UI dashboard.

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Conda (recommended)
- AWS account with S3 bucket + EC2 (for production)
- Groq API key (free tier available)
- Gemini API key (fallback)
- Tavily API key (web search)
- LangSmith account (evaluation, optional)

### Install

```bash
git clone <your-repo-url>
cd agentic-ai

conda create -n agentic-rag python=3.10 -y
conda activate agentic-rag

pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Groq (Primary LLM Provider)
GROQ_API_KEY=gsk_...
GROQ_API_KEY_BACKUP=gsk_...          # Optional: for load balancing

# Google Gemini (Fallback LLM Provider)
GEMINI_API_KEY=AIza...

# Tavily (Web Search)
TAVILY_API_KEY=tvly-...

# AWS S3 (Knowledge Base Storage)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=eu-north-1
AWS_S3_BUCKET_NAME=your-bucket-name

# LangSmith (Evaluation — optional)
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-rag
```

---

## Running the System

### 1. Build the Knowledge Base (first time)

Upload your PDF files to the configured S3 bucket, then run:

```bash
conda activate agentic-rag
python ingestion.py
```

This downloads PDFs from S3, splits them into chunks, generates HuggingFace embeddings (`all-MiniLM-L6-v2`), and saves the FAISS index to `./vector_store/research_papers/`.

### 2. Start the Flask Backend

```bash
python app.py
# Running on http://0.0.0.0:5050
```

### 3. Start the Streamlit Frontend

```bash
streamlit run frontend.py
# Running on http://localhost:8501
```

### 4. (Optional) CLI Mode

```bash
python main.py
```

Runs an interactive terminal chat session with the full agent pipeline.

### 5. (Optional) Run Evaluation

```bash
python evaluate_system.py
```

Pushes the synthetic dataset to LangSmith and runs all 4 judge metrics. View results at [smith.langchain.com](https://smith.langchain.com).

---

## API Endpoints

### `POST /api/chat`

```json
// Request
{
  "message": "What is the latest news on AI?",
  "thread_id": "session_001"
}

// Response
{
  "response": "Here are the latest AI developments...",
  "logs": [
    "⏳ [Gateway] Scanning user input...",
    "⏳ [Engine] Activating Agentic Orchestrator...",
    "[Gateway Metrics] Query Cost: $0.000042",
    "✅ [System] Gateway approval verified."
  ]
}
```

### `POST /api/update_knowledge`

```json
// Response
{
  "success": true,
  "logs": [
    "Connecting to AWS S3 Bucket: your-bucket...",
    "Downloading paper.pdf from S3...",
    "Successfully processed PDF: paper.pdf",
    "Generating embeddings for 142 chunks...",
    "✅ Vector index hot-swapped and live in active memory."
  ]
}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Orchestration** | LangGraph, LangChain |
| **LLM Providers** | Groq (Llama-3.1-8b, Llama-3.3-70b), Google Gemini |
| **Vector DB** | FAISS + HuggingFace `all-MiniLM-L6-v2` |
| **Web Search** | Tavily API, BeautifulSoup4 |
| **Cloud Storage** | AWS S3 |
| **Deployment** | AWS EC2 |
| **Backend API** | Flask, Flask-CORS |
| **Frontend** | Streamlit |
| **Evaluation** | LangSmith, Groq judge LLM |
| **Security** | Regex guardrails, LLM cognitive guardrails, Pydantic validation |

# restart system

nohup python -u app.py > backend.log 2>&1 &