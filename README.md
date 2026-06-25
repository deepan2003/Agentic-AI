## Architecture

```text
agentic-rag-system/
│
├── projects/
│   └── agent.md                 # Global system rules and preferences
│
├── skills/                      # On-demand custom instructions folder
│
├── tools/
│   ├── __init__.py
│   ├── web_tools.py             # Tavily search and markdown web fetch
│   └── vector_tools.py          # Local Vector DB query functions
│
├── gateway.py                   # Anti-injection and LLM safety middleware
├── sub_agents.py                # Research sub-agent definition & Pydantic schemas
├── ingestion.py                 # Pulls PDFs from S3 -> splits -> embeds locally
├── main.py                      # Main LangGraph/Deep-Agent Orchestrator
├── .env                         # Your API keys (OpenAI, Tavily, AWS credentials)
└── requirements.txt             # Project library dependencies
```

[ USER PROMPT ]
              │
              ▼
┌───────────────────────────┐
│ 1. INPUT GATEWAY GUARD    │ ◄─── (gateway.py -> verify_user_input)
└─────────────┬─────────────┘
              │ (If Secure)
              ▼
┌───────────────────────────┐
│ 2. MAIN ORCHESTRATOR NODE │ ◄─── (main.py -> main_orchestrator_node)
│   (Groq Llama 3.1 / 3.3)  │
└─────────────┬─────────────┘
              │
              ▼
    [ INTENT CLASSIFIER ] ───────► (Is it casual greeting / chat?)
              │                                   │
              │ (If RESEARCH)                     │ (If CHITCHAT)
              ▼                                   ▼
┌───────────────────────────┐           ┌───────────────────────────┐
│ 3. HYBRID RESEARCH WORKER │           │ FAST CONVERSATIONAL PATH  │
│  (sub_agents.py Pipeline) │           │   (Direct friendly reply) │
└─────────────┬─────────────┘           └─────────────┬─────────────┘
              │                                       │
              ├─► Local PDF DB (HF Embeddings+FAISS)  │
              ├─► Live Web Data (Tavily Search API)    │
              │                                       │
              ▼                                       │
┌───────────────────────────┐                         │
│ 4. ACTIVE ReAct TOOL LOOP │                         │
│   (create_react_agent)    │                         │
└─────────────┬─────────────┘                         │
              ├─► [Tool] E2B Sandbox (Code Run)       │
              └─► [Tool] Save to Disk (./projects)    │
              │                                       │
              ▼                                       │
┌───────────────────────────┐                         │
│ 5. Pydantic FactSheet     │                         │
│   (Strict Fact Extraction)│                         │
└─────────────┬─────────────┘                         │
              │                                       │
              ▼                                       │
┌───────────────────────────┐                         │
│ 6. MAIN AGENT ANSWER COMP │                         │
└─────────────┬─────────────┘                         │
              │                                       │
              ▼                                       │
┌───────────────────────────┐                         │
│ 7. OUTPUT GATEWAY GUARD   │ ◄───────────────────────┘
│   (Hallucination Check)   │ ◄─── (gateway.py -> verify_agent_output)
└─────────────┬─────────────┘
              │ (If Validated)
              ▼
      [ AGENT RESPONSE ]



# to start the system

conda activate agentic-rag

# Worflow image Details

Here is the updated architectural diagram that accurately reflects the structure of your current system. This new visualization incorporates several key changes to match your production workflow:

Intent Classifier (The Decision Fork): Unlike the original image, which showed a linear path, your system now has an Intent Router (Diamond Shape) immediately after the main agent. This routes simple greetings ("Chitchat") on a fast track and only activates the complex "Research Worker" when necessary.

Local Workspace (No AWS): All references to AWS S3 have been removed. The storage block now clearly shows Local Project Workspaces with hard drive icons, representing where your .md global rules are stored and where your generated files are permanently saved to disk (such as ai_news_report.md or bomb_precautions.md).

Active Tool Loop: In the "Hybrid Research Worker" block, we have added a dedicated Active ReAct Tool Agent wrapper. This visualizes the create_react_agent loop we built, which actively calls execute_python_code in the E2B Secure Cloud Sandbox and save_file_to_disk on your local hard drive.

Bookended Guards: The system is clearly bookended by the Input Gateway Guard (verify_user_input) and the Output Gateway Guard (verify_agent_output), ensuring that both the user’s request and the agent’s final response are scanned for safety and hallucinations, respectively.

# Gateway Architecture 

[ User Query ]
              │
              ▼
    ┌───────────────────┐
    │    LLM Gateway    │ ───► [ Check Cache ] ─── (Hit: Return immediately)
    └───────────────────┘
              │ (Miss)
              ▼
     [ Smart Router ] ────► Simple Task? ──► Route to Llama-3-8b (Load balanced keys)
              │           │
              │           └► Primary Fails? ──► [ Fallback to Gemini / Groq Backup ]
              │
              └───────────► Complex Task? ──► Route to Llama-3.3-70b / Claude
                          │
                          └► Primary Fails? ──► [ Fallback to Backup Large Model ]
              │
              ▼
     [ Cost Tracker ] ───► Logs input/output tokens & USD dollar costs
              │
              ▼
     [ Final Output ]

# Whole system working principle after the gateway added 

1. Step 1: The Intent Router (First Model)
Model Used: llama-3.1-8b-instant

Action: The prompt goes here first. This lightweight model instantly reads the text and decides its category: CHITCHAT, SAVE_HISTORY, or RESEARCH.

2. Step 2: The Branching Decision
Depending on what Step 1 decides, the system splits into different paths:

Path A (CHITCHAT): Stays on llama-3.1-8b-instant. It generates a quick, friendly response immediately. The process ends here.

Path B & C (RESEARCH / SAVE_HISTORY): The system passes the prompt down to the hybrid_research_worker sub-agent to look up facts or trigger code tools.

3. Step 3: The Deep Reasoner (Next Model)
Model Used: llama-3.3-70b-versatile

Action: For research, coding, or file actions, this heavy-duty model takes over. It analyzes the search results, processes the data structure, ensures facts are accurate, and drafts the final complex response.

4. The Invisible Safety Net (Fallback Model)
Model Used: gemini-1.5-flash-backup

Action: If Groq's servers timeout or crash during any of the steps above, the gateway intercepts the error and routes that exact step to Gemini seamlessly.
