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

# to start the system

conda activate agentic-rag