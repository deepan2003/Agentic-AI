import os
import re
import json
import time
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from llm_gateway import smart_gateway
from tools.research import query_local_research_papers
from tools.web_search import custom_web_search, custom_web_fetch
from tools.file_system import save_file_to_disk

load_dotenv()

class FactSheet(BaseModel):
    summary: str = Field(description="Comprehensive summary of the findings answering the user's query.")
    verified_facts: List[str] = Field(description="List of specific, verified factual statements extracted from the source context.")
    sources_used: List[str] = Field(description="List of source filenames or URLs used.")


def _is_document_or_resume_query(query: str) -> bool:
    """Returns True if the user is asking about an uploaded PDF, resume, CV, or candidate profile."""
    q = query.lower()
    doc_keywords = [
        "resume", "cv", "curriculum vitae", "deepan", "uploaded", "pdf",
        "paper", "candidate", "his skills", "his experience", "his projects",
        "his education", "according to his", "accoridng to his", "in the document"
    ]
    return any(k in q for k in doc_keywords)


def _extract_factsheet_from_context(user_query: str, raw_context: str, sources: List[str]) -> FactSheet:
    """
    Compiles raw context into a verified FactSheet in a SINGLE fast LLM call via smart_gateway
    (avoids slow multi-turn tool loops and 60s structured_output stalls).
    """
    t0 = time.time()
    print("[Sub-Agent] Compiling facts into structured schema...", flush=True)

    prompt = (
        "You are a precise research extraction engine.\n"
        "Extract the relevant facts from the SOURCE CONTEXT below to answer the USER QUERY.\n"
        "Respond ONLY with a valid JSON object matching this exact schema (no markdown fences, no extra text):\n"
        '{\n'
        '  "summary": "Direct, clear summary answering the user query based strictly on the context",\n'
        '  "verified_facts": ["Fact 1", "Fact 2", "Fact 3", "Fact 4"],\n'
        '  "sources_used": ["Source 1"]\n'
        '}\n\n'
        f"USER QUERY: {user_query}\n\n"
        f"AVAILABLE SOURCES: {', '.join(sources) if sources else 'Local Knowledge Base'}\n\n"
        f"SOURCE CONTEXT:\n{raw_context[:6000]}"
    )

    try:
        res = smart_gateway.completion(messages=[{"role": "user", "content": prompt}])
        text = res.content.strip()
        # Strip markdown json fences if present
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text).strip()

        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            data = json.loads(text[start_idx:end_idx + 1])
            summary = str(data.get("summary") or "").strip()
            facts = [str(f).strip() for f in data.get("verified_facts", []) if str(f).strip()]
            used_sources = [str(s).strip() for s in data.get("sources_used", []) if str(s).strip()] or sources
            if summary or facts:
                print(f"[Sub-Agent] ✅ Compiled {len(facts)} verified facts in {time.time() - t0:.2f}s.", flush=True)
                return FactSheet(
                    summary=summary or " ".join(facts[:3]),
                    verified_facts=facts if facts else [summary],
                    sources_used=used_sources
                )
    except Exception as e:
        print(f"[Sub-Agent Warning] Fast JSON parse fallback ({e}). Using direct text extraction.", flush=True)

    # Instant deterministic fallback so the sub-agent NEVER hangs or crashes
    snippets = [line.strip() for line in raw_context.splitlines() if len(line.strip()) > 25][:6]
    if not snippets:
        snippets = [raw_context[:500].strip()] if raw_context.strip() else ["No relevant data found."]
    return FactSheet(
        summary=snippets[0],
        verified_facts=snippets,
        sources_used=sources or ["Knowledge Base"]
    )


def hybrid_research_worker(user_query: str, thread_id: str = "default") -> FactSheet:
    """
    Fast Hybrid Research Sub-Agent:
    1. Immediately searches user's local FAISS RAG database (<10ms).
    2. If relevant PDF chunks exist, extracts FactSheet in 1 fast LLM call (~1.5s).
    3. If query requires live web data (and is not a private resume query), searches Tavily.
    """
    t_start = time.time()
    print(f"\n[Sub-Agent] Starting research for: '{user_query}' (User: '{thread_id}')", flush=True)
    print(f"[Sub-Agent] Checking user '{thread_id}' RAG database first...", flush=True)

    local_results = query_local_research_papers(user_query, top_k=5, thread_id=thread_id)
    valid_chunks = [r for r in local_results if isinstance(r, dict) and "content_snippet" in r and r.get("content_snippet", "").strip()]

    is_doc_query = _is_document_or_resume_query(user_query)

    if valid_chunks:
        combined_pdf_text = "\n\n".join(
            f"[Source: {r.get('source_file', 'Uploaded PDF')}]\n{r['content_snippet']}"
            for r in valid_chunks
        )
        pdf_sources = list(dict.fromkeys(r.get("source_file", "Uploaded PDF") for r in valid_chunks))

        # If user is explicitly asking about the resume/PDF, immediately use the PDF chunks!
        if is_doc_query:
            print("[Sub-Agent] ✅ RAG Match: Resume/Document query matched uploaded PDF(s). Bypassing live web crawling.", flush=True)
            sheet = _extract_factsheet_from_context(user_query, combined_pdf_text, pdf_sources)
            print(f"[Sub-Agent] 🏁 Research complete in {time.time() - t_start:.2f}s. Extracted {len(sheet.verified_facts)} facts.", flush=True)
            return sheet

        # Otherwise do a quick relevance check in <1s
        gate_prompt = (
            "Does the following PDF excerpt contain information that answers the user's question?\n"
            "Reply with ONLY 'YES' or 'NO'.\n\n"
            f"Question: {user_query}\n\n"
            f"PDF Excerpt:\n{combined_pdf_text[:2500]}"
        )
        try:
            gate_res = smart_gateway.completion(messages=[{"role": "user", "content": gate_prompt}])
            if "YES" in gate_res.content.strip().upper():
                print("[Sub-Agent] ✅ RAG Match: Context found in uploaded PDF(s). Bypassing live web crawling.", flush=True)
                sheet = _extract_factsheet_from_context(user_query, combined_pdf_text, pdf_sources)
                print(f"[Sub-Agent] 🏁 Research complete in {time.time() - t_start:.2f}s. Extracted {len(sheet.verified_facts)} facts.", flush=True)
                return sheet
        except Exception as e:
            print(f"[Sub-Agent Warning] Gate check error: {e}", flush=True)

    # If the user asked about a resume/document and no PDF chunks were found, do not leak/confuse with public web search
    if is_doc_query and not valid_chunks:
        print("[Sub-Agent] 🛡️ Resume/Document query with no indexed text. Skipping web search.", flush=True)
        return FactSheet(
            summary="No uploaded PDF or resume was found in the knowledge base for this user session. Please upload the PDF in the left sidebar or click 'Sync Cloud Assets'.",
            verified_facts=["No indexed PDF chunks were found in the local vector store for this session."],
            sources_used=["Local RAG Database"]
        )

    # Stage 4: Live Web Search via Tavily
    print(f"[Sub-Agent] 🌐 Querying live web search (Tavily) for: '{user_query}'...", flush=True)
    search_results = custom_web_search(user_query, max_results=3)
    valid_urls = [r for r in search_results if isinstance(r, dict) and "url" in r]

    web_contexts = []
    sources_used = []
    for item in valid_urls[:2]:
        url = item["url"]
        title = item.get("title", url)
        print(f"[Sub-Agent] 📄 Fetching web content from: {url}", flush=True)
        fetched = custom_web_fetch(url)
        content = fetched.get("content", "")
        if content and not content.startswith("Error:") and not content.startswith("Failed"):
            web_contexts.append(f"[Title: {title} | URL: {url}]\n{content[:2500]}")
            sources_used.append(url)

    if web_contexts:
        combined_web = "\n\n".join(web_contexts)
        sheet = _extract_factsheet_from_context(user_query, combined_web, sources_used)
        print(f"[Sub-Agent] 🏁 Web research complete in {time.time() - t_start:.2f}s. Extracted {len(sheet.verified_facts)} facts.", flush=True)
        return sheet

    # Final fallback: answer using general model knowledge if web search returned nothing
    print("[Sub-Agent] ⚠️ No external web pages fetched; synthesizing from model knowledge...", flush=True)
    fallback_res = smart_gateway.completion(
        messages=[{"role": "user", "content": f"Provide a concise, factual answer to: {user_query}"}]
    )
    answer_text = fallback_res.content.strip()
    lines = [l.strip("- *•") for l in answer_text.splitlines() if len(l.strip()) > 15][:5]
    return FactSheet(
        summary=answer_text,
        verified_facts=lines or [answer_text],
        sources_used=["AI Knowledge Base"]
    )