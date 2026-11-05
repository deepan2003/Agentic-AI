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

load_dotenv()


class FactSheet(BaseModel):
    summary: str = Field(description="Comprehensive summary of the findings answering the user's query.")
    verified_facts: List[str] = Field(description="List of specific, verified factual statements extracted from the source context.")
    sources_used: List[str] = Field(description="List of source filenames or URLs used.")


def _analyze_research_source_with_llm(user_query: str, pdf_excerpt: str, has_uploaded_pdfs: bool) -> dict:
    """
    100% LLM-driven decision on how to answer a research query (Zero hardcoded keyword lists):
    - Determines whether the user is asking about their uploaded document/PDF/resume/knowledge base
      or asking a general/live-web question.
    - Evaluates whether the retrieved PDF chunks from the user's private FAISS index actually answer the question.
    - Generates an optimized search query if live web search (Tavily) is needed.
    """
    prompt = (
        "You are the Semantic Source Router for a Hybrid RAG + Web Search Sub-Agent.\n"
        "Analyze the USER QUERY and the RETRIEVED LOCAL PDF EXCERPTS (if any) from the user's private knowledge base.\n"
        "Do NOT rely on rigid keyword matching — understand the user's actual intent.\n\n"
        "Return ONLY a valid JSON object with these exact keys:\n"
        "{\n"
        '  "is_asking_about_uploaded_docs": true | false,\n'
        '  "pdf_context_answers_query": true | false,\n'
        '  "optimized_web_query": "concise search engine query if web search is needed"\n'
        "}\n\n"
        "DECISION RULES:\n"
        "1. 'is_asking_about_uploaded_docs': Set to true if the user's question is referring to their uploaded PDF, "
        "document, paper, resume/CV, candidate profile, or private knowledge base (even if phrased indirectly like "
        "'what are the skills mentioned?', 'summarize the file', 'who is the author?', 'what projects did he build?'). "
        "Set to false if the user is asking for live external information (like current news in a city, weather, stock prices, "
        "or general world/coding facts not about their uploaded file).\n"
        "2. 'pdf_context_answers_query': Set to true if the RETRIEVED LOCAL PDF EXCERPTS contain relevant information "
        "that answers or addresses the user's question. Set to false if no PDF excerpts are provided or if the PDF excerpts "
        "are completely unrelated to the question (for example, the PDF is a resume, but the user asks 'what is the current news in Karur').\n"
        "3. 'optimized_web_query': If web search is needed, write a clean, focused search engine query.\n\n"
        f"USER HAS UPLOADED PDFS IN SESSION: {has_uploaded_pdfs}\n"
        f"USER QUERY: {user_query}\n\n"
        f"RETRIEVED LOCAL PDF EXCERPTS:\n{pdf_excerpt[:3000] if pdf_excerpt else 'NONE (User has no indexed PDF chunks)'}\n\n"
        "Respond ONLY with the raw JSON object."
    )

    try:
        res = smart_gateway.completion(messages=[{"role": "user", "content": prompt}])
        raw = res.content.strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw).strip()
        s_idx, e_idx = raw.find("{"), raw.rfind("}")
        if s_idx != -1 and e_idx != -1:
            data = json.loads(raw[s_idx:e_idx + 1])
            return {
                "is_asking_about_uploaded_docs": bool(data.get("is_asking_about_uploaded_docs", False)),
                "pdf_context_answers_query": bool(data.get("pdf_context_answers_query", False)),
                "optimized_web_query": str(data.get("optimized_web_query") or user_query).strip(),
            }
    except Exception as e:
        print(f"[Sub-Agent Warning] LLM source router fallback: {e}", flush=True)

    return {
        "is_asking_about_uploaded_docs": False,
        "pdf_context_answers_query": bool(pdf_excerpt),
        "optimized_web_query": user_query,
    }


def _extract_factsheet_from_context(user_query: str, raw_context: str, sources: List[str]) -> FactSheet:
    """
    Compiles raw context into a verified FactSheet in a SINGLE fast LLM call via smart_gateway.
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
    100% LLM-Guided Hybrid Research Sub-Agent:
    1. Queries the active user's (`thread_id`) isolated FAISS RAG database.
    2. Uses the LLM Semantic Source Router (`_analyze_research_source_with_llm`) to understand whether
       the user is asking about their uploaded document or external web knowledge, and whether the PDF chunks
       answer the question.
    3. Routes to Private RAG extraction, Empty-Vault Notice (if asking about an uploaded doc when none is uploaded),
       or Live Tavily Web Search accordingly.
    """
    t_start = time.time()
    print(f"\n[Sub-Agent] Starting research for: '{user_query}' (User: '{thread_id}')", flush=True)
    print(f"[Sub-Agent] Checking user '{thread_id}' private RAG database...", flush=True)

    local_results = query_local_research_papers(user_query, top_k=5, thread_id=thread_id)
    valid_chunks = [
        r for r in local_results
        if isinstance(r, dict) and "content_snippet" in r and r.get("content_snippet", "").strip()
    ]

    combined_pdf_text = ""
    pdf_sources = []
    if valid_chunks:
        combined_pdf_text = "\n\n".join(
            f"[Source: {r.get('source_file', 'Uploaded PDF')}]\n{r['content_snippet']}"
            for r in valid_chunks
        )
        pdf_sources = list(dict.fromkeys(r.get("source_file", "Uploaded PDF") for r in valid_chunks))

    # Ask the LLM Semantic Router to evaluate the query and retrieved PDF context
    routing_decision = _analyze_research_source_with_llm(
        user_query=user_query,
        pdf_excerpt=combined_pdf_text,
        has_uploaded_pdfs=bool(valid_chunks)
    )
    print(
        f"[Sub-Agent] LLM Source Decision -> AskingAboutDocs: {routing_decision['is_asking_about_uploaded_docs']} "
        f"| PDFMatchesQuery: {routing_decision['pdf_context_answers_query']}",
        flush=True
    )

    # Case 1: The user's private PDF chunks answer the query
    if valid_chunks and (routing_decision["pdf_context_answers_query"] or routing_decision["is_asking_about_uploaded_docs"]):
        print("[Sub-Agent] ✅ RAG Match: Answering from user's private uploaded PDF(s). Bypassing web search.", flush=True)
        sheet = _extract_factsheet_from_context(user_query, combined_pdf_text, pdf_sources)
        print(f"[Sub-Agent] 🏁 Research complete in {time.time() - t_start:.2f}s. Extracted {len(sheet.verified_facts)} facts.", flush=True)
        return sheet

    # Case 2: The user is asking about their uploaded PDF/document/resume, but this user has NOT uploaded any PDF
    if routing_decision["is_asking_about_uploaded_docs"] and not valid_chunks:
        print(f"[Sub-Agent] 🛡️ User '{thread_id}' asked about an uploaded document, but their RAG store is empty.", flush=True)
        return FactSheet(
            summary=f"No uploaded PDF documents were found in the private knowledge base for user '{thread_id}'. Please upload a PDF in the left sidebar first to ask questions about your documents.",
            verified_facts=[f"User '{thread_id}' currently has 0 indexed PDF documents in their private RAG knowledge base."],
            sources_used=["Private RAG Database (Empty)"]
        )

    # Case 3: Live Web Search via Tavily using the LLM's optimized search query
    web_query = routing_decision["optimized_web_query"] or user_query
    print(f"[Sub-Agent] 🌐 Querying live web search (Tavily) for: '{web_query}'...", flush=True)
    search_results = custom_web_search(web_query, max_results=3)
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