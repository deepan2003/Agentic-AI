import os
import time
import hashlib
from typing import List, Dict, Any
from itertools import cycle
from langchain_openai import ChatOpenAI

# Cost tracking configuration per 1M tokens
PRICING_TIERS = {
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "openai/gpt-oss-20b": {"input": 0.05, "output": 0.08},
    "openai/gpt-oss-120b": {"input": 0.59, "output": 0.79},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30}
}

class SmartLLMGateway:
    def __init__(self):
        self.execution_logs = []
        self.cache: Dict[str, Any] = {}
        
        groq_keys = [os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY_BACKUP")]
        self.active_keys = cycle([k for k in groq_keys if k])
        
        self.total_cost_usd = 0.0
        self.total_queries = 0

    def _generate_cache_key(self, prompt: str, model_name: str) -> str:
        return hashlib.md5(f"{model_name}:{prompt}".encode('utf-8')).hexdigest()

    def _smart_route(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        if any(keyword in prompt_lower for keyword in ["code", "def ", "class ", "import ", "calculate", "math"]):
            return "llama-3.3-70b-versatile"
        return "llama-3.1-8b-instant"

    def _track_cost(self, model_name: str, input_tokens: int, output_tokens: int):
        tier = PRICING_TIERS.get(model_name, {"input": 0.1, "output": 0.1})
        cost = ((input_tokens / 1_000_000) * tier["input"]) + ((output_tokens / 1_000_000) * tier["output"])
        self.total_cost_usd += cost
        
        print(f"[Gateway Metrics] Query Cost: ${cost:.6f} | Session Total: ${self.total_cost_usd:.6f}", flush=True)
        self.execution_logs.append(f"[Gateway Metrics] Query Cost: ${cost:.6f}")

    def completion(self, messages: List[Dict[str, str]], override_model: str = None) -> Any:
        """
        Unified API function wrapper with strict 15-second timeout and max_retries=0 per model
        so a rate-limited or stalled model never blocks the backend for 2+ minutes.
        """
        self.total_queries += 1
        flat_prompt = " ".join([str(m.get("content", "")) for m in messages])
        
        # Map slow/harmony models to Groq's fastest 800-tok/s Llama models first
        if override_model in (None, "openai/gpt-oss-20b"):
            target_model = "llama-3.1-8b-instant"
        elif override_model == "openai/gpt-oss-120b":
            target_model = "llama-3.3-70b-versatile"
        else:
            target_model = override_model
        
        cache_key = self._generate_cache_key(flat_prompt, target_model)
        if cache_key in self.cache:
            print("[Gateway Cache] ⚡ Cache hit! Returning instant response (0ms).", flush=True)
            return self.cache[cache_key]

        current_key = next(self.active_keys, os.getenv("GROQ_API_KEY"))
        gemini_key = os.getenv("GEMINI_API_KEY")

        models_to_try = [
            {"model": target_model, "base_url": "https://api.groq.com/openai/v1", "api_key": current_key},
            {"model": "llama-3.3-70b-versatile", "base_url": "https://api.groq.com/openai/v1", "api_key": current_key},
            {"model": "llama-3.1-8b-instant", "base_url": "https://api.groq.com/openai/v1", "api_key": current_key},
            {"model": "openai/gpt-oss-20b", "base_url": "https://api.groq.com/openai/v1", "api_key": current_key},
        ]

        if gemini_key:
            models_to_try.extend([
                {"model": "gemini-2.5-flash", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "api_key": gemini_key},
                {"model": "gemini-2.0-flash", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "api_key": gemini_key},
            ])

        seen_models = set()
        for attempt in models_to_try:
            model_id = attempt["model"]
            if model_id in seen_models or not attempt.get("api_key"):
                continue
            seen_models.add(model_id)

            t0 = time.time()
            try:
                print(f"[Gateway Router] 🚀 Calling {model_id}...", flush=True)
                llm = ChatOpenAI(
                    base_url=attempt["base_url"],
                    model=model_id,
                    api_key=attempt["api_key"],
                    temperature=0.1,
                    timeout=15.0,
                    max_retries=0
                )
                
                response = llm.invoke(messages)
                elapsed = time.time() - t0
                print(f"[Gateway Router] ✅ {model_id} responded in {elapsed:.2f}s", flush=True)
                
                if response.response_metadata and "token_usage" in response.response_metadata:
                    usage = response.response_metadata["token_usage"]
                    self._track_cost(model_id, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

                self.cache[cache_key] = response
                return response

            except Exception as e:
                elapsed = time.time() - t0
                print(f"[Gateway Warning] {model_id} failed after {elapsed:.2f}s ({str(e)[:120]}). Switching to next model...", flush=True)
                continue
        
        raise RuntimeError("CRITICAL CRASH: All primary and fallback models failed to respond.")

smart_gateway = SmartLLMGateway()