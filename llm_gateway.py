import os
import time
import hashlib
from typing import List, Dict, Any
from itertools import cycle
from langchain_openai import ChatOpenAI

# Cost tracking configuration per 1M tokens (Approximate values for demonstration)
PRICING_TIERS = {
    "openai/gpt-oss-20b": {"input": 0.05, "output": 0.08},
    "openai/gpt-oss-120b": {"input": 0.59, "output": 0.79},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30}
}

class SmartLLMGateway:
    def __init__(self):
        # 1. Caching Layer Initialization
        self.execution_logs = []
        self.cache: Dict[str, Any] = {}
        
        # 2. Load Balancing (Multiple Groq API keys rotated seamlessly)
        groq_keys = [os.getenv("GROQ_API_KEY"), os.getenv("GROQ_API_KEY_BACKUP")]
        # Filter out missing keys to prevent runtime issues
        self.active_keys = cycle([k for k in groq_keys if k])
        
        # 3. Cost Tracking Totals
        self.total_cost_usd = 0.0
        self.total_queries = 0

    def _generate_cache_key(self, prompt: str, model_name: str) -> str:
        """Creates a unique MD5 hash for the combination of prompt and model choice."""
        return hashlib.md5(f"{model_name}:{prompt}".encode('utf-8')).hexdigest()

    def _smart_route(self, prompt: str) -> str:
        """Pillar 5: Smart Routing based on task complexity."""
        prompt_lower = prompt.lower()
        # Hard tasks: Code execution, complex reasoning, system building
        if any(keyword in prompt_lower for keyword in ["code", "def ", "class ", "import ", "calculate", "math"]):
            return "openai/gpt-oss-120b"
        # Cheap tasks: Summarization, routing, chit-chat, greetings
        return "openai/gpt-oss-20b"

    def _track_cost(self, model_name: str, input_tokens: int, output_tokens: int):
        """Pillar 4: Automatically log usage costs dynamically."""
        tier = PRICING_TIERS.get(model_name, {"input": 0.1, "output": 0.1})
        cost = ((input_tokens / 1_000_000) * tier["input"]) + ((output_tokens / 1_000_000) * tier["output"])
        self.total_cost_usd += cost
        
        print(f"[Gateway Metrics] Query Cost: ${cost:.6f} | Session Total: ${self.total_cost_usd:.6f}")
        self.execution_logs.append(f"[Gateway Metrics] Query Cost: ${cost:.6f}")

    def completion(self, messages: List[Dict[str, str]], override_model: str = None) -> Any:
        """
        Pillar 1: Unified API function wrapper.
        - Checks memory cache first.
        - Uses Smart Routing to pick the primary Groq model.
        - Prevents 'tool_use_failed' (HTTP 400) by falling back across Groq models (llama-3.3-70b-versatile, llama-3.1-8b-instant).
        - Uses active Gemini 2.5 Flash / 2.0 Flash endpoints instead of the retired gemini-1.5-flash (HTTP 404).
        """
        self.execution_logs = []
        self.total_queries += 1
        
        # Flatten messages array into a flat string for caching evaluation
        flat_prompt = " ".join([str(m.get("content", "")) for m in messages])
        
        # Determine target model
        target_model = override_model if override_model else self._smart_route(flat_prompt)
        
        # Pillar 3: Check Memory Cache
        cache_key = self._generate_cache_key(flat_prompt, target_model)
        if cache_key in self.cache:
            print("[Gateway Cache] Hit! Returning recorded response instantly. Saved 100% tokens.")
            return self.cache[cache_key]

        # Pillar 6: Rotate API key to balance high traffic load
        current_key = next(self.active_keys, os.getenv("GROQ_API_KEY"))
        gemini_key = os.getenv("GEMINI_API_KEY")

        # Setup primary and multi-tier fallback configuration parameters
        models_to_try = [
            {"model": target_model, "base_url": "https://api.groq.com/openai/v1", "api_key": current_key},
            # Secondary Groq fallbacks (immune to gpt-oss-20b harmony tool_use_failed 400 errors)
            {"model": "llama-3.3-70b-versatile", "base_url": "https://api.groq.com/openai/v1", "api_key": current_key},
            {"model": "llama-3.1-8b-instant", "base_url": "https://api.groq.com/openai/v1", "api_key": current_key},
        ]

        # Add Gemini cross-provider fallbacks using current supported models (gemini-1.5-flash is retired/404)
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

            try:
                print(f"[Gateway Router] Directing request to {model_id}...")
                llm = ChatOpenAI(
                    base_url=attempt["base_url"],
                    model=model_id,
                    api_key=attempt["api_key"],
                    temperature=0.1
                )
                
                response = llm.invoke(messages)
                
                # Extract token metrics safely from the LLM execution metadata
                if response.response_metadata and "token_usage" in response.response_metadata:
                    usage = response.response_metadata["token_usage"]
                    self._track_cost(model_id, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

                # Update cache registry with successful results
                self.cache[cache_key] = response
                return response

            except Exception as e:
                print(f"[Gateway Warning] Primary API crash on {model_id}: {str(e)}. Triggering automated fallback route...")
                continue
        
        raise RuntimeError("CRITICAL CRASH: All primary and fallback models failed to respond.")

# Initialize a single system global proxy instance
smart_gateway = SmartLLMGateway()