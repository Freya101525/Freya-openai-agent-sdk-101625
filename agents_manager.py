# agents_manager.py
import os
import yaml
import json
import traceback
from typing import Dict, Any
from openai import OpenAI

# --- Google Generative AI (GenAI) sample import (uses code pattern from user) ---
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

# environment keys:
# OPENAI_API_KEY (for OpenAI)
# GEMINI_API_KEY (for genai)
# GEMINI_PROJECT / GEMINI_ENDPOINT (optional)

_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client

def load_agents_config(path="agents.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_openai_completion(model_name: str, prompt: str, temperature: float = 0.0, max_tokens: int = 512):
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role":"system","content":"You are a helpful assistant."},
                  {"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    # safe navigation for different client versions
    try:
        return resp.choices[0].message["content"]
    except Exception:
        # older/newer client shapes: try alternatives
        try:
            return resp.choices[0].message.content
        except Exception:
            return str(resp)

def run_genai_completion(model_name: str, prompt: str, temperature: float = 0.0, max_tokens: int = 512):
    """
    Uses google.generativeai library per your sample. This is a simple wrapper
    that follows the sample pattern: genai.configure(api_key=...) and `model.generate_content(prompt)`
    If your environment or GenAI SDK version differs, replace with official Google Cloud client usage.
    """
    gem_key = os.getenv("GEMINI_API_KEY", "")
    if not gem_key:
        raise ValueError("GEMINI_API_KEY not set in environment.")
    if not GENAI_AVAILABLE:
        raise RuntimeError("google.generativeai package not available in environment.")
    # configure
    genai.configure(api_key=gem_key)
    # The user sample used a GenerativeModel helper. Use same pattern:
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        # sample field name is `text` per user sample
        return getattr(response, "text", str(response))
    except Exception as e:
        # fallback to genai.generate if available
        try:
            resp = genai.generate(model=model_name, prompt=prompt, temperature=temperature, max_output_tokens=max_tokens)
            # resp may be dict-like
            if isinstance(resp, dict) and "candidates" in resp:
                return resp["candidates"][0].get("content", "")
            return str(resp)
        except Exception as ee:
            traceback.print_exc()
            raise RuntimeError(f"GenAI call failed: {e} / {ee}")

def execute_agent(agent_def: Dict[str, Any], inputs: Dict[str, Any]):
    """
    agent_def structure expected similar to entries in agents.yaml:
      - model: provider:modelname  (e.g. openai:chatgpt-4o-mini OR google:gemini-2.5-flash)
      - prompt_template: multiline template using {document_markdown}, etc.
      - temperature, max_tokens
    """
    model_selector = agent_def.get("model", agent_def.get("default_model", "openai:chatgpt-4o-mini"))
    if ":" in model_selector:
        provider, model_name = model_selector.split(":", 1)
    else:
        provider = "openai"
        model_name = model_selector

    prompt_template = agent_def.get("prompt_template", "")
    # Basic safe interpolation - if keys missing, str.format will throw. Use .format_map with fallback.
    try:
        prompt_filled = prompt_template.format_map(DefaultDict(inputs))
    except Exception:
        # last-resort: manual .format with safe replacements for some common keys
        prompt_filled = prompt_template
        for k, v in inputs.items():
            prompt_filled = prompt_filled.replace("{" + k + "}", json.dumps(v) if not isinstance(v, str) else v)

    temperature = float(agent_def.get("temperature", 0.0))
    max_tokens = int(agent_def.get("max_tokens", 512))

    if provider.lower() in ("openai", "oai"):
        return run_openai_completion(model_name, prompt_filled, temperature, max_tokens)
    elif provider.lower() in ("google", "genai", "gemini"):
        return run_genai_completion(model_name, prompt_filled, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported model provider: {provider}")

class DefaultDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"  # leave placeholder unchanged
