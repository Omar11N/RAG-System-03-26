"""
generation/llm_factory.py — Pluggable LLM backend.

Swap strategy: change cfg.generation.provider in config.py.
Supported: "openai" | "anthropic" | "gemini" | "local_hf" | "ollama"
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable

from config import cfg
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
class OpenAIClient:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=cfg.generation.openai_api_key)
        self.model = cfg.generation.openai_model
        log.info(f"LLM: OpenAI ({self.model})")

    def generate(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=cfg.generation.max_new_tokens,
            temperature=cfg.generation.temperature,
        )
        return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Anthropic (Claude)
# ---------------------------------------------------------------------------
class AnthropicClient:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=cfg.generation.anthropic_api_key)
        self.model = cfg.generation.anthropic_model
        log.info(f"LLM: Anthropic ({self.model})")

    def generate(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=cfg.generation.max_new_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------
class GeminiClient:
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=cfg.generation.gemini_api_key)
        self.model = genai.GenerativeModel(cfg.generation.gemini_model)
        log.info(f"LLM: Gemini ({cfg.generation.gemini_model})")

    def generate(self, prompt: str) -> str:
        resp = self.model.generate_content(prompt)
        return resp.text.strip()


# ---------------------------------------------------------------------------
# Local HuggingFace (Qwen2.5, DeepSeek-R1, Llama 4, etc.)
# ---------------------------------------------------------------------------
class LocalHFClient:
    def __init__(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        model_id = cfg.generation.local_hf_model_id
        log.info(f"Loading local HF model: {model_id}")

        quant_cfg = None
        if cfg.generation.local_hf_load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            quantization_config=quant_cfg,
            device_map="auto",
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Local model loaded")

    def generate(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=cfg.generation.max_new_tokens,
                temperature=cfg.generation.temperature,
                do_sample=True,
            )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip the prompt echo
        return decoded[len(prompt):].strip()


# ---------------------------------------------------------------------------
# Ollama (local server)
# ---------------------------------------------------------------------------
class OllamaClient:
    def __init__(self):
        import requests
        self.base_url = cfg.generation.ollama_base_url
        self.model = cfg.generation.ollama_model
        # Quick health check
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=3)
            log.info(f"LLM: Ollama ({self.model}) at {self.base_url}")
        except Exception:
            log.warning(f"Ollama server not reachable at {self.base_url}")

    def generate(self, prompt: str) -> str:
        import requests
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": cfg.generation.temperature,
                    "num_predict": cfg.generation.max_new_tokens,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_PROVIDERS: dict[str, type] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "gemini": GeminiClient,
    "local_hf": LocalHFClient,
    "ollama": OllamaClient,
}

_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        provider = cfg.generation.provider
        if provider not in _PROVIDERS:
            raise ValueError(
                f"Unknown LLM provider '{provider}'. "
                f"Choose from: {list(_PROVIDERS)}"
            )
        _client = _PROVIDERS[provider]()
    return _client
