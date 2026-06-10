"""
llm.py — Ollama LLM client for ContractRAG.
"""

import json
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, config: dict):
        llm_cfg = config["llm"]
        self.base_url = llm_cfg["base_url"].rstrip("/")
        self.model_name = llm_cfg["model_name"]
        self.temperature = llm_cfg["temperature"]
        self.max_tokens = llm_cfg["max_tokens"]
        self.generate_url = f"{self.base_url}/api/generate"

        logger.info(
            "OllamaClient ready — model=%s temperature=%s",
            self.model_name,
            self.temperature,
        )

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a prompt to Ollama and return the raw response string.
        Raises RuntimeError on connection failure or bad status.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.temperature,
                "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
            },
        }

        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=600,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama request timed out after 120s.")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}")

        data = response.json()
        text = data.get("response", "").strip()

        if not text:
            logger.warning("Ollama returned empty response for model=%s", self.model_name)

        logger.debug("Ollama response length=%d chars", len(text))
        return text

    def health_check(self) -> bool:
        """Returns True if Ollama is reachable and model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            available = any(self.model_name in m for m in models)
            if not available:
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s",
                    self.model_name,
                    models,
                )
            return available
        except Exception as e:
            logger.error("Ollama health check failed: %s", e)
            return False