from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from .config import Config

logger = logging.getLogger(__name__)


class LLMInterface:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.base_url = config.ollama_base_url.rstrip("/")

    def _request(
        self,
        endpoint: str,
        payload: dict[str, Any],
        retries: int = 3,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/api/{endpoint}"
        for attempt in range(retries):
            try:
                resp = requests.post(url, json=payload, timeout=120)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                logger.warning(
                    "Ollama request failed (attempt %d/%d): %s",
                    attempt + 1, retries, e,
                )
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Ollama request to {endpoint} failed after {retries} retries")

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model or self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None
                else self.config.ollama_temperature,
                "num_predict": max_tokens if max_tokens is not None
                else self.config.ollama_max_tokens,
            },
        }
        if system:
            payload["system"] = system
        data = self._request("generate", payload)
        return data.get("response", "")

    def extract_structured(
        self,
        prompt: str,
        model: str | None = None,
    ) -> dict[str, Any]:
        system = (
            "You are a precise information extraction system. "
            "Respond ONLY with valid JSON. No markdown, no explanation."
        )
        text = self.generate(prompt, model=model, system=system, temperature=0.0)
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON from LLM response: %s", text[:500])
            return {}

    def embed(self, text: str, model: str | None = None) -> list[float]:
        payload = {
            "model": model or self.config.ollama_embed_model,
            "prompt": text,
        }
        data = self._request("embeddings", payload)
        return data.get("embedding", [])

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": model or self.config.ollama_model,
            "messages": messages,
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except requests.RequestException as e:
            logger.error("Chat request failed: %s", e)
            return ""
