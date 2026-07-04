from __future__ import annotations

import json
import logging
import re
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
            pass
        # Attempt to repair truncated JSON
        repaired = self._repair_json(text)
        if repaired is not None:
            logger.warning("Repaired truncated JSON from LLM (%d chars) keys=%s",
                           len(text), list(repaired.keys()))
            return repaired
        logger.error("Failed to parse JSON from LLM response (len=%d): %s",
                     len(text), text[:500])
        return {}

    @staticmethod
    def _repair_json(text: str) -> dict[str, Any] | None:
        """Try to fix common truncation patterns in LLM JSON output."""
        if not text or text[0] != '{':
            return None
        text = text.strip()

        # --- Pass 1: structural fixes ---

        # Remove trailing commas before ] or }
        text = re.sub(r',(\s*[}\]])', r'\1', text)

        # Fix unquoted values containing letters (e.g. 0.93x-1.45x, 2.04x)
        text = re.sub(
            r':\s*(\d[\w.\-+]*[a-zA-Z][\w.\-+]*)\s*([,}\]])',
            r': "\1"\2',
            text,
        )

        # Fix objects whose values are just strings without keys (no :)
        # e.g. "findings": { "str1", "str2" } → "findings": ["str1", "str2"]
        text = re.sub(
            r'"(\w+)":\s*\{\s*("[^"]*"\s*(?:,\s*"[^"]*"\s*)*)\s*\}',
            r'"\1": [\2]',
            text,
        )

        # --- Pass 2: close open brackets / strings ---
        stack: list[str] = []
        in_str = False
        escaped = False
        for ch in text:
            if escaped:
                escaped = False
                continue
            if ch == '\\':
                escaped = True
                continue
            if ch == '"' and not escaped:
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch in '([{':
                stack.append(ch)
            elif ch == ')':
                if stack and stack[-1] == '(':
                    stack.pop()
            elif ch == ']':
                if stack and stack[-1] == '[':
                    stack.pop()
            elif ch == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
        if in_str:
            text += '"'
        close_map = {'{': '}', '[': ']', '(': ')'}
        for ch in reversed(stack):
            text += close_map.get(ch, '}')
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # --- Pass 3: last resort — extract known text fields by char scanning ---
        fields = {}
        for key in ('summary', 'notes', 'lineage_notes'):
            key_pattern = f'"{key}"'
            start = text.find(key_pattern)
            if start < 0:
                continue
            pos = start + len(key_pattern)
            # skip whitespace and colon
            while pos < len(text) and text[pos] in ' \t\n\r:':
                pos += 1
            if pos >= len(text) or text[pos] != '"':
                continue
            pos += 1  # skip opening quote
            value_chars: list[str] = []
            while pos < len(text):
                ch = text[pos]
                if ch == '\\':
                    pos += 1
                    if pos < len(text):
                        value_chars.append(text[pos])
                    pos += 1
                    continue
                if ch == '"':
                    break
                value_chars.append(ch)
                pos += 1
            if value_chars:
                fields[key] = ''.join(value_chars)
        if fields:
            return fields

        return None

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
