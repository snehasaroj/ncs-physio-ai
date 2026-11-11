"""Minimal multi-provider LLM client (Google Gemini, OpenAI, Anthropic) over plain HTTPS.

Using raw REST calls keeps dependencies small and makes the request/response handling
explicit. API keys are read from Streamlit secrets or environment variables and are
never written to disk or logs.
"""
from __future__ import annotations

import json
import os
import re
import time

import requests

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
}
KEY_NAMES = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


class LLMError(RuntimeError):
    pass


def get_key(provider: str) -> str | None:
    name = KEY_NAMES[provider]
    try:
        import streamlit as st  # optional: only present when running the app
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


def available_providers() -> list[str]:
    return [p for p in KEY_NAMES if get_key(p)]


def complete(prompt: str, system: str, provider: str, model: str | None = None,
             temperature: float = 0.1, json_mode: bool = True, timeout: int = 60,
             retries: int = 2) -> str:
    key = get_key(provider)
    if not key:
        raise LLMError(f"No API key configured for {provider} ({KEY_NAMES[provider]}).")
    model = model or DEFAULT_MODELS[provider]
    for attempt in range(retries + 1):
        try:
            if provider == "gemini":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                body = {"systemInstruction": {"parts": [{"text": system}]},
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": temperature,
                                             **({"responseMimeType": "application/json"} if json_mode else {})}}
                r = requests.post(url, params={"key": key}, json=body, timeout=timeout)
                r.raise_for_status()
                return "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
            if provider == "openai":
                body = {"model": model, "temperature": temperature,
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
                if json_mode:
                    body["response_format"] = {"type": "json_object"}
                r = requests.post("https://api.openai.com/v1/chat/completions", json=body, timeout=timeout,
                                  headers={"Authorization": f"Bearer {key}"})
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
            if provider == "anthropic":
                body = {"model": model, "max_tokens": 1500, "temperature": temperature, "system": system,
                        "messages": [{"role": "user", "content": prompt}]}
                r = requests.post("https://api.anthropic.com/v1/messages", json=body, timeout=timeout,
                                  headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
                r.raise_for_status()
                return "".join(b.get("text", "") for b in r.json()["content"])
            raise LLMError(f"Unknown provider {provider}")
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (429, 500, 502, 503) and attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            detail = e.response.text[:300] if e.response is not None else str(e)
            raise LLMError(f"{provider} HTTP {status}: {detail}") from e
        except requests.RequestException as e:
            if attempt < retries:
                time.sleep(2)
                continue
            raise LLMError(f"{provider} request failed: {e}") from e
    raise LLMError("unreachable")


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences or leading prose."""
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise LLMError("Model did not return JSON.")
    return json.loads(m.group(0))
