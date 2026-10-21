"""Run agents on free LLM tiers without falling over at rate limits, and without sending private data to free
tiers that may train on it.

Backends, tried in turn until one answers:
  groq   - Groq's free tier (GROQ_API_KEY). Fast. Groq says it doesn't train on API data.
  gemini - Google's free tier (GEMINI_API_KEY). A strong writer, but daily limits are low and Google may use
           free-tier prompts to improve its products, so private=True never goes there.
  local  - a model on your own machine through Ollama. Always there, tried last.

    from free_llm_router import Router
    ai = Router()                       # keys from GROQ_API_KEY / GEMINI_API_KEY
    ai.ask("You are concise.", "Explain RAG in one line.")
    ai.ask("...", "Summarise my emails: ...", private=True)   # never reaches Gemini's free tier
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time

import requests

__all__ = ["Router", "AIError"]
__version__ = "0.1.0"

log = logging.getLogger("free_llm_router")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


class AIError(RuntimeError):
    """No backend could answer."""


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


class Router:
    def __init__(self, *, groq_key: str | None = None, gemini_key: str | None = None,
                 groq_model: str | None = None, groq_web_model: str | None = None,
                 groq_vision_model: str | None = None, gemini_model: str | None = None,
                 gemini_lite_model: str | None = None, ollama_url: str | None = None,
                 ollama_model: str | None = None, vision_on_gemini: bool | None = None):
        self.groq_key = groq_key if groq_key is not None else _env("GROQ_API_KEY")
        self.gemini_key = gemini_key if gemini_key is not None else _env("GEMINI_API_KEY")
        self.groq_model = groq_model or _env("GROQ_MODEL", "openai/gpt-oss-120b")
        # Groq's built-in browser_search tool: only a model that really searches may answer web questions.
        self.groq_web_model = groq_web_model or _env("GROQ_WEB_MODEL", "openai/gpt-oss-120b")
        self.groq_vision_model = groq_vision_model if groq_vision_model is not None else _env("GROQ_VISION_MODEL")
        self.gemini_model = gemini_model or _env("GEMINI_MODEL", "gemini-3.8-flash")
        self.gemini_lite_model = gemini_lite_model or _env("GEMINI_LITE_MODEL", "gemini-3.5-flash-lite")
        self.ollama_url = (ollama_url or _env("OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.ollama_model = ollama_model or _env("OLLAMA_MODEL", "qwen2.5:7b")
        # Private pictures (a screen, a webcam) go to Gemini's free tier only if you opt in.
        self.vision_on_gemini = vision_on_gemini if vision_on_gemini is not None else _env("VISION_GEMINI") == "1"
        self._rest: dict[str, float] = {}   # "groq:model" -> monotonic time it may be tried again
        self._lock = threading.Lock()

    # --- resting a backend that refused ---------------------------------------------------------------
    def usable(self, name: str) -> bool:
        return time.monotonic() >= self._rest.get(name, 0.0)

    def rest(self, name: str, seconds: float) -> None:
        with self._lock:
            self._rest[name] = max(self._rest.get(name, 0.0), time.monotonic() + seconds)

    def rest_for(self, name: str, r: requests.Response) -> None:
        """As long as the backend asked (retry-after), an hour for a daily limit, a day for a bad key."""
        if r.status_code in (401, 403):
            self.rest(name, 24 * 3600)
        elif r.status_code == 429:
            try:
                wait = float(r.headers.get("retry-after") or 0)
            except ValueError:
                wait = 0
            per_day = any(s in r.text.lower() for s in ("per day", "perday", "rpd", "tpd"))
            self.rest(name, wait or (3600 if per_day else 60))

    # --- planning ---------------------------------------------------------------------------------------
    def plan(self, quality: bool = False, private: bool = False, images: bool = False,
             web: bool = False) -> list[tuple[str, str]]:
        """The (backend, model) order for one request."""
        g, gem = bool(self.groq_key), bool(self.gemini_key)
        out: list[tuple[str, str]] = []
        if images:
            if g and self.groq_vision_model:
                out.append(("groq", self.groq_vision_model))
            if gem and (not private or self.vision_on_gemini):
                out.append(("gemini", self.gemini_lite_model))
            return [p for p in out if self.usable(":".join(p))]
        if web:   # a model that can't search would invent the results
            return [("groq", self.groq_web_model)] if g and self.usable(f"groq:{self.groq_web_model}") else []
        if quality and gem and not private:
            out.append(("gemini", self.gemini_model))
        if g:
            out.append(("groq", self.groq_model))
        if gem and not private:
            out.append(("gemini", self.gemini_lite_model))
        return [p for p in out if self.usable(":".join(p))] + [("local", self.ollama_model)]

    # --- calls ------------------------------------------------------------------------------------------
    def _openai_call(self, url: str, key: str, name: str, body: dict, timeout: float) -> str:
        try:
            r = requests.post(url, json=body, timeout=timeout, headers={"Authorization": f"Bearer {key}"})
        except requests.RequestException as e:
            raise AIError(f"{name}: can't reach it ({type(e).__name__})") from e
        if r.status_code == 400 and "reasoning" in r.text.lower() and "reasoning_effort" in body:
            body = {k: v for k, v in body.items() if k != "reasoning_effort"}   # a model without that setting
            r = requests.post(url, json=body, timeout=timeout, headers={"Authorization": f"Bearer {key}"})
        if r.status_code != 200:
            self.rest_for(name, r)
            raise AIError(f"{name}: error {r.status_code} {r.text[:160]}")
        msg = (r.json().get("choices") or [{}])[0].get("message") or {}
        text = re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.S).strip()
        if not text:
            raise AIError(f"{name}: empty answer")
        return text

    def _local(self, system: str, prompt: str, json_mode: bool, max_tokens: int, temperature: float,
               timeout: float) -> str:
        body = {"model": self.ollama_model, "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": 8192}}
        if json_mode:
            body["format"] = "json"
        try:
            r = requests.post(f"{self.ollama_url}/api/chat", json=body, timeout=max(timeout, 300))
        except requests.RequestException as e:
            raise AIError(f"local: Ollama isn't answering ({type(e).__name__})") from e
        if r.status_code != 200:
            raise AIError(f"local: Ollama error {r.status_code}")
        text = re.sub(r"<think>.*?</think>", "", (r.json().get("message") or {}).get("content", ""), flags=re.S)
        if not text.strip():
            raise AIError("local: empty answer")
        return text.strip()

    @staticmethod
    def _user_content(prompt: str, images: list[bytes] | None):
        if not images:
            return prompt
        return [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()}}
            for j in images]

    def ask(self, system: str, prompt: str, *, quality: bool = False, private: bool = False,
            json_mode: bool = False, max_tokens: int = 1200, temperature: float = 0.4,
            images: list[bytes] | None = None, web: bool = False, local: bool = True,
            timeout: float = 120) -> str:
        """One answer from the best free model available right now. Raises AIError if none could answer.

        quality   - try Gemini's stronger model first (for the few jobs a day that need the best writing)
        private   - never send this to Gemini's free tier
        web       - answer with live web search (Groq's browser_search tool) or not at all
        local     - False leaves out the Ollama model (for callers with their own local fallback)
        """
        errors = []
        for backend, model in self.plan(quality, private, bool(images), web):
            if backend == "local" and (not local or images):
                break
            name = f"{backend}:{model}"
            try:
                if backend == "local":
                    return self._local(system, prompt, json_mode, max_tokens, temperature, timeout)
                body = {"model": model, "temperature": temperature,
                        "messages": [{"role": "system", "content": system},
                                     {"role": "user", "content": self._user_content(prompt, images)}]}
                if backend == "groq":
                    body["max_tokens"] = max_tokens + (1500 if "gpt-oss" in model else 0)   # room to think first
                    if "gpt-oss" in model:
                        body["reasoning_effort"] = "medium" if quality else "low"
                    if web:
                        body["tools"] = [{"type": "browser_search"}]
                    elif json_mode:
                        body["response_format"] = {"type": "json_object"}
                    return self._openai_call(GROQ_URL, self.groq_key, name, body, timeout)
                body["max_tokens"] = max_tokens + 2000   # Gemini's thinking counts against this
                body["reasoning_effort"] = "low"
                if json_mode:
                    body["response_format"] = {"type": "json_object"}
                return self._openai_call(GEMINI_URL, self.gemini_key, name, body,
                                         max(timeout, 60) if quality else timeout)
            except AIError as e:
                errors.append(str(e)[:160])
                log.info("%s didn't answer (%s); trying the next", name, str(e)[:160])
        raise AIError("no free model could answer: " + " | ".join(errors or ["nothing set up"]))

    def ask_json(self, system: str, prompt: str, **kw):
        """ask() for a JSON answer, parsed. Raises AIError, or ValueError if the reply has no JSON."""
        text = self.ask(system, prompt + "\n\nReply with the JSON only.", json_mode=True, **kw)
        t = re.sub(r"```(?:json)?", "", text).strip()
        starts = [i for i in (t.find("{"), t.find("[")) if i >= 0]
        if not starts:
            raise ValueError("no JSON in the reply")
        return json.JSONDecoder().raw_decode(t[min(starts):])[0]

    def status(self) -> str:
        """Which backends are set up, in one line."""
        return ", ".join([f"Groq ({self.groq_model})" if self.groq_key else "Groq: no GROQ_API_KEY",
                          f"Gemini ({self.gemini_model}, not for private requests)" if self.gemini_key
                          else "Gemini: no GEMINI_API_KEY",
                          f"local ({self.ollama_model} at {self.ollama_url})"])
