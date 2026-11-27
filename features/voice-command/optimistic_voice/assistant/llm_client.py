"""Multi-provider LLM client for OptiMistic Smart Glasses Voice Assistant.

Supports:
1. Grok (xAI Grok-2 / Grok-2-mini / Grok-beta) - Advanced reasoning and real-time knowledge from xAI.
2. Google Gemini (Gemini 2.0 Flash) - Fast, smart, generous free tier.
3. Groq (Llama 3.3 70B / Llama 3.1 8B) - Fastest inference in the world (~150ms TTFT).
4. Ollama (Llama 3.2 3B / Qwen 2.5 3B) - 100% offline edge privacy without internet or API keys.
5. OpenAI (GPT-4o Mini) - Fast, reliable alternative.
6. Mock - Offline deterministic fallback for tests or when no key is set.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any
import requests

from optimistic_voice.automation import ExecutionResult
from optimistic_voice.config import OptimisticConfig

VOICE_SYSTEM_PROMPT = (
    "You are OptiMistic, an intelligent AI voice assistant built into augmented reality smart glasses.\n"
    "Guidelines for your response:\n"
    "1. Speak naturally, warmly, and concisely in 1 to 3 spoken sentences.\n"
    "2. Never use markdown, bullet points, asterisks, numbered lists, citations, or emojis, "
    "because your response is read aloud by a text-to-speech voice synthesizer.\n"
    "3. Answer the user's question directly without preamble like 'Sure, here is...' or 'As an AI...'.\n"
    "4. If asked to perform an action outside your capabilities, explain briefly in one sentence."
)


def clean_for_speech(text: str) -> str:
    """Sanitize LLM output text for clean, natural Text-to-Speech synthesis."""
    if not text:
        return ""
    # Strip markdown bold/italics (e.g. **text** -> text, *text* -> text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # Strip headers (# Header -> Header)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Strip bullet points and list markers (- Item, 1. Item)
    text = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)
    # Strip code fences and inline backticks
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Strip URLs
    text = re.sub(r"https?://\S+", "", text)
    # Collapse multiple whitespace/newlines into single line
    text = " ".join(text.split())
    return text.strip()


class LLMClient:
    """Unified client providing low-latency conversational intelligence to OptiMistic."""

    def __init__(
        self,
        config: OptimisticConfig | None = None,
        provider: str | None = None,
        system_prompt: str = VOICE_SYSTEM_PROMPT,
    ) -> None:
        self.config = config or OptimisticConfig()
        self.provider = (provider or self.config.llm_provider).lower()
        self.system_prompt = system_prompt
        self.history: list[dict[str, str]] = []  # Stores recent turns for conversational context

    def _detect_provider(self) -> str:
        """Auto-detect provider if set to auto."""
        if self.provider != "auto":
            return self.provider
        if self.config.xai_api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY"):
            return "grok"
        if self.config.gemini_api_key or os.getenv("GEMINI_API_KEY"):
            return "gemini"
        if self.config.groq_api_key or os.getenv("GROQ_API_KEY"):
            return "groq"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        return "grok" if (self.config.xai_api_key or os.getenv("XAI_API_KEY")) else "gemini"

    def ask(self, query: str) -> ExecutionResult:
        """Query the LLM and return a speech-ready ExecutionResult."""
        if not query or not query.strip():
            return ExecutionResult(success=False, message="I didn't hear a question.")

        query = query.strip()
        provider = self._detect_provider()

        try:
            if provider == "mock":
                return self._ask_mock(query)
            if provider in ("grok", "xai"):
                return self._ask_grok(query)
            if provider in ("gemini", "google"):
                return self._ask_gemini(query)
            if provider == "groq":
                return self._ask_groq(query)
            if provider == "ollama":
                return self._ask_ollama(query)
            if provider == "openai":
                return self._ask_openai(query)
            # Default to grok if provider was grok-like, else gemini
            if "grok" in provider or "xai" in provider:
                return self._ask_grok(query)
            return self._ask_gemini(query)
        except requests.exceptions.Timeout:
            return ExecutionResult(
                success=False,
                message="I timed out waiting for an answer. Please check your internet connection.",
            )
        except requests.exceptions.ConnectionError:
            return ExecutionResult(
                success=False,
                message="I couldn't connect to the AI service. Please check your network connection.",
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                message=f"I encountered an error while processing that: {str(e)}",
            )

    def _ask_grok(self, query: str) -> ExecutionResult:
        """Query xAI Grok model via standard REST completions endpoint."""
        api_key = self.config.xai_api_key or os.getenv("XAI_API_KEY", "") or os.getenv("GROK_API_KEY", "")
        if not api_key:
            return ExecutionResult(
                success=True,
                message=(
                    "I am ready to use the Grok LLM. Please set your XAI_API_KEY or GROK_API_KEY "
                    "in your dot env file to enable Grok answers."
                ),
            )

        model = self.config.grok_model or self.config.llm_model or "grok-2-mini"
        url = "https://api.x.ai/v1/chat/completions"

        messages = [{"role": "system", "content": self.system_prompt}]
        for turn in self.history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": query})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 160,
            "temperature": 0.7,
            "stream": False,
        }

        resp = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=8.0,
        )
        if resp.status_code != 200:
            if resp.status_code in (401, 403):
                try:
                    err_json = resp.json()
                    err_str = err_json.get("error", "")
                    if "credits" in err_str.lower() or "license" in err_str.lower():
                        return ExecutionResult(
                            success=False,
                            message="Your xAI team doesn't have any credits yet. Please add credits on console.x.ai to use Grok.",
                        )
                except Exception:
                    pass
                return ExecutionResult(
                    success=False,
                    message="Your Grok API key appears invalid or expired. Please check your xAI key.",
                )
            return ExecutionResult(
                success=False,
                message=f"Grok API returned an error with status code {resp.status_code}.",
            )

        data = resp.json()
        try:
            raw_text = data["choices"][0]["message"]["content"]
            clean_text = clean_for_speech(raw_text)
            self._update_history(query, clean_text)
            return ExecutionResult(success=True, message=clean_text)
        except (KeyError, IndexError):
            return ExecutionResult(success=False, message="I received an empty response from Grok.")

    def _ask_gemini(self, query: str) -> ExecutionResult:
        api_key = self.config.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            return ExecutionResult(
                success=True,
                message=(
                    "I am ready to answer your questions. Please set your Gemini API key in your "
                    "dot env file or environment variable to enable AI answers."
                ),
            )

        model = self.config.llm_model or "gemini-2.0-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        # Build contents including short conversation history
        contents: list[dict[str, Any]] = []
        for turn in self.history[-4:]:  # keep last 2 exchanges
            role = "user" if turn["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": turn["content"]}]})
        contents.append({"role": "user", "parts": [{"text": query}]})

        payload = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": self.system_prompt}]},
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 160,
            },
        }

        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=8.0)
        if resp.status_code != 200:
            err_details = resp.text[:200]
            if resp.status_code in (401, 403):
                return ExecutionResult(
                    success=False,
                    message="Your Gemini API key appears invalid or expired. Please check your key.",
                )
            return ExecutionResult(
                success=False,
                message=f"Gemini service returned an error with status {resp.status_code}.",
            )

        data = resp.json()
        try:
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
            clean_text = clean_for_speech(raw_text)
            self._update_history(query, clean_text)
            return ExecutionResult(success=True, message=clean_text)
        except (KeyError, IndexError):
            return ExecutionResult(
                success=False,
                message="I received an empty response from the AI model.",
            )

    def _ask_groq(self, query: str) -> ExecutionResult:
        api_key = self.config.groq_api_key or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            return ExecutionResult(
                success=True,
                message=(
                    "I am ready to answer with Groq lightning speed. Please set your GROQ_API_KEY "
                    "in your dot env file to enable AI answers."
                ),
            )

        model = self.config.groq_model or self.config.llm_model or "llama-3.3-70b-versatile"
        url = "https://api.groq.com/openai/v1/chat/completions"

        messages = [{"role": "system", "content": self.system_prompt}]
        for turn in self.history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": query})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 160,
            "temperature": 0.7,
        }

        resp = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=6.0,
        )
        if resp.status_code != 200:
            if resp.status_code in (401, 403):
                return ExecutionResult(
                    success=False,
                    message="Your Groq API key appears invalid. Please check your key.",
                )
            return ExecutionResult(
                success=False,
                message=f"Groq returned an error with status code {resp.status_code}.",
            )

        data = resp.json()
        try:
            raw_text = data["choices"][0]["message"]["content"]
            clean_text = clean_for_speech(raw_text)
            self._update_history(query, clean_text)
            return ExecutionResult(success=True, message=clean_text)
        except (KeyError, IndexError):
            return ExecutionResult(success=False, message="I received an empty response from Groq.")

    def _ask_ollama(self, query: str) -> ExecutionResult:
        model = self.config.llm_model or self.config.ollama_model or "llama3.2:3b"
        base_url = self.config.ollama_url.rstrip("/")
        url = f"{base_url}/api/chat"

        messages = [{"role": "system", "content": self.system_prompt}]
        for turn in self.history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": query})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=12.0)
        if resp.status_code != 200:
            return ExecutionResult(
                success=False,
                message=f"Ollama returned status code {resp.status_code}. Is model '{model}' installed?",
            )

        data = resp.json()
        try:
            raw_text = data["message"]["content"]
            clean_text = clean_for_speech(raw_text)
            self._update_history(query, clean_text)
            return ExecutionResult(success=True, message=clean_text)
        except KeyError:
            return ExecutionResult(success=False, message="Empty response from local Ollama model.")

    def _ask_openai(self, query: str) -> ExecutionResult:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return ExecutionResult(
                success=True,
                message="Please set your OPENAI_API_KEY in your environment to enable OpenAI answers.",
            )

        model = self.config.llm_model or "gpt-4o-mini"
        url = "https://api.openai.com/v1/chat/completions"

        messages = [{"role": "system", "content": self.system_prompt}]
        for turn in self.history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": query})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 160,
            "temperature": 0.7,
        }

        resp = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=8.0,
        )
        if resp.status_code != 200:
            return ExecutionResult(success=False, message=f"OpenAI error with status code {resp.status_code}.")

        data = resp.json()
        try:
            raw_text = data["choices"][0]["message"]["content"]
            clean_text = clean_for_speech(raw_text)
            self._update_history(query, clean_text)
            return ExecutionResult(success=True, message=clean_text)
        except (KeyError, IndexError):
            return ExecutionResult(success=False, message="Empty response from OpenAI.")

    def _ask_mock(self, query: str) -> ExecutionResult:
        """Mock answer generator for testing and offline environments."""
        clean_q = query.lower()
        if "joke" in clean_q:
            answer = "Why do computer scientists love smart glasses? Because they give them a clear perspective on code!"
        elif "capital of france" in clean_q:
            answer = "The capital of France is Paris."
        elif "who are you" in clean_q or "what are you" in clean_q:
            answer = "I am OptiMistic, your augmented reality voice assistant running right on your glasses."
        elif "speed of light" in clean_q:
            answer = "The speed of light in a vacuum is approximately 299,792 kilometers per second."
        else:
            answer = f"I heard your question about {query}. I am OptiMistic and ready to help."
        self._update_history(query, answer)
        return ExecutionResult(success=True, message=answer)

    def _update_history(self, user_msg: str, assistant_msg: str) -> None:
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_msg})
        if len(self.history) > 10:
            self.history = self.history[-10:]

    def clear_history(self) -> None:
        self.history.clear()
