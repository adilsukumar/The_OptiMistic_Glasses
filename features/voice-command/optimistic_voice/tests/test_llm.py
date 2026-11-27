"""Tests for LLM conversational intelligence, multi-provider routing, and speech sanitization."""

from __future__ import annotations

import pytest

from optimistic_voice.assistant.command_parser import CommandParser
from optimistic_voice.assistant.command_router import CommandRouter
from optimistic_voice.assistant.intent_classifier import IntentType, ParsedCommand
from optimistic_voice.assistant.llm_client import LLMClient, clean_for_speech
from optimistic_voice.config import OptimisticConfig
from optimistic_voice.main import OptimisticAssistant


def test_clean_for_speech_removes_markdown() -> None:
    raw_markdown = (
        "**Quantum computing** is revolutionary!\n\n"
        "- It uses *qubits* instead of bits.\n"
        "- Learn more at https://example.com/quantum\n"
        "```python\nprint('hello')\n```\n"
        "### Key takeaway:\n1. Superposition is awesome."
    )
    cleaned = clean_for_speech(raw_markdown)
    assert "**" not in cleaned
    assert "*" not in cleaned
    assert "https://" not in cleaned
    assert "```" not in cleaned
    assert "#" not in cleaned
    assert "Quantum computing is revolutionary!" in cleaned
    assert "It uses qubits instead of bits." in cleaned
    assert "Superposition is awesome." in cleaned


def test_llm_client_mock_answers_questions() -> None:
    cfg = OptimisticConfig()
    client = LLMClient(config=cfg, provider="mock")

    # Joke query
    joke_res = client.ask("Tell me a funny joke")
    assert joke_res.success is True
    assert "smart glasses" in joke_res.message.lower() or "perspective" in joke_res.message.lower()

    # Fact query
    cap_res = client.ask("What is the capital of France?")
    assert cap_res.success is True
    assert "Paris" in cap_res.message

    # Multi-turn history tracking
    assert len(client.history) >= 4


def test_llm_client_missing_key_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure without API key it doesn't crash but provides helpful guidance
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    cfg = OptimisticConfig(gemini_api_key="", groq_api_key="", xai_api_key="")

    client_grok = LLMClient(config=cfg, provider="grok")
    res_grok = client_grok.ask("What is quantum entanglement?")
    assert res_grok.success is True
    assert "Grok" in res_grok.message or "XAI_API_KEY" in res_grok.message

    client_gemini = LLMClient(config=cfg, provider="gemini")
    res_gemini = client_gemini.ask("What is machine learning?")
    assert res_gemini.success is True
    assert "Gemini API key" in res_gemini.message

    client_groq = LLMClient(config=cfg, provider="groq")
    res_groq = client_groq.ask("What is machine learning?")
    assert res_groq.success is True
    assert "GROQ_API_KEY" in res_groq.message


def test_grok_auto_detection_from_key() -> None:
    cfg = OptimisticConfig(llm_provider="auto", xai_api_key="xai-test-key-12345")
    client = LLMClient(config=cfg)
    assert client._detect_provider() == "grok"


def test_command_parser_routes_questions_to_llm() -> None:
    parser = CommandParser(enable_llm=True)

    test_queries = [
        "What is the capital of France?",
        "Tell me a joke",
        "Who was Albert Einstein?",
        "How do airplanes fly?",
        "Why is the sky blue?",
        "Can you explain quantum computing?",
        "What can you do?",
        "Hello, how are you today?",
    ]

    for q in test_queries:
        cmd = parser.parse(q)
        assert cmd.intent == IntentType.LLM_QUERY, f"Failed on {q!r}: got {cmd.intent}"
        assert cmd.confidence >= 0.85


def test_command_parser_preserves_deterministic_automation_with_llm_enabled() -> None:
    parser = CommandParser(enable_llm=True)

    # Core OS automation must STILL parse as local commands, NOT LLM queries
    assert parser.parse("Open Google").intent == IntentType.OPEN_WEBSITE
    assert parser.parse("Open Notepad").intent == IntentType.OPEN_APPLICATION
    assert parser.parse("Open note").intent == IntentType.OPEN_APPLICATION
    assert parser.parse("Open pad").intent == IntentType.OPEN_APPLICATION
    assert parser.parse("Type hello world").intent == IntentType.TYPE_TEXT
    assert parser.parse("Search for Interstellar on YouTube").intent == IntentType.YOUTUBE_SEARCH
    assert parser.parse("Volume up").intent == IntentType.VOLUME
    assert parser.parse("Minimize window").intent == IntentType.DESKTOP_ACTION


def test_command_router_executes_llm_query() -> None:
    cfg = OptimisticConfig()
    mock_llm = LLMClient(config=cfg, provider="mock")
    router = CommandRouter(config=cfg, llm=mock_llm)

    cmd = ParsedCommand(
        intent=IntentType.LLM_QUERY,
        raw_text="What is the speed of light?",
        params={"query": "What is the speed of light?"},
    )
    result = router.execute(cmd)
    assert result.success is True
    assert "299,792" in result.message


def test_assistant_end_to_end_llm_execution() -> None:
    cfg = OptimisticConfig(
        execution_mode="local",
        wake_word_enabled=True,
        display_mode="console",
        tts_engine="mock",
    )
    mock_llm = LLMClient(config=cfg, provider="mock")
    router = CommandRouter(config=cfg, llm=mock_llm)
    assistant = OptimisticAssistant(config=cfg, router=router)

    # Single-breath command with wake prefix
    result = assistant.process_utterance("OptiMistic what is the capital of France")
    assert result is not None
    assert result.success is True
    assert "Paris" in result.message
