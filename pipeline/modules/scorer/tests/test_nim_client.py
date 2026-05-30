"""Unit tests for NIMTextClient — the production TextClient for the Scorer.

All tests mock the openai client so nothing hits the network.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pipeline.modules.scorer import NIMTextClient, NIMUnavailableError, TextClient


def _mock_openai_returning(content: str) -> MagicMock:
    """Build a MagicMock that mimics openai.OpenAI(...).chat.completions.create(...)."""
    fake_choice = MagicMock()
    fake_choice.message.content = content
    fake_response = MagicMock(choices=[fake_choice])
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response
    return fake_client


def test_satisfies_textclient_protocol() -> None:
    client = NIMTextClient(api_key="dummy")
    assert isinstance(client, TextClient)
    assert isinstance(client.model_id, str) and client.model_id


def test_model_id_defaults() -> None:
    client = NIMTextClient(api_key="dummy")
    # Default is the documented one — not asserting exact value, just non-empty.
    assert "/" in client.model_id  # NIM model slugs look like "vendor/name"


def test_model_id_from_explicit_arg() -> None:
    client = NIMTextClient(api_key="dummy", model_id="meta/llama-3.1-8b-instruct")
    assert client.model_id == "meta/llama-3.1-8b-instruct"


def test_model_id_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCORER_NIM_MODEL_ID", "meta/llama-3.1-405b-instruct")
    client = NIMTextClient(api_key="dummy")
    assert client.model_id == "meta/llama-3.1-405b-instruct"


def test_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_BASE_URL", "http://localhost:8081/v1")
    client = NIMTextClient(api_key="dummy")
    assert client._base_url == "http://localhost:8081/v1"


def test_complete_returns_message_content() -> None:
    client = NIMTextClient(api_key="dummy", model_id="meta/llama-3.1-8b-instruct")
    fake = _mock_openai_returning("the model said this")
    with patch.object(client, "_get_client", return_value=fake):
        result = client.complete("hello")
    assert result == "the model said this"

    # Verify the call shape matches OpenAI text-completion contract
    fake.chat.completions.create.assert_called_once()
    kwargs = fake.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "meta/llama-3.1-8b-instruct"
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert kwargs["temperature"] == 0.0


def test_complete_handles_none_content() -> None:
    """If the API returns None content, complete() returns empty string, not None."""
    client = NIMTextClient(api_key="dummy")
    fake = _mock_openai_returning(None)
    with patch.object(client, "_get_client", return_value=fake):
        result = client.complete("hello")
    assert result == ""


def test_complete_raises_nim_unavailable_on_failure(capsys: pytest.CaptureFixture[str]) -> None:
    client = NIMTextClient(api_key="dummy", model_id="m/x")
    fake = MagicMock()
    fake.chat.completions.create.side_effect = RuntimeError("connection refused")
    with patch.object(client, "_get_client", return_value=fake):
        with pytest.raises(NIMUnavailableError) as exc:
            client.complete("hello")
    assert exc.value.model == "m/x"
    assert "connection refused" in exc.value.detail


def test_missing_api_key_warns_at_construction(capsys: pytest.CaptureFixture[str],
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    NIMTextClient()  # no api_key arg, no env
    stderr = capsys.readouterr().err
    assert "NVIDIA_API_KEY is not set" in stderr


# ---------------------------------------------------------------------------
# Timeout — must be passed to the OpenAI client so a stuck NIM call cannot
# hang a worker thread (Scorer.score_batch uses ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def test_timeout_defaults_to_600_seconds() -> None:
    """10-minute ceiling matches the documented hygiene fix."""
    client = NIMTextClient(api_key="dummy")
    assert client._timeout == 600.0


def test_timeout_constructor_arg_overrides_default() -> None:
    client = NIMTextClient(api_key="dummy", timeout=30.0)
    assert client._timeout == 30.0


def test_timeout_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_NIM_TIMEOUT_SECONDS", "120")
    client = NIMTextClient(api_key="dummy")
    assert client._timeout == 120.0


def test_timeout_is_passed_to_openai_constructor() -> None:
    """The timeout must reach the OpenAI client — that's what the fix is for."""
    client = NIMTextClient(api_key="dummy", timeout=42.0)
    with patch("openai.OpenAI") as fake_openai:
        client._get_client()
    fake_openai.assert_called_once()
    assert fake_openai.call_args.kwargs.get("timeout") == 42.0
