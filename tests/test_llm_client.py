"""Provider adapters, token budgets and runtime failures with HTTP mocks."""

import json
import httpx
import pandas as pd
import pytest
from llm_forecasting.client import (
    DEFAULT_MODEL,
    EXPECTED_DIGEST,
    EXPECTED_OLLAMA,
    GENERATION,
    GenerationError,
    LLMClient,
    Settings,
)
from llm_forecasting.prompts import response_schema, inspect_reply

DATES = pd.date_range("2026-07-07", periods=2)


def reply():
    return {"days": {"2026-07-07": 24.0, "2026-07-08": 24.0}}


def parse_daily(text):
    checked = inspect_reply(text, DATES)
    if checked["failed_days"]:
        raise ValueError("Invalid daily response")
    return checked["accepted_days"]


def ollama_result(value, **overrides):
    return {
        "message": {"content": json.dumps(value)},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 100,
        "eval_count": 200,
        **overrides,
    }


def mock_client(chat, *, provider="ollama", api_key="", sleep=lambda _: None):
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path.endswith("/api/generate"):
            return httpx.Response(200, json={"done": True})
        return chat(request)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    settings = Settings(
        provider,
        (
            "http://127.0.0.1:11434"
            if provider == "ollama"
            else "https://provider.invalid/v1"
        ),
        DEFAULT_MODEL if provider == "ollama" else "test-model",
        api_key,
    )
    return LLMClient(settings, http=http, sleep=sleep), calls


@pytest.mark.parametrize(
    "context,output,inputs,valid",
    [
        (16384, 8192, 5518, True),
        (8192, 4096, 4096, True),
        (8192, 4096, 4097, False),
        (16384, 8192, 8192, True),
        (16384, 8192, 8193, False),
        (16384, 4096, 10000, True),
    ],
)
def test_prompt_budget_reserves_actual_output_allowance(
    monkeypatch, context, output, inputs, valid
):
    monkeypatch.setitem(GENERATION, "context_length", context)
    monkeypatch.setitem(GENERATION, "max_output_tokens", output)
    client, calls = mock_client(
        lambda _: httpx.Response(
            200, json=ollama_result(reply(), prompt_eval_count=inputs)
        )
    )
    events = []
    if valid:
        value = client.generate(
            [], response_schema(DATES), parse_daily, record=events.append
        )
        assert len(value) == 2 and len(events) == 1
    else:
        with pytest.raises(
            GenerationError, match=f"{context-output}-token prompt budget"
        ):
            client.generate(
                [], response_schema(DATES), parse_daily, record=events.append
            )
        assert len(events) == 4
    assert all(row["input_token_budget"] == context - output for row in events)
    body = json.loads(calls[0].content)
    assert (
        body["options"]["num_ctx"] == context
        and body["options"]["num_predict"] == output
    )


def test_explicit_output_allowance_controls_input_budget(monkeypatch):
    monkeypatch.setitem(GENERATION, "context_length", 16384)
    monkeypatch.setitem(GENERATION, "max_output_tokens", 8192)
    client, _ = mock_client(
        lambda _: httpx.Response(
            200, json=ollama_result(reply(), prompt_eval_count=10000)
        )
    )
    events = []
    client.generate(
        [],
        response_schema(DATES),
        parse_daily,
        max_output_tokens=4096,
        record=events.append,
    )
    assert events[0]["input_token_budget"] == 12288
    assert events[0]["request"]["options"]["num_predict"] == 4096


def test_invalid_context_allocation_blocks_provider_call():
    client, calls = mock_client(
        lambda _: pytest.fail("Invalid allocation must block inference")
    )
    with pytest.raises(ValueError, match="leave room"):
        client.generate(
            [],
            response_schema(DATES),
            parse_daily,
            max_output_tokens=GENERATION["context_length"],
        )
    assert not calls


def test_provider_error_cannot_leak_key_into_saved_evidence():
    client, _ = mock_client(
        lambda _: httpx.Response(401, json={"error": "Invalid key private-test-key"}),
        provider="openai",
        api_key="private-test-key",
    )
    with pytest.raises(GenerationError) as caught:
        client.generate(
            [],
            response_schema(DATES),
            parse_daily,
        )
    assert "private-test-key" not in json.dumps(caught.value.records)
    assert "private-test-key" not in str(caught.value)
    assert "[redacted]" in json.dumps(caught.value.records)


@pytest.mark.parametrize(
    "failure", ["transport", "http", "truncated", "context", "unfinished"]
)
def test_runtime_and_response_failures(failure):
    def chat(request):
        if failure == "transport":
            raise httpx.ReadTimeout("test timeout", request=request)
        if failure == "http":
            return httpx.Response(503, json={"error": "unavailable"})
        options = {
            "truncated": {"done_reason": "length"},
            "context": {
                "prompt_eval_count": GENERATION["context_length"]
                - GENERATION["max_output_tokens"]
                + 1
            },
            "unfinished": {"done": False},
        }[failure]
        return httpx.Response(200, json=ollama_result(reply(), **options))

    client, _ = mock_client(chat)
    with pytest.raises(GenerationError) as error:
        client.generate(
            [],
            response_schema(DATES),
            parse_daily,
        )
    assert len(error.value.records) == 4
    assert "Last failure" in str(error.value)
    assert sum(row["status"] == "transport_error" for row in error.value.records) == (
        4 if failure == "transport" else 0
    )
    assert sum(row["status"] == "http_error" for row in error.value.records) == (
        4 if failure == "http" else 0
    )


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_hosted_adapter_auth_schema_usage_and_key_redaction(provider):
    def chat(request):
        body = json.loads(request.content)
        if provider == "openai":
            assert request.headers["authorization"] == "Bearer private-test-key"
            assert body["text"]["format"]["type"] == "json_schema"
            assert body["store"] is False
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": json.dumps(reply())}
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 12, "output_tokens": 20},
                },
            )
        assert request.headers["x-goog-api-key"] == "private-test-key"
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": json.dumps(reply())}]},
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 18,
                    "thoughtsTokenCount": 2,
                },
            },
        )

    client, calls = mock_client(chat, provider=provider, api_key="private-test-key")
    events = []
    result = client.generate(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "predict"},
        ],
        response_schema(DATES),
        parse_daily,
        record=events.append,
    )
    assert len(result) == 2 and len(calls) == 1
    assert events[0]["input_tokens"] == 12 and events[0]["output_tokens"] == 20
    assert events[0]["api_cost_usd"] is None
    assert "private-test-key" not in json.dumps(events)


def test_generic_configuration_and_cloud_requirements(monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert Settings.from_env() == Settings()
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="require"):
        Settings.from_env()
    monkeypatch.setenv("LLM_MODEL", "configured-model")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    assert Settings.from_env().public()["model"] == "configured-model"
    assert "secret" not in json.dumps(Settings.from_env().public())
    monkeypatch.setenv("LLM_BASE_URL", "http://insecure.invalid")
    with pytest.raises(ValueError, match="HTTPS"):
        Settings.from_env()


@pytest.mark.parametrize("problem", ["version", "digest", "missing", "hosted"])
def test_model_identity_guard(problem):
    def handler(request):
        if request.url.path.endswith("version"):
            return httpx.Response(
                200,
                json={"version": "other" if problem == "version" else EXPECTED_OLLAMA},
            )
        models = (
            []
            if problem == "missing"
            else [
                {
                    "name": DEFAULT_MODEL,
                    "digest": "other" if problem == "digest" else EXPECTED_DIGEST,
                }
            ]
        )
        return httpx.Response(200, json={"models": models})

    settings = (
        Settings()
        if problem != "hosted"
        else Settings("openai", "https://provider.invalid/v1", "test", "key")
    )
    client = LLMClient(
        settings, http=httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ValueError):
        client.identity()


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_interrupted_generation_records_evidence_without_retrying(interruption):
    events, sleeps = [], []

    def chat(_):
        raise interruption()

    client, calls = mock_client(chat, sleep=sleeps.append)
    with pytest.raises(interruption):
        client.generate(
            [],
            response_schema(DATES),
            parse_daily,
            record=events.append,
        )
    assert len(events) == 1 and events[0]["status"] == "interrupted"
    assert events[0]["retry_delay_seconds"] == 0 and sleeps == []
    assert len(calls) == 2  # One generation, followed by model unloading.
    assert calls[-1].url.path.endswith("/api/generate")
    assert events[0]["status"] == "interrupted"
