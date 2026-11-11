"""HTTPX adapters and bounded retries. No provider SDK or automatic server startup."""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import os
import threading
import time
from urllib.parse import quote, urlsplit

import httpx

DEFAULT_MODEL = "qwen3.5:4b-q4_K_M"
EXPECTED_OLLAMA = "0.40.15"
EXPECTED_DIGEST = "d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04"
GENERATION = {
    "temperature": 0.2,
    "context_length": 16384,
    "max_output_tokens": 8192,
    "thinking": False,
    "retries": 3,
    "seed": None,
    "keep_alive": 0,
}
INFERENCE_LOCK = threading.Lock()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Settings:
    provider: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = DEFAULT_MODEL
    api_key: str = ""

    @classmethod
    def from_env(cls):
        provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
        defaults = {
            "ollama": "http://127.0.0.1:11434",
            "openai": "https://api.openai.com/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta",
        }
        if provider not in defaults:
            raise ValueError("LLM_PROVIDER must be ollama, openai or gemini")
        model = os.getenv(
            "LLM_MODEL", DEFAULT_MODEL if provider == "ollama" else ""
        ).strip()
        key = os.getenv("LLM_API_KEY", "")
        base = (os.getenv("LLM_BASE_URL") or defaults[provider]).rstrip("/")
        if not model or (provider != "ollama" and not key):
            raise ValueError("Hosted providers require LLM_MODEL and LLM_API_KEY")
        parsed = urlsplit(base)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "LLM_BASE_URL must be an HTTP(S) URL without credentials or query parameters"
            )
        if provider != "ollama" and parsed.scheme != "https":
            raise ValueError("Hosted providers require HTTPS")
        return cls(provider, base, model, key)

    def public(self):
        return {k: v for k, v in asdict(self).items() if k != "api_key"}


class GenerationError(Exception):
    def __init__(self, records):
        message = "LLM request failed after the initial call and three retries"
        if records:
            last = records[-1]
            detail = str(last.get("error", "Unknown failure")).replace("\n", " ")[:400]
            message += f". Last failure ({last.get('status', 'unknown')}): {detail}"
        super().__init__(message)
        self.records = records


class PartialResponseError(ValueError):
    """A usable subset was retained, but more requested values are needed."""


class LLMClient:
    def __init__(self, settings=None, *, http=None, sleep=time.sleep):
        self.settings = settings or Settings.from_env()
        self.http = http
        self.sleep = sleep

    def _request(self, method, path, *, timeout=120, **kwargs):
        settings = self.settings
        headers = {}
        if settings.provider == "openai":
            headers["Authorization"] = f"Bearer {settings.api_key}"
        elif settings.provider == "gemini":
            headers["x-goog-api-key"] = settings.api_key
        kwargs["headers"] = headers
        kwargs["timeout"] = httpx.Timeout(timeout, connect=3)
        url = f"{settings.base_url.rstrip('/')}/{path.lstrip('/')}"
        if self.http is not None:
            return self.http.request(method, url, **kwargs)
        with httpx.Client(trust_env=settings.provider != "ollama") as client:
            return client.request(method, url, **kwargs)

    def identity(self):
        if self.settings.provider != "ollama":
            raise ValueError(
                "Assignment experiments use only the installed local Qwen model"
            )
        if self.settings.model != DEFAULT_MODEL or urlsplit(
            self.settings.base_url
        ).hostname not in ("127.0.0.1", "localhost", "host.docker.internal"):
            raise ValueError("Assignment experiments require local qwen3.5:4b-q4_K_M")
        version = (
            self._request("GET", "api/version", timeout=5)
            .raise_for_status()
            .json()["version"]
        )
        models = (
            self._request("GET", "api/tags", timeout=5)
            .raise_for_status()
            .json()["models"]
        )
        model = next(
            (row for row in models if row["name"] == self.settings.model), None
        )
        if model is None:
            raise ValueError("Required model is missing; pull qwen3.5:4b-q4_K_M")
        if version != EXPECTED_OLLAMA or model["digest"] != EXPECTED_DIGEST:
            raise ValueError(
                "Ollama version or installed model digest differs from the agreed experiment"
            )
        return {
            "ollama_version": version,
            "model": model["name"],
            "digest": model["digest"],
            "details": model.get("details", {}),
        }

    def payload(self, messages, schema, max_output_tokens):
        settings = self.settings
        if settings.provider == "ollama":
            return "api/chat", {
                "model": settings.model,
                "messages": messages,
                "format": schema,
                "think": GENERATION["thinking"],
                "stream": False,
                "keep_alive": GENERATION["keep_alive"],
                "options": {
                    "temperature": GENERATION["temperature"],
                    "num_ctx": GENERATION["context_length"],
                    "num_predict": max_output_tokens,
                },
            }
        if settings.provider == "openai":
            return "responses", {
                "model": settings.model,
                "input": messages,
                "temperature": GENERATION["temperature"],
                "max_output_tokens": max_output_tokens,
                "store": False,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "result",
                        "strict": True,
                        "schema": schema,
                    }
                },
            }
        contents = [
            {"role": "user", "parts": [{"text": row["content"]}]}
            for row in messages
            if row["role"] != "system"
        ]
        system = "\n".join(
            row["content"] for row in messages if row["role"] == "system"
        )
        return f"models/{quote(settings.model, safe='')}:generateContent", {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {
                "temperature": GENERATION["temperature"],
                "maxOutputTokens": max_output_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            },
        }

    def _normalize(self, data):
        provider = self.settings.provider
        if provider == "ollama":
            return (
                data["message"]["content"],
                data.get("prompt_eval_count"),
                data.get("eval_count"),
                data.get("done") is not True or data.get("done_reason") == "length",
            )
        if provider == "openai":
            text = "".join(
                part["text"]
                for item in data.get("output", [])
                if item.get("type") == "message"
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            )
            usage = data.get("usage") or {}
            return (
                text,
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                data.get("status") != "completed",
            )
        candidate = data["candidates"][0]
        text = "".join(
            part.get("text", "")
            for part in candidate["content"]["parts"]
            if not part.get("thought")
        )
        usage = data.get("usageMetadata", {})
        output = usage.get("candidatesTokenCount")
        if output is not None:
            output += usage.get("thoughtsTokenCount", 0)
        return (
            text,
            usage.get("promptTokenCount"),
            output,
            candidate.get("finishReason") != "STOP",
        )

    def unload(self):
        if self.settings.provider == "ollama":
            self._request(
                "POST",
                "api/generate",
                timeout=5,
                json={"model": self.settings.model, "keep_alive": 0},
            ).raise_for_status()

    def generate(self, messages, schema, validator, **kwargs):
        with INFERENCE_LOCK:
            return self._generate(messages, schema, validator, **kwargs)

    def _generate(
        self,
        messages,
        schema,
        validator,
        *,
        record=None,
        before_attempt=None,
        timeout=120,
        max_output_tokens=None,
        request_factory=None,
    ):
        if max_output_tokens is None:
            max_output_tokens = GENERATION["max_output_tokens"]
        input_budget = None
        if self.settings.provider == "ollama":
            input_budget = GENERATION["context_length"] - max_output_tokens
            if max_output_tokens <= 0 or input_budget <= 0:
                raise ValueError(
                    "Ollama context must leave room for both input and output tokens"
                )
        records = []
        maximum = GENERATION["retries"] + 1
        for attempt in range(1, maximum + 1):
            if before_attempt is not None:
                before_attempt()
            if request_factory is not None:
                messages, schema = request_factory()
            path, payload = self.payload(messages, schema, max_output_tokens)
            started = time.perf_counter()
            event = {
                "attempt": attempt,
                "started_at": datetime.now(UTC).isoformat(),
                "provider": self.settings.provider,
                "model": self.settings.model,
                "base_url": self.settings.base_url,
                "request": payload,
                "request_sha256": digest(payload),
                "received_reply": False,
                "input_tokens": None,
                "output_tokens": None,
                "input_token_budget": input_budget,
                "api_cost_usd": 0.0 if self.settings.provider == "ollama" else None,
            }
            valid = False
            interruption = None
            try:
                response = self._request("POST", path, timeout=timeout, json=payload)
                event["http_status"] = response.status_code
                event["raw_response"] = response.text
                response.raise_for_status()
                event["received_reply"] = True
                text, inputs, outputs, truncated = self._normalize(response.json())
                event.update(text=text, input_tokens=inputs, output_tokens=outputs)
                if truncated:
                    raise ValueError(
                        "Response was truncated or did not finish normally"
                    )
                if (
                    self.settings.provider == "ollama"
                    and inputs is not None
                    and inputs > input_budget
                ):
                    raise ValueError(
                        f"Input uses {inputs} tokens, exceeding the {input_budget}-token "
                        f"prompt budget (context {GENERATION['context_length']} minus "
                        f"output allowance {max_output_tokens}); shorten development prompts before freezing"
                    )
                value = validator(text)
                event["status"] = "valid"
                valid = True
            except httpx.HTTPStatusError as error:
                event.update(status="http_error", error=str(error))
            except httpx.RequestError as error:
                event.update(status="transport_error", error=str(error))
            except PartialResponseError as error:
                event.update(status="partial", error=str(error))
            except (ValueError, KeyError, TypeError, IndexError) as error:
                event.update(status="invalid", error=str(error))
            except (KeyboardInterrupt, SystemExit) as error:
                interruption = error
                event.update(status="interrupted", error=type(error).__name__)
            finally:
                # Unload/cancel on failures too; do not introduce another inference.
                try:
                    self.unload()
                except (httpx.HTTPError, ValueError) as error:
                    event["unload_error"] = str(error)
                except (KeyboardInterrupt, SystemExit) as error:
                    interruption = error
                    event.update(status="interrupted", error=type(error).__name__)
            event["latency_seconds"] = time.perf_counter() - started
            event["retry_delay_seconds"] = (
                0
                if valid or interruption is not None or attempt == maximum
                else 2 ** (attempt - 1)
            )
            # Provider errors occasionally echo credentials. Never save keys.
            if self.settings.api_key:
                event = json.loads(
                    json.dumps(event).replace(self.settings.api_key, "[redacted]")
                )
            records.append(event)
            if record is not None:
                record(event)
            if interruption is not None:
                raise interruption
            if valid:
                return value
            if attempt < maximum:
                self.sleep(2 ** (attempt - 1))
        raise GenerationError(records)
