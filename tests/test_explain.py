"""Explanation facts, numeric checking and API fallbacks without a live model."""

import json

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from forecasting.api import create_app, load_artifacts
from llm_forecasting.client import LLMClient, Settings
from llm_forecasting.explain import (
    TEMPLATE,
    explanation_facts,
    render,
    validate_explanation,
)


def valid_reply():
    sentences = TEMPLATE.split(". ")
    return {
        "sentences": [
            sentence if sentence.endswith(".") else sentence + "."
            for sentence in sentences
        ]
    }


def client_with_response(value=None, offline=False):
    calls = []

    def handler(request):
        if request.url.path.endswith("/api/generate"):
            return httpx.Response(200, json={"done": True})
        calls.append(request)
        if offline:
            raise httpx.ConnectError("Deliberate unavailable provider", request=request)
        return httpx.Response(
            200,
            json={
                "done": True,
                "done_reason": "stop",
                "message": {
                    "content": json.dumps(value if value is not None else valid_reply())
                },
                "prompt_eval_count": 200,
                "eval_count": 150,
            },
        )

    return (
        LLMClient(
            Settings(),
            http=httpx.Client(transport=httpx.MockTransport(handler)),
            sleep=lambda _: None,
        ),
        calls,
    )


def test_explanation_facts_match_part1_daily_ranges(artifact_root):
    hourly, daily, _ = load_artifacts(artifact_root)
    facts = explanation_facts(hourly, daily)
    assert facts["period_total"] == f"{hourly.forecast.sum():,.1f}"
    day = daily.forecast.idxmax()
    assert facts["daily_low"] == f"{daily.loc[day, 'p10']:,.1f}"
    assert facts["daily_low"] != f"{hourly.loc[str(day.date())].p10.sum():,.1f}"
    text = validate_explanation(json.dumps(valid_reply()), facts)
    assert text == render(TEMPLATE, facts)
    assert "{{" not in text


@pytest.mark.parametrize("mode", ["literal", "mixed"])
def test_correct_literal_figures_in_template_roles_are_accepted(artifact_root, mode):
    hourly, daily, _ = load_artifacts(artifact_root)
    facts = explanation_facts(hourly, daily)
    value = valid_reply()
    if mode == "literal":
        value["sentences"] = [
            render(sentence, facts) for sentence in value["sentences"]
        ]
    else:
        value["sentences"][0] = render(value["sentences"][0], facts)
        value["sentences"][2] = value["sentences"][2].replace(
            "{{daily_low}}", facts["daily_low"]
        )
    assert validate_explanation(json.dumps(value), facts) == render(TEMPLATE, facts)
    provider, calls = client_with_response(value)
    with TestClient(create_app(artifact_root, llm_client=provider)) as api:
        response = api.get("/explain")
        assert response.json()["source"] == "llm"
        assert response.json()["explanation"] == render(TEMPLATE, facts)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "problem",
    [
        "wrong_total",
        "swapped_totals",
        "wrong_date",
        "swapped_dates",
        "reversed_range",
        "wrong_daily_point",
        "wrong_bound",
        "extra_number",
        "extra_claim",
        "altered_context",
    ],
)
def test_literal_figures_must_match_their_specific_fact_roles(artifact_root, problem):
    hourly, daily, _ = load_artifacts(artifact_root)
    facts = explanation_facts(hourly, daily)
    value = valid_reply()
    if problem == "wrong_total":
        value["sentences"][0] = value["sentences"][0].replace(
            "{{period_total}}", "999,999.0"
        )
    elif problem == "swapped_totals":
        value["sentences"][0] = value["sentences"][0].replace(
            "{{period_total}}", facts["busiest_period_total"]
        )
    elif problem == "wrong_date":
        value["sentences"][0] = value["sentences"][0].replace(
            "{{start_date}}", "2026-09-30"
        )
    elif problem == "swapped_dates":
        value["sentences"][0] = value["sentences"][0].replace(
            "{{start_date}}", facts["end_date"]
        )
    elif problem == "reversed_range":
        value["sentences"][2] = (
            value["sentences"][2]
            .replace("{{daily_low}}", facts["daily_high"])
            .replace("{{daily_high}}", facts["daily_low"])
        )
    elif problem == "wrong_daily_point":
        value["sentences"][2] = value["sentences"][2].replace(
            "{{daily_point}}", facts["period_total"]
        )
    elif problem == "wrong_bound":
        value["sentences"][2] = value["sentences"][2].replace(
            "{{daily_low}}", "999,999.0"
        )
    elif problem == "extra_number":
        value["sentences"][0] = value["sentences"][0][:-1] + " over 31 days."
    elif problem == "extra_claim":
        value["sentences"][0] += " Another 31 studies."
    else:
        # Reusing a correct number in a new claim does not verify its meaning.
        value["sentences"][0] = f"Staffing capacity is {facts['period_total']} studies."
    value["sentences"] = [render(sentence, facts) for sentence in value["sentences"]]
    with pytest.raises(ValueError):
        validate_explanation(json.dumps(value), facts)


@pytest.mark.parametrize(
    "problem",
    [
        "wrong_number",
        "number_word",
        "unicode_number",
        "unknown_fact",
        "missing_range",
        "reversed_range",
        "too_many_sentences",
        "one_sentence",
        "two_in_one",
        "extra_field",
        "malformed",
    ],
)
def test_bad_explanations_rejected(artifact_root, problem):
    hourly, daily, _ = load_artifacts(artifact_root)
    facts = explanation_facts(hourly, daily)
    value = valid_reply()
    if problem == "wrong_number":
        value["sentences"][0] = value["sentences"][0].replace(
            "{{period_total}}", "999,999.0"
        )
    elif problem == "number_word":
        value["sentences"][0] = "There are fifty studies."
    elif problem == "unicode_number":
        value["sentences"][0] = "The workload is ½ of normal."
    elif problem == "unknown_fact":
        value["sentences"][0] += " {{invented}}"
    elif problem == "missing_range":
        value["sentences"][2] = "The forecast remains uncertain."
    elif problem == "reversed_range":
        value["sentences"][2] = value["sentences"][2].replace(
            "{{daily_low}} to {{daily_high}}", "{{daily_high}} to {{daily_low}}"
        )
    elif problem == "too_many_sentences":
        value["sentences"].append("Another sentence.")
    elif problem == "one_sentence":
        value["sentences"] = ["One sentence."]
    elif problem == "two_in_one":
        value["sentences"][0] += " Extra claim."
    elif problem == "extra_field":
        value["extra"] = "ignored?"
    with pytest.raises(ValueError):
        validate_explanation(
            "{" if problem == "malformed" else json.dumps(value), facts
        )


@pytest.mark.parametrize("mode", ["valid", "invalid", "offline"])
def test_explain_endpoint_and_four_attempt_fallback(artifact_root, mode):
    value = (
        {"sentences": ["There are 999 studies.", "A fabricated range is 1 to 2."]}
        if mode == "invalid"
        else None
    )
    provider, calls = client_with_response(value, offline=mode == "offline")
    with TestClient(create_app(artifact_root, llm_client=provider)) as api:
        response = api.get("/explain")
        assert response.status_code == 200
        result = response.json()
        assert result["source"] == ("llm" if mode == "valid" else "template")
        assert result["metadata"] == api.get("/forecast/hourly").json()["metadata"]
        hourly, daily, _ = load_artifacts(artifact_root)
        assert result["explanation"] == render(
            TEMPLATE, explanation_facts(hourly, daily)
        )
        assert len(calls) == (1 if mode == "valid" else 4)


@pytest.mark.parametrize(
    "query,code",
    [
        ("start_date=bad", 422),
        ("start_date=2026-10-20&end_date=2026-10-10", 422),
        ("end_date=2026-11-01", 404),
        ("start_date=2026-09-30", 404),
    ],
)
def test_explain_date_errors_do_not_call_model(artifact_root, query, code):
    provider, calls = client_with_response()
    with TestClient(create_app(artifact_root, llm_client=provider)) as api:
        assert api.get("/explain?" + query).status_code == code
    assert not calls


def test_filtered_explanation_uses_inclusive_part1_range_only(artifact_root):
    provider, calls = client_with_response()
    with TestClient(create_app(artifact_root, llm_client=provider)) as api:
        response = api.get("/explain?start_date=2026-10-25&end_date=2026-10-31")
        assert response.status_code == 200 and response.json()["source"] == "llm"
    user = json.loads(json.loads(calls[0].content)["messages"][1]["content"])
    facts = user["verified_facts"]
    saved = (
        pd.read_csv(artifact_root / "forecast.csv", parse_dates=["hour_start"])
        .set_index("hour_start")
        .loc["2026-10-25":"2026-10-31"]
    )
    assert facts["start_date"] == "2026-10-25" and facts["end_date"] == "2026-10-31"
    assert facts["period_total"] == f"{saved.forecast.sum():,.1f}"


def test_startup_readiness_and_original_endpoints_need_no_llm(
    artifact_root, monkeypatch
):
    for name in ("LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    provider, calls = client_with_response(offline=True)
    with TestClient(create_app(artifact_root, llm_client=provider)) as api:
        assert api.get("/health").json()["status"] == "ready"
        assert len(api.get("/forecast/hourly").json()["data"]) == 744
        assert len(api.get("/forecast/daily").json()["data"]) == 31
    assert not calls


def test_missing_hosted_configuration_returns_template(artifact_root, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with TestClient(create_app(artifact_root)) as api:
        assert api.get("/health").status_code == 200
        result = api.get("/explain").json()
    assert result["source"] == "template" and result["model"] is None
