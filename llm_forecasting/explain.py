"""LLM-written sentences with computed Part 1 facts and a fixed fallback."""

import re

import httpx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .client import GenerationError, LLMClient
from .prompts import strict_json

PLACEHOLDER = re.compile(r"\{\{([a-z_]+)\}\}")
NUMERIC_WORD = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"billion|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"half|quarter|double|twice|triple|dozen|percent|percentage)\b",
    re.I,
)
REQUIRED_FACTS = {
    "period_total",
    "busiest_period",
    "busiest_period_total",
    "busiest_day",
    "daily_low",
    "daily_high",
}
TEMPLATE = (
    "For {{start_date}} through {{end_date}}, the forecast totals {{period_total}} recorded studies. "
    "The busiest forecast period is {{busiest_period}}, with {{busiest_period_total}} studies. "
    "On {{busiest_day}}, the daily point forecast is {{daily_point}} and its range is {{daily_low}} to {{daily_high}} studies."
)


class Sentences(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    sentences: list[str] = Field(min_length=2, max_length=3)


def explanation_facts(hourly, daily):
    if hourly.empty or daily.empty:
        raise ValueError("Cannot explain an empty forecast range")
    point = float(hourly.forecast.sum())
    day = daily.forecast.idxmax()
    peak = hourly.forecast.idxmax()
    weeks = (daily.index - pd.Timestamp("2026-10-01")).days // 7
    groups = [
        (frame.index[0], frame.index[-1], float(frame.forecast.sum()))
        for _, frame in daily.groupby(weeks)
    ]
    first, last, total = max(groups, key=lambda row: row[2])
    return {
        "start_date": str(hourly.index[0].date()),
        "end_date": str(hourly.index[-1].date()),
        "period_total": f"{point:,.1f}",
        "busiest_period": f"{first.date()} through {last.date()}",
        "busiest_period_total": f"{total:,.1f}",
        "busiest_day": str(day.date()),
        "daily_point": f"{daily.loc[day, 'forecast']:,.1f}",
        "daily_low": f"{daily.loc[day, 'p10']:,.1f}",
        "daily_high": f"{daily.loc[day, 'p90']:,.1f}",
        "busiest_hour": peak.strftime("%Y-%m-%d %H:%M"),
        "hourly_point": f"{hourly.loc[peak, 'forecast']:,.1f}",
    }


def render(text, facts):
    return PLACEHOLDER.sub(lambda match: facts[match[1]], text)


def bind_template_facts(sentence, facts):
    """Recognize literal figures only in their exact, known template roles."""
    for fragment in TEMPLATE.split(". "):
        template = fragment if fragment.endswith(".") else fragment + "."
        parts, previous = [], 0
        for match in PLACEHOLDER.finditer(template):
            parts.append(re.escape(template[previous : match.start()]))
            # A total cannot be accepted merely because it appears elsewhere
            # in facts: this slot must match this particular named fact.
            parts.append(
                "(?:" + re.escape(match[0]) + "|" + re.escape(facts[match[1]]) + ")"
            )
            previous = match.end()
        parts.append(re.escape(template[previous:]))
        if re.fullmatch("".join(parts), sentence.strip(), flags=re.I):
            return template
    return sentence


def validate_explanation(text, facts):
    response = Sentences.model_validate(strict_json(text))
    sentences = [
        bind_template_facts(sentence, facts) for sentence in response.sentences
    ]
    used = set()
    for sentence in sentences:
        placeholders = PLACEHOLDER.findall(sentence)
        if not set(placeholders) <= facts.keys():
            raise ValueError("Unknown explanation fact")
        remainder = PLACEHOLDER.sub("", sentence)
        if (
            re.search(r"\d|[{}\n\r]", remainder)
            or any(char.isnumeric() for char in remainder)
            or NUMERIC_WORD.search(remainder)
        ):
            raise ValueError("A numeric claim was not supplied through a verified fact")
        if re.fullmatch(r"[^.!?]+[.!?]", remainder.strip()) is None:
            raise ValueError("Each list item must contain exactly one plain sentence")
        if "daily_low" in placeholders or "daily_high" in placeholders:
            if not {"daily_low", "daily_high"} <= set(placeholders) or sentence.find(
                "{{daily_low}}"
            ) > sentence.find("{{daily_high}}"):
                raise ValueError(
                    "Daily range must use verified lower then upper bounds"
                )
        if "busiest_period" in placeholders and "busiest" not in sentence.lower():
            raise ValueError("Busiest-period fact must be identified correctly")
        used.update(placeholders)
    if not REQUIRED_FACTS <= used:
        raise ValueError("Explanation omitted required workload or uncertainty facts")
    original = " ".join(sentence.strip() for sentence in sentences)
    result = render(original, facts)
    # Every rendered numeric span must originate in one of the supplied values.
    number_pattern = r"\d+(?:,\d{3})*(?:\.\d+)?"
    allowed = {
        number for name in used for number in re.findall(number_pattern, facts[name])
    }
    if set(re.findall(number_pattern, result)) - allowed or "{{" in result:
        raise ValueError("Rendered explanation contains an unverified figure")
    return result


def explanation_messages(facts):
    import json

    return [
        {
            "role": "system",
            "content": "Write two or three short plain-language forecast sentences as a JSON sentences list. Use ONLY the supplied facts. Prefer copying fact placeholders exactly, for example {{period_total}}. If writing literal figures, copy example_structure exactly and substitute each placeholder only with its corresponding verified value. Literal figures in other wording are rejected. Never add number words, percentages, causes, staffing advice or confidence claims. Include period_total, busiest_period with busiest_period_total, and busiest_day with daily_low then daily_high in the same range sentence. Each list item must be exactly one sentence ending with a period. Busiest periods are source-calendar seven-day blocks, possibly partial at the edges. Daily ranges are separately calibrated Part 1 ranges, not sums of hourly bounds. The timezone is unspecified.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {"verified_facts": facts, "example_structure": TEMPLATE},
                separators=(",", ":"),
            ),
        },
    ]


def explain_forecast(hourly, daily, client=None):
    facts = explanation_facts(hourly, daily)
    result = {
        "explanation": render(TEMPLATE, facts),
        "source": "template",
        "model": None,
        "facts": facts,
        "records": [],
    }
    try:
        client = client or LLMClient()
        result["model"] = client.settings.model
        text = client.generate(
            explanation_messages(facts),
            Sentences.model_json_schema(),
            lambda text: validate_explanation(text, facts),
            record=result["records"].append,
            timeout=20,
            max_output_tokens=512,
        )
        result.update(explanation=text, source="llm")
    except (GenerationError, httpx.HTTPError, ValueError, OSError, RuntimeError):
        pass  # The fixed template uses the same verified facts, without an LLM.
    return result


def explain_demo(root=None, client=None):
    from forecasting.api import load_artifacts
    from .pipeline import ROOT, frozen_protocol, output, write_json

    root = ROOT if root is None else root
    protocol = frozen_protocol(root)
    client = client or LLMClient()
    if client.settings.public() != protocol["provider"]:
        raise ValueError(
            "Use the frozen local Qwen configuration for the demonstration"
        )
    hourly, daily, metadata = load_artifacts(root)
    live = explain_forecast(hourly, daily, client)

    def unavailable(request):
        raise httpx.ConnectError(
            "Deliberately simulated provider failure", request=request
        )

    with httpx.Client(transport=httpx.MockTransport(unavailable)) as http:
        offline = explain_forecast(
            hourly, daily, LLMClient(client.settings, http=http, sleep=lambda _: None)
        )
    write_json(
        output(root) / "explanations.json",
        {
            "metadata": metadata.model_dump(mode="json"),
            "live": live,
            "simulated_provider_failure": offline,
        },
    )
    if live["source"] != "llm":
        failures = [
            record["error"] for record in live["records"] if record.get("error")
        ]
        detail = f". Last failure: {failures[-1]}" if failures else ""
        raise ValueError(
            "Saved fallback evidence, but a genuine validated LLM explanation is still required"
            + detail
        )
    print("Saved a genuine LLM explanation and a labelled simulated-failure fallback.")
