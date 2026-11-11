"""Hybrid contracts and complete manual stages, isolated from real model calls."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import httpx
import numpy as np
import pandas as pd
import pytest

from forecasting.artifacts import validate_hourly
from llm_forecasting.client import (
    DEFAULT_MODEL,
    EXPECTED_DIGEST,
    EXPECTED_OLLAMA,
    GENERATION,
    GenerationError,
    LLMClient,
    Settings,
)
from llm_forecasting import pipeline as p
from llm_forecasting.prompts import (
    allocate,
    batches,
    date_keys,
    empirical_ranges,
    history_before,
    history_summary,
    hourly_proportions,
    inspect_reply,
    messages,
    response_schema,
)

DATES = pd.date_range("2026-07-07", periods=2)
EXPERIMENT = "a" * 64


def make_client(chat=None, sleep=lambda _: None):
    calls = []

    def handler(request):
        if request.url.path.endswith("/api/version"):
            return httpx.Response(200, json={"version": EXPECTED_OLLAMA})
        if request.url.path.endswith("/api/tags"):
            return httpx.Response(
                200,
                json={"models": [{"name": DEFAULT_MODEL, "digest": EXPECTED_DIGEST}]},
            )
        if request.url.path.endswith("/api/generate"):
            return httpx.Response(200, json={"done": True})
        calls.append(json.loads(request.content))
        result = (
            chat(calls[-1])
            if chat
            else {"days": {key: 24.0 for key in user(calls[-1])["response_dates"]}}
        )
        if isinstance(result, BaseException):
            raise result
        if isinstance(result, httpx.Response):
            return result
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps(result)},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 120,
                "eval_count": 25,
            },
        )

    client = LLMClient(
        Settings(),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=sleep,
    )
    return client, calls


def user(body):
    return json.loads(body["messages"][1]["content"])


def recover(root, client, dates=DATES, summary=None, key=None):
    return p.recover_batch(
        summary or {"history_end": "2026-07-06 23:00:00"},
        dates,
        root,
        client,
        key or "development/2026-07-07/1/main/2026-07-07T00:00:00/2026-07-07T00:00:00",
        "development",
        "main",
        "2026-07-07",
        1,
        EXPERIMENT,
    )


def test_summary_proportions_and_complete_prediction_leakage(hourly_history, tmp_path):
    origin = "2026-07-21"
    altered = hourly_history.copy()
    altered.loc[origin:] = 9999
    summary = history_summary(hourly_history, origin)
    assert summary == history_summary(altered, origin)
    assert len(summary["recent_28_daily_totals"]) == 28
    assert "weekday_hour_profiles" not in summary
    assert summary["unusual_days_removed"] is False
    assert summary["sparse_days"] and summary["empty_days"]
    proportions = hourly_proportions(hourly_history, origin)
    pd.testing.assert_frame_equal(proportions, hourly_proportions(altered, origin))
    client, _ = make_client()
    a = p.forecast_window(
        hourly_history, origin, 2, 1, "development", tmp_path / "a", client, EXPERIMENT
    )
    b = p.forecast_window(
        altered, origin, 2, 1, "development", tmp_path / "b", client, EXPERIMENT
    )
    pd.testing.assert_frame_equal(a["hourly"], b["hourly"])
    pd.testing.assert_frame_equal(a["daily"], b["daily"])


@pytest.mark.parametrize("problem", ["aware", "missing", "negative", "nan", "short"])
def test_invalid_history(hourly_history, problem):
    series = hourly_history.copy()
    if problem == "aware":
        series.index = series.index.tz_localize("UTC")
    elif problem == "missing":
        series = series.drop(series.index[2])
    elif problem == "negative":
        series.iloc[0] = -1
    elif problem == "nan":
        series = series.astype(float)
        series.iloc[0] = np.nan
    else:
        series = series.loc["2026-07-15":]
    with pytest.raises(ValueError):
        history_before(series, "2026-07-21")


@pytest.mark.parametrize(
    "days,count,hours", [(28, 14, 672), (30, 15, 720), (31, 16, 744), (7, 4, 168)]
)
def test_batches_and_hour_counts(hourly_history, days, count, hours):
    parts = list(batches("2026-07-21", days))
    assert len(parts) == count
    assert len(parts[-1]) == (1 if days % 2 else 2)
    dates = pd.DatetimeIndex([day for part in parts for day in part])
    assert dates.equals(pd.date_range("2026-07-21", periods=days))
    result = allocate(
        {str(day.date()): 24.0 for day in dates},
        hourly_proportions(hourly_history, "2026-07-21"),
    )
    assert len(result) == hours
    assert result.index.equals(pd.date_range("2026-07-21", periods=hours, freq="h"))


def test_batches_cross_month_and_retry_calendar():
    dates = list(batches("2026-07-31", 3))
    assert date_keys(dates[0]) == ["2026-07-31", "2026-08-01"]
    prompt = user({"messages": messages({"fixed": True}, dates[0], dates[0][1:])})
    assert len(prompt["calendar"]) == 2
    assert prompt["response_dates"] == ["2026-08-01"]
    assert prompt["calendar"][1]["weekday"] == 5
    schema = response_schema(dates[0][1:])
    assert schema["properties"]["days"]["required"] == ["2026-08-01"]
    assert schema["properties"]["days"]["additionalProperties"] is False


@pytest.mark.parametrize(
    "value", [True, "2", -1, None, [2], {}, float("nan"), float("inf"), 10**400]
)
def test_invalid_daily_values_are_not_accepted(value):
    reply = {"days": {"2026-07-07": 24.5, "2026-07-08": value}}
    if isinstance(value, float) and not np.isfinite(value):
        with pytest.raises(ValueError):
            inspect_reply(json.dumps(reply), DATES)
    else:
        checked = inspect_reply(json.dumps(reply), DATES)
        assert checked["accepted_days"] == {"2026-07-07": 24.5}
        assert list(checked["failed_days"]) == ["2026-07-08"]


@pytest.mark.parametrize(
    "text",
    [
        "not-json",
        "{",
        "[]",
        '{"days":[]}',
        '{"days":{},"extra":1}',
        '{"days":{"2026-07-09":1}}',
        '{"days":{"2026-07-07":1,"2026-07-07":2}}',
        '{"days":{},"days":{}}',
    ],
)
def test_invalid_envelopes_reject_whole_response(text):
    with pytest.raises(ValueError):
        inspect_reply(text, DATES)


def test_shuffled_dates_missing_dates_and_zero():
    checked = inspect_reply('{"days":{"2026-07-08":0,"2026-07-07":2.5}}', DATES)
    assert list(checked["accepted_days"]) == date_keys(DATES)
    assert not checked["failed_days"]
    assert list(inspect_reply('{"days":{"2026-07-08":0}}', DATES)["failed_days"]) == [
        "2026-07-07"
    ]


def test_proportion_formula_fallbacks_and_conservation(hourly_history):
    history = hourly_history.copy()
    history.loc[history.index.day_of_week == 0] = 0
    proportions = hourly_proportions(history, "2026-07-21")
    assert set(proportions.loc[proportions.weekday == 0, "profile_source"]) == {
        "overall"
    }
    np.testing.assert_allclose(proportions.groupby("weekday").proportion.sum(), 1)
    totals = {"2026-07-21": 1234.56, "2026-07-22": 0.0}
    allocated = allocate(totals, proportions)
    np.testing.assert_allclose(
        allocated.forecast.resample("D").sum(), list(totals.values()), rtol=0, atol=1e-6
    )
    assert (allocated.loc["2026-07-22"].forecast == 0).all()
    weekday = history.loc[
        (history.index < "2026-07-21") & (history.index.day_of_week == 1)
    ]
    expected = weekday.groupby(weekday.index.hour).sum() / weekday.sum()
    np.testing.assert_allclose(
        proportions.loc[proportions.weekday == 1, "proportion"], expected
    )
    uniform = hourly_proportions(history * 0, "2026-07-21")
    assert set(uniform.profile_source) == {"uniform"}
    np.testing.assert_allclose(uniform.proportion, 1 / 24)
    with pytest.raises(ValueError, match="normalized"):
        allocate(totals, proportions.assign(proportion=2.0))


def test_residual_calibration_separate_daily_intervals():
    index = pd.date_range("2026-06-16", periods=504, freq="h")
    # Alternating within-day errors cancel; daily bands must not sum hourly bands.
    residuals = pd.Series(
        np.tile([-4.0, 4.0], 252) * np.repeat(np.arange(1, 22), 24), index=index
    )
    future = pd.DataFrame(
        {"forecast": 100.0}, index=pd.date_range("2026-07-07", periods=48, freq="h")
    )
    hourly, daily = empirical_ranges(future, residuals)
    assert (daily.p10 == 2400).all() and (daily.p90 == 2400).all()
    assert not np.allclose(hourly.p10.resample("D").sum(), daily.p10)
    assert hourly.p10.iloc[0] == pytest.approx(24.0)
    assert hourly.p90.iloc[1] == pytest.approx(176.0)
    with pytest.raises(ValueError):
        empirical_ranges(future, residuals.iloc[1:])
    shifted = residuals.copy()
    shifted.index += pd.Timedelta(days=30)
    with pytest.raises(ValueError):
        empirical_ranges(future, shifted)


def test_partial_retry_only_failed_dates_and_cache(tmp_path):
    sleeps = []

    def chat(body):
        keys = user(body)["response_dates"]
        return {
            "days": {
                key: (
                    123.5 if key.endswith("07") else (-1 if len(keys) == 2 else 456.25)
                )
                for key in keys
            }
        }

    client, calls = make_client(chat, sleeps.append)
    values = recover(tmp_path, client)
    assert values == {"2026-07-07": 123.5, "2026-07-08": 456.25}
    assert [user(body)["response_dates"] for body in calls] == [
        date_keys(DATES),
        ["2026-07-08"],
    ]
    assert sleeps == [1]
    assert user(calls[0])["history"] == user(calls[1])["history"]
    assert user(calls[0])["calendar"] == user(calls[1])["calendar"]
    assert recover(tmp_path, client) == values and len(calls) == 2
    rows = p.records(tmp_path)
    assert [r["status"] for r in rows] == ["partial", "valid"]
    assert p.reliability(rows)["invalid_reply_share"] == 0.5
    assert calls[0]["options"]["num_predict"] == GENERATION["max_output_tokens"]


def test_four_attempts_partial_resume_and_first_value_protection(tmp_path):
    sleeps = []
    client, calls = make_client(
        lambda body: (
            {"days": {"2026-07-07": 12.5}}
            if len(user(body)["response_dates"]) == 2
            else {"days": {}}
        ),
        sleeps.append,
    )
    with pytest.raises(GenerationError):
        recover(tmp_path, client)
    assert len(calls) == 4 and sleeps == [1, 2, 4]
    assert all(user(body)["response_dates"] == ["2026-07-08"] for body in calls[1:])
    replacement, retried = make_client()
    values = recover(tmp_path, replacement)
    assert values == {"2026-07-07": 12.5, "2026-07-08": 24.0}
    assert len(retried) == 1 and user(retried[0])["response_dates"] == ["2026-07-08"]
    bad, _ = make_client()
    with pytest.raises(ValueError, match="mismatch"):
        recover(tmp_path, bad, summary={"changed": True})


def test_cache_isolation_and_malformed_journal(tmp_path):
    client, calls = make_client()
    recover(tmp_path, client)
    other = pd.date_range("2026-07-09", periods=2)
    recover(
        tmp_path,
        client,
        dates=other,
        key="development/2026-07-07/1/main/2026-07-07T00:00:00/2026-07-09T00:00:00",
    )
    assert len(calls) == 2
    path = p.output(tmp_path) / "requests.jsonl"
    path.write_text(path.read_text() + "broken\n")
    with pytest.raises(ValueError, match="Malformed"):
        p.records(tmp_path)
    assert path.read_text().endswith("broken\n")


def test_global_budget_and_interrupt_preserve_evidence(tmp_path, monkeypatch):
    client, calls = make_client()
    recover(tmp_path, client)
    monkeypatch.setattr(p, "MAX_ATTEMPTS", 1)
    with pytest.raises(ValueError, match="budget"):
        recover(
            tmp_path,
            client,
            dates=DATES + pd.Timedelta(days=2),
            key="development/2026-07-07/1/main/2026-07-07T00:00:00/2026-07-09T00:00:00",
        )
    assert len(calls) == 1
    monkeypatch.setattr(p, "MAX_ATTEMPTS", 748)
    interrupted, _ = make_client(lambda _: KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        recover(
            tmp_path,
            interrupted,
            dates=DATES + pd.Timedelta(days=2),
            key="development/2026-07-07/1/main/2026-07-07T00:00:00/2026-07-09T00:00:00",
        )
    assert p.records(tmp_path)[-1]["status"] == "interrupted"


def test_truncation_and_transport_failures(tmp_path):
    response = httpx.Response(
        200,
        json={
            "message": {"content": '{"days":{"2026-07-07":24}}'},
            "done": True,
            "done_reason": "length",
        },
    )
    client, calls = make_client(lambda _: response)
    with pytest.raises(GenerationError):
        recover(tmp_path, client)
    assert len(calls) == 4 and all(
        row["day_validation"] is None for row in p.records(tmp_path)
    )
    client, _ = make_client(lambda _: httpx.ConnectError("offline"))
    with pytest.raises(GenerationError):
        recover(tmp_path, client)
    assert p.reliability(p.records(tmp_path))["transport_failures"] == 4


def test_concurrent_pipeline_locks(tmp_path):
    script = "from pathlib import Path; from llm_forecasting.pipeline import exclusive; import sys\nwith exclusive(Path(sys.argv[1])): pass"
    with p.exclusive(tmp_path):
        result = subprocess.run(
            [__import__("sys").executable, "-B", "-c", script, str(tmp_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0 and "Another LLM" in result.stderr
        with pytest.raises(ValueError, match="Another LLM"):
            with p.exclusive(tmp_path):
                pass
    with p.exclusive(tmp_path):
        with pytest.raises(ValueError, match="Another LLM"):
            with p.exclusive(tmp_path):
                pass


def setup_root(root, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    for name in {
        *p.SOURCE_FILES,
        "llm_forecasting/reporting.py",
        "llm_forecasting/explain.py",
        "forecasting/__init__.py",
        "forecasting/api.py",
    }:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / name, destination)
    (root / "data").mkdir()
    index = pd.date_range("2026-06-01", "2026-10-01", inclusive="left", freq="h")
    pd.DataFrame(
        {
            "study_id": [f"S{i}" for i in range(len(index))],
            "arrival_time": index.strftime("%Y-%m-%d %H:%M:%S"),
            "hospital_id": "H",
            "modality": "CT",
            "specialty_required": "GENERAL",
            "criticality": "ROUTINE",
            "patient_location": "ED",
            "day_total_studies": 24,
        }
    ).to_csv(root / "data/studies.csv", index=False)
    path = root / "artifacts/selection.json"
    p.write_json(path, {"selected_candidate": "seasonal_naive"})
    monkeypatch.setattr(
        p,
        "baseline_files",
        lambda target: p.hashes(target, [target / "forecast.csv", path]),
    )
    return project


@pytest.mark.parametrize("partial", [False, True])
def test_full_187_batch_workflow_and_freeze(artifact_root, monkeypatch, partial):
    root = artifact_root
    setup_root(root, monkeypatch)

    def chat(body):
        keys = user(body)["response_dates"]
        return {
            "days": {
                key: (True if partial and len(keys) == 2 and i == 1 else 24.0)
                for i, key in enumerate(keys)
            }
        }

    client, calls = make_client(chat)
    protected = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            root / "forecast.csv",
            root / "artifacts/daily_forecast.csv",
            root / "artifacts/selection.json",
        ]
    }
    p.develop(root, client)
    assert len(calls) == (147 if partial else 78)
    protocol = p.read_json(p.output(root) / "protocol.json")
    assert protocol["expected_batches"] == 187 and protocol["maximum_attempts"] == 748
    assert len(pd.read_csv(p.output(root) / "development/metrics.csv")) == 6
    with pytest.raises(ValueError, match="Commit"):
        p.evaluate_final(root, client)

    def git(*args):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "Temporary hybrid test")
    git("config", "user.email", "test@example.invalid")
    git("add", ".")
    git("commit", "-qm", "Mock hybrid development")
    p.validate_protocol(root)
    source = root / "llm_forecasting/prompts.py"
    content = source.read_bytes()
    source.write_bytes(content + b"\n# changed\n")
    with pytest.raises(ValueError, match="Frozen file"):
        p.evaluate_final(root, client)
    source.write_bytes(content)
    p.evaluate_final(root, client)
    assert len(calls) == (300 if partial else 159)
    final = [row for row in p.records(root) if row["stage"] == "final"]
    prompts = [
        [row["request"] for row in final if row["run"] == run] for run in (1, 2, 3)
    ]
    assert prompts[0] == prompts[1] == prompts[2]
    assert (
        p.read_json(p.output(root) / "final/manifest.json")["evaluation_status"]
        == p.EXPLORATORY
    )
    for run in (1, 2, 3):
        assert (
            len(pd.read_csv(p.output(root) / f"final/2026-09-01_run{run}_hourly.csv"))
            == 720
        )
    with pytest.raises(ValueError, match="frozen"):
        p.develop(root, client)
    p.forecast(root, client)
    assert len(calls) == (352 if partial else 187)
    assert len(p.records(root)) == len(calls)
    submitted = root / p.EXPORT_NAME
    validate_hourly(pd.read_csv(submitted))
    october = pd.read_csv(p.output(root) / "october/2026-10-01_run1_daily.csv")
    assert len(october) == 31 and october.coherent.all()
    np.testing.assert_allclose(october.forecast, october.llm_total, atol=1e-6)
    assert p.read_json(p.output(root) / "october/reliability.json")["calibration"][
        "attempts"
    ] == (21 if partial else 12)
    before = submitted.read_bytes()
    p.evaluate_final(root, client)
    p.forecast(root, client)
    assert submitted.read_bytes() == before
    with pytest.raises(ValueError, match="overwrite"):
        p.export_october(
            pd.read_csv(submitted, parse_dates=["hour_start"]).set_index("hour_start"),
            submitted,
        )
    for name, checksum in protected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == checksum
    for row in p.records(root):
        context = user(row["request"])
        first = pd.Timestamp(context["calendar"][0]["date"])
        assert pd.Timestamp(context["history"]["history_end"]) < first
    stats = p.read_json(p.output(root) / "final/reliability.json")
    assert stats["combined"]["input_tokens"] == len(final) * 120
    assert all(
        window["complete_forecast_seconds"] > 0
        and window["last_invocation_seconds"] > 0
        for window in stats["windows"]
    )
    repeatability = p.read_json(p.output(root) / "final/repeatability.json")
    assert repeatability["monthly_total_range"] == 0
    assert pd.read_csv(p.output(root) / "final/metrics.csv").hourly_wape.max() < 1e-9


def test_incomplete_development_continues_all_windows_and_resumes(
    artifact_root, monkeypatch
):
    root = artifact_root
    setup_root(root, monkeypatch)
    failing = True

    def chat(body):
        keys = user(body)["response_dates"]
        return {
            "days": {
                key: (-1 if failing and key == "2026-07-07" else 24.0) for key in keys
            }
        }

    client, calls = make_client(chat)
    with pytest.raises(p.IncompleteForecastError):
        p.develop(root, client)
    assert not (p.output(root) / "protocol.json").exists()
    assert not (p.output(root) / "development").exists()
    assert any("2026-08-04" in row["key"] for row in p.records(root))
    count = len(calls)
    failing = False
    p.develop(root, client)
    assert len(calls) == count + 2  # Main July 7 and one later window's calibration.


def test_forecast_export_validation_and_protection(tmp_path):
    hourly = pd.DataFrame(
        {"forecast": 1.25, "p10": 0.25, "p90": 2.75},
        index=pd.date_range("2026-10-01", periods=744, freq="h"),
    )
    path = tmp_path / p.EXPORT_NAME
    p.export_october(hourly, path)
    assert len(validate_hourly(pd.read_csv(path))) == 744
    for frame in (hourly.iloc[:-1], hourly.assign(p10=3), hourly.assign(p90=np.inf)):
        target = tmp_path / "invalid.csv"
        with pytest.raises(ValueError):
            p.export_october(frame, target)
        assert not target.exists()


def test_existing_submission_without_manifest_stops_before_inference(
    artifact_root, monkeypatch
):
    # Run the same orchestration guards without contacting an actual provider.
    monkeypatch.setattr(p, "validate_protocol", lambda _: {})
    monkeypatch.setattr(p, "verify_manifest", lambda *_: None)
    p.write_json(p.output(artifact_root) / "final/manifest.json", {"complete": True})
    target = artifact_root / p.EXPORT_NAME
    target.write_text("preserve me")
    client, calls = make_client()
    with pytest.raises(ValueError, match="no verified manifest"):
        p.forecast(artifact_root, client)
    assert not calls and target.read_text() == "preserve me"


def test_cache_rejects_modified_raw_evidence(tmp_path):
    client, _ = make_client()
    recover(tmp_path, client)
    path = p.output(tmp_path) / "requests.jsonl"
    rows = p.records(tmp_path)
    rows[0]["text"] = '{"days":{"2026-07-07":999,"2026-07-08":999}}'
    rows[0]["day_validation"] = inspect_reply(rows[0]["text"], DATES)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="raw provider response"):
        recover(tmp_path, client)


def test_permanent_http_error_stops_before_later_batches(hourly_history, tmp_path):
    client, calls = make_client(
        lambda _: httpx.Response(401, json={"error": "unauthorized"})
    )
    with pytest.raises(GenerationError):
        p.forecast_window(
            hourly_history,
            "2026-07-21",
            2,
            1,
            "development",
            tmp_path,
            client,
            EXPERIMENT,
        )
    assert len(calls) == 4
    assert len({row["key"] for row in p.records(tmp_path)}) == 1


@pytest.mark.parametrize("content", [[], {}, {"version": 999}])
def test_malformed_protocols_are_preserved(tmp_path, content):
    path = p.output(tmp_path) / "protocol.json"
    p.write_json(path, content)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Malformed"):
        p.validate_protocol(tmp_path)
    assert path.read_bytes() == before


def prepare_report_fixtures(root):
    """Saved comparison baselines only; never copy real submission results."""
    import nbformat

    for stage in ("development", "final"):
        hybrid = pd.read_csv(p.output(root) / stage / "metrics.csv")
        classical = hybrid.loc[hybrid.run == 1].assign(candidate="seasonal_naive")
        p.write_frame(
            root / "artifacts" / stage / "metrics.csv",
            classical.set_index(["origin", "candidate", "scope"]),
        )
    shutil.copyfile(
        p.output(root) / "final/2026-09-01_run1_hourly.csv",
        root / "artifacts/final/2026-09-01_seasonal_naive_hourly.csv",
    )
    from llm_forecasting.explain import TEMPLATE, explain_demo

    sentences = [
        sentence if sentence.endswith(".") else sentence + "."
        for sentence in TEMPLATE.split(". ")
    ]
    client, _ = make_client(lambda _: {"sentences": sentences})
    explain_demo(root, client)
    preserved = nbformat.v4.new_code_cell(
        "raise AssertionError('Existing cells must not be executed')"
    )
    preserved.execution_count = 1
    preserved.outputs = [
        nbformat.v4.new_output("display_data", data={"image/png": "cHJlc2VydmVk"})
    ]
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell("Existing Part 1 charts"), preserved],
        metadata={
            "kernelspec": {
                "name": "main_env",
                "display_name": "Python (main_env)",
                "language": "python",
            }
        },
    )
    nbformat.write(notebook, root / "report.ipynb")


def test_offline_report_data_and_tagged_cell_preservation(artifact_root, monkeypatch):
    import nbformat
    import sys
    from types import SimpleNamespace
    from llm_forecasting import reporting

    root = artifact_root
    test_full_187_batch_workflow_and_freeze(root, monkeypatch, False)
    prepare_report_fixtures(root)
    monkeypatch.setattr(
        LLMClient,
        "generate",
        lambda *a, **k: pytest.fail("Reports must not call a model"),
    )
    data = reporting.saved_report_data(root)
    assert set(data["comparison"].method_source) == {
        "Part 1",
        "LLM hybrid",
    }
    assert len(data["diagnostics"]) == 6
    assert len(data["failure_examples"]) == 3
    assert data["diagnostics"].oracle_daily_volume_hourly_wape.max() < 1e-9
    original = nbformat.read(root / "report.ipynb", as_version=4)
    for cell in reporting.notebook_cells(root):
        if cell.cell_type == "code":
            compile(cell.source, "hybrid report", "exec")

    class NotebookStub:
        def __init__(self, notebook, **kwargs):
            self.nb = notebook

        def execute(self, **kwargs):
            return self.nb

    monkeypatch.setitem(
        sys.modules, "nbclient", SimpleNamespace(NotebookClient=NotebookStub)
    )
    reporting.build_report(root)
    reporting.build_report(root)
    result = nbformat.read(root / "report.ipynb", as_version=4)
    assert result.cells[:2] == original.cells
    assert len(result.cells) == 6
    assert all(reporting.TAG in cell.metadata.tags for cell in result.cells[2:])
    assert (p.output(root) / "comparison.csv").exists()


def test_promoted_development_retains_bytes_and_accepts_new_stage(
    artifact_root, monkeypatch
):
    root = artifact_root
    setup_root(root, monkeypatch)
    client, calls = make_client()
    p.develop(root, client)
    original = p.read_json(p.output(root) / "protocol.json")
    original["version"] = 1
    excluded = {
        "experiment_sha256",
        "development_files",
        "journal_bytes",
        "journal_sha256",
    }
    original["experiment_sha256"] = p.digest(
        {k: v for k, v in original.items() if k not in excluded}
    )
    rows = p.records(root)
    for row in rows:
        row.update(protocol_version=1, experiment_sha256=original["experiment_sha256"])
    journal = p.output(root) / "requests.jsonl"
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original.update(**p.journal_snapshot(root))
    original_text = json.dumps(original)
    promoted = {k: v for k, v in original.items() if k not in excluded}
    promoted["version"] = p.PROTOCOL_VERSION
    promoted["promotion"] = {
        "original_protocol": original,
        "original_protocol_text": original_text,
        "original_protocol_sha256": hashlib.sha256(original_text.encode()).hexdigest(),
    }
    promoted.update(
        experiment_sha256=p.digest(promoted),
        development_files=original["development_files"],
        **p.journal_snapshot(root),
    )
    p.write_json(p.output(root) / "protocol.json", promoted)
    before = journal.read_bytes()
    p.validate_protocol(root, committed=False)
    p.develop(root, client)
    assert journal.read_bytes() == before and len(calls) == 78
    dates = pd.date_range("2026-09-01", periods=2)
    p.recover_batch(
        {},
        dates,
        root,
        client,
        "final/2026-09-01/1/main/2026-09-01T00:00:00/2026-09-01T00:00:00",
        "final",
        "main",
        "2026-09-01",
        1,
        promoted["experiment_sha256"],
    )
    assert journal.read_bytes().startswith(before)
    recovered = p.records(root)
    assert (
        len(recovered) == 79 and recovered[-1]["protocol_version"] == p.PROTOCOL_VERSION
    )
    assert recovered[-1]["experiment_sha256"] == promoted["experiment_sha256"]
    journal.write_bytes(b" " + journal.read_bytes())
    with pytest.raises(ValueError, match="Malformed"):
        p.records(root)


def test_nonpromoted_legacy_journal_is_rejected(tmp_path):
    client, calls = make_client()
    recover(tmp_path, client)
    path = p.output(tmp_path) / "requests.jsonl"
    row = json.loads(path.read_text())
    row["protocol_version"] = 1
    path.write_text(json.dumps(row) + "\n")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Malformed"):
        recover(tmp_path, client)
    assert len(calls) == 1 and path.read_bytes() == before
