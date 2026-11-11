"""HTTP behaviour and saved-forecast contracts, without a running server."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from forecasting.api import create_app, load_artifacts
from forecasting.artifacts import validate_daily, validate_hourly


def assert_rows_match(rows, expected, key):
    assert len(rows) == len(expected)
    received = pd.DataFrame(rows)
    expected_keys = expected[key].astype(str)
    if key == "hour_start":
        expected_keys = pd.to_datetime(expected_keys).dt.strftime("%Y-%m-%dT%H:%M:%S")
    assert received[key].tolist() == expected_keys.tolist()
    for column in ("forecast", "p10", "p90"):
        np.testing.assert_allclose(
            received[column], expected[column], rtol=0, atol=1e-9
        )


def test_health_and_full_forecasts_match_saved_artifacts(artifact_root):
    with TestClient(create_app(artifact_root)) as client:
        assert client.get("/health").json() == {
            "status": "ready",
            "hours": 744,
            "days": 31,
        }
        hourly = client.get("/forecast/hourly")
        daily = client.get("/forecast/daily")
        assert hourly.status_code == daily.status_code == 200
        assert_rows_match(
            hourly.json()["data"],
            pd.read_csv(artifact_root / "forecast.csv"),
            "hour_start",
        )
        assert_rows_match(
            daily.json()["data"],
            pd.read_csv(artifact_root / "artifacts/daily_forecast.csv"),
            "date",
        )
        saved = json.loads(
            (artifact_root / "artifacts/forecast_metadata.json").read_text()
        )
        expected_metadata = {
            name: saved[name]
            for name in (
                "model",
                "forecast_origin",
                "timezone",
                "point_estimate",
                "daily_point_estimate",
                "hourly_interval",
                "daily_interval",
            )
        }
        assert (
            hourly.json()["metadata"] == daily.json()["metadata"] == expected_metadata
        )
        assert sum(row["forecast"] for row in hourly.json()["data"]) == sum(
            row["forecast"] for row in daily.json()["data"]
        )


@pytest.mark.parametrize(
    "endpoint,key,filename,rows_per_day",
    [
        ("hourly", "hour_start", "forecast.csv", 24),
        ("daily", "date", "artifacts/daily_forecast.csv", 1),
    ],
)
@pytest.mark.parametrize(
    "query,start,end",
    [
        ("start_date=2026-10-01&end_date=2026-10-01", "2026-10-01", "2026-10-01"),
        ("start_date=2026-10-31&end_date=2026-10-31", "2026-10-31", "2026-10-31"),
        ("start_date=2026-10-25&end_date=2026-10-31", "2026-10-25", "2026-10-31"),
        ("end_date=2026-10-02", "2026-10-01", "2026-10-02"),
        ("start_date=2026-10-30", "2026-10-30", "2026-10-31"),
    ],
)
def test_inclusive_date_filters_and_default_bounds(
    artifact_root, endpoint, key, filename, rows_per_day, query, start, end
):
    expected = pd.read_csv(artifact_root / filename)
    dates = pd.to_datetime(expected[key]).dt.normalize()
    expected = expected.loc[dates.between(pd.Timestamp(start), pd.Timestamp(end))]
    with TestClient(create_app(artifact_root)) as client:
        response = client.get(f"/forecast/{endpoint}?{query}")
        assert response.status_code == 200
        assert (
            len(response.json()["data"])
            == ((pd.Timestamp(end) - pd.Timestamp(start)).days + 1) * rows_per_day
        )
        assert_rows_match(response.json()["data"], expected, key)


@pytest.mark.parametrize("endpoint", ["hourly", "daily"])
@pytest.mark.parametrize(
    "query,status",
    [
        ("start_date=invalid", 422),
        ("end_date=2026-10-32", 422),
        ("start_date=2026-10-20&end_date=2026-10-10", 422),
        ("start_date=2026-09-30", 404),
        ("end_date=2026-11-01", 404),
        ("start_date=2026-11-01&end_date=2026-11-03", 404),
    ],
)
def test_date_validation_returns_sensible_errors(
    artifact_root, endpoint, query, status
):
    with TestClient(create_app(artifact_root)) as client:
        response = client.get(f"/forecast/{endpoint}?{query}")
        assert response.status_code == status
        assert response.json()["detail"]


@pytest.mark.parametrize(
    "problem",
    [
        "missing_hour",
        "duplicate_hour",
        "reordered_hours",
        "timestamp_format",
        "negative",
        "nan",
        "infinity",
        "lower_crossing",
        "upper_crossing",
        "extra_column",
    ],
)
def test_invalid_hourly_forecasts_rejected(artifact_root, problem):
    frame = pd.read_csv(artifact_root / "forecast.csv")
    if problem == "missing_hour":
        frame = frame.iloc[:-1]
    elif problem == "duplicate_hour":
        frame.loc[1, "hour_start"] = frame.loc[0, "hour_start"]
    elif problem == "reordered_hours":
        frame = frame.iloc[::-1]
    elif problem == "timestamp_format":
        frame.loc[0, "hour_start"] = "2026-10-01T00:00:00Z"
    elif problem == "extra_column":
        frame["unexpected"] = 1
    else:
        column, value = {
            "negative": ("p10", -1),
            "nan": ("forecast", np.nan),
            "infinity": ("p90", np.inf),
            "lower_crossing": ("p10", frame.loc[0, "forecast"] + 1),
            "upper_crossing": ("p90", frame.loc[0, "forecast"] - 1),
        }[problem]
        frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_hourly(frame)


def test_daily_totals_and_independent_ranges(artifact_root):
    hourly = validate_hourly(pd.read_csv(artifact_root / "forecast.csv"))
    frame = pd.read_csv(artifact_root / "artifacts/daily_forecast.csv")
    daily = validate_daily(frame, hourly)
    np.testing.assert_allclose(daily.forecast, hourly.forecast.resample("D").sum())
    assert not np.allclose(daily.p10, hourly.p10.resample("D").sum())
    frame.loc[0, "forecast"] += 1
    with pytest.raises(ValueError, match="must equal sums"):
        validate_daily(frame, hourly)
    with pytest.raises(ValueError, match="31 ordered"):
        validate_daily(frame.iloc[:-1], hourly)


@pytest.mark.parametrize(
    "problem", ["missing", "modified_hourly", "modified_daily", "invalid_metadata"]
)
def test_invalid_service_artifacts_fail_startup(artifact_root, problem):
    root = artifact_root
    message = "missing or invalid"
    if problem == "missing":
        root = artifact_root / "not-created"
    elif problem in {"modified_hourly", "modified_daily"}:
        path = artifact_root / (
            "forecast.csv"
            if problem == "modified_hourly"
            else "artifacts/daily_forecast.csv"
        )
        frame = pd.read_csv(path)
        frame.loc[0, "p10"] /= (
            2  # Keep bounds/totals valid to isolate checksum detection.
        )
        frame.to_csv(path, index=False)
        message = "checksum mismatch"
    else:
        (artifact_root / "artifacts/forecast_metadata.json").write_text("not JSON")
    with pytest.raises(RuntimeError, match=message), TestClient(create_app(root)):
        pass


def test_submitted_forecast_contract_read_only():
    root = Path(__file__).resolve().parents[1]
    hourly, daily, metadata = load_artifacts(root)
    assert len(hourly) == 744 and len(daily) == 31
    assert hourly.index[0] == pd.Timestamp("2026-10-01")
    assert hourly.index[-1] == pd.Timestamp("2026-10-31 23:00:00")
    assert metadata.forecast_origin.isoformat() == "2026-10-01"
