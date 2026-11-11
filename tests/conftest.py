"""Small synthetic inputs; all writable artifacts live in pytest's temp directory."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def isolated_llm_environment(monkeypatch):
    """Tests never inherit a user's real provider configuration or API key."""
    for name in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def study_frame():
    return pd.DataFrame(
        {
            "study_id": ["S1", "S2", "S3"],
            "arrival_time": [
                "2026-06-01 01:10:00",
                "2026-06-01 01:20:00",
                "2026-06-03 23:59:00",
            ],
            "hospital_id": ["H01"] * 3,
            "modality": ["CT"] * 3,
            "specialty_required": ["GENERAL"] * 3,
            "criticality": ["ROUTINE"] * 3,
            "patient_location": ["ED"] * 3,
            "day_total_studies": [2, 2, 1],
        }
    )


@pytest.fixture
def hourly_history():
    index = pd.date_range("2026-06-01", periods=70 * 24, freq="h", name="hour_start")
    counts = 10 + index.hour.to_numpy() + 5 * (index.day_of_week.to_numpy() < 5)
    counts = np.where((index.hour >= 1) & (index.hour <= 17), counts, 0)
    history = pd.Series(counts, index=index, name="actual")
    history.loc["2026-07-08"] = 0
    history.loc["2026-07-09"] = 1
    return history


@pytest.fixture
def artifact_root(tmp_path):
    index = pd.date_range(
        "2026-10-01", "2026-11-01", inclusive="left", freq="h", name="hour_start"
    )
    point = 10.0 + index.day + index.hour / 2
    hourly = pd.DataFrame(
        {"forecast": point, "p10": point / 2, "p90": point * 1.5}, index=index
    )
    hourly_path = tmp_path / "forecast.csv"
    hourly.to_csv(hourly_path, date_format="%Y-%m-%d %H:%M:%S")
    output = tmp_path / "artifacts"
    output.mkdir()
    daily = hourly.forecast.resample("D").sum().to_frame().rename_axis("date")
    # Daily intervals deliberately differ from sums of hourly quantiles.
    daily["p10"] = daily.forecast - 36
    daily["p90"] = daily.forecast + 72
    daily_path = output / "daily_forecast.csv"
    daily.to_csv(daily_path, date_format="%Y-%m-%d")
    metadata = {
        "model": "seasonal_naive",
        "forecast_origin": "2026-10-01",
        "timezone": "unspecified",
        "point_estimate": "hourly_point_forecast",
        "daily_point_estimate": "sum_of_hourly_point_forecasts",
        "hourly_interval": "hour-conditioned residual percentiles",
        "daily_interval": "whole-day residual percentiles",
        "forecast_csv_sha256": hashlib.sha256(hourly_path.read_bytes()).hexdigest(),
        "daily_csv_sha256": hashlib.sha256(daily_path.read_bytes()).hexdigest(),
    }
    (output / "forecast_metadata.json").write_text(json.dumps(metadata))
    return tmp_path
