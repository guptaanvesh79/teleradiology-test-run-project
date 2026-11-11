"""Shared validation for the submission CSV and saved daily forecast."""

import numpy as np
import pandas as pd

from .config import FORECAST_END, FORECAST_ORIGIN

HOURLY_COLUMNS = ["hour_start", "forecast", "p10", "p90"]
DAILY_COLUMNS = ["date", "forecast", "p10", "p90"]


def validate_bounds(frame):
    try:
        values = frame[["forecast", "p10", "p90"]].to_numpy(dtype=float)
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError("Forecast values must be numeric") from error
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Forecasts and bounds must be finite and nonnegative")
    if not ((values[:, 1] <= values[:, 0]) & (values[:, 0] <= values[:, 2])).all():
        raise ValueError("Require p10 <= forecast <= p90 in every row")


def validate_hourly(frame):
    if list(frame.columns) != HOURLY_COLUMNS:
        raise ValueError(f"Hourly CSV columns must be {HOURLY_COLUMNS}")
    strings = frame.hour_start.astype(str)
    if not strings.str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}").all():
        raise ValueError("hour_start must use YYYY-MM-DD HH:MM:SS without a timezone")
    try:
        index = pd.DatetimeIndex(
            pd.to_datetime(strings, format="%Y-%m-%d %H:%M:%S", errors="raise")
        )
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid hourly timestamp") from error
    expected = pd.date_range(FORECAST_ORIGIN, FORECAST_END, freq="h", inclusive="left")
    if len(frame) != 744 or not index.equals(expected):
        raise ValueError(
            "Require exactly 744 ordered, unique, consecutive October hours"
        )
    validate_bounds(frame)
    return frame.set_index(index)[["forecast", "p10", "p90"]].rename_axis("hour_start")


def validate_daily(frame, hourly):
    if list(frame.columns) != DAILY_COLUMNS:
        raise ValueError(f"Daily CSV columns must be {DAILY_COLUMNS}")
    try:
        index = pd.DatetimeIndex(
            pd.to_datetime(frame.date, format="%Y-%m-%d", errors="raise")
        )
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid daily date") from error
    expected = pd.date_range(FORECAST_ORIGIN, FORECAST_END, freq="D", inclusive="left")
    if not index.equals(expected):
        raise ValueError("Require exactly 31 ordered October days")
    validate_bounds(frame)
    result = frame.set_index(index)[["forecast", "p10", "p90"]].rename_axis("date")
    if not np.allclose(
        result.forecast, hourly.forecast.resample("D").sum(), rtol=0, atol=1e-6
    ):
        raise ValueError(
            "Daily point forecasts must equal sums of saved hourly point forecasts"
        )
    return result
