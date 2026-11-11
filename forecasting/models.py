"""Forecast whole horizons using only observations before their origin."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from .config import CALIBRATION_WEEKS, PARAMETERS, Candidate
from .data import day_audit
from .features import calendar_features


def training_mask(history, candidate):
    mask = pd.Series(True, index=history.index)
    if candidate.exclude_sparse:
        audit = day_audit(history)
        excluded = audit.index[audit.near_empty]
        mask.loc[history.index.normalize().isin(excluded)] = False
    return mask


def validate_history(history, future):
    if len(history) < 14 * 24:
        raise ValueError("At least 14 complete days of history are required")
    if not history.index.equals(
        pd.date_range(history.index[0], periods=len(history), freq="h")
    ):
        raise ValueError("History must be a complete, ordered hourly series")
    if not np.isfinite(history).all() or (history < 0).any():
        raise ValueError("History counts must be finite and nonnegative")
    expected = pd.date_range(
        history.index[-1] + pd.Timedelta(hours=1), periods=len(future), freq="h"
    )
    if len(future) == 0 or not future.equals(expected):
        raise ValueError("Future must be consecutive hours beginning after history")


def fit_predict(history, future, candidate: Candidate):
    """Return point/direct quantile predictions; interval calibration is separate."""
    future = pd.DatetimeIndex(future)
    validate_history(history, future)
    if candidate.family == "naive":
        point = np.resize(history.iloc[-168:].to_numpy(dtype=float), len(future))
        values = np.column_stack([point, point, point])
        crossings = np.zeros(len(future), dtype=bool)
    else:
        mask = training_mask(history, candidate)
        x_train = calendar_features(history.index[mask])
        y_train = history.loc[mask].to_numpy(dtype=float)
        x_future = calendar_features(future)
        if not len(y_train):
            raise ValueError(
                "No training observations remain after sparse-day filtering"
            )
        with threadpool_limits(limits=1):
            if not y_train.any():
                values = np.zeros((len(future), 3))
            elif candidate.family == "poisson":
                model = HistGradientBoostingRegressor(loss="poisson", **PARAMETERS)
                point = model.fit(x_train, y_train).predict(x_future)
                values = np.column_stack([point, point, point])
            elif candidate.family == "quantile":
                values = np.column_stack(
                    [
                        HistGradientBoostingRegressor(
                            loss="quantile", quantile=q, **PARAMETERS
                        )
                        .fit(x_train, y_train)
                        .predict(x_future)
                        for q in (0.1, 0.5, 0.9)
                    ]
                )
            else:
                raise ValueError(f"Unknown model family: {candidate.family}")
        crossings = (values[:, 0] > values[:, 1]) | (values[:, 1] > values[:, 2])
        values = np.sort(np.maximum(values, 0), axis=1)
    return pd.DataFrame(
        {
            "forecast": values[:, 1],
            "p10": values[:, 0],
            "p90": values[:, 2],
            "crossed_before_sort": crossings,
        },
        index=future,
    ).rename_axis("hour_start")


def calibration_residuals(history, candidate):
    """Three disjoint prior weeks; each forecast uses its own preceding history."""
    origin = history.index[-1] + pd.Timedelta(hours=1)
    pieces = []
    for weeks in range(CALIBRATION_WEEKS, 0, -1):
        inner_origin = origin - pd.Timedelta(weeks=weeks)
        future = pd.date_range(inner_origin, periods=168, freq="h")
        prediction = fit_predict(
            history.loc[history.index < inner_origin], future, candidate
        )
        pieces.append(
            (history.reindex(future) - prediction.forecast).rename("residual")
        )
    residuals = pd.concat(pieces)
    if residuals.isna().any() or residuals.index.max() >= origin:
        raise ValueError("Calibration must contain only observed pre-origin hours")
    return residuals


def ordered_band(point, low, high):
    """Keep nonnegative ranges containing the supplied point estimate."""
    return np.maximum(0, np.minimum(point, low)), np.maximum(point, high)


def forecast_candidate(history, future, candidate):
    prediction = fit_predict(history, future, candidate)
    residuals = calibration_residuals(history, candidate)
    if candidate.family != "quantile":
        grouped = residuals.groupby(residuals.index.hour)
        low = prediction.index.hour.map(grouped.quantile(0.1)).to_numpy()
        high = prediction.index.hour.map(grouped.quantile(0.9)).to_numpy()
        prediction["p10"], prediction["p90"] = ordered_band(
            prediction.forecast, prediction.forecast + low, prediction.forecast + high
        )
    daily = prediction.forecast.resample("D").sum().to_frame()
    daily_residuals = residuals.resample("D").sum()
    daily["p10"], daily["p90"] = ordered_band(
        daily.forecast,
        daily.forecast + daily_residuals.quantile(0.1),
        daily.forecast + daily_residuals.quantile(0.9),
    )
    daily.index.name = "date"
    return prediction, daily


def run_origin(series, origin, days, candidate):
    """The sole outer split: later observations cannot enter fitting/calibration."""
    origin = pd.Timestamp(origin)
    history = series.loc[series.index < origin].copy()
    future = pd.date_range(origin, periods=days * 24, freq="h", name="hour_start")
    return forecast_candidate(history, future, candidate)
