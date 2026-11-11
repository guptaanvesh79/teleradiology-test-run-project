"""Metrics use original validation counts, including empty and abnormal days."""

import numpy as np
import pandas as pd


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def pinball(actual, prediction, quantile):
    error = np.asarray(actual) - np.asarray(prediction)
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def summarize(actual, prediction, daily, quantile_model=False):
    if not actual.index.equals(prediction.index) or actual.isna().any():
        raise ValueError(
            "Actuals and predictions must align without missing observations"
        )
    daily_actual = actual.resample("D").sum()
    daily = daily.reindex(daily_actual.index)
    actual_total = float(actual.sum())
    forecast_total = float(prediction.forecast.sum())
    error = forecast_total - actual_total
    result = {
        "hours": len(actual),
        "days": len(daily_actual),
        "actual_total": actual_total,
        "forecast_total": forecast_total,
        "hourly_wape": fraction(
            (prediction.forecast - actual).abs().sum(), actual_total
        ),
        "daily_wape": fraction(
            (daily.forecast - daily_actual).abs().sum(), actual_total
        ),
        "bias": fraction(error, actual_total),
        "total_error": error,
        "absolute_total_error": abs(error),
        "absolute_total_error_fraction": fraction(abs(error), actual_total),
        "coverage": float(
            ((actual >= prediction.p10) & (actual <= prediction.p90)).mean()
        ),
        "mean_interval_width": float((prediction.p90 - prediction.p10).mean()),
        "daily_coverage": float(
            ((daily_actual >= daily.p10) & (daily_actual <= daily.p90)).mean()
        ),
        "mean_daily_interval_width": float((daily.p90 - daily.p10).mean()),
        "quantile_crossing_rate": float(prediction.crossed_before_sort.mean()),
    }
    for name, q in (("p10", 0.1), ("forecast", 0.5), ("p90", 0.9)):
        result[f"pinball_{int(q * 100)}"] = (
            pinball(actual, prediction[name], q) if quantile_model else None
        )
    return result


def evaluate(actual, prediction, daily, quantile_model=False):
    last_week = prediction.index[-1].normalize() - pd.Timedelta(days=6)
    return {
        "all": summarize(actual, prediction, daily, quantile_model),
        "last_7_days": summarize(
            actual.loc[last_week:],
            prediction.loc[last_week:],
            daily.loc[last_week:],
            quantile_model,
        ),
    }
