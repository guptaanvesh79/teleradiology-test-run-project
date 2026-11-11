"""Origin-safe daily contracts, seasonal proportions and empirical ranges."""

import json
import math

import numpy as np
import pandas as pd

from forecasting.artifacts import validate_bounds
from forecasting.data import day_audit


def strict_json(text):
    def constant(value):
        raise ValueError(f"Non-JSON numeric constant: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, parse_constant=constant, object_pairs_hook=unique)


BATCH_DAYS = 2
TOLERANCE = 1e-6
SYSTEM_PROMPT = """Forecast daily recorded teleradiology study arrivals from synthetic history.
Use ONLY the supplied pre-origin daily summaries and requested dates and weekdays.
Recorded zeros and sparse days are not proven outages. Do not invent holidays,
timezones, hospitals, staffing or future observations. Predict each requested daily
point total directly; Python will allocate it to hours separately.
Return ONLY strict JSON: {"days":{"requested-date":nonnegative-number}}.
Use exactly response_dates as keys, without extra fields. An initial request has
up to two dates; retries may request only one. Copy dates exactly. Do not return
hourly predictions, quantiles, explanations or a combined total."""
ALLOCATION_POLICY = {
    "history": "all recorded pre-origin observations; no sparse-day exclusion",
    "proportion": "weekday/hour volume divided by total volume of that weekday",
    "fallback": ["overall hour-of-day proportions", "uniform 1/24"],
    "point": "daily LLM point total times historical hourly proportion; not an hourly median",
}
CALIBRATION_POLICY = {
    "origin_offsets_days": [21, 14, 7],
    "window_days": 7,
    "hourly_residual_groups": "clock hour, 21 observations each",
    "daily_residuals": "21 whole-day errors, independent of hourly bounds",
    "percentiles": [0.1, 0.9],
    "bounds": "nonnegative and containing the point; empirical, no coverage guarantee",
}


def history_before(series, origin):
    origin = pd.Timestamp(origin)
    if origin.tz is not None or pd.isna(origin) or origin != origin.normalize():
        raise ValueError("Origin must be a valid timezone-naive midnight")
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.tz is not None:
        raise ValueError("History must use timezone-naive hourly timestamps")
    history = series.loc[series.index < origin].copy()
    if (
        len(history) < 14 * 24
        or not history.index.equals(
            pd.date_range(history.index[0], origin, freq="h", inclusive="left")
        )
        or history.index[0] != history.index[0].normalize()
    ):
        raise ValueError("Require at least fourteen complete pre-origin days")
    values = history.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("History must be finite and nonnegative")
    return history


def history_summary(series, origin):
    history = history_before(series, origin)
    daily = history.resample("D").sum()
    profiles = []
    for weekday, counts in daily.groupby(daily.index.day_of_week):
        profiles.append(
            {
                "weekday": int(weekday),
                "mean": float(counts.mean()),
                "median": float(counts.median()),
                "p10": float(counts.quantile(0.1)),
                "p90": float(counts.quantile(0.9)),
                "n": len(counts),
            }
        )
    audit = day_audit(history)
    return {
        "history_start": str(history.index[0]),
        "history_end": str(history.index[-1]),
        "timezone": "unspecified",
        "weekday_definition": "Monday=0, Sunday=6",
        "recent_28_daily_totals": {
            str(day.date()): float(total) for day, total in daily.tail(28).items()
        },
        "weekday_daily_profiles": profiles,
        "empty_days": int(daily.eq(0).sum()),
        "sparse_day_rule": "below 10% of positive historical weekday median; diagnostic only",
        "sparse_days": {
            str(day.date()): float(row.recorded_total)
            for day, row in audit.loc[audit.near_empty].iterrows()
        },
        "unusual_days_removed": False,
    }


def date_keys(dates):
    dates = pd.DatetimeIndex(dates)
    if (
        not 1 <= len(dates) <= BATCH_DAYS
        or dates.tz is not None
        or dates.hasnans
        or not dates.is_unique
        or not dates.is_monotonic_increasing
        or not dates.equals(dates.normalize())
    ):
        raise ValueError("Require one or two unique ordered timezone-naive dates")
    return [str(day.date()) for day in dates]


def batches(origin, days):
    if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
        raise ValueError("Forecast days must be a positive integer")
    index = pd.date_range(origin, periods=days, freq="D")
    date_keys(index[:1])
    for offset in range(0, days, BATCH_DAYS):
        yield index[offset : offset + BATCH_DAYS]


def response_schema(dates):
    return {
        "type": "object",
        "properties": {
            "days": {
                "type": "object",
                "properties": {
                    key: {"type": "number", "minimum": 0} for key in date_keys(dates)
                },
                "required": date_keys(dates),
                "additionalProperties": False,
            }
        },
        "required": ["days"],
        "additionalProperties": False,
    }


def messages(summary, full_dates, requested_dates):
    if not set(requested_dates) <= set(full_dates):
        raise ValueError("Retry dates must belong to the original batch")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "history": summary,
                    "calendar": [
                        {"date": key, "weekday": int(day.day_of_week)}
                        for key, day in zip(date_keys(full_dates), full_dates)
                    ],
                    "response_dates": date_keys(requested_dates),
                },
                allow_nan=False,
                separators=(",", ":"),
            ),
        },
    ]


def inspect_reply(text, dates):
    keys = date_keys(dates)
    value = strict_json(text)
    if (
        not isinstance(value, dict)
        or set(value) != {"days"}
        or not isinstance(value["days"], dict)
    ):
        raise ValueError("Require exactly one days object")
    if set(value["days"]) - set(keys):
        raise ValueError("Response contains an unrequested date")
    accepted, failed = {}, {}
    for key in keys:
        number = value["days"].get(key)
        try:
            valid = (
                type(number) in (int, float) and math.isfinite(number) and number >= 0
            )
        except OverflowError:
            valid = False
        if not valid:
            failed[key] = "Require a finite nonnegative numeric daily total"
        else:
            accepted[key] = float(number)
    return {"accepted_days": accepted, "failed_days": failed}


def hourly_proportions(series, origin):
    history = history_before(series, origin)
    table = history.to_frame("count")
    table["weekday"], table["hour"] = history.index.day_of_week, history.index.hour
    overall = (
        table.groupby("hour")["count"]
        .sum()
        .reindex(range(24), fill_value=0)
        .to_numpy(dtype=float)
    )
    overall = overall / overall.sum() if overall.sum() > 0 else np.full(24, 1 / 24)
    rows = []
    for weekday in range(7):
        counts = (
            table.loc[table.weekday == weekday]
            .groupby("hour")["count"]
            .sum()
            .reindex(range(24), fill_value=0)
            .to_numpy(dtype=float)
        )
        fallback = (
            "weekday" if counts.sum() else ("overall" if history.sum() else "uniform")
        )
        weights = counts / counts.sum() if counts.sum() else overall.copy()
        weights /= weights.sum()
        rows.extend(
            {
                "weekday": weekday,
                "hour": hour,
                "proportion": float(weight),
                "profile_source": fallback,
            }
            for hour, weight in enumerate(weights)
        )
    return pd.DataFrame(rows)


def allocate(daily_totals, proportions):
    dates = pd.DatetimeIndex(pd.to_datetime(list(daily_totals)))
    if (
        dates.empty
        or dates.tz is not None
        or not dates.equals(pd.date_range(dates[0], periods=len(dates), freq="D"))
    ):
        raise ValueError(
            "Daily predictions must be consecutive ordered naive midnights"
        )
    if not dates.equals(dates.normalize()):
        raise ValueError("Daily predictions must start at midnight")
    lookup = proportions.pivot(
        index="weekday", columns="hour", values="proportion"
    ).reindex(index=range(7), columns=range(24))
    values = lookup.to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or not np.allclose(values.sum(axis=1), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError(
            "Require normalized nonnegative complete weekday/hour proportions"
        )
    totals = np.array(list(daily_totals.values()), dtype=float)
    if not np.isfinite(totals).all() or (totals < 0).any():
        raise ValueError("Daily forecasts must be finite and nonnegative")
    points = (totals[:, None] * values[dates.day_of_week]).ravel()
    if not np.isfinite(points).all():
        raise ValueError("Allocated forecasts overflowed")
    hourly = pd.DataFrame(
        {"forecast": points},
        index=pd.date_range(dates[0], periods=len(points), freq="h", name="hour_start"),
    )
    if not np.allclose(
        hourly.forecast.resample("D").sum(), totals, atol=TOLERANCE, rtol=0
    ):
        raise ValueError("Allocated hours do not preserve daily totals")
    return hourly


def empirical_ranges(hourly, residuals):
    if (
        len(residuals) != 21 * 24
        or residuals.index.tz is not None
        or not residuals.index.equals(
            pd.date_range(
                hourly.index[0] - pd.Timedelta(days=21), periods=21 * 24, freq="h"
            )
        )
        or residuals.index[-1] >= hourly.index[0]
    ):
        raise ValueError("Require 21 complete pre-origin days of calibration residuals")
    if not np.isfinite(residuals).all():
        raise ValueError("Calibration errors must be finite")
    hourly = hourly.copy()
    daily = hourly.forecast.resample("D").sum().to_frame().rename_axis("date")
    groups = residuals.groupby(residuals.index.hour)
    low = hourly.index.hour.map(groups.quantile(0.1)).to_numpy()
    high = hourly.index.hour.map(groups.quantile(0.9)).to_numpy()
    hourly["p10"] = np.maximum(0, hourly.forecast + np.minimum(0, low))
    hourly["p90"] = hourly.forecast + np.maximum(0, high)
    daily_errors = residuals.resample("D").sum()
    daily["p10"] = np.maximum(0, daily.forecast + min(0, daily_errors.quantile(0.1)))
    daily["p90"] = daily.forecast + max(0, daily_errors.quantile(0.9))
    validate_bounds(hourly)
    validate_bounds(daily)
    return hourly, daily
