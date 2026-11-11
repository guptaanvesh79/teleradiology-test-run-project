"""Validate the original file and construct recorded-arrival counts."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_END, DATA_START, SPARSE_DAY_RATIO

REQUIRED_COLUMNS = (
    "study_id",
    "arrival_time",
    "hospital_id",
    "modality",
    "specialty_required",
    "criticality",
    "patient_location",
    "day_total_studies",
)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_studies(path):
    frame = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame.empty or frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(
            "The studies file must be nonempty with no missing required values"
        )
    if frame.study_id.duplicated().any():
        raise ValueError("Duplicate study_id values")
    try:
        timestamps = pd.to_datetime(
            frame.arrival_time, format="%Y-%m-%d %H:%M:%S", errors="raise"
        )
    except (ValueError, TypeError) as error:
        raise ValueError(
            "Invalid arrival_time; expected timezone-naive YYYY-MM-DD HH:MM:SS"
        ) from error
    if timestamps.dt.tz is not None:
        raise ValueError(
            "Timezone-aware arrivals are not supported; preserve the source clock"
        )
    if not timestamps.between(DATA_START, DATA_END, inclusive="left").all():
        raise ValueError("Arrival timestamp outside June 1 through September 30, 2026")
    totals = pd.to_numeric(frame.day_total_studies, errors="coerce")
    if (
        not np.isfinite(totals).all()
        or (totals < 0).any()
        or (totals != np.floor(totals)).any()
    ):
        raise ValueError("day_total_studies must contain nonnegative integer counts")
    frame["arrival_time"] = timestamps
    frame["day_total_studies"] = totals.astype("int64")
    return frame


def hourly_counts(frame, start=DATA_START, end=DATA_END):
    """Count rows on [start, end), including missing boundary hours as zeros."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start >= end or start != start.floor("h") or end != end.floor("h"):
        raise ValueError("Require an increasing pair of whole-hour boundaries")
    arrivals = frame.loc[
        frame.arrival_time.between(start, end, inclusive="left"), "arrival_time"
    ]
    index = pd.date_range(start, end, freq="h", inclusive="left", name="hour_start")
    return (
        arrivals.groupby(arrivals.dt.floor("h"))
        .size()
        .reindex(index, fill_value=0)
        .rename("actual")
    )


def day_audit(history):
    """An origin-local diagnostic; no classification asserts a known outage."""
    totals = history.resample("D").sum()
    medians = totals.groupby(totals.index.day_of_week).median()
    reference = pd.Series(totals.index.day_of_week.map(medians), index=totals.index)
    return pd.DataFrame(
        {
            "recorded_total": totals,
            "weekday_median": reference,
            "sparse_threshold": SPARSE_DAY_RATIO * reference,
            "near_empty": (reference > 0) & (totals < SPARSE_DAY_RATIO * reference),
            "empty_hours": history.eq(0).resample("D").sum(),
        }
    ).rename_axis("date")


def source_consistency(frame):
    """Historical diagnostic only: never a feature or repaired target."""
    grouped = frame.groupby(frame.arrival_time.dt.normalize())
    totals = grouped.day_total_studies.agg(["first", "nunique"])
    counts = grouped.size()
    return {
        "days_with_inconsistent_source_totals": int((totals["nunique"] != 1).sum()),
        "days_source_total_differs_from_rows": int((totals["first"] != counts).sum()),
    }
