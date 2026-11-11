"""Calendar inputs, recorded-count preparation, and origin-safe forecasting."""

import numpy as np
import pandas as pd
import pytest

from forecasting.config import CANDIDATES
from forecasting.data import day_audit, hourly_counts, load_studies, source_consistency
from forecasting.features import calendar_features
from forecasting.models import run_origin, training_mask


def test_known_calendar_features():
    index = pd.DatetimeIndex(["2026-06-01 00:00:00", "2026-10-03 06:00:00"])
    features = calendar_features(index)
    assert list(features.columns) == [
        "hour",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "days_since_start",
    ]
    pd.testing.assert_index_equal(features.index, index)
    assert features.hour.tolist() == [0, 6]
    assert features.day_of_week.tolist() == [0, 5]
    assert features.is_weekend.tolist() == [0, 1]
    assert features.days_since_start.tolist() == [0, 124]
    assert features.iloc[0].hour_sin == pytest.approx(0)
    assert features.iloc[0].hour_cos == pytest.approx(1)
    assert features.iloc[0].weekday_sin == pytest.approx(0)
    assert features.iloc[0].weekday_cos == pytest.approx(1)
    assert features.iloc[1].hour_sin == pytest.approx(1)
    assert features.iloc[1].hour_cos == pytest.approx(0, abs=1e-12)


def test_cyclical_features_connect_midnight_and_week_boundaries():
    index = pd.DatetimeIndex(
        ["2026-06-07 23:00:00", "2026-06-08 00:00:00", "2026-06-11 12:00:00"]
    )
    features = calendar_features(index)
    for columns in (("hour_sin", "hour_cos"), ("weekday_sin", "weekday_cos")):
        positions = features[list(columns)].to_numpy()
        np.testing.assert_allclose((positions**2).sum(axis=1), 1)
        assert np.linalg.norm(positions[0] - positions[1]) < np.linalg.norm(
            positions[0] - positions[2]
        )


@pytest.mark.parametrize(
    "index",
    [
        pd.DatetimeIndex([pd.NaT]),
        pd.date_range("2026-10-01", periods=2, tz="UTC"),
        ["not-a-timestamp"],
    ],
)
def test_invalid_feature_timestamps_rejected(index):
    with pytest.raises(ValueError):
        calendar_features(index)


def test_hourly_counts_include_empty_hours_and_whole_days(study_frame, tmp_path):
    path = tmp_path / "studies.csv"
    study_frame.to_csv(path, index=False)
    frame = load_studies(path)
    counts = hourly_counts(frame)
    assert len(counts) == 2928
    assert counts.index[0] == pd.Timestamp("2026-06-01")
    assert counts.index[-1] == pd.Timestamp("2026-09-30 23:00:00")
    assert counts.iloc[0] == counts.iloc[-1] == 0
    assert counts.loc["2026-06-01 01:00:00"] == 2
    assert counts.loc["2026-06-02"].eq(0).all()
    assert counts.loc["2026-06-03 23:00:00"] == 1
    assert counts.sum() == len(study_frame)
    assert source_consistency(frame) == {
        "days_with_inconsistent_source_totals": 0,
        "days_source_total_differs_from_rows": 0,
    }


@pytest.mark.parametrize(
    "problem",
    [
        "missing_column",
        "duplicate_id",
        "missing_value",
        "bad_timestamp",
        "timezone",
        "outside_history",
        "negative_total",
        "fractional_total",
        "infinite_total",
        "empty_file",
    ],
)
def test_invalid_study_data_rejected(study_frame, tmp_path, problem):
    frame = study_frame.copy()
    if problem == "missing_column":
        frame = frame.drop(columns="modality")
    elif problem == "duplicate_id":
        frame.loc[1, "study_id"] = frame.loc[0, "study_id"]
    elif problem == "missing_value":
        frame.loc[0, "hospital_id"] = None
    elif problem == "bad_timestamp":
        frame.loc[0, "arrival_time"] = "invalid"
    elif problem == "timezone":
        frame.loc[0, "arrival_time"] = "2026-06-01 01:10:00+00:00"
    elif problem == "outside_history":
        frame.loc[0, "arrival_time"] = "2026-10-01 00:00:00"
    elif problem == "empty_file":
        frame = frame.iloc[:0]
    else:
        frame["day_total_studies"] = frame.day_total_studies.astype(float)
        frame.loc[0, "day_total_studies"] = {
            "negative_total": -1,
            "fractional_total": 1.5,
            "infinite_total": np.inf,
        }[problem]
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_studies(path)


def test_sparse_days_use_training_weekday_medians(hourly_history):
    history = hourly_history.loc[:"2026-07-20 23:00:00"].copy()
    history.loc["2026-07-06"] *= 4
    audit = day_audit(history)
    assert audit.loc["2026-07-08", "weekday_median"] == 408
    assert audit.loc["2026-07-08", "sparse_threshold"] == pytest.approx(40.8)
    assert audit.loc["2026-07-08", "near_empty"]
    assert audit.loc["2026-07-09", "near_empty"]
    assert not audit.loc["2026-07-06", "near_empty"]
    candidate = next(c for c in CANDIDATES if c.name == "quantile_filtered")
    mask = training_mask(history, candidate)
    assert not mask.loc["2026-07-08"].any()
    assert not mask.loc["2026-07-09"].any()
    assert mask.loc["2026-07-06"].all()
    raw = next(c for c in CANDIDATES if c.name == "quantile_raw")
    assert training_mask(history, raw).all()


def test_zero_weekday_median_does_not_remove_all_training_rows(hourly_history):
    history = hourly_history.iloc[: 35 * 24] * 0
    candidate = next(c for c in CANDIDATES if c.name == "quantile_filtered")
    assert not day_audit(history).near_empty.any()
    assert training_mask(history, candidate).all()


def test_future_counts_cannot_change_predictions_or_calibration(
    hourly_history, monkeypatch
):
    from forecasting.models import PARAMETERS

    # Isolated synthetic model fits are small; production parameters/files stay unchanged.
    monkeypatch.setitem(PARAMETERS, "max_iter", 8)
    origin = pd.Timestamp("2026-07-21")
    changed = hourly_history.copy()
    changed.loc[changed.index >= origin] = 1000000
    candidate = next(c for c in CANDIDATES if c.name == "quantile_filtered")
    first, first_daily = run_origin(hourly_history, origin, 28, candidate)
    second, second_daily = run_origin(changed, origin, 28, candidate)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_daily, second_daily)
    assert len(first) == 672 and len(first_daily) == 28
    assert np.isfinite(first[["forecast", "p10", "p90"]]).all().all()
    assert (first.p10 >= 0).all()
    assert (first.p10 <= first.forecast).all() and (first.forecast <= first.p90).all()
    np.testing.assert_allclose(first_daily.forecast, first.forecast.resample("D").sum())
